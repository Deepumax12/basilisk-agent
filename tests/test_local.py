"""The agent loop with a free local model (mock Ollama server) and an OpenAI-compatible server (mock)."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AGENT = os.environ.get("BA_AGENT") or str(Path(__file__).resolve().parents[1] / "basilisk_agent.py")
ROOT = Path(tempfile.mkdtemp(prefix="ba-local-"))
HOME, WS, FAKE = ROOT / "home", ROOT / "ws", ROOT / "basilisk" / "src"
for d in (HOME, WS, FAKE / "examples"):
    d.mkdir(parents=True, exist_ok=True)
(FAKE / "qcc").write_text('#!/bin/bash\nexec gcc "$@"\n')
(FAKE / "qcc").chmod(0o755)
FAILS, OKS = [], []


def expect(name, cond, detail=""):
    (OKS if cond else FAILS).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else " -> " + str(detail)[:500]), flush=True)


SIM = r'''#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main (int argc, char * argv[]) {
  int n = argc > 1 ? atoi (argv[1]) : 10;
  fprintf (stderr, "# i t ke\n");
  for (int i = 0; i <= n; i++) { fprintf (stderr, "%d %g %g\n", i, 0.1*i, 1./(1. + 0.01*i*i)); usleep (30000); }
  return 0;
}
'''
PLOT = ("import numpy as np, matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt, glob\n"
        "d = np.loadtxt(sorted(glob.glob('.runs/*/stderr.log'))[-1], comments='#')\n"
        "plt.plot(d[:,1], d[:,2]); plt.savefig('ke.png', dpi=80); print('final', d[-1,2])\n")

STATE = {"scenario": None, "step": 0, "run_id": None, "caps": ["completion", "tools", "vision"]}


def tool_msgs(body):
    return [m for m in body["messages"] if m["role"] == "tool"]


# ---------------- Ollama scenario ----------------
def o1(body):
    expect("ollama: num_ctx sent", body["options"]["num_ctx"] == 32768, body.get("options"))
    expect("ollama: tools in function format", len(body["tools"]) == 15 and body["tools"][0]["type"] == "function"
           and "parameters" in body["tools"][0]["function"], body["tools"][0])
    expect("ollama: system prompt with tool-use addendum", body["messages"][0]["role"] == "system"
           and "Working with tools" in body["messages"][0]["content"])
    expect("ollama: task in user message", "cavity" in body["messages"][1]["content"]
           and "<session_context>" in body["messages"][1]["content"])
    return {"content": "Checking the machine.", "tool_calls": [{"function": {"name": "environment", "arguments": {}}}]}


def o2(body):
    asst = [m for m in body["messages"] if m["role"] == "assistant"][-1]
    tid = asst["tool_calls"][0]["id"]
    expect("ollama: generated tool-call id passed back", tid.startswith("call_"), asst)
    tm = tool_msgs(body)[-1]
    expect("ollama: tool result message", tm["tool_call_id"] == tid and tm["tool_name"] == "environment"
           and "BASILISK" in tm["content"], tm)
    text_call = json.dumps({"name": "write_file", "arguments": {"path": "case/sim.c", "content": SIM}})
    return {"content": "I will write the code.\n<tool_call>\n%s\n</tool_call>" % text_call}


def o3(body):
    tm = tool_msgs(body)[-1]
    expect("ollama: tool call written as text was recovered", "Wrote" in tm["content"], tm)
    return {"content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": {"command": "qcc -O2 sim.c -o sim -lm && echo built", "cwd": "case"}}},
        {"function": {"name": "sim_start", "arguments": {"command": "./sim 15", "cwd": "case", "label": "local"}}}]}


def o4(body):
    tms = tool_msgs(body)
    expect("ollama: compiled", "built" in tms[-2]["content"], tms[-2])
    m = re.search(r"Started run id: (\S+)", tms[-1]["content"])
    expect("ollama: run started", m is not None, tms[-1])
    STATE["run_id"] = m.group(1) if m else "x"
    return {"content": "", "tool_calls": [{"function": {"name": "sim_wait", "arguments": {
        "run_id": STATE["run_id"], "max_minutes": 1, "poll_seconds": 5}}}]}


def o5(body):
    expect("ollama: run finished", "[finished]" in tool_msgs(body)[-1]["content"], tool_msgs(body)[-1])
    return {"content": "", "tool_calls": [
        {"function": {"name": "write_file", "arguments": {"path": "case/plot.py", "content": PLOT}}},
        {"function": {"name": "bash", "arguments": {"command": "python3 plot.py", "cwd": "case"}}},
        {"function": {"name": "view_image", "arguments": {"path": "case/ke.png"}}}]}


def o6(body):
    imgs = [m for m in body["messages"] if m["role"] == "user" and m.get("images")]
    if STATE["caps"] and "vision" in STATE["caps"]:
        expect("ollama vision: image sent to the model", len(imgs) == 1 and len(imgs[0]["images"][0]) > 500, imgs)
    else:
        expect("ollama no-vision: no image sent", not imgs)
        expect("ollama no-vision: told to check numerically", "cannot see images" in tool_msgs(body)[-1]["content"],
               tool_msgs(body)[-1])
    return {"content": "Done. KE decays; see case/ke.png."}


# ---------------- Ollama compaction scenario (adaptive) ----------------
COMPACT = {"summary_seen": False, "calls": 0}


def c_any(body):
    if "tools" not in body:
        COMPACT["summary_seen"] = True
        expect("local compaction: transcript sent", "<transcript>" in body["messages"][-1]["content"])
        return {"content": "SUMMARY: printed number sequences; next: finish."}
    users = [m for m in body["messages"] if m["role"] == "user"]
    if any("SUMMARY: printed number sequences" in m["content"] for m in users):
        expect("local compaction: fresh history with summary", len(body["messages"]) == 2,
               [m["role"] for m in body["messages"]])
        return {"content": "Finished."}
    COMPACT["calls"] += 1
    if COMPACT["calls"] > 10:
        return {"content": "Giving up (no compaction happened)."}
    n = COMPACT["calls"]
    return {"content": "", "tool_calls": [{"function": {"name": "bash", "arguments": {
        "command": "seq %d %d" % (n * 10000, n * 10000 + 3000)}}}]}


SCEN = {"ollama": [o1, o2, o3, o4, o5, o6], "ollama_compact": [c_any] * 14}


# ---------------- OpenAI-compatible scenario ----------------
def p1(body):
    expect("openai: tool_choice auto not forced / tools present", "tools" in body and body["stream"] is True)
    return {"tool_calls": [{"id": "call_a", "name": "bash",
                            "arguments": json.dumps({"command": "python3 -c \"from PIL import Image; Image.new('RGB',(40,30)).save('x.png')\" && echo ok"})}],
            "text": "Making an image."}


def p2(body):
    tm = [m for m in body["messages"] if m["role"] == "tool"][-1]
    expect("openai: tool result with id", tm["tool_call_id"] == "call_a" and "ok" in tm["content"], tm)
    asst = [m for m in body["messages"] if m["role"] == "assistant"][-1]
    expect("openai: arguments passed back as JSON string", isinstance(asst["tool_calls"][0]["function"]["arguments"], str))
    return {"tool_calls": [{"id": "call_b", "name": "view_image", "arguments": json.dumps({"path": "x.png"})}]}


def p3(body):
    parts = [m for m in body["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    ok = parts and any(p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/png;base64,")
                       for p in parts[-1]["content"])
    expect("openai vision: image sent as data URL", bool(ok), parts)
    return {"text": "All done."}


SCEN["openai"] = [p1, p2, p3]
SCEN["ping"] = [lambda body: {"content": "ready", "text": "ready"}]


def ndjson(res):
    lines = []
    content = res.get("content", "")
    for k in range(0, len(content), 15):
        lines.append({"message": {"role": "assistant", "content": content[k:k + 15]}, "done": False})
    if res.get("tool_calls"):
        lines.append({"message": {"role": "assistant", "content": "", "tool_calls": res["tool_calls"]}, "done": False})
    lines.append({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                  "prompt_eval_count": 900, "eval_count": 40})
    return "".join(json.dumps(x) + "\n" for x in lines).encode()


def sse_openai(res):
    chunks = []
    if res.get("text"):
        for k in range(0, len(res["text"]), 7):
            chunks.append({"choices": [{"index": 0, "delta": {"content": res["text"][k:k + 7]}, "finish_reason": None}]})
    for i, tc in enumerate(res.get("tool_calls", [])):
        args = tc["arguments"]
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                                                          "function": {"name": tc["name"], "arguments": ""}}]}}]})
        for k in range(0, len(args), 10):
            chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "function": {"arguments": args[k:k + 10]}}]}}]})
    chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if res.get("tool_calls") else "stop"}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 30}})
    return ("".join("data: %s\n\n" % json.dumps(c) for c in chunks) + "data: [DONE]\n\n").encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, json.dumps({"models": [{"name": "qwen3-coder:30b"}, {"name": "gpt-oss:20b"}]}).encode(),
                       "application/json")
        else:
            self._send(404, b"{}", "application/json")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path == "/api/show":
            self._send(200, json.dumps({"capabilities": STATE["caps"]}).encode(), "application/json")
            return
        steps = SCEN[STATE["scenario"]]
        if STATE["step"] >= len(steps):
            FAILS.append("extra request")
            res = {"content": "(extra)", "text": "(extra)"}
        else:
            res = steps[STATE["step"]](body)
            STATE["step"] += 1
        if self.path == "/api/chat":
            self._send(200, ndjson(res), "application/x-ndjson")
        elif self.path == "/v1/chat/completions":
            if self.headers.get("authorization") != "Bearer test-key":
                FAILS.append("openai auth header missing")
            self._send(200, sse_openai(res), "text/event-stream")
        else:
            self._send(404, b"{}", "application/json")


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = "http://127.0.0.1:%d" % srv.server_address[1]
env = dict(os.environ, BASILISK_AGENT_HOME=str(HOME), BASILISK=str(FAKE), PATH=os.environ["PATH"] + ":" + str(FAKE),
           NO_COLOR="1", OPENAI_API_KEY="test-key")
env.pop("ANTHROPIC_API_KEY", None)


def run(scenario, args):
    STATE.update(scenario=scenario, step=0)
    print("\n=== %s ===" % scenario, flush=True)
    p = subprocess.run([sys.executable, AGENT, "--workdir", str(WS), "--yes"] + args, env=env, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    o = p.stdout.decode(errors="replace")
    print(o[-2500:])
    expect("%s: exit 0" % scenario, p.returncode == 0, p.returncode)
    expect("%s: all steps used" % scenario, STATE["step"] == len(SCEN[scenario]), STATE["step"])
    return o


o = run("ollama", ["--provider", "ollama", "--base-url", URL, "Run a tiny cavity test and plot KE"])
expect("ollama: banner shows local model", "qwen3-coder:30b via ollama" in o)
expect("ollama: usage line says free", "local, free" in o)
shutil.rmtree(str(WS / ".basilisk_agent"), ignore_errors=True)
STATE["caps"] = ["completion", "tools"]
run("ollama", ["--provider", "ollama", "--base-url", URL, "Run a tiny cavity test and plot KE"])
shutil.rmtree(str(WS / ".basilisk_agent"), ignore_errors=True)
STATE.update(scenario="ollama_compact", step=0)
print("\n=== ollama_compact ===", flush=True)
p = subprocess.run([sys.executable, AGENT, "--workdir", str(WS), "--yes", "--provider", "ollama", "--base-url", URL,
                    "--num-ctx", "16384", "Print numbers"], env=env, stdin=subprocess.DEVNULL,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
o = p.stdout.decode(errors="replace")
print(o[-1500:])
expect("local compaction happened", COMPACT["summary_seen"] and "Compacting" in o, o[-600:])
expect("local compaction run finished", p.returncode == 0 and "Finished." in o, o[-400:])
shutil.rmtree(str(WS / ".basilisk_agent"), ignore_errors=True)
run("openai", ["--provider", "openai", "--base-url", URL + "/v1", "--model", "some-model", "--vision", "on",
               "Make and look at an image"])

# auto-detect: no Claude key, Ollama running -> ollama is chosen
env2 = dict(env, OLLAMA_HOST=URL)
STATE.update(scenario="ping", step=0)
p = subprocess.run([sys.executable, AGENT, "--workdir", str(WS), "--check"], env=env2, stdin=subprocess.DEVNULL,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
o = p.stdout.decode(errors="replace")
expect("auto-detects Ollama when no Claude key", "AI provider: ollama" in o, o[-800:])
expect("--check pings the local model", "AI connection OK" in o and STATE["step"] == 1, o[-500:])
# missing model
env3 = dict(env2)
p = subprocess.run([sys.executable, AGENT, "--workdir", str(WS), "--provider", "ollama", "--model", "nope:1b", "--yes", "x"],
                   env=env3, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
o = p.stdout.decode(errors="replace")
expect("missing local model explained", p.returncode == 1 and "ollama pull nope:1b" in o, o)
# nothing configured
env4 = dict(env)
env4["OLLAMA_HOST"] = "http://127.0.0.1:9"
p = subprocess.run([sys.executable, AGENT, "--workdir", str(WS), "--yes", "x"], env=env4, stdin=subprocess.DEVNULL,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
o = p.stdout.decode(errors="replace")
expect("no AI configured: clear options shown", p.returncode == 1 and "Free local model" in o and "sim --help" in o, o)

srv.shutdown()
print("\n%d passed, %d failed" % (len(OKS), len(FAILS)))
if FAILS:
    print("FAILED:", FAILS)
shutil.rmtree(str(ROOT), ignore_errors=True)
sys.exit(1 if FAILS else 0)
