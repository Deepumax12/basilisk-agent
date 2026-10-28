"""End-to-end test of the agent loop against a mock Claude Messages API (SSE streaming)."""
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
ROOT = Path(tempfile.mkdtemp(prefix="ba-loop-"))
HOME, WS, FAKE = ROOT / "home", ROOT / "ws", ROOT / "basilisk" / "src"
for d in (HOME, WS, FAKE / "examples", FAKE / "gl"):
    d.mkdir(parents=True, exist_ok=True)
(FAKE / "qcc").write_text('#!/bin/bash\nexec gcc "$@"\n')
(FAKE / "qcc").chmod(0o755)
(FAKE / "examples" / "karman.c").write_text('#include "embed.h"\n#include "navier-stokes/centered.h"\n')

FAILS, NOTES = [], []


def expect(name, cond, detail=""):
    (NOTES if cond else FAILS).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else " -> " + str(detail)[:400]), flush=True)


SIM = r'''#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main (int argc, char * argv[]) {
  int n = argc > 1 ? atoi (argv[1]) : 10;
  fprintf (stderr, "# i t ke\n");
  for (int i = 0; i <= n; i++) { double t = 0.05*i; fprintf (stderr, "%d %g %g\n", i, t, 1./(1. + t*t)); usleep (50000); }
  return 0;
}
'''
PLOT = r'''import sys, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
log = sorted(glob.glob(".runs/*/stderr.log"))[-1]
d = np.loadtxt(log, comments="#")
plt.figure(figsize=(6, 4)); plt.plot(d[:, 1], d[:, 2], "o-"); plt.xlabel("t"); plt.ylabel("KE")
plt.tight_layout(); plt.savefig("ke.png", dpi=150)
print("points", len(d), "final KE %.4f" % d[-1, 2])
'''

SIGNATURE = "sig-ABC123-opaque"


def tool(tid, name, inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def text(t):
    return {"type": "text", "text": t}


def last_user(body):
    return body["messages"][-1]


def results_by_id(body):
    out = {}
    for b in last_user(body)["content"]:
        if b.get("type") == "tool_result":
            out[b["tool_use_id"]] = b
    return out


def rtext(r):
    c = r["content"]
    return c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if x.get("type") == "text")


STATE = {"run_id": None, "overloaded_once": False}


# ---------------------------------------------------------------- scenario 1
def s1_r1(body):
    expect("request: streaming on", body.get("stream") is True)
    expect("request: automatic prompt caching", body.get("cache_control") == {"type": "ephemeral"}, body.get("cache_control"))
    expect("request: effort set", body.get("output_config") == {"effort": "high"}, body.get("output_config"))
    expect("request: thinking left to model default", "thinking" not in body)
    expect("request: 15 tools", len(body.get("tools", [])) == 15, len(body.get("tools", [])))
    expect("request: system prompt", "Basilisk Agent" in body["system"] and "{long_run_hours}" not in body["system"])
    first = body["messages"][0]["content"]
    expect("request: session context first", "<session_context>" in first[0]["text"] and "Environment" in first[0]["text"])
    expect("request: task included", "lid" in first[-1]["text"])
    return [{"type": "thinking", "thinking": "", "signature": SIGNATURE},
            text("I'll check the machine first."), tool("tu_1", "environment", {})]


def s1_r2(body):
    asst = body["messages"][-2]
    expect("thinking block passed back unchanged",
           asst["content"][0] == {"type": "thinking", "thinking": "", "signature": SIGNATURE}, asst["content"][0])
    r = results_by_id(body)
    expect("environment tool result", "tu_1" in r and "BASILISK" in rtext(r["tu_1"]), r)
    return [text("Writing and compiling the test program."),
            tool("tu_2", "write_file", {"path": "case1/sim.c", "content": SIM}),
            tool("tu_3", "bash", {"command": "qcc -O2 -Wall sim.c -o sim -lm && echo built", "cwd": "case1"})]


def s1_r3(body):
    if not STATE["overloaded_once"]:
        STATE["overloaded_once"] = True
        return ("http", 529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
                {"retry-after": "0"})
    r = results_by_id(body)
    expect("two tool results in one message", set(r) == {"tu_2", "tu_3"}, list(r))
    expect("compile succeeded", "built" in rtext(r["tu_3"]) and not r["tu_3"]["is_error"], rtext(r["tu_3"]))
    return [tool("tu_4", "sim_start", {"command": "./sim 30", "cwd": "case1", "threads": 1, "label": "demo"})]


def s1_r4(body):
    r = results_by_id(body)["tu_4"]
    m = re.search(r"Started run id: (\S+)", rtext(r))
    expect("sim_start returned run id", m is not None, rtext(r))
    STATE["run_id"] = m.group(1) if m else "x"
    return [tool("tu_5", "sim_wait", {"run_id": STATE["run_id"], "max_minutes": 1, "poll_seconds": 5})]


def s1_r5(body):
    r = results_by_id(body)["tu_5"]
    expect("sim_wait saw the run finish", "[finished]" in rtext(r), rtext(r))
    return [tool("tu_6", "write_file", {"path": "case1/plot.py", "content": PLOT}),
            tool("tu_7", "bash", {"command": "python3 plot.py", "cwd": "case1"}),
            tool("tu_8", "view_image", {"path": "case1/ke.png"})]


def s1_r6(body):
    r = results_by_id(body)
    expect("plot script ran", "final KE" in rtext(r["tu_7"]), rtext(r["tu_7"]))
    img = [x for x in r["tu_8"]["content"] if isinstance(r["tu_8"]["content"], list) and x.get("type") == "image"]
    expect("image block in tool result", len(img) == 1 and img[0]["source"]["media_type"] == "image/png"
           and len(img[0]["source"]["data"]) > 1000, r["tu_8"])
    return [tool("tu_9", "bash", {"command": "sudo apt install ffmpeg"}),
            tool("tu_10", "write_file", {"path": "case1/REPORT.md", "content": "# Report\nKE decays.\n"}),
            tool("tu_11", "notes", {"action": "append", "which": "project", "text": "Run %s finished." % STATE["run_id"]})]


def s1_r7(body):
    r = results_by_id(body)
    expect("sudo refused in unattended mode", r["tu_9"]["is_error"] and "Not run" in rtext(r["tu_9"]), rtext(r["tu_9"]))
    return [text("Done: the test run finished and REPORT.md is in case1/.")]


# ---------------------------------------------------------------- scenario 2 (resume)
def s2_r1(body):
    n = len(body["messages"])
    expect("resume: history reloaded", n >= 13, n)
    expect("resume: new task appended", "What did you find" in body["messages"][-1]["content"][-1]["text"])
    return [text("Kinetic energy decays as 1/(1+t^2).")]


# ---------------------------------------------------------------- scenario 3 (compaction + truncated tool call)
def s3_r1(body):
    return {"blocks": [text("Starting."), tool("tu_c1", "list_dir", {"path": "."})], "usage_in": 5000}


def s3_r2(body):
    expect("compaction request has tool_choice none", body.get("tool_choice") == {"type": "none"}, body.get("tool_choice"))
    expect("compaction prompt sent", "handoff summary" in json.dumps(body["messages"][-1]))
    return {"blocks": [text("SUMMARY: user wants a cavity run; case1 has sim.c; next: run it.")], "usage_in": 100}


def s3_r3(body):
    msgs = body["messages"]
    expect("after compaction: single user message", len(msgs) == 1 and msgs[0]["role"] == "user", len(msgs))
    expect("after compaction: summary included", "SUMMARY: user wants a cavity run" in json.dumps(msgs))
    return {"blocks": [tool("tu_c2", "write_file", {"path": "x.txt", "content": "abc"})], "truncate_tool": True,
            "stop": "max_tokens", "usage_in": 100}


def s3_r4(body):
    r = results_by_id(body)
    expect("truncated tool call reported as error", "tu_c2" in r and r["tu_c2"]["is_error"] and "cut off" in rtext(r["tu_c2"]), r)
    return {"blocks": [text("OK, done.")], "usage_in": 100}


SCENARIOS = {"1": [s1_r1, s1_r2, s1_r3, s1_r4, s1_r5, s1_r6, s1_r7], "2": [s2_r1], "3": [s3_r1, s3_r2, s3_r3, s3_r4]}
CURRENT = {"name": "1", "step": 0}


def sse(blocks, stop="end_turn", usage_in=1000, truncate_tool=False):
    ev = [{"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [],
                                                 "model": "claude-opus-5-5", "stop_reason": None,
                                                 "usage": {"input_tokens": usage_in, "cache_read_input_tokens": 0,
                                                           "cache_creation_input_tokens": 0, "output_tokens": 1}}}]
    has_tool = any(b["type"] == "tool_use" for b in blocks)
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            for k in range(0, len(b["text"]), 12):
                ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"][k:k + 12]}})
        elif b["type"] == "thinking":
            ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": ""}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": b["signature"]}})
        elif b["type"] == "tool_use":
            ev.append({"type": "content_block_start", "index": i,
                       "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
            js = json.dumps(b["input"])
            if truncate_tool:
                js = js[: len(js) // 2]
            for k in range(0, len(js), 25):
                ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": js[k:k + 25]}})
        ev.append({"type": "content_block_stop", "index": i})
        ev.append({"type": "ping"})
    ev.append({"type": "message_delta", "delta": {"stop_reason": stop if not (has_tool and stop == "end_turn") else "tool_use",
                                                   "stop_sequence": None}, "usage": {"output_tokens": 42}})
    ev.append({"type": "message_stop"})
    return "".join("event: %s\ndata: %s\n\n" % (e["type"], json.dumps(e)) for e in ev).encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        if self.headers.get("x-api-key") != "sk-test" or self.headers.get("anthropic-version") != "2023-06-01":
            FAILS.append("bad headers")
        steps = SCENARIOS[CURRENT["name"]]
        if CURRENT["step"] >= len(steps):
            FAILS.append("unexpected extra request in scenario %s" % CURRENT["name"])
            res = [text("(no more steps)")]
        else:
            res = steps[CURRENT["step"]](body)
        if isinstance(res, tuple) and res[0] == "http":
            _, status, payload, headers = res
            data = json.dumps(payload).encode()
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        CURRENT["step"] += 1
        if isinstance(res, dict):
            data = sse(res["blocks"], res.get("stop", "end_turn"), res.get("usage_in", 1000), res.get("truncate_tool", False))
        else:
            data = sse(res)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(data)


srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
env = dict(os.environ, ANTHROPIC_BASE_URL="http://127.0.0.1:%d" % srv.server_address[1], ANTHROPIC_API_KEY="sk-test",
           BASILISK_AGENT_HOME=str(HOME), BASILISK=str(FAKE), PATH=os.environ["PATH"] + ":" + str(FAKE), NO_COLOR="1")


def run_agent(scenario, args):
    CURRENT.update(name=scenario, step=0)
    print("\n=== scenario %s ===" % scenario, flush=True)
    p = subprocess.run([sys.executable, AGENT, "--workdir", str(WS)] + args, env=env, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    outp = p.stdout.decode(errors="replace")
    print(outp[-3500:])
    expect("scenario %s: exit code 0" % scenario, p.returncode == 0, p.returncode)
    expect("scenario %s: all steps used" % scenario, CURRENT["step"] == len(SCENARIOS[scenario]), CURRENT["step"])
    return outp


o1 = run_agent("1", ["--yes", "Run a tiny lid test and plot the kinetic energy"])
expect("streamed text printed", "I'll check the machine first." in o1)
expect("retry message printed", "retrying" in o1, o1[-800:])
expect("report written", (WS / "case1" / "REPORT.md").exists())
expect("session saved", (WS / ".basilisk_agent" / "session.json").exists())
expect("notes written", "finished" in (WS / "NOTES.md").read_text())
o2 = run_agent("2", ["--yes", "--resume", "What did you find?"])
expect("resume answer printed", "1/(1+t^2)" in o2)
o3 = run_agent("3", ["--yes", "--compact-at", "1000", "Start a new cavity case"])
expect("compaction announced", "Compacting" in o3)

srv.shutdown()
print("\n%d checks passed, %d failed" % (len(NOTES), len(FAILS)))
if FAILS:
    print("FAILED:", FAILS)
shutil.rmtree(str(ROOT), ignore_errors=True)
sys.exit(1 if FAILS else 0)
