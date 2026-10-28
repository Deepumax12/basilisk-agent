"""Drive the agent through a pseudo-terminal: slash commands, ask_user, and an approval prompt."""
import json
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AGENT = os.environ.get("BA_AGENT") or str(Path(__file__).resolve().parents[1] / "basilisk_agent.py")
ROOT = Path(tempfile.mkdtemp(prefix="ba-tty-"))
HOME, WS = ROOT / "home", ROOT / "ws"
HOME.mkdir()
WS.mkdir()
FAILS, OKS = [], []


def expect(name, cond, detail=""):
    (OKS if cond else FAILS).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else " -> " + str(detail)[-600:]), flush=True)


def sse(blocks):
    ev = [{"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "content": [],
                                                 "model": "claude-opus-5-5", "stop_reason": None,
                                                 "usage": {"input_tokens": 900, "output_tokens": 1}}}]
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}})
        else:
            ev.append({"type": "content_block_start", "index": i, "content_block": dict(b, input={})})
            ev.append({"type": "content_block_delta", "index": i,
                       "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})
        ev.append({"type": "content_block_stop", "index": i})
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    ev.append({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 20}})
    ev.append({"type": "message_stop"})
    return "".join("event: %s\ndata: %s\n\n" % (e["type"], json.dumps(e)) for e in ev).encode()


def last_result(body):
    for b in body["messages"][-1]["content"]:
        if b.get("type") == "tool_result":
            c = b["content"]
            return c if isinstance(c, str) else json.dumps(c), b.get("is_error")
    return "", None


STEP = {"n": 0}


def respond(body):
    n = STEP["n"]
    STEP["n"] += 1
    if n == 0:
        return [{"type": "text", "text": "One question first."},
                {"type": "tool_use", "id": "q1", "name": "ask_user", "input": {"question": "Which Reynolds number?"}}]
    if n == 1:
        res, _ = last_result(body)
        expect("ask_user answer reached the model", "User's answer: Re = 1000" in res, res)
        return [{"type": "tool_use", "id": "b1", "name": "bash", "input": {"command": "ssh -V 2>&1 || echo no-ssh"}}]
    if n == 2:
        res, err = last_result(body)
        expect("approved command ran", "exit code" in res and "Not run" not in res, res)
        return [{"type": "tool_use", "id": "b2", "name": "bash", "input": {"command": "sudo true"}}]
    if n == 3:
        res, err = last_result(body)
        expect("declined command not run", "Not run" in res and err, res)
        return [{"type": "text", "text": "All good, Re = 1000 noted."}]
    return [{"type": "text", "text": "extra"}]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        data = sse(respond(body))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(data)


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
env = dict(os.environ, ANTHROPIC_BASE_URL="http://127.0.0.1:%d" % srv.server_address[1], ANTHROPIC_API_KEY="sk-test",
           BASILISK_AGENT_HOME=str(HOME), NO_COLOR="1", TERM="dumb")
env.pop("BASILISK", None)

master, slave = pty.openpty()
p = subprocess.Popen([sys.executable, AGENT, "--workdir", str(WS)], stdin=slave, stdout=slave, stderr=slave,
                     env=env, close_fds=True)
os.close(slave)
buf = {"text": ""}


def read_until(pattern, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if re.search(pattern, buf["text"]):
            return True
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try:
                chunk = os.read(master, 65536).decode(errors="replace")
            except OSError:
                break
            buf["text"] += chunk
    return bool(re.search(pattern, buf["text"]))


def send(line):
    mark = len(buf["text"])
    os.write(master, (line + "\n").encode())
    return mark


expect("prompt shown", read_until(r"you > "), buf["text"])
m = send("/help")
expect("/help works", read_until(r"Commands:[\s\S]*you > "), buf["text"][m:])
m = send("Please set up a cavity case")
expect("ask_user question shown", read_until(r"Question from the agent:[\s\S]*your answer > "), buf["text"][m:])
m = send("Re = 1000")
expect("approval prompt shown", read_until(r"Approval needed[\s\S]*Allow\?"), buf["text"][m:])
expect("approval reason shown", "remote access" in buf["text"][m:], buf["text"][m:])
m = send("y")
expect("second approval prompt (sudo)", read_until(r"uses sudo[\s\S]*Allow\?"), buf["text"][m:])
m = send("n")
expect("final answer printed", read_until(r"All good, Re = 1000 noted\.[\s\S]*you > "), buf["text"][m:])
m = send("/cost")
expect("/cost works", read_until(r"calls 4"), buf["text"][m:])
m = send("/exit")
expect("exit summary", read_until(r"Session: 4 model calls"), buf["text"][m:])
try:
    p.wait(timeout=10)
except subprocess.TimeoutExpired:
    p.kill()
expect("process exited cleanly", p.returncode == 0, p.returncode)
srv.shutdown()
print("\n--- transcript (tail) ---\n" + buf["text"][-1800:])
print("\n%d passed, %d failed" % (len(OKS), len(FAILS)))
shutil.rmtree(str(ROOT), ignore_errors=True)
sys.exit(1 if FAILS else 0)
