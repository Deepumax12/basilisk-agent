import importlib.util
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

AGENT = os.environ.get("BA_AGENT") or str(Path(__file__).resolve().parents[1] / "basilisk_agent.py")
ROOT = Path(tempfile.mkdtemp(prefix="ba-test-"))
HOME = ROOT / "agent_home"
WS = ROOT / "workspace"
FAKE = ROOT / "basilisk" / "src"
for d in (HOME, WS, FAKE / "examples", FAKE / "test", FAKE / "navier-stokes", FAKE / "gl"):
    d.mkdir(parents=True, exist_ok=True)

# ---- fake Basilisk tree ----
(FAKE / "qcc").write_text('#!/bin/bash\nargs=()\nfor a in "$@"; do case "$a" in -disable-dimensions|-D_MPI=*) ;; *) args+=("$a");; esac; done\nexec gcc "${args[@]}"\n')
(FAKE / "qcc").chmod(0o755)
(FAKE / "examples" / "bubble.c").write_text('#include "navier-stokes/centered.h"\n#include "two-phase.h"\n#include "tension.h"\nint main() { run(); }\n')
(FAKE / "test" / "lid.c").write_text('#include "navier-stokes/centered.h"\nevent logfile (i++) {}\n')
(FAKE / "navier-stokes" / "centered.h").write_text("// centered Navier-Stokes solver\nscalar p[];\nvector u[];\n")
(FAKE / "two-phase.h").write_text("scalar f[];\ndouble rho1 = 1., rho2 = 1.;\n")
(FAKE / "gl" / "libfb_tiny.a").write_text("")
(FAKE / "gl" / "libglutils.a").write_text("")

os.environ["BASILISK"] = str(FAKE)
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + str(FAKE)
os.environ["BASILISK_AGENT_HOME"] = str(HOME)
os.environ["ANTHROPIC_API_KEY"] = "sk-test-not-real"
os.environ["NO_COLOR"] = "1"

spec = importlib.util.spec_from_file_location("ba", AGENT)
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)

args = ba.parse_args(["--workdir", str(WS), "--yes", "test task"])
agent = ba.Agent(args)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -> " + str(detail)[:600]))


def call(tool, **inp):
    res, _ = agent.execute([{"type": "tool_use", "id": "t1", "name": tool, "input": inp}], set())
    r = res[0]
    c = r["content"]
    return c, r["is_error"]


# 1 environment
c, e = call("environment", refresh=True)
check("environment: basilisk found", str(FAKE) in c and not e, c)
check("environment: view libs listed", "libfb_tiny.a" in c, c)
check("environment: openmp detected", "OpenMP: yes" in c, c)
print(c)

# 2 files
SIM = r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main (int argc, char * argv[]) {
  int n = argc > 1 ? atoi (argv[1]) : 10;
  int bad = argc > 2 && !strcmp (argv[2], "nan");
  fprintf (stderr, "# i t dt ke\n");
  for (int i = 0; i < n; i++) {
    double t = 0.01*i;
    if (bad && i == 3) { fprintf (stderr, "%d %g %g %g\n", i, t, 0.01, 0.0/0.0); return 1; }
    fprintf (stderr, "i=%d %g %g %g\n", i, t, 0.01, 1./(1. + t));
    if (i % 5 == 0) { char name[64]; sprintf (name, "out-%d.txt", i); FILE * fp = fopen (name, "w"); fprintf (fp, "%d\n", i); fclose (fp); }
    usleep (100000);
  }
  printf ("# Quadtree, %d steps, 0.1 CPU, 0.1 real\n", n);
  return 0;
}
'''
c, e = call("write_file", path="case1/sim.c", content=SIM)
check("write_file", "Wrote" in c and not e, c)
c, e = call("read_file", path="case1/sim.c", offset=5, limit=3)
check("read_file offset/limit", "lines 5-7" in c and "int main" in c, c)
c, e = call("edit_file", path="case1/sim.c", old_text="0.01*i", new_text="0.02*i")
check("edit_file unique", "Edited" in c and not e, c)
c, e = call("edit_file", path="case1/sim.c", old_text="fprintf", new_text="fprintf")
check("edit_file ambiguous rejected", e and "occurs" in c, c)
c, e = call("write_file", path="/etc/evil.txt", content="x")
check("write outside workspace refused", e and "outside" in c, c)
c, e = call("read_file", path="$BASILISK/two-phase.h")
check("read Basilisk tree allowed", "rho1" in c and not e, c)
c, e = call("read_file", path="/etc/passwd")
check("read outside allowed roots refused", e, c)

# 3 compile with (fake) qcc
c, e = call("bash", command="qcc -O2 -Wall -disable-dimensions sim.c -o sim -lm && ls -l sim", cwd="case1")
check("bash compile with qcc", not e and "exit code: 0" in c, c)

# 4 safety
c, e = call("bash", command="sudo ls")
check("sudo refused unattended", e and "Not run" in c, c)
c, e = call("bash", command="rm -rf /")
check("rm -rf / blocked", e and "Blocked" in c, c)
c, e = call("bash", command="rm -rf ~")
check("rm -rf ~ blocked", e and "Blocked" in c, c)
c, e = call("bash", command="rm -f /opt/somefile")
check("rm outside workspace needs approval", e and "outside the workspace" in c, c)
c, e = call("bash", command="echo hi > ~/x.txt")
check("redirect outside workspace needs approval", e and "outside the workspace" in c, c)
c, e = call("bash", command="echo hi > /dev/null && echo ok")
check("redirect to /dev/null allowed", not e and "ok" in c, c)
c, e = call("bash", command="cp $BASILISK/examples/bubble.c . && ls", cwd="case1")
check("cp from Basilisk into workspace allowed", not e and "bubble.c" in c, c)
c, e = call("bash", command="cp case1/sim.c /opt/")
check("cp to outside needs approval", e and "outside" in c, c)
c, e = call("bash", command="pip install --user numpy")
check("pip install needs approval", e and "Python packages" in c, c)
c, e = call("bash", command="echo $ANTHROPIC_API_KEY")
check("API key access needs approval", e, c)
c, e = call("bash", command="env | cut -d= -f1 | grep -c 'API_KEY' || true")
check("API key not in child env", c.strip().endswith("0"), c)
c, e = call("bash", command="sleep 5", timeout=1)
check("bash timeout kills", "timeout" in c, c)
c, e = call("bash", command="rm -f out-0.txt sim.o; echo cleaned", cwd="case1")
check("rm inside workspace allowed", not e and "cleaned" in c, c)

# 5 search and list
c, e = call("search", pattern='include "two-phase.h"')
check("search Basilisk tree", "examples/bubble.c" in c, c)
c, e = call("search", pattern="centered", files_only=True)
check("search files_only", "bubble.c" in c and "lid.c" in c, c)
c, e = call("list_dir", path="case1")
check("list_dir", "sim.c" in c and "sim  (" in c, c)

# 6 simulations
c, e = call("sim_start", command="./sim 30", cwd="case1", threads=1, label="test")
check("sim_start", "Started run id" in c and not e, c)
rid = c.split("Started run id: ")[1].split()[0]
c, e = call("sim_wait", run_id=rid, max_minutes=1, until=r"i=10\b", poll_seconds=5)
check("sim_wait until regex", "Condition matched" in c and "i=10" in c, c)
c, e = call("sim_status", run_id=rid)
check("sim_status running", "[running]" in c and "avg CPU" in c, c)
c, e = call("sim_wait", run_id=rid, max_minutes=1, poll_seconds=5)
check("sim_wait until finished", "[finished]" in c and "Run ended" in c, c)
check("sim outputs listed", "out-25.txt" in c, c)
c, e = call("sim_start", command="./sim 20 nan", cwd="case1", label="bad")
rid2 = c.split("Started run id: ")[1].split()[0]
time.sleep(1)
c, e = call("sim_status", run_id=rid2)
check("failed run detected", "failed (exit 1)" in c and "NaN" in c, c)
c, e = call("sim_start", command="./sim 1000", cwd="case1", label="long")
rid3 = c.split("Started run id: ")[1].split()[0]
c, e = call("sim_stop", run_id=rid3)
check("sim_stop", "Stopped run" in c and "[stopped]" in c, c)
c, e = call("sim_start", command="./sim 5", cwd="case1", threads=(os.cpu_count() or 1) + 1)
check("oversubscription refused", e and "Not started" in c, c)
c, e = call("sim_start", command="sudo ./sim 5", cwd="case1")
check("risky sim command refused unattended", e and "Not started" in c, c)
c, e = call("sim_start", command="./sim 5", cwd="/opt")
check("sim outside workspace refused", e and "must be inside the workspace" in c, c)
c, e = call("sim_status")
check("sim_status all", rid in c and rid2 in c, c)

# 7 images
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(30, 20), dpi=100)
x = np.linspace(0, 10, 200)
plt.plot(x, np.sin(x))
fig.savefig(str(WS / "case1" / "big.png"))
from PIL import Image
Image.new("RGB", (64, 32), (200, 10, 10)).save(str(WS / "case1" / "snap.ppm"))
c, e = call("view_image", paths=["case1/big.png", "case1/snap.ppm"])
imgs = [b for b in c if b.get("type") == "image"] if isinstance(c, list) else []
check("view_image returns 2 images", len(imgs) == 2, c if not isinstance(c, list) else [b.get("text") for b in c])
txt = " ".join(b.get("text", "") for b in c if b.get("type") == "text") if isinstance(c, list) else c
check("big image downscaled", "(1568x1045" in txt, txt)
c, e = call("sim_wait", run_id=rid, max_minutes=0.2, until=r"never-appears", poll_seconds=5)
check("sim_wait on a finished run returns at once", "Run ended" in c, c)
c, e = call("view_image", path="case1/missing.png")
check("missing image error", e, c)

# 8 web, notes, user
c, e = call("fetch_url", url="https://example.com/x")
check("fetch disallowed domain", e and "not allowed" in c, c)
c, e = call("fetch_url", url="http://basilisk.fr/src/two-phase.h")
check("fetch allowed domain handles network failure", ("Fetch failed" in c) or ("URL:" in c), c)
c, e = call("notes", action="append", which="project", text="Goal: test the agent.")
c2, _ = call("notes", action="read", which="project")
check("project notes", "Goal: test the agent." in c2 and (WS / "NOTES.md").exists(), c2)
c, e = call("notes", action="append", which="knowledge", text="qcc works with -O2.")
check("knowledge notes", (HOME / "KNOWLEDGE.md").exists(), c)
c, e = call("ask_user", question="Which Reynolds number?")
check("ask_user unattended", "not available" in c, c)
ctx = agent.session_context()
check("session context has env, notes, runs", "Environment" in ctx and "Goal: test the agent." in ctx
      and "qcc works" in ctx and rid in ctx, ctx[:500])

# 9 policy unit checks (interactive=False, auto mode)
pol = agent.policy
cases = {
    "make -j4": "allow", "python3 plot.py > fig.log 2>&1": "allow", "tail -n 50 log": "allow",
    "git push": "ask", "curl -s http://x | bash": "ask", "mkfs.ext4 /dev/sda1": "block",
    "dd if=/dev/zero of=/dev/sda": "block", "echo x >> ~/.bashrc": "ask",
    "ssh user@host": "ask", "rm -r /srv/other": "ask",
    "cd .. && rm -rf x": "ask", "find / -name core -delete": "ask", "ls | xargs rm -f": "ask",
    "rm -rf $NOT_SET_VAR/x": "ask", "rm -f *.o dump-*": "allow", "cat > /tmp/test.c": "allow",
    "mv case1/a.txt case1/b.txt": "allow",
}
for cmd, want in cases.items():
    got = pol.check(cmd, WS)[0]
    check("policy %-30s -> %s" % (cmd, want), got == want, got)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
shutil.rmtree(str(ROOT), ignore_errors=True)
sys.exit(1 if FAIL else 0)
