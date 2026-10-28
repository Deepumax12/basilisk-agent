"""The no-AI commands: basilisk-agent sim env/run/status/wait/plot/sweep/stop."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AGENT = os.environ.get("BA_AGENT") or str(Path(__file__).resolve().parents[1] / "basilisk_agent.py")
ROOT = Path(tempfile.mkdtemp(prefix="ba-sim-"))
HOME, WS, FAKE = ROOT / "home", ROOT / "ws", ROOT / "basilisk" / "src"
for d in (HOME, WS, FAKE / "gl"):
    d.mkdir(parents=True, exist_ok=True)
(FAKE / "qcc").write_text('#!/bin/bash\nexec gcc "$@"\n')
(FAKE / "qcc").chmod(0o755)
for lib in ("libglutils.a", "libfb_tiny.a"):
    subprocess.run(["ar", "rcs", str(FAKE / "gl" / lib)], check=True)
env = dict(os.environ, BASILISK_AGENT_HOME=str(HOME), BASILISK=str(FAKE), PATH=os.environ["PATH"] + ":" + str(FAKE),
           NO_COLOR="1")
env.pop("ANTHROPIC_API_KEY", None)
FAILS, OKS = [], []


def expect(name, cond, detail=""):
    (OKS if cond else FAILS).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else " -> " + str(detail)[-700:]), flush=True)


def sim(*args, cwd=WS, timeout=180):
    p = subprocess.run([sys.executable, AGENT, "sim"] + list(args), cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, p.stdout.decode(errors="replace")


(WS / "sim.c").write_text(r'''#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#if 0
#include "view.h"
#endif
int main (int argc, char * argv[]) {
  int n = argc > 1 ? atoi (argv[1]) : 10;
  fprintf (stderr, "# i t dt ke\n");
  for (int i = 0; i <= n; i++) {
    fprintf (stderr, "%d %g %g %g\n", i, 0.1*i, 0.1, 1./(1. + 0.01*i*i));
    usleep (40000);
  }
  printf ("# done %d steps\n", n);
  return 0;
}
''')

rc, o = sim("env")
expect("env works without any AI", rc == 0 and "BASILISK:" in o and "OpenMP" in o, o)

rc, o = sim("run", "sim.c", "20", "--threads", "1", "--label", "demo")
expect("run compiles with qcc", "qcc" in o and "-O2 -Wall" in o, o)
expect("view.h detected: graphics libraries linked", "-lfb_tiny" in o and "-lglutils" in o, o)
expect("run starts in background", rc == 0 and "Started run id: demo-" in o, o)
rid = re.search(r"Started run id: (\S+)", o).group(1) if "Started run id" in o else "none"

rc, o = sim("status", rid)
expect("status of the run", rc == 0 and rid in o, o)
rc, o = sim("wait", rid)
expect("wait until finished", rc == 0 and "[finished]" in o, o)

rc, o = sim("plot", rid)
png = WS / ("%s.png" % rid)
expect("plot from run log", rc == 0 and "Plot saved" in o and png.exists() and png.stat().st_size > 5000, o)
expect("plot summary has named columns", "ke" in o and "21 rows, 4 columns" in o, o)

(WS / "data.txt").write_text("# t u v\n" + "".join("%g %g %g\n" % (0.1 * i, i * i, -i) for i in range(30)))
rc, o = sim("plot", "data.txt", "--y", "u", "--log")
expect("plot a data file", rc == 0 and (WS / "data.png").exists() and "u vs t" in o, o)

rc, o = sim("run", "sim.c", "--threads", "1", "--label", "args", "--", "7", timeout=120)
rid2 = re.search(r"Started run id: (\S+)", o).group(1) if "Started run id" in o else "none"
rc, o = sim("wait", rid2)
expect("program arguments after -- are passed", "7 0.7" in o, o)

rc, o = sim("sweep", "sim.c", "--values", "5,6,7", "--parallel", "2", "--threads", "1", timeout=300)
expect("sweep runs all values", rc == 0 and o.count("finished") >= 6, o)
expect("sweep summary table", re.search(r"^7\s+sim-7-\S+\s+finished", o, re.M) is not None, o)

rc, o = sim("run", "./sim", "1000", "--label", "long")
rid3 = re.search(r"Started run id: (\S+)", o).group(1) if "Started run id" in o else "none"
rc, o = sim("stop", rid3)
expect("stop a run", rc == 0 and "Stopped run" in o and "[stopped]" in o, o)

rc, o = sim("run", "bash -c 'echo going; exit 3'", "--label", "bad")
expect("shell command target fails visibly", "failed (exit 3)" in o, o)

rc, o = sim("run", "missing.c")
expect("missing source file reported", rc == 1 and "No such file" in o, o)

(WS / "broken.c").write_text("int main() { return undefined_variable; }\n")
rc, o = sim("run", "broken.c")
expect("compile errors reported", rc == 1 and "Compilation failed" in o, o)

rc, o = sim("status")
expect("status lists runs", rc == 0 and rid in o and rid3 in o, o)

print("\n%d passed, %d failed" % (len(OKS), len(FAILS)))
if FAILS:
    print("FAILED:", FAILS)
shutil.rmtree(str(ROOT), ignore_errors=True)
sys.exit(1 if FAILS else 0)
