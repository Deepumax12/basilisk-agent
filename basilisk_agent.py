#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basilisk Agent - an autonomous CFD assistant for Basilisk (http://basilisk.fr).

It runs on your own machine (Linux or WSL), and for any kind of flow problem it:
  * reads your environment (compilers, qcc, OpenMP/MPI, cores, RAM, Python tools),
  * finds the closest Basilisk example and checks the real headers,
  * writes the Basilisk C code, compiles it with qcc and fixes errors,
  * runs a cheap smoke test, then long simulations in the background,
  * monitors runs (NaN / crash detection, resource use, waiting without cost),
  * post-processes with Python, looks at its own plots, compares with theory,
  * writes REPORT.md and keeps a project notebook (NOTES.md) across sessions.

The "brain" can be:
  * Claude (best results; needs an API key),
  * a free local model running on your own computer through Ollama (no key, no cost),
  * any OpenAI-compatible server (LM Studio, vLLM, llama.cpp server, cloud providers).

Without any AI, the `sim` commands still compile, run, monitor, plot and sweep:
    basilisk-agent sim run cavity.c 1000 --threads 4

Single file, Python 3.8+ standard library only (Pillow is optional, for image
resizing; matplotlib for `sim plot`). See README.md for setup.

    cd ~/cfd/my_problem
    basilisk-agent "Simulate a 2D lid-driven cavity at Re=1000 and compare with Ghia et al."
    basilisk-agent --provider ollama "..."      # free local model
"""

import argparse
import base64
import datetime
import html
import http.client
import io
import json
import os
import platform
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

VERSION = "1.1.1"

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

API_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
API_URL = API_BASE + "/v1/messages"
API_VERSION = "2023-06-01"

MODEL_ALIASES = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "fable": "claude-fable-5-1",
    "haiku": "claude-haiku-5-5",
}
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 64000

# Approximate USD prices per million tokens: (input, output, cache-read multiplier).
# Used only for the cost estimate shown in the terminal; edit if prices change.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.025),
    "claude-opus-5-5": (4.0, 20.0, 0.05),
    "claude-sonnet-5-5": (2.0, 10.0, 0.05),
}
CACHE_WRITE_MULT = {"5m": 1.25, "1h": 2.0}

HOME_DIR = Path(os.environ.get("BASILISK_AGENT_HOME", "~/.basilisk_agent")).expanduser()

MAX_TOOL_OUTPUT = 30000        # characters of tool output sent back to the model
MAX_IMAGE_EDGE = 1568          # pixels; plots are downscaled to this before sending
MAX_IMAGES_BEFORE_COMPACT = 16
DEFAULT_COMPACT_AT = 350000    # prompt tokens
MAX_REQUEST_BYTES = 24 * 1024 * 1024

# Local / other AI providers
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL
DEFAULT_LOCAL_MODEL = "qwen3-coder:30b"
DEFAULT_NUM_CTX = 32768
DEFAULT_LOCAL_MAX_TOKENS = 16384
# Rough suggestions by available memory, used by --check (see README for the trade-offs).
LOCAL_MODEL_SUGGESTIONS = [
    (20, "qwen3-coder:30b", "coding-agent model, about 19 GB"),
    (14, "gpt-oss:20b", "about 14 GB"),
    (8, "qwen2.5-coder:7b", "small; expect many mistakes"),
]


# --------------------------------------------------------------------------
# Terminal helpers
# --------------------------------------------------------------------------

class Style:
    def __init__(self, enabled):
        def c(code):
            return code if enabled else ""
        self.reset = c("\033[0m")
        self.bold = c("\033[1m")
        self.dim = c("\033[2m")
        self.italic = c("\033[3m")
        self.red = c("\033[31m")
        self.green = c("\033[32m")
        self.yellow = c("\033[33m")
        self.blue = c("\033[34m")
        self.magenta = c("\033[35m")
        self.cyan = c("\033[36m")


S = Style(sys.stdout.isatty() and not os.environ.get("NO_COLOR"))


def out(text="", end="\n"):
    sys.stdout.write(text + end)
    sys.stdout.flush()


def info(text):
    out(S.dim + text + S.reset)


def warn(text):
    out(S.yellow + text + S.reset)


def error(text):
    out(S.red + text + S.reset)


def clip(text, limit=MAX_TOOL_OUTPUT):
    """Keep the head and the tail of long text."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.35)
    tail = limit - head
    return (text[:head] + "\n\n[... %d characters omitted ...]\n\n" % (len(text) - limit)
            + text[-tail:])


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return ("%.0f %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0


def human_time(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%dh%02dm" % (h, m)
    if m:
        return "%dm%02ds" % (m, s)
    return "%ds" % s


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def slug(text, maxlen=24):
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (s[:maxlen].strip("-")) or "run"


def is_within(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def resolve_path(p, base):
    """Resolve a user/agent path (relative to base) to an absolute real path."""
    p = os.path.expandvars(os.path.expanduser(str(p)))
    q = Path(p)
    if not q.is_absolute():
        q = Path(base) / q
    return Path(os.path.realpath(str(q)))


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, str(path))


def read_json(path, default):
    try:
        with open(str(path), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def clean_env():
    """Environment for child processes: no API keys, Basilisk on PATH."""
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("ANTHROPIC_") and ("KEY" in k or "TOKEN" in k):
            del env[k]
    bas = env.get("BASILISK")
    if bas and os.path.isfile(os.path.join(bas, "qcc")):
        if bas not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = env.get("PATH", "") + os.pathsep + bas
    return env


# --------------------------------------------------------------------------
# Process helpers (Linux /proc)
# --------------------------------------------------------------------------

def _proc_table():
    """Return {pid: (ppid, pgrp, state, cpu_seconds, rss_bytes)} from /proc."""
    table = {}
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not d.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % d) as f:
                stat = f.read()
            rest = stat.rsplit(")", 1)[1].split()
            state, ppid, pgrp = rest[0], int(rest[1]), int(rest[2])
            cpu = (int(rest[11]) + int(rest[12])) / float(tick)
            rss = int(rest[21]) * page
            table[int(d)] = (ppid, pgrp, state, cpu, rss)
        except (OSError, IndexError, ValueError):
            continue
    return table


def descendants(pid, table=None):
    table = table if table is not None else _proc_table()
    children = {}
    for p, (pp, _, _, _, _) in table.items():
        children.setdefault(pp, []).append(p)
    result, stack = [], [pid]
    while stack:
        cur = stack.pop()
        for ch in children.get(cur, []):
            result.append(ch)
            stack.append(ch)
    return result


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open("/proc/%d/stat" % pid) as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
        return state not in ("Z", "X")
    except (OSError, IndexError):
        return True


def kill_tree(pid, sig=signal.SIGTERM):
    pids = descendants(pid) + [pid]
    for p in pids:
        try:
            os.kill(p, sig)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def signal_name(code):
    try:
        return signal.Signals(code).name
    except (ValueError, AttributeError):
        return "signal %d" % code


# --------------------------------------------------------------------------
# Environment detection
# --------------------------------------------------------------------------

def _cmd_output(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout, env=clean_env())
        return r.returncode, r.stdout.decode(errors="replace").strip()
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def detect_environment(workdir):
    env = {"detected": now_str(), "warnings": []}
    w = env["warnings"]
    u = platform.uname()
    env["kernel"] = "%s %s %s" % (u.system, u.release, u.machine)
    pretty = ""
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    env["os"] = pretty or u.system
    env["wsl"] = ("microsoft" in u.release.lower()) or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
    env["python"] = platform.python_version()

    cpu_model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    env["cpu"] = cpu_model or platform.processor()
    env["logical_cpus"] = os.cpu_count() or 1
    rc, lscpu = _cmd_output(["lscpu"])
    cores, sockets = None, None
    if rc == 0:
        for line in lscpu.splitlines():
            if line.startswith("Core(s) per socket:"):
                cores = int(line.split(":")[1])
            elif line.startswith("Socket(s):"):
                sockets = int(line.split(":")[1])
    env["physical_cores"] = (cores * sockets) if (cores and sockets) else None

    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0]) * 1024
        env["ram_total"] = human_bytes(mem.get("MemTotal", 0))
        env["ram_available"] = human_bytes(mem.get("MemAvailable", 0))
        env["ram_total_gb"] = round(mem.get("MemTotal", 0) / 1024.0 ** 3, 1)
    except (OSError, ValueError):
        pass
    try:
        du = shutil.disk_usage(str(workdir))
        env["disk_free_workspace"] = human_bytes(du.free)
    except OSError:
        pass

    tools = {}
    for name in ("gcc", "cc", "clang", "make", "qcc", "mpicc", "mpirun", "python3", "ffmpeg",
                 "gnuplot", "convert", "magick", "pvpython", "paraview", "git", "darcs",
                 "timeout", "nvidia-smi", "bview2D", "bview3D", "ollama"):
        p = shutil.which(name, path=clean_env().get("PATH"))
        if p:
            tools[name] = p
    env["tools"] = tools
    if "gcc" in tools:
        env["gcc_version"] = _cmd_output(["gcc", "--version"])[1].splitlines()[0]
    else:
        w.append("gcc not found: install build tools (sudo apt install build-essential).")

    bas = os.environ.get("BASILISK", "")
    if not bas and "qcc" in tools:
        bas = os.path.dirname(os.path.realpath(tools["qcc"]))
    env["BASILISK"] = bas or None
    if bas and os.path.isdir(bas):
        env["basilisk_dirs"] = [d for d in ("examples", "test", "navier-stokes", "gl", "layered",
                                            "compressible", "ehd", "grid")
                                if os.path.isdir(os.path.join(bas, d))]
        qcc = os.path.join(bas, "qcc")
        if os.path.isfile(qcc):
            env["qcc_built"] = datetime.datetime.fromtimestamp(os.path.getmtime(qcc)).strftime("%Y-%m-%d")
        gl = os.path.join(bas, "gl")
        env["view_libs"] = sorted(f for f in os.listdir(gl) if f.endswith(".a")) if os.path.isdir(gl) else []
        if not env["view_libs"]:
            w.append("No compiled graphics libraries in $BASILISK/gl: view.h output (save(), movies) "
                     "will not link until they are built (see $BASILISK/gl and basilisk.fr/src/gl/INSTALL).")
        if not os.environ.get("BASILISK"):
            w.append("BASILISK is not exported; set `export BASILISK=%s` in ~/.bashrc." % bas)
    else:
        w.append("Basilisk not found (no qcc on PATH and no $BASILISK). Install it from basilisk.fr "
                 "(darcs get http://basilisk.fr/basilisk or the tarball), then export BASILISK and PATH.")

    if "gcc" in tools:
        d = tempfile.mkdtemp(prefix="ba-omp-")
        src = os.path.join(d, "t.c")
        with open(src, "w") as f:
            f.write("#include <omp.h>\n#include <stdio.h>\nint main(){printf(\"%d\",omp_get_max_threads());return 0;}\n")
        rc, _ = _cmd_output(["gcc", "-fopenmp", src, "-o", os.path.join(d, "t")])
        env["openmp"] = rc == 0
        if rc != 0:
            w.append("OpenMP test compile failed; parallel runs will need MPI or a compiler with -fopenmp.")
        shutil.rmtree(d, ignore_errors=True)
    if "mpicc" in tools:
        env["mpi"] = _cmd_output(["mpirun", "--version"])[1].splitlines()[0] if "mpirun" in tools else "mpicc only"

    probe = ("import importlib.util as u;"
             "print(','.join(m for m in ['numpy','scipy','matplotlib','PIL','pandas','h5py','vtk']"
             " if u.find_spec(m)))")
    rc, mods = _cmd_output(["python3", "-c", probe])
    env["python_packages"] = mods.split(",") if rc == 0 and mods else []
    for m in ("numpy", "matplotlib"):
        if m not in env["python_packages"]:
            w.append("Python package %s missing (sudo apt install python3-%s, or pip install --user %s)." % (m, m, m))
    if "PIL" not in env["python_packages"]:
        w.append("Pillow not installed: images can still be viewed but are not downscaled (python3-pil).")
    if "ffmpeg" not in tools:
        w.append("ffmpeg not found: Basilisk movies (.mp4) cannot be written.")
    if "nvidia-smi" in tools:
        rc, g = _cmd_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        gpus, best = [], 0.0
        if rc == 0:
            for line in g.splitlines():
                parts = [p.strip() for p in line.split(",")]
                try:
                    gb = float(parts[1]) / 1024.0
                except (IndexError, ValueError):
                    continue
                gpus.append("%s (%.0f GB)" % (parts[0], gb))
                best = max(best, gb)
        if gpus:
            env["gpu"] = "; ".join(gpus)
            env["gpu_mem_gb"] = round(best, 1)
    if str(workdir).startswith("/mnt/"):
        w.append("The workspace is on the Windows filesystem (/mnt/...): file I/O is slow in WSL. "
                 "Prefer a folder under ~ (ParaView can still open it via \\\\wsl$\\).")
    env["workspace"] = str(workdir)
    return env


def environment_text(env):
    lines = []
    lines.append("OS: %s (%s)%s" % (env.get("os"), env.get("kernel"), " [WSL]" if env.get("wsl") else ""))
    lines.append("CPU: %s | logical CPUs: %s | physical cores: %s" % (
        env.get("cpu"), env.get("logical_cpus"), env.get("physical_cores") or "?"))
    lines.append("RAM: %s total, %s available | disk free (workspace): %s" % (
        env.get("ram_total", "?"), env.get("ram_available", "?"), env.get("disk_free_workspace", "?")))
    if env.get("gpu"):
        lines.append("GPU: %s" % env["gpu"])
    lines.append("Compiler: %s | OpenMP: %s | MPI: %s" % (
        env.get("gcc_version", "none"), "yes" if env.get("openmp") else "no", env.get("mpi") or "no"))
    lines.append("BASILISK: %s%s" % (env.get("BASILISK") or "NOT FOUND",
                                      (" (qcc built %s)" % env["qcc_built"]) if env.get("qcc_built") else ""))
    if env.get("basilisk_dirs"):
        lines.append("Basilisk subdirs: %s" % ", ".join(env["basilisk_dirs"]))
    if env.get("BASILISK"):
        lines.append("view.h libs in $BASILISK/gl: %s" % (", ".join(env.get("view_libs") or []) or "none"))
    lines.append("Python %s, packages: %s" % (env.get("python"), ", ".join(env.get("python_packages") or []) or "none"))
    lines.append("Tools: %s" % ", ".join(sorted(env.get("tools", {}))))
    lines.append("Workspace: %s" % env.get("workspace"))
    for wmsg in env.get("warnings", []):
        lines.append("WARNING: " + wmsg)
    lines.append("(detected %s)" % env.get("detected"))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Safety policy
# --------------------------------------------------------------------------

CATASTROPHIC = [
    r"\brm\s+(-\S+\s+)*(/|/\*|~|~/|\$HOME|\$HOME/|\$\{HOME\}/?)(\s|$)",
    r"--no-preserve-root",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\b[^\n]*\bof=/dev/",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\b(shutdown|reboot|poweroff|halt)\b",
    r">\s*/dev/(sd|nvme|hd|vd)",
    r"\bch(mod|own)\s+(-\S+\s+)*\S+\s+/(\s|$)",
]

RISKY = [
    (r"\bsudo\b", "uses sudo"),
    (r"\b(apt|apt-get|yum|dnf|pacman|zypper|snap|brew)\s+(install|remove|purge|upgrade|update|dist-upgrade)\b",
     "system package manager"),
    (r"\b(pip3?|python3?\s+-m\s+pip|conda|mamba)\s+(install|uninstall)\b", "installs/removes Python packages"),
    (r"\bnpm\s+(install|i)\b.*\s-g\b", "global npm install"),
    (r"\bgit\s+(push|reset\s+--hard|clean\s+-\S*f)", "git push/reset/clean"),
    (r"\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", "pipes a download into a shell"),
    (r"\bcurl\b.*\s(-d|--data\S*|-F|--form|-T|--upload-file)(\s|=)", "uploads data"),
    (r"(>|>>|tee\s+(-a\s+)?)\s*~?/?(\S*/)?\.(bashrc|bash_profile|profile|zshrc)\b", "edits shell startup files"),
    (r"\b(killall|pkill)\b", "kills processes by name"),
    (r"\bkill\s+(-\S+\s+)*-1\b", "kills all processes"),
    (r"\bcrontab\b", "edits crontab"),
    (r"\b(ssh|scp|rsync|sftp)\b", "remote access/copy"),
    (r"(\.ssh/|\.aws/|\.netrc|id_rsa|id_ed25519|\.gnupg|\.git-credentials)", "touches credentials"),
    (r"ANTHROPIC_API_KEY", "touches the API key"),
    (r"\bchmod\s+-R\b|\bchown\b", "changes permissions/ownership recursively"),
    (r"\bfind\b[^|;&]*\s(-delete|-exec\s+(rm|mv|shred))\b", "find with deletion"),
    (r"\bxargs\b[^|;&]*\b(rm|mv|shred)\b", "xargs with deletion"),
    (r"/mnt/[a-z]/(Windows|Program Files)", "touches Windows system folders"),
]

DELETE_CMDS = ("rm", "rmdir", "unlink", "shred", "mv", "truncate")
DEST_CMDS = ("cp", "install", "ln", "rsync")          # only the destination (last argument) matters
SHELL_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")"}
SAFE_REDIRECTS = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty")


class Policy:
    """Decides which shell commands and file operations need the user's approval."""

    def __init__(self, work, mode, interactive, read_roots, write_roots, run_roots=None):
        self.work = Path(work)
        self.mode = mode                     # "ask" | "auto" | "full"
        self.interactive = interactive
        self.read_roots = [Path(os.path.realpath(str(r))) for r in read_roots]
        self.write_roots = [Path(os.path.realpath(str(r))) for r in write_roots]
        self.run_roots = [Path(os.path.realpath(str(r))) for r in (run_roots or [work])]
        self.always_allowed = set()

    def can_run_in(self, path):
        return any(is_within(path, r) for r in self.run_roots)

    def can_read(self, path):
        return any(is_within(path, r) for r in self.read_roots + self.write_roots)

    def can_write(self, path):
        return any(is_within(path, r) for r in self.write_roots)

    def _outside(self, arg, cwd):
        p = os.path.expandvars(os.path.expanduser(arg))
        p = p if os.path.isabs(p) else os.path.join(str(cwd), p)
        p = os.path.normpath(p)
        target = os.path.realpath(p) if os.path.exists(p) else p
        return not any(is_within(target, r) for r in self.write_roots)

    def _outside_paths(self, command, cwd):
        """Paths outside the writable folders that the command would delete, move or write to."""
        bad = []
        for m in re.finditer(r"(?<![0-9&<])>>?\s*([^\s;&|<>()]+)", command):
            target = m.group(1)
            if target not in SAFE_REDIRECTS and not target.startswith("&") and self._outside(target, cwd):
                bad.append(target)
        for m in re.finditer(r"\btee\s+((?:-\S+\s+)*)([^\s;&|<>()]+)", command):
            if self._outside(m.group(2), cwd):
                bad.append(m.group(2))
        try:
            tokens = shlex.split(command, comments=True, posix=True)
        except ValueError:
            return bad + ["(could not parse the command)"]
        names = {os.path.basename(t) for t in tokens}
        if names & set(DELETE_CMDS) and names & {"cd", "pushd"}:
            return bad + ["(paths after a cd cannot be checked)"]
        i = 0
        while i < len(tokens):
            name = os.path.basename(tokens[i])
            if name in DELETE_CMDS or name in DEST_CMDS:
                j = i + 1
                args = []
                while j < len(tokens) and tokens[j] not in SHELL_SEPARATORS:
                    if not tokens[j].startswith("-"):
                        args.append(tokens[j])
                    j += 1
                if name in DEST_CMDS:
                    args = args[-1:] if len(args) >= 2 else []
                for a in args:
                    expanded = os.path.expandvars(a)
                    if "$" in expanded or "`" in expanded:
                        bad.append(a + " (unexpanded variable)")
                    elif self._outside(a, cwd):
                        bad.append(a)
                i = j
            else:
                i += 1
        return bad

    def check(self, command, cwd):
        """Return (decision, reason) with decision in allow / ask / block."""
        for pat in CATASTROPHIC:
            if re.search(pat, command):
                return "block", "catastrophic command pattern"
        if self.mode == "full":
            return "allow", ""
        reasons = [why for pat, why in RISKY if re.search(pat, command)]
        outside = self._outside_paths(command, cwd)
        if outside:
            reasons.append("changes files outside the workspace: %s" % ", ".join(outside[:3]))
        if self.mode == "ask":
            return "ask", "; ".join(reasons) or "ask mode: every command needs approval"
        if reasons:
            return "ask", "; ".join(reasons)
        return "allow", ""

    def confirm(self, title, detail, reason):
        if reason in self.always_allowed:
            return True
        if not self.interactive:
            return False
        out("")
        out(S.yellow + S.bold + "Approval needed: " + S.reset + S.yellow + title +
            (" (%s)" % reason if reason else "") + S.reset)
        for line in detail.splitlines()[:30]:
            out("    " + line)
        while True:
            try:
                ans = input(S.yellow + "Allow? [y]es / [n]o / [a]lways for this kind: " + S.reset).strip().lower()
            except EOFError:
                return False
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no", ""):
                return False
            if ans in ("a", "always"):
                self.always_allowed.add(reason)
                return True


# --------------------------------------------------------------------------
# Simulation manager
# --------------------------------------------------------------------------

WRAPPER = ('{timeout}bash -c "$BA_CMD" > "$BA_OUT" 2> "$BA_ERR"; '
           'echo $? > "$BA_EXIT"')

PROBLEM_PATTERNS = [
    (re.compile(r"(?<![A-Za-z])(-?nan|inf)(?![A-Za-z])", re.I), "NaN/Inf in the output"),
    (re.compile(r"Segmentation fault|SIGSEGV", re.I), "segmentation fault"),
    (re.compile(r"Floating point exception|SIGFPE", re.I), "floating point exception"),
    (re.compile(r"Assertion .* failed|assertion failed", re.I), "assertion failure"),
    (re.compile(r"core dumped", re.I), "core dumped"),
    (re.compile(r"\bKilled\b|Out of memory|oom-kill", re.I), "killed (possibly out of memory)"),
    (re.compile(r"Warning: .*(convergence|not converge)|max iterations", re.I), "solver convergence warning"),
]


def tail_file(path, nlines=15, max_bytes=200000):
    try:
        with open(str(path), "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode(errors="replace")
    except OSError:
        return ""
    lines = data.splitlines()
    return "\n".join(lines[-nlines:])


class SimManager:
    def __init__(self, home, work, policy):
        self.registry = Path(home) / "runs.json"
        self.work = Path(work)
        self.policy = policy
        self.procs = {}

    def _load(self):
        return read_json(self.registry, [])

    def _save(self, runs):
        atomic_write(self.registry, json.dumps(runs[-500:], indent=1))

    def _find(self, run_id):
        runs = self._load()
        for r in runs:
            if r["id"] == run_id:
                return r
        matches = [r for r in runs if r["id"].startswith(run_id)]
        return matches[-1] if len(matches) == 1 else None

    def state(self, r):
        p = self.procs.get(r["id"])
        if p is not None:
            p.poll()
        exit_file = Path(r["run_dir"]) / "exit_code"
        code = None
        if exit_file.exists():
            try:
                code = int(exit_file.read_text().strip() or "-1")
            except (OSError, ValueError):
                code = -1
        alive = pid_alive(r["pid"]) if code is None else False
        if alive:
            return "running", None
        if code is None:
            return ("stopped" if r.get("stopped") else "terminated (no exit code)"), None
        if code == 0:
            return "finished", 0
        if code == 124 and r.get("max_hours"):
            return "time limit reached", code
        if code > 128:
            return "failed (%s)" % signal_name(code - 128), code
        return "failed (exit %d)" % code, code

    def _usage(self, r):
        table = _proc_table()
        pids = [r["pid"]] + descendants(r["pid"], table)
        cpu = sum(table[p][3] for p in pids if p in table)
        rss = sum(table[p][4] for p in pids if p in table)
        return cpu, rss

    def start(self, command, cwd, threads=1, label=None, force=False, max_hours=None):
        cwd = resolve_path(cwd or ".", self.work)
        if not self.policy.can_run_in(cwd):
            return "Error: the run directory must be inside the workspace (%s)." % self.work, True
        if not cwd.is_dir():
            return "Error: directory %s does not exist." % cwd, True
        threads = max(1, int(threads or 1))
        ncpu = os.cpu_count() or 1
        runs = self._load()
        active = [r for r in runs if self.state(r)[0] == "running"]
        used = sum(int(r.get("threads", 1)) for r in active)
        if used + threads > ncpu and not force:
            lst = "\n".join("  %s: %s threads, %s" % (r["id"], r.get("threads"), r["command"]) for r in active)
            return ("Not started: %d threads requested but %d of %d logical CPUs are already used by:\n%s\n"
                    "Wait for a run to finish (sim_wait), reduce threads, or pass force=true."
                    % (threads, used, ncpu, lst)), True
        decision, reason = self.policy.check(command, cwd)
        if decision == "block":
            return "Blocked: this command matches a destructive pattern and will not be run.", True
        if decision == "ask" and not self.policy.confirm("start background run in %s" % cwd, command, reason):
            return "Not started: the user did not approve (%s)." % reason, True
        base = "%s-%s" % (slug(label or command.split()[0].lstrip("./")), datetime.datetime.now().strftime("%m%d-%H%M%S"))
        taken = {r["id"] for r in runs}
        rid, k = base, 2
        while rid in taken or (cwd / ".runs" / rid).exists():
            rid, k = "%s-%d" % (base, k), k + 1
        run_dir = cwd / ".runs" / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        env = clean_env()
        env["OMP_NUM_THREADS"] = str(threads)
        env["BA_CMD"] = command
        env["BA_OUT"] = str(run_dir / "stdout.log")
        env["BA_ERR"] = str(run_dir / "stderr.log")
        env["BA_EXIT"] = str(run_dir / "exit_code")
        tmo = ""
        if max_hours:
            tmo = "timeout -s TERM -k 60 %d " % int(float(max_hours) * 3600)
        p = subprocess.Popen(["bash", "-c", WRAPPER.format(timeout=tmo)], cwd=str(cwd), env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        self.procs[rid] = p
        entry = {"id": rid, "command": command, "cwd": str(cwd), "threads": threads, "pid": p.pid,
                 "start": time.time(), "started": now_str(), "run_dir": str(run_dir),
                 "max_hours": max_hours, "workspace": str(self.work)}
        runs = self._load()
        runs.append(entry)
        self._save(runs)
        time.sleep(1.5)
        st, _ = self.state(entry)
        msg = ("Started run id: %s\ncommand: %s\ncwd: %s\nthreads (OMP_NUM_THREADS): %d\n"
               "logs: %s/stdout.log and stderr.log\n"
               % (rid, command, cwd, threads, run_dir))
        if st != "running":
            msg += "\nThe run already ended: " + self.describe(entry)
            return msg, st != "finished"
        return msg + "Status: running. Use sim_wait to wait for it or sim_status to inspect.", False

    def describe(self, r, ntail=15):
        st, code = self.state(r)
        elapsed = (time.time() - r["start"]) if st == "running" else None
        if elapsed is None:
            ef = Path(r["run_dir"]) / "exit_code"
            end = os.path.getmtime(str(ef)) if ef.exists() else time.time()
            elapsed = end - r["start"]
        lines = ["run %s [%s]  pid %d, threads %s, elapsed %s" % (
            r["id"], st, r["pid"], r.get("threads"), human_time(elapsed))]
        if st == "running":
            cpu, rss = self._usage(r)
            lines[0] += ", avg CPU %.1f cores, memory %s" % (cpu / max(elapsed, 1.0), human_bytes(rss))
        lines.append("command: %s   (cwd: %s)" % (r["command"], r["cwd"]))
        err = tail_file(Path(r["run_dir"]) / "stderr.log", ntail)
        so = tail_file(Path(r["run_dir"]) / "stdout.log", 6)
        if err:
            lines.append("--- stderr (last lines) ---\n" + err)
        if so:
            lines.append("--- stdout (last lines) ---\n" + so)
        recent = tail_file(Path(r["run_dir"]) / "stderr.log", 200) + "\n" + tail_file(
            Path(r["run_dir"]) / "stdout.log", 50)
        flags = sorted({why for rx, why in PROBLEM_PATTERNS if rx.search(recent)})
        if flags:
            lines.append("PROBLEMS DETECTED: " + "; ".join(flags))
        try:
            files = [f for f in Path(r["cwd"]).iterdir() if f.is_file()]
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                lines.append("newest files in cwd: " + ", ".join(
                    "%s (%s, %s ago)" % (f.name, human_bytes(f.stat().st_size),
                                         human_time(time.time() - f.stat().st_mtime)) for f in files[:8]))
        except OSError:
            pass
        return "\n".join(lines)

    def status(self, run_id=None):
        runs = self._load()
        if run_id:
            r = self._find(run_id)
            return (self.describe(r), False) if r else ("Unknown run id: %s" % run_id, True)
        recent = [r for r in runs if self.state(r)[0] == "running" or time.time() - r["start"] < 3 * 86400]
        if not recent:
            return "No running or recent runs.", False
        return "\n\n".join(self.describe(r, 6) for r in recent[-12:]), False

    def short_list(self):
        runs = self._load()
        rows = []
        for r in runs[-30:]:
            st, _ = self.state(r)
            if st == "running" or time.time() - r["start"] < 3 * 86400:
                rows.append("%s [%s] %s (cwd %s, started %s)" % (r["id"], st, r["command"], r["cwd"], r.get("started")))
        return "\n".join(rows) or "none"

    def wait(self, run_id, max_minutes=30, until=None, poll=30):
        r = self._find(run_id)
        if not r:
            return "Unknown run id: %s" % run_id, True
        max_minutes = min(max(float(max_minutes or 30), 0.1), 24 * 60)
        poll = min(max(int(poll or 30), 5), 600)
        regex = None
        if until:
            try:
                regex = re.compile(until)
            except re.error as e:
                return "Invalid regex for 'until': %s" % e, True
        files = [Path(r["run_dir"]) / "stderr.log", Path(r["run_dir"]) / "stdout.log"]
        offsets = [0, 0]      # the whole log is searched, so a match just before the wait is not missed
        deadline = time.time() + max_minutes * 60
        tty = sys.stdout.isatty()
        try:
            while True:
                st, _ = self.state(r)
                if st != "running":
                    if tty:
                        out("")
                    return "Run ended.\n" + self.describe(r), False
                if regex:
                    for k, f in enumerate(files):
                        try:
                            with open(str(f), "rb") as fh:
                                fh.seek(offsets[k])
                                new = fh.read().decode(errors="replace")
                                offsets[k] = fh.tell()
                        except OSError:
                            continue
                        m = regex.search(new)
                        if m:
                            line = new[new.rfind("\n", 0, m.start()) + 1:].split("\n", 1)[0]
                            if tty:
                                out("")
                            return "Condition matched in the log: %s\n%s" % (line, self.describe(r)), False
                left = deadline - time.time()
                if left <= 0:
                    if tty:
                        out("")
                    return ("Still running after waiting %s.\n" % human_time(max_minutes * 60)
                            + self.describe(r)), False
                last = tail_file(files[0], 1) or tail_file(files[1], 1)
                msg = "  waiting on %s | %s elapsed | %s" % (
                    r["id"], human_time(time.time() - r["start"]), last[:90])
                if tty:
                    sys.stdout.write("\r" + S.dim + msg.ljust(120)[:120] + S.reset)
                    sys.stdout.flush()
                time.sleep(min(poll, max(left, 1)))
        except KeyboardInterrupt:
            if tty:
                out("")
            return "Waiting interrupted by the user (the run continues).\n" + self.describe(r), False

    def stop(self, run_id):
        r = self._find(run_id)
        if not r:
            return "Unknown run id: %s" % run_id, True
        if self.state(r)[0] != "running":
            return "Run %s is not running.\n%s" % (r["id"], self.describe(r)), False
        kill_tree(r["pid"], signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.5)
            if not pid_alive(r["pid"]) and not any(pid_alive(p) for p in descendants(r["pid"])):
                break
        else:
            kill_tree(r["pid"], signal.SIGKILL)
        runs = self._load()
        for x in runs:
            if x["id"] == r["id"]:
                x["stopped"] = True
        self._save(runs)
        r["stopped"] = True
        return "Stopped run %s.\n%s" % (r["id"], self.describe(r)), False


# --------------------------------------------------------------------------
# Images and web pages
# --------------------------------------------------------------------------

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".gif": "image/gif", ".webp": "image/webp"}


def load_image(path):
    """Return (media_type, base64_data, (w, h)) ready for the API, downscaling if possible."""
    try:
        from PIL import Image  # optional dependency
        im = Image.open(str(path))
        im.load()
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, float(MAX_IMAGE_EDGE) / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        data, mtype = buf.getvalue(), "image/png"
        if len(data) > 3500000:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=88)
            data, mtype = buf.getvalue(), "image/jpeg"
        return mtype, base64.b64encode(data).decode(), im.size
    except ImportError:
        pass
    ext = Path(path).suffix.lower()
    if ext in IMAGE_TYPES and os.path.getsize(str(path)) < 7000000:
        with open(str(path), "rb") as f:
            return IMAGE_TYPES[ext], base64.b64encode(f.read()).decode(), None
    conv = shutil.which("magick") or shutil.which("convert")
    if conv:
        tmp = tempfile.mktemp(suffix=".png")
        cmd = [conv, str(path), "-resize", "%dx%d>" % (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), tmp]
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            with open(tmp, "rb") as f:
                data = f.read()
            os.remove(tmp)
            return "image/png", base64.b64encode(data).decode(), None
    raise ValueError("cannot load %s: install Pillow (python3-pil) or ImageMagick, or save as PNG/JPEG" % path)


def html_to_text(page):
    page = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", "", page)
    page = re.sub(r"(?i)<br\s*/?>", "\n", page)
    page = re.sub(r"(?i)</(p|div|li|tr|h[1-6]|pre|table|section|blockquote)>", "\n", page)
    page = re.sub(r"(?i)<li\b[^>]*>", "- ", page)
    page = re.sub(r"(?s)<[^>]+>", "", page)
    page = html.unescape(page)
    page = re.sub(r"[ \t]+\n", "\n", page)
    page = re.sub(r"\n{3,}", "\n\n", page)
    return page.strip()


# --------------------------------------------------------------------------
# Log tables (for the no-AI `sim plot` command)
# --------------------------------------------------------------------------

def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_table(path):
    """Read a whitespace-separated numeric log (Basilisk style). Handles '# name name' headers,
    a header line without '#', and 'key=value' tokens. Returns (column_names, rows)."""
    names, rows = None, []
    with open(str(path), encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                toks = s.lstrip("#").replace(",", " ").split()
                if toks and not any(_is_number(t) for t in toks) and names is None:
                    names = toks
                continue
            vals, keys, ok = [], [], True
            for t in s.replace(",", " ").split():
                k = None
                if "=" in t:
                    k, t = t.split("=", 1)
                try:
                    vals.append(float(t))
                except ValueError:
                    ok = False
                    break
                keys.append(k)
            if ok and vals:
                rows.append((vals, keys))
            elif not rows and names is None:
                toks = s.split()
                if toks and not any(_is_number(t) for t in toks):
                    names = toks
    if not rows:
        return [], []
    counts = {}
    for v, _ in rows:
        counts[len(v)] = counts.get(len(v), 0) + 1
    n = max(counts, key=lambda c: (counts[c], c))
    good = [(v, k) for v, k in rows if len(v) == n]
    keys = good[0][1]
    if names and len(names) == n:
        cols = list(names)
    else:
        cols = [keys[i] or (names[i] if names and i < len(names) else "col%d" % (i + 1)) for i in range(n)]
    return cols, [v for v, _ in good]


def table_summary(cols, data):
    lines = ["%d rows, %d columns" % (len(data), len(cols)),
             "%-14s %14s %14s %14s %14s" % ("column", "first", "last", "min", "max")]
    for j, c in enumerate(cols):
        col = [r[j] for r in data]
        finite = [x for x in col if x == x and abs(x) != float("inf")]
        bad = len(col) - len(finite)
        if finite:
            lines.append("%-14s %14.6g %14.6g %14.6g %14.6g%s" % (
                c[:14], col[0], col[-1], min(finite), max(finite), ("   (%d NaN/Inf!)" % bad) if bad else ""))
        else:
            lines.append("%-14s all NaN/Inf!" % c[:14])
    return "\n".join(lines)


def plot_table(cols, data, out, xname=None, ynames=None, title="", logy=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if xname and xname in cols:
        xi = cols.index(xname)
    else:
        xi = cols.index("t") if "t" in cols else 0
    if ynames:
        ys = [cols.index(y) for y in ynames if y in cols]
    else:
        ys = [j for j, c in enumerate(cols) if j != xi and c.lower() not in ("i", "step", "it", "iter")][:8]
    if not ys:
        ys = [j for j in range(len(cols)) if j != xi][:1]
    x = [r[xi] for r in data]
    fig, axes = plt.subplots(len(ys), 1, figsize=(8, 1.9 * len(ys) + 0.9), sharex=True, squeeze=False)
    for ax, j in zip(axes[:, 0], ys):
        ax.plot(x, [r[j] for r in data], lw=1.3)
        ax.set_ylabel(cols[j])
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
    axes[-1, 0].set_xlabel(cols[xi])
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out), dpi=130)
    plt.close(fig)
    return [cols[j] for j in ys], cols[xi]


# --------------------------------------------------------------------------
# Claude API client (streaming, stdlib only)
# --------------------------------------------------------------------------

class APIError(Exception):
    def __init__(self, status, etype, message, retry_after=None):
        Exception.__init__(self, "%s %s: %s" % (status, etype, message))
        self.status = status
        self.etype = etype
        self.message = message
        self.retry_after = retry_after


RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}


def iter_sse(resp):
    event, data = None, []
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, "\n".join(data)


def clean_block(b):
    t = b.get("type")
    if t == "text":
        return {"type": "text", "text": b.get("text", "")}
    if t == "tool_use":
        return {"type": "tool_use", "id": b["id"], "name": b["name"], "input": b.get("input") or {}}
    if t == "thinking":
        return {"type": "thinking", "thinking": b.get("thinking", ""), "signature": b.get("signature", "")}
    if t == "redacted_thinking":
        return {"type": "redacted_thinking", "data": b.get("data", "")}
    return {k: v for k, v in b.items() if not k.startswith("_")}


class StreamPrinter:
    """Prints the assistant's text (and optional thinking summaries) as it streams."""

    def __init__(self, show_thinking=False, quiet=False):
        self.show_thinking = show_thinking
        self.quiet = quiet
        self.kind = None
        self.wrote = False

    def _switch(self, kind):
        if self.kind != kind:
            if self.wrote:
                out(S.reset)
            self.kind = kind
            if kind == "thinking":
                sys.stdout.write(S.dim + S.italic + "thinking: ")

    def text(self, t):
        if self.quiet or not t:
            return
        self._switch("text")
        sys.stdout.write(t)
        sys.stdout.flush()
        self.wrote = True

    def thinking(self, t):
        if self.quiet or not self.show_thinking or not t:
            return
        self._switch("thinking")
        sys.stdout.write(t)
        sys.stdout.flush()
        self.wrote = True

    def finish(self):
        if self.wrote:
            out(S.reset)
        self.kind, self.wrote = None, False


class Claude:
    provider = "claude"

    def __init__(self, api_key, model, effort, max_tokens, cache_ttl="5m", show_thinking=False):
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self.show_thinking = show_thinking
        self.timeout = 600
        self.supports_vision = True
        self.max_tool_output = MAX_TOOL_OUTPUT
        self.context_limit = 1000000

    def summarize(self, system, tools, messages, prompt):
        msgs = [dict(m) for m in messages]
        ask = {"type": "text", "text": prompt}
        if msgs and msgs[-1]["role"] == "user":
            c = msgs[-1]["content"]
            msgs[-1]["content"] = (c if isinstance(c, list) else [{"type": "text", "text": c}]) + [ask]
        else:
            msgs.append({"role": "user", "content": [ask]})
        r = self.create(system, tools, msgs, tool_choice={"type": "none"})
        return r, "\n".join(b["text"] for b in r["content"] if b.get("type") == "text").strip()

    def body(self, system, tools, messages, tool_choice=None):
        b = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
             "messages": messages, "stream": True}
        if tools:
            b["tools"] = tools
        cc = {"type": "ephemeral"}
        if self.cache_ttl == "1h":
            cc["ttl"] = "1h"
        b["cache_control"] = cc
        if self.effort:
            b["output_config"] = {"effort": self.effort}
        if self.show_thinking:
            b["thinking"] = {"type": "adaptive", "display": "summarized"}
        if tool_choice:
            b["tool_choice"] = tool_choice
        return b

    def create(self, system, tools, messages, tool_choice=None, printer=None):
        payload = json.dumps(self.body(system, tools, messages, tool_choice)).encode("utf-8")
        if len(payload) > MAX_REQUEST_BYTES + 6 * 1024 * 1024:
            raise APIError(413, "request_too_large", "request is %s" % human_bytes(len(payload)))
        printer = printer or StreamPrinter(quiet=True)
        delay = 2.0
        for attempt in range(9):
            try:
                return self._stream_once(payload, printer)
            except APIError as e:
                if e.status not in RETRYABLE or attempt == 8:
                    raise
                wait = delay
                if e.retry_after:
                    try:
                        wait = max(wait, float(e.retry_after))
                    except ValueError:
                        pass
                printer.finish()
                warn("  API %s (%s); retrying in %.0fs..." % (e.status, e.etype, wait))
            except (urllib.error.URLError, socket.timeout, ConnectionError, http.client.HTTPException, OSError) as e:
                if attempt == 8:
                    raise APIError(0, "network_error", str(e))
                printer.finish()
                warn("  Network problem (%s); retrying in %.0fs..." % (e, delay))
                wait = delay
            time.sleep(wait + random.uniform(0, 1))
            delay = min(delay * 2, 60)
        raise APIError(0, "retry_exhausted", "giving up")

    def _stream_once(self, payload, printer):
        req = urllib.request.Request(API_URL, data=payload, method="POST", headers={
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
            "user-agent": "basilisk-agent/%s" % VERSION,
        })
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                j = json.loads(raw)
                et, msg = j["error"]["type"], j["error"]["message"]
            except (ValueError, KeyError, TypeError):
                et, msg = "http_error", raw[:500]
            raise APIError(e.code, et, msg, e.headers.get("retry-after"))
        blocks, partial, usage, stop_reason = {}, {}, {}, None
        with resp:
            for event, data in iter_sse(resp):
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                t = ev.get("type") or event
                if t == "message_start":
                    usage.update(ev.get("message", {}).get("usage") or {})
                elif t == "content_block_start":
                    idx = ev["index"]
                    cb = dict(ev["content_block"])
                    blocks[idx] = cb
                    if cb.get("type") == "tool_use":
                        partial[idx] = []
                        cb["input"] = {}
                    elif cb.get("type") == "text" and cb.get("text"):
                        printer.text(cb["text"])
                elif t == "content_block_delta":
                    idx = ev["index"]
                    b = blocks.setdefault(idx, {"type": "text", "text": ""})
                    d = ev.get("delta", {})
                    dt = d.get("type")
                    if dt == "text_delta":
                        b["text"] = b.get("text", "") + d.get("text", "")
                        printer.text(d.get("text", ""))
                    elif dt == "input_json_delta":
                        partial.setdefault(idx, []).append(d.get("partial_json", ""))
                    elif dt == "thinking_delta":
                        b["thinking"] = b.get("thinking", "") + d.get("thinking", "")
                        printer.thinking(d.get("thinking", ""))
                    elif dt == "signature_delta":
                        b["signature"] = b.get("signature", "") + d.get("signature", "")
                elif t == "content_block_stop":
                    idx = ev["index"]
                    if idx in partial:
                        s = "".join(partial.pop(idx))
                        try:
                            blocks[idx]["input"] = json.loads(s) if s.strip() else {}
                        except ValueError:
                            blocks[idx]["input"] = {}
                            blocks[idx]["_invalid"] = True
                elif t == "message_delta":
                    stop_reason = ev.get("delta", {}).get("stop_reason") or stop_reason
                    for k, v in (ev.get("usage") or {}).items():
                        if v is not None:
                            usage[k] = v
                elif t == "message_stop":
                    break
                elif t == "error":
                    e = ev.get("error", {})
                    status = 529 if e.get("type") == "overloaded_error" else 500
                    raise APIError(status, e.get("type", "stream_error"), e.get("message", ""))
        printer.finish()
        for idx in list(partial):   # stream ended inside a tool call
            blocks[idx]["_invalid"] = True
        content, invalid_ids = [], set()
        for i in sorted(blocks):
            b = blocks[i]
            if b.get("type") == "text" and not b.get("text"):
                continue
            if b.get("_invalid") and b.get("type") == "tool_use":
                invalid_ids.add(b["id"])
            content.append(clean_block(b))
        return {"content": content, "stop_reason": stop_reason, "usage": usage, "invalid_tool_ids": invalid_ids}


# ---------------- helpers shared by the Ollama and OpenAI-compatible clients ----------------

def _new_call_id():
    return "call_" + os.urandom(6).hex()


def _tool_result_parts(content):
    """Split a tool_result content into (text, [(media_type, base64)])."""
    if isinstance(content, str):
        return content, []
    texts, images = [], []
    for c in content or []:
        if c.get("type") == "text":
            texts.append(c.get("text", ""))
        elif c.get("type") == "image":
            src = c.get("source", {})
            images.append((src.get("media_type", "image/png"), src.get("data", "")))
    return "\n".join(texts), images


def chat_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


def to_chat_messages(system, messages, style, vision):
    """Convert the internal (Anthropic-style) history into Ollama ('ollama') or OpenAI ('openai') chat messages."""
    out = [{"role": "system", "content": system}]
    names = {}
    for m in messages:
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        if m["role"] == "assistant":
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
            calls = [b for b in content if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text}
            if calls:
                for b in calls:
                    names[b["id"]] = b["name"]
                if style == "ollama":
                    msg["tool_calls"] = [{"id": b["id"], "function": {"name": b["name"], "arguments": b.get("input") or {}}}
                                         for b in calls]
                else:
                    msg["tool_calls"] = [{"id": b["id"], "type": "function",
                                          "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                                         for b in calls]
            out.append(msg)
            continue
        texts, images = [], []
        for b in content:
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "image":
                src = b.get("source", {})
                images.append((src.get("media_type", "image/png"), src.get("data", "")))
            elif t == "tool_result":
                txt, imgs = _tool_result_parts(b.get("content"))
                if b.get("is_error"):
                    txt = "ERROR: " + txt
                tm = {"role": "tool", "content": txt or "(no output)", "tool_call_id": b["tool_use_id"]}
                if style == "ollama":
                    tm["tool_name"] = names.get(b["tool_use_id"], "")
                out.append(tm)
                images += imgs
        if images and vision:
            note = "Image(s) returned by the last tool call:"
            if style == "ollama":
                out.append({"role": "user", "content": note, "images": [d for _, d in images]})
            else:
                parts = [{"type": "text", "text": note}] + [
                    {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mt, d)}} for mt, d in images]
                out.append({"role": "user", "content": parts})
        if texts:
            out.append({"role": "user", "content": "\n\n".join(texts)})
    return out


def parse_text_tool_calls(text):
    """Some local models write tool calls as text instead of real function calls. Recover them.
    Returns (remaining_text, [(name, arguments)])."""
    names = {t["name"] for t in TOOLS}
    calls, spans = [], []

    def take(obj):
        if not isinstance(obj, dict):
            return False
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
        name = obj.get("name") or fn.get("name")
        args = obj.get("arguments", obj.get("parameters", fn.get("arguments")))
        if name not in names:
            return False
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return False
        calls.append((name, args if isinstance(args, dict) else {}))
        return True

    for rx in (r"<tool_call>\s*(\{.*?\})\s*</tool_call>", r"```(?:json|tool_call)?\s*(\{.*?\})\s*```"):
        for m in re.finditer(rx, text, re.S):
            try:
                if take(json.loads(m.group(1))):
                    spans.append(m.span())
            except ValueError:
                pass
        if calls:
            break
    if not calls:
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                if take(json.loads(s)):
                    spans.append((0, len(text)))
            except ValueError:
                pass
    for a, b in sorted(spans, reverse=True):
        text = text[:a] + text[b:]
    return text.strip(), calls


def build_response(text, raw_calls, truncated, usage):
    """Build the agent's internal response from text and (id, name, arguments) tool calls."""
    calls = []
    for cid, name, args in raw_calls:
        bad = False
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args, bad = {}, True
        if not isinstance(args, dict):
            args, bad = {}, True
        calls.append((cid or _new_call_id(), name or "unknown", args, bad))
    if not calls and text:
        text, parsed = parse_text_tool_calls(text)
        calls = [(_new_call_id(), n, a, False) for n, a in parsed]
    content, invalid = [], set()
    if text.strip():
        content.append({"type": "text", "text": text})
    for cid, name, args, bad in calls:
        content.append({"type": "tool_use", "id": cid, "name": name, "input": args})
        if bad:
            invalid.add(cid)
    stop = "tool_use" if calls else ("max_tokens" if truncated else "end_turn")
    return {"content": content, "stop_reason": stop, "usage": usage, "invalid_tool_ids": invalid}


def flatten_transcript(messages, limit):
    """Plain-text transcript of the conversation, used to summarise it with any model."""
    lines = []
    for m in messages:
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for b in content:
            t = b.get("type")
            if t == "text":
                txt = b.get("text", "")
                if "<session_context>" in txt:
                    continue
                lines.append(("USER: " if m["role"] == "user" else "ASSISTANT: ") + txt)
            elif t == "tool_use":
                lines.append("ASSISTANT called %s %s" % (b["name"], json.dumps(b.get("input"))[:600]))
            elif t == "tool_result":
                txt, _ = _tool_result_parts(b.get("content"))
                lines.append("RESULT%s: %s" % (" (error)" if b.get("is_error") else "", clip(txt, 1500)))
    return clip("\n".join(lines), limit)


def estimate_tokens(system, messages):
    """Rough token count of a request (about 3.5 characters per token; images about 1200 tokens)."""
    chars, images = len(system) + 14000, 0
    for m in messages:
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for b in content:
            t = b.get("type")
            if t == "text":
                chars += len(b.get("text", ""))
            elif t == "tool_use":
                chars += len(json.dumps(b.get("input")))
            elif t == "tool_result":
                txt, imgs = _tool_result_parts(b.get("content"))
                chars += len(txt)
                images += len(imgs)
            elif t == "image":
                images += 1
    return int(chars / 3.5) + 1200 * images


class LocalClientBase:
    effort = None
    cache_ttl = "5m"

    def summarize(self, system, tools, messages, prompt):
        flat = flatten_transcript(messages, int(self.context_limit * 2.0))
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt + "\n\n<transcript>\n" + flat
                                              + "\n</transcript>"}]}]
        r = self.create(system, None, msgs)
        return r, "\n".join(b["text"] for b in r["content"] if b.get("type") == "text").strip()


class OllamaClient(LocalClientBase):
    """A model running on this computer through Ollama (https://ollama.com): free, private, no API key."""
    provider = "ollama"

    def __init__(self, base_url, model, num_ctx, max_tokens, show_thinking=False, vision="auto"):
        self.base = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.show_thinking = show_thinking
        self.vision_mode = vision
        self.supports_vision = vision == "on"
        self.context_limit = num_ctx
        self.max_tool_output = int(min(MAX_TOOL_OUTPUT, max(4000, num_ctx * 0.4)))
        self.timeout = 1800
        self.capabilities = None

    def _request(self, path, payload=None, timeout=10):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data else "GET",
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def probe(self):
        """Check that Ollama is running and the model is installed. Returns (ok, message)."""
        try:
            tags = self._request("/api/tags", timeout=5)
        except (urllib.error.URLError, socket.timeout, OSError, ValueError) as e:
            return False, ("Ollama is not reachable at %s (%s).\nInstall it (curl -fsSL https://ollama.com/install.sh | sh), "
                           "start it (ollama serve), and pull a model (ollama pull %s)." % (self.base, e, self.model))
        names = [m.get("name", "") for m in tags.get("models", [])]
        if self.model not in names and self.model + ":latest" not in names:
            return False, "Model %s is not installed. Run:  ollama pull %s\nInstalled models: %s" % (
                self.model, self.model, ", ".join(names) or "none")
        try:
            caps = self._request("/api/show", {"model": self.model}, timeout=20).get("capabilities")
            if isinstance(caps, list):
                self.capabilities = caps
        except (urllib.error.URLError, socket.timeout, OSError, ValueError):
            pass
        if self.vision_mode == "auto":
            self.supports_vision = bool(self.capabilities and "vision" in self.capabilities)
        msg = "Ollama model %s is ready (capabilities: %s; vision %s)." % (
            self.model, ", ".join(self.capabilities) if self.capabilities else "unknown",
            "on" if self.supports_vision else "off")
        if self.capabilities is not None and "tools" not in self.capabilities:
            msg += "\nWARNING: this model does not support tool calling, which the agent needs. Choose another model."
        return True, msg

    def create(self, system, tools, messages, tool_choice=None, printer=None):
        printer = printer or StreamPrinter(quiet=True)
        body = {"model": self.model, "stream": True, "keep_alive": "30m",
                "messages": to_chat_messages(system, messages, "ollama", self.supports_vision),
                "options": {"num_ctx": self.num_ctx, "temperature": 0.2, "num_predict": self.max_tokens}}
        if tools and tool_choice != {"type": "none"}:
            body["tools"] = chat_tools(tools)
        payload = json.dumps(body).encode()
        for attempt in range(3):
            try:
                return self._stream(payload, printer)
            except APIError as e:
                if e.status not in (500, 502, 503) or attempt == 2:
                    raise
                printer.finish()
                warn("  Local model error (%s); retrying..." % e.message[:150])
                time.sleep(3)

    def _stream(self, payload, printer):
        req = urllib.request.Request(self.base + "/api/chat", data=payload, method="POST",
                                     headers={"content-type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                msg = str(json.loads(raw).get("error", raw))
            except ValueError:
                msg = raw
            if e.code == 404:
                msg += "  (install the model with: ollama pull %s)" % self.model
            raise APIError(e.code, "ollama_error", msg[:600])
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            raise APIError(0, "network_error", "cannot reach Ollama at %s: %s (is `ollama serve` running?)" % (self.base, e))
        text, calls, usage, done_reason = "", [], {}, None
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise APIError(500, "ollama_error", str(ev["error"]))
                msg = ev.get("message") or {}
                if msg.get("thinking"):
                    printer.thinking(msg["thinking"])
                if msg.get("content"):
                    text += msg["content"]
                    printer.text(msg["content"])
                calls += msg.get("tool_calls") or []
                if ev.get("done"):
                    done_reason = ev.get("done_reason")
                    usage = {"input_tokens": ev.get("prompt_eval_count") or 0, "output_tokens": ev.get("eval_count") or 0}
                    break
        printer.finish()
        raw_calls = [(tc.get("id"), (tc.get("function") or {}).get("name"), (tc.get("function") or {}).get("arguments"))
                     for tc in calls]
        return build_response(text, raw_calls, done_reason == "length", usage)


class OpenAIClient(LocalClientBase):
    """Any OpenAI-compatible chat-completions server: LM Studio, vLLM, llama.cpp server, or a cloud provider."""
    provider = "openai"

    def __init__(self, base_url, api_key, model, max_tokens, show_thinking=False, vision="auto", context=128000):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.show_thinking = show_thinking
        self.supports_vision = vision == "on"
        self.context_limit = context
        self.max_tool_output = int(min(MAX_TOOL_OUTPUT, max(4000, context * 0.25)))
        self.timeout = 900

    def create(self, system, tools, messages, tool_choice=None, printer=None):
        printer = printer or StreamPrinter(quiet=True)
        body = {"model": self.model, "stream": True, "temperature": 0.2, "max_tokens": self.max_tokens,
                "messages": to_chat_messages(system, messages, "openai", self.supports_vision)}
        if tools and tool_choice != {"type": "none"}:
            body["tools"] = chat_tools(tools)
        payload = json.dumps(body).encode()
        delay = 2.0
        for attempt in range(6):
            try:
                return self._stream(payload, printer)
            except APIError as e:
                if e.status not in RETRYABLE and e.status != 0 or attempt == 5:
                    raise
                printer.finish()
                warn("  Server %s (%s); retrying in %.0fs..." % (e.status, e.etype, delay))
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def _stream(self, payload, printer):
        headers = {"content-type": "application/json", "accept": "text/event-stream",
                   "user-agent": "basilisk-agent/%s" % VERSION}
        if self.api_key:
            headers["authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base + "/chat/completions", data=payload, method="POST", headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                err = json.loads(raw).get("error", raw)
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            except ValueError:
                msg = raw
            raise APIError(e.code, "server_error", msg[:600], e.headers.get("retry-after"))
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            raise APIError(0, "network_error", "cannot reach %s: %s" % (self.base, e))
        text, calls, usage, finish = "", {}, {}, None
        with resp:
            for _, data in iter_sse(resp):
                if data.strip() == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise APIError(500, "server_error", str(ev["error"]))
                u = ev.get("usage")
                if u:
                    usage = {"input_tokens": u.get("prompt_tokens") or 0, "output_tokens": u.get("completion_tokens") or 0}
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or ch.get("message") or {}
                    thought = d.get("reasoning_content") or d.get("reasoning")
                    if isinstance(thought, str) and thought:
                        printer.thinking(thought)
                    if d.get("content"):
                        text += d["content"]
                        printer.text(d["content"])
                    for k, tc in enumerate(d.get("tool_calls") or []):
                        slot = calls.setdefault(tc.get("index", k), {"id": None, "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        f = tc.get("function") or {}
                        if f.get("name") and not slot["name"]:
                            slot["name"] = f["name"]
                        a = f.get("arguments")
                        if a:
                            slot["args"] += a if isinstance(a, str) else json.dumps(a)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        printer.finish()
        raw_calls = [(c["id"], c["name"], c["args"]) for _, c in sorted(calls.items())]
        return build_response(text, raw_calls, finish == "length", usage)


def make_client(cfg):
    if cfg.provider == "ollama":
        return OllamaClient(cfg.base_url or OLLAMA_URL, cfg.model, cfg.num_ctx, cfg.local_max_tokens,
                            cfg.show_thinking, cfg.vision)
    if cfg.provider == "openai":
        return OpenAIClient(cfg.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                            os.environ.get("OPENAI_API_KEY", ""), cfg.model, cfg.local_max_tokens,
                            cfg.show_thinking, cfg.vision, cfg.context)
    return Claude(cfg.api_key, cfg.model, cfg.effort, cfg.max_tokens, cfg.cache_ttl, cfg.show_thinking)


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Basilisk Agent, an autonomous computational fluid dynamics (CFD) engineer working directly on the user's own computer. You set up, run, and analyse simulations with Basilisk (http://basilisk.fr), the adaptive C solver framework by Stephane Popinet, for any kind of flow problem: single- and multiphase flows, interfaces and surface tension, heat and mass transfer, embedded solid boundaries, axisymmetric and 3D problems, shallow water and multilayer flows, compressible flows, viscoelastic and electrohydrodynamic flows, and plain PDE solves. You work for a CFD researcher: be rigorous, quantitative and honest about uncertainty.

You act through tools that run on the user's machine (a shell, file tools, a Basilisk source search, a background simulation manager, an image viewer for your own figures, a web fetcher for basilisk.fr, persistent notes, and a way to ask the user). The user may be away: carry the work through the whole pipeline yourself, and involve the user only when a decision genuinely needs them.

# Workflow for a new problem
1. Understand the physics. State the governing equations, geometry, boundary and initial conditions, the dimensionless groups (Re, We, Bo, Oh, Ca, Pe, Ra, Pr, Fr, Ma, St, density and viscosity ratios...) and the quantities of interest. Choose and state a non-dimensionalisation. If something essential is missing, make a reasonable, explicitly documented assumption; use ask_user only when a wrong guess would waste a long run or change the physics.
2. Know the machine. Read the environment summary in the session context (call environment with refresh=true if something looks stale) and the KNOWLEDGE notes, which hold lessons learned on this machine.
3. Start from the closest existing Basilisk code. Search the local tree ($BASILISK is the Basilisk src directory: examples in $BASILISK/examples, tests and benchmarks in $BASILISK/test, solvers and modules in $BASILISK/*.h and its subdirectories) and read the closest examples and every header you include. Never invent Basilisk functions, variables or arguments: confirm each non-trivial API in the local source, or on basilisk.fr with fetch_url (for example sandbox contributions that are not in the local tree).
4. Write the code in a dedicated case directory inside the workspace (one directory per problem, descriptive name). Put every physical and numerical parameter in a commented block at the top, and read the key ones from the command line (argv) so parameter sweeps need no recompilation. Log diagnostics to stderr every few steps in a whitespace-separated format with a header line (for example: i t dt and the quantities of interest). Include the initial condition, boundary conditions, mesh adaptation with an explicit tolerance for each field component, restartable snapshots (dump), lightweight outputs for analysis, and a clear end time.
5. Compile with qcc and fix every error and every warning that matters. Typical: qcc -O2 -Wall -fopenmp case.c -o case -lm. Add -disable-dimensions only if qcc's dimensional analysis complains and the issue cannot be fixed properly. Programs that include view.h must also link the graphics libraries found in $BASILISK/gl (for example -L$BASILISK/gl -lglutils -lfb_tiny); check which ones exist first.
6. Smoke test before anything expensive: run a cheap version (low maximum level, short end time) in the foreground with a timeout. Check that it runs without NaN or crashes, the log is sane, conserved quantities are conserved (for example volume via statsf on the VOF field), the time step behaves, and the fields look physical: make a quick plot and look at it with view_image.
7. Estimate the production cost from the smoke test: wall time per step, and how it scales (one more level of refinement multiplies cells by about 4 in 2D and 8 in 3D, and roughly halves the time step; adaptivity reduces this). Tell the user the estimate. If one run would take longer than {long_run_hours} hours, get approval with ask_user first unless the user already approved it.
8. Production runs go through sim_start (background, OMP_NUM_THREADS set from threads; leave a core or two free if the user works on the machine). Use sim_wait to wait cheaply until a run finishes or a pattern appears in its log, and sim_status to inspect. If a run fails or diverges, diagnose (resolution of thin films and boundary layers, time-step limits, boundary conditions, parameter ranges, adaptation tolerances, solver tolerance), fix the cause, and rerun. Parameter sweeps: one compiled binary, several runs with different argv, never more threads in total than logical CPUs.
9. Analyse with Python scripts saved in the case directory (numpy and matplotlib; check they are installed). Produce clear figures (PNG about 1200 px wide, labelled axes with units or dimensionless groups, legends), and look at every figure with view_image before drawing conclusions. Compare with theory, correlations, benchmarks or experiments whenever they exist and quantify the error. When results matter, do a resolution (grid) convergence study.
10. Report. Write REPORT.md in the case directory: problem, setup and parameters, numerical method, validation, results with figures and key numbers, limitations, next steps. Keep the project notebook (notes, which="project") up to date with goals, decisions, run ids and results, and record reusable lessons about this machine or Basilisk in the knowledge notes (which="knowledge").

Give the user short progress messages at milestones (plan, smoke test result, run launched with an ETA, results). Be concise: numbers over adjectives. Your final message for a task summarises what was done, the key results, where the files are, and what remains.

# Basilisk reference (verify details in the source before relying on them)
- Build and run: qcc -O2 -Wall [-fopenmp] file.c -o file -lm, then OMP_NUM_THREADS=N ./file [args] (sim_start sets the threads). MPI: CC99='mpicc -std=c99' qcc -D_MPI=1 -O2 file.c -o file -lm, then mpirun -np N ./file. Basilisk prints a timing summary at exit (steps, CPU and real time, points.step/s): use it for cost estimates.
- Program layout: #include lines, global parameters, boundary conditions, int main (int argc, char * argv[]) { domain size (L0 / size()), origin(), init_grid (1 << level), physical properties, run(); }, then events. Events: event init (t = 0), event logfile (i++), event snapshots (t += dtout; t <= tend), event end (t = tend). Events with the same name in several included files all run, in definition order.
- Grids: tree grids (quadtree in 2D, octree in 3D) by default; grid/multigrid.h for uniform grids; grid/octree.h for 3D; grid/bitree.h for 1D.
- Incompressible Navier-Stokes: navier-stokes/centered.h (fields u, p, face velocity uf, face pressure pf; DT bounds the time step; TOLERANCE for the Poisson solver; stokes = true drops the advection term). For large density ratios with VOF, navier-stokes/conserving.h gives momentum-conserving advection.
- Two-phase flows: two-phase.h (VOF tracer f, phase 1 where f = 1, properties rho1, rho2, mu1, mu2), tension.h (surface tension coefficient f.sigma), reduced.h (gravity through the reduced-gravity formulation, vector G and reference Z), contact.h (contact angles through height functions), tag.h (connected regions, for example counting drops). Initial interfaces: fraction (f, expression) or fractions() of a vertex distance function.
- Geometry: embed.h for embedded solid boundaries (cs, fs; boundary conditions such as u.n[embed] = dirichlet (0)); axi.h for axisymmetric problems (include it first; x is the axis of symmetry and y the radial coordinate).
- Scalars: tracer.h (advected tracers), diffusion.h (implicit diffusion), heat or species transport with tracer plus diffusion, Boussinesq buoyancy through an acceleration event.
- Other solvers to look for in the tree: saint-venant.h (shallow water), layered/ (multilayer hydrostatic and non-hydrostatic), all-mach.h and compressible/ (compressible), log-conform.h (viscoelastic), ehd/ (electrohydrodynamics), poisson.h (Poisson and Helmholtz multigrid solvers), advection.h, green-naghdi.h. Check what exists before using it.
- Boundary conditions: u.n[left] = dirichlet (U0); u.t[left] = dirichlet (0); p[right] = dirichlet (0); pf[right] = dirichlet (0); u.n[right] = neumann (0); periodic (right). A common outflow is neumann on velocity with dirichlet zero on p and pf.
- Mesh adaptation: adapt_wavelet ({f, u}, (double[]){fErr, uErr, uErr, uErr}, maxlevel, minlevel) takes one tolerance per scalar component in the list (u counts as 2 components in 2D, 3 in 3D).
- Diagnostics: statsf (gives sum, min, max, volume...), normf, interpolate, foreach() loops with reduction (+:x).
- Output: dump () and restore (file = ...) for snapshots and restarts (a common init pattern: if (!restore (file = "restart")) { set the initial condition }); output_facets (f, fp) for interfaces; output_field ({...}, fp, n, linear) for interpolated uniform fields; output_ppm and view.h (squares, draw_vof, box, save ("file.png")) for images and movies (need ffmpeg or ImageMagick for mp4/png).
- Resolution: adapt first and keep uniform grids small, especially in 3D (a uniform level L in 3D has 8^L cells). Reason in cells per characteristic length (for example cells per bubble diameter, per film thickness, per boundary-layer thickness) and justify the maximum level. Basilisk is unit-agnostic: work non-dimensionally unless dimensional output is requested.

# Starting points in the Basilisk tree (paths relative to $BASILISK; confirm they exist in the user's version)
- Rising bubbles: test/rising.c (2D benchmark of Hysing et al. 2009, reference results in test/rising.ref, axisymmetric variants rising-axi*), examples/bubble.c (3D bubble in a large tank), test/axi_rising_bubble.c (axisymmetric bubble with soluble gas).
- Drops: test/oscillation.c (capillary shape oscillation of a droplet), test/sessile.c (sessile drop and contact angle), test/fall.c (axisymmetric viscoelastic drop impact on a solid), examples/atomisation.c (jet atomisation).
- Waves: test/capwave.c (damped capillary wave vs Prosperetti's solution), test/dispersion_gravitocap.c (gravito-capillary dispersion relation), examples/breaking.c (breaking Stokes wave).
- Pipe and channel flows: test/poiseuille-axi.c (axisymmetric Poiseuille pipe flow), test/pipe-axi-embed.c (developing pipe flow with embedded walls), test/poiseuille-periodic.c, test/poiseuille45.c (gravity-driven flow in a channel inclined at 45 degrees).
- Single-phase classics: test/lid.c (lid-driven cavity at Re 1000), test/taylor-green.c, examples/karman.c (vortex street behind a cylinder with embed.h), examples/isotropic.c (3D turbulence), test/rt.c (Rayleigh-Taylor instability).
- Shallow water and long waves: examples/tsunami.c, test/bore.c (undular bores, Green-Naghdi).
- No direct example; combine pieces and validate carefully:
  - Taylor bubbles in tubes: axisymmetric two-phase as in the rising-bubble tests, with no-slip conditions on the tube wall (the top boundary y = R in axi.h) or embed.h walls as in pipe-axi-embed.c, and a domain long enough for the bubble and its wake. Validate the Froude number against correlations (for example Viana et al. 2003).
  - Falling films on inclined planes: two-phase in a periodic domain with gravity tilted through the G vector of reduced.h. Validate the flat-film velocity profile against the Nusselt solution.
  - If the user has their own validated code for such a problem (folders allowed with --allow-read), adapt it rather than starting from scratch.
- Basilisk's own regression tests run with make <name>.tst inside $BASILISK/test and compare with the .ref files. Copy a case into the workspace before modifying it; never edit the Basilisk tree.

# Analysis and visualisation
- Basilisk outputs are whitespace-separated columns (gnuplot-style blocks separated by blank lines); output_facets writes segment end points as pairs of lines.
- For ParaView on the Windows side of WSL, write VTK files (for example legacy STRUCTURED_POINTS from an interpolated uniform grid, or a sandbox VTU writer if one is available) and give the user the Windows path (\\\\wsl$\\<distro>\\home\\...).
- Look at every figure (view_image): check axes, units, trends and artefacts before reporting. Quote numbers with sensible precision.

# Rules
- Work inside the workspace directory. Never modify the Basilisk source tree or files outside the workspace unless the user explicitly asks.
- Never use sudo. Ask the user before installing packages; prefer pip install --user or a virtual environment.
- Do not start background processes with the bash tool (they block it): use sim_start.
- Tool outputs are truncated: use grep, tail, or read_file with offset and limit for large files.
- Background runs keep going after you finish; record their run ids in the project notes.
- If results disagree with theory or a benchmark, say so plainly and investigate; never tune parameters until the plot looks right without a physical reason.
"""


LOCAL_ADDENDUM = """

# Working with tools (important)
Act only through the provided tools by emitting real function calls; never just describe a command you would run. Work in small steps: call one or a few tools, read the results, then continue until the task is done. Keep tool outputs short (tail, grep, head, read_file with limit). Prefer copying and adapting an existing Basilisk example over writing code from memory. If view_image says the model cannot see images, check results numerically instead (print key values, extrema and trends)."""


COMPACT_PROMPT = """The conversation is getting long and will be compacted now. Write a handoff summary that lets you continue seamlessly with an empty context. Include: the user's goals and any instructions or preferences they gave; the physical problem and chosen parameters; every case directory and important file path (code, scripts, figures, reports) with what it contains; compile commands that work on this machine; run ids of simulations (running or finished) with their parameters and status; results obtained so far with key numbers; problems found and how they were solved; and the exact next steps. Do not call any tools; reply with the summary only."""


# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------

TOOLS = [
    {
        "name": "bash",
        "description": ("Run a shell command (bash) on the user's machine and return exit code, stdout and stderr. "
                        "Use it to compile with qcc, run short tests and post-processing scripts, inspect files "
                        "(ls, head, tail, grep), and check tools. It blocks until the command finishes or the "
                        "timeout expires. Do not start background processes here; use sim_start for anything "
                        "that runs longer than a few minutes. Risky commands may need the user's approval."),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line to run with bash -c."},
                "cwd": {"type": "string", "description": "Working directory, relative to the workspace (default: workspace root)."},
                "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 600, max 7200)."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": ("Read a text file with line numbers. Allowed inside the workspace, the Basilisk tree, "
                        "the agent's home folder and extra folders the user allowed. Use offset/limit for long files."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative to the workspace, absolute, or using $BASILISK)."},
                "offset": {"type": "integer", "description": "First line to read, 1-based (default 1)."},
                "limit": {"type": "integer", "description": "Number of lines to read (default 400)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file inside the workspace (parent folders are created).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": ("Replace an exact piece of text in a workspace file. old_text must match exactly once "
                        "(include surrounding lines to make it unique) unless replace_all is true."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "list_dir",
        "description": "List a directory (names, sizes, modification times). Optionally recursive and filtered by a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory (default: workspace root)."},
                "pattern": {"type": "string", "description": "Glob filter such as *.c (default *)."},
                "recursive": {"type": "boolean"},
            },
        },
    },
    {
        "name": "search",
        "description": ("Search file contents with an extended regular expression (grep -rnE). Defaults to the "
                        "Basilisk source tree ($BASILISK). Use it to find examples that include a given header, "
                        "the definition of a function, or how an event or variable is used."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Extended regex, e.g. 'include \"two-phase.h\"' or 'adapt_wavelet'."},
                "path": {"type": "string", "description": "File or directory to search (default $BASILISK)."},
                "glob": {"type": "string", "description": "Comma-separated file globs, e.g. '*.c,*.h' (default '*.c,*.h')."},
                "context": {"type": "integer", "description": "Lines of context around matches (default 0, max 10)."},
                "max_results": {"type": "integer", "description": "Maximum matching lines (default 80)."},
                "files_only": {"type": "boolean", "description": "Return only the names of matching files."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "environment",
        "description": ("Report the machine environment: OS/WSL, CPU and cores, RAM, disk, compilers, qcc and "
                        "$BASILISK, OpenMP/MPI, graphics libraries for view.h, Python packages, tools like ffmpeg "
                        "and ParaView, and warnings. Cached; use refresh=true after installing something."),
        "input_schema": {"type": "object", "properties": {"refresh": {"type": "boolean"}}},
    },
    {
        "name": "sim_start",
        "description": ("Start a simulation (or any long command) in the background. It keeps running after this "
                        "session ends. stdout/stderr go to log files in <cwd>/.runs/<run id>/. Returns a run id. "
                        "Refuses to oversubscribe the CPUs unless force=true."),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command line, e.g. './bubble 0.5 11' or 'mpirun -np 4 ./case'."},
                "cwd": {"type": "string", "description": "Case directory, relative to the workspace."},
                "threads": {"type": "integer", "description": "OMP_NUM_THREADS for the run (default 1)."},
                "label": {"type": "string", "description": "Short label used in the run id."},
                "max_hours": {"type": "number", "description": "Optional wall-time limit; the run is terminated after it."},
                "force": {"type": "boolean", "description": "Start even if CPUs would be oversubscribed."},
            },
            "required": ["command", "cwd"],
        },
    },
    {
        "name": "sim_status",
        "description": ("Status of a background run (or of all running and recent runs if run_id is omitted): state, "
                        "exit code, elapsed time, CPU and memory use, last log lines, detected problems such as NaN "
                        "or crashes, and the newest output files."),
        "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}},
    },
    {
        "name": "sim_wait",
        "description": ("Block until a background run finishes, until a regular expression appears in its log "
                        "(stdout or stderr, searched from the start of the run), or until max_minutes have passed; "
                        "then return its status. Waiting costs nothing, so prefer this over repeated sim_status "
                        "calls. Up to 1440 minutes per call."),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "max_minutes": {"type": "number", "description": "Maximum wait in minutes (default 30)."},
                "until": {"type": "string", "description": "Optional regex; stop waiting as soon as it appears in the run's stdout/stderr."},
                "poll_seconds": {"type": "integer", "description": "Polling interval (default 30)."},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "sim_stop",
        "description": "Stop a background run (SIGTERM to the whole process tree, then SIGKILL if needed).",
        "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
    },
    {
        "name": "view_image",
        "description": ("Look at one or more images (PNG, JPEG, GIF, WebP; PPM and others if Pillow or ImageMagick "
                        "is installed): your plots, Basilisk snapshots, movie frames. Use it to check every figure."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Image path."},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Up to 4 image paths to compare."},
            },
        },
    },
    {
        "name": "fetch_url",
        "description": ("Fetch a web page as text from an allowed domain (basilisk.fr by default): documentation, "
                        "source files, sandbox contributions. Long pages are paged with offset."),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "offset": {"type": "integer", "description": "Character offset for paging (default 0)."},
                "max_chars": {"type": "integer", "description": "Characters to return (default 20000)."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "notes",
        "description": ("Persistent notes that survive across sessions. which='project': the workspace notebook "
                        "(NOTES.md: goals, decisions, run ids, results). which='knowledge': lessons about this "
                        "machine and Basilisk that help every future session (compile flags that work, pitfalls). "
                        "action read, append (adds a dated entry) or write (replaces the file)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read", "append", "write"]},
                "which": {"type": "string", "enum": ["project", "knowledge"]},
                "text": {"type": "string"},
            },
            "required": ["action", "which"],
        },
    },
    {
        "name": "ask_user",
        "description": ("Ask the user a question and wait for the answer. Use sparingly: only when a decision "
                        "genuinely needs them (missing physics, approval of a very long run). If the user is not "
                        "available you get told so; then proceed with a documented assumption."),
        "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    },
]


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work = Path(cfg.workdir).resolve()
        HOME_DIR.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.work / ".basilisk_agent"
        self.session_file = self.state_dir / "session.json"
        self.notes_file = self.work / "NOTES.md"
        self.knowledge_file = HOME_DIR / "KNOWLEDGE.md"
        self.user_file = HOME_DIR / "USER.md"
        self.env_file = HOME_DIR / "env.json"
        bas = os.environ.get("BASILISK") or ""
        if not bas and shutil.which("qcc"):
            bas = os.path.dirname(os.path.realpath(shutil.which("qcc")))
        self.basilisk = bas
        read_roots = [HOME_DIR, Path(tempfile.gettempdir())] + [Path(p).expanduser() for p in cfg.allow_read]
        if bas:
            read_roots.append(Path(bas).parent)
        extra_write = [Path(p).expanduser() for p in cfg.allow_write]
        write_roots = [self.work, Path(tempfile.gettempdir())] + extra_write
        self.policy = Policy(self.work, cfg.mode, cfg.interactive, read_roots, write_roots,
                             run_roots=[self.work] + extra_write)
        self.sims = SimManager(HOME_DIR, self.work, self.policy)
        self.client = make_client(cfg)
        self.system = SYSTEM_PROMPT.replace("{long_run_hours}", ("%g" % cfg.long_run_hours))
        if self.client.provider != "claude":
            self.system += LOCAL_ADDENDUM
        self.max_out = self.client.max_tool_output
        self.messages = []
        self.totals = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "cost": 0.0, "calls": 0}
        self.last_prompt_tokens = 0
        self.handlers = {
            "bash": self.t_bash, "read_file": self.t_read_file, "write_file": self.t_write_file,
            "edit_file": self.t_edit_file, "list_dir": self.t_list_dir, "search": self.t_search,
            "environment": self.t_environment, "sim_start": self.t_sim_start, "sim_status": self.t_sim_status,
            "sim_wait": self.t_sim_wait, "sim_stop": self.t_sim_stop, "view_image": self.t_view_image,
            "fetch_url": self.t_fetch_url, "notes": self.t_notes, "ask_user": self.t_ask_user,
        }

    # ---------------- session context ----------------

    def environment(self, refresh=False):
        env = read_json(self.env_file, None)
        stale = (env is None or env.get("workspace") != str(self.work)
                 or time.time() - os.path.getmtime(str(self.env_file)) > 7 * 86400)
        if refresh or stale:
            env = detect_environment(self.work)
            atomic_write(self.env_file, json.dumps(env, indent=1))
        return env

    def session_context(self):
        parts = ["<session_context>",
                 "Date: %s" % now_str(),
                 "Workspace (create all files and case directories inside it): %s" % self.work,
                 "Approval mode: %s (%s)" % (self.cfg.mode, {
                     "ask": "every shell command needs the user's approval",
                     "auto": "risky shell commands need the user's approval",
                     "full": "no approvals; only destructive commands are blocked"}[self.cfg.mode]),
                 "User available for questions: %s" % ("yes" if self.cfg.interactive else
                                                       "no (unattended run: decide and document)"),
                 "",
                 "## Environment",
                 environment_text(self.environment()),
                 "",
                 "## Simulation runs (running or recent)",
                 self.sims.short_list()]
        for title, path, limit in (("About the user (USER.md)", self.user_file, 4000),
                                   ("Knowledge notes for this machine (KNOWLEDGE.md)", self.knowledge_file, 8000),
                                   ("Project notebook (NOTES.md, most recent part)", self.notes_file, 12000)):
            try:
                txt = path.read_text(encoding="utf-8").strip()
            except OSError:
                txt = ""
            if txt:
                if len(txt) > limit:
                    txt = "[... earlier part omitted ...]\n" + txt[-limit:]
                parts += ["", "## " + title, txt]
        parts.append("</session_context>")
        return "\n".join(parts)

    # ---------------- persistence ----------------

    def save(self):
        try:
            atomic_write(self.session_file, json.dumps({"model": self.client.model, "messages": self.messages}))
        except OSError as e:
            warn("Could not save the session: %s" % e)

    def load(self):
        data = read_json(self.session_file, None)
        if not data or not data.get("messages"):
            return False
        self.messages = data["messages"]
        last = self.messages[-1]
        if last["role"] == "assistant":
            pending = [b for b in last["content"] if b.get("type") == "tool_use"]
            if pending:
                self.messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b["id"], "is_error": True,
                     "content": "Interrupted: the session was closed before this tool ran."} for b in pending]})
        if data.get("model") and data["model"] != self.client.model:
            warn("Session was created with %s; strip its thinking blocks to continue with %s."
                 % (data["model"], self.client.model))
            self.strip_thinking()
        return True

    def strip_thinking(self):
        for m in self.messages:
            if m["role"] == "assistant" and isinstance(m["content"], list):
                kept = [b for b in m["content"] if b.get("type") not in ("thinking", "redacted_thinking")]
                m["content"] = kept or [{"type": "text", "text": "(continued)"}]

    # ---------------- message helpers ----------------

    def append_user(self, blocks):
        if self.messages and self.messages[-1]["role"] == "user":
            c = self.messages[-1]["content"]
            if isinstance(c, str):
                c = [{"type": "text", "text": c}]
            self.messages[-1]["content"] = c + blocks
        else:
            self.messages.append({"role": "user", "content": blocks})

    def count_images(self):
        n = 0
        for m in self.messages:
            if not isinstance(m["content"], list):
                continue
            for b in m["content"]:
                if b.get("type") == "image":
                    n += 1
                elif b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                    n += sum(1 for x in b["content"] if x.get("type") == "image")
        return n

    def needs_compaction(self):
        limit = min(self.cfg.compact_at, int(self.client.context_limit * 0.7))
        tokens = self.last_prompt_tokens
        if self.client.provider != "claude":
            tokens = max(tokens, estimate_tokens(self.system, self.messages))
        if tokens > limit:
            return "context %d tokens" % tokens
        if self.count_images() > MAX_IMAGES_BEFORE_COMPACT:
            return "many images in context"
        if len(json.dumps(self.messages)) > MAX_REQUEST_BYTES:
            return "request size"
        return None

    def account(self, usage):
        inp = int(usage.get("input_tokens") or 0)
        cr = int(usage.get("cache_read_input_tokens") or 0)
        cw = int(usage.get("cache_creation_input_tokens") or 0)
        o = int(usage.get("output_tokens") or 0)
        self.last_prompt_tokens = inp + cr + cw
        t = self.totals
        t["input"] += inp
        t["cache_read"] += cr
        t["cache_write"] += cw
        t["output"] += o
        t["calls"] += 1
        price = PRICES.get(self.client.model)
        cost = 0.0
        if price:
            pin, pout, mread = price
            cost = (inp * pin + cr * pin * mread + cw * pin * CACHE_WRITE_MULT.get(self.client.cache_ttl, 1.25)
                    + o * pout) / 1e6
        t["cost"] += cost
        return cost

    def usage_line(self, turn_cost):
        t = self.totals
        if self.client.provider == "claude":
            price_note = ("  ~$%.2f this turn, ~$%.2f session" % (turn_cost, t["cost"])
                          if self.client.model in PRICES else "")
            return ("[%s | context %.1fk tokens | output %.1fk tokens this session%s]"
                    % (self.client.model, self.last_prompt_tokens / 1000.0, t["output"] / 1000.0, price_note))
        ctx = max(self.last_prompt_tokens, estimate_tokens(self.system, self.messages))
        return ("[%s %s | context ~%.1fk of %dk tokens | output %.1fk tokens this session%s]"
                % (self.client.provider, self.client.model, ctx / 1000.0, self.client.context_limit // 1000,
                   t["output"] / 1000.0, " | local, free" if self.client.provider == "ollama" else ""))

    # ---------------- compaction ----------------

    def local_handoff(self):
        texts = []
        for m in self.messages[-40:]:
            if m["role"] == "assistant" and isinstance(m["content"], list):
                for b in m["content"]:
                    if b.get("type") == "text" and b.get("text", "").strip():
                        texts.append(b["text"].strip())
                    elif b.get("type") == "tool_use":
                        texts.append("[tool %s %s]" % (b["name"], json.dumps(b.get("input"))[:300]))
        return ("(Automatic summary: the model summary was unavailable.)\nRecent activity:\n"
                + clip("\n".join(texts), 20000))

    def compact(self, reason):
        info("  Compacting the conversation (%s)..." % reason)
        last_user_texts = []
        if self.messages and self.messages[-1]["role"] == "user":
            c = self.messages[-1]["content"]
            if isinstance(c, list) and not any(b.get("type") == "tool_result" for b in c):
                last_user_texts = [b for b in c if b.get("type") == "text" and "<session_context>" not in b.get("text", "")]
        summary = None
        try:
            r, summary = self.client.summarize(self.system, TOOLS, self.messages, COMPACT_PROMPT)
            self.account(r["usage"])
        except APIError as e:
            warn("  Summary request failed (%s); using a local summary." % e.etype)
        if not summary:
            summary = self.local_handoff()
        blocks = [{"type": "text", "text": self.session_context()},
                  {"type": "text", "text": "<handoff_summary>\n%s\n</handoff_summary>\nThe conversation was compacted. "
                   "Continue from where you left off; re-read files instead of relying on memory for exact code."
                   % summary}]
        self.messages = [{"role": "user", "content": blocks + last_user_texts}]
        self.last_prompt_tokens = 0
        self.save()
        info("  Compacted. Continuing.")

    # ---------------- main loop ----------------

    def run_turn(self, text):
        blocks = []
        if not self.messages:
            blocks.append({"type": "text", "text": self.session_context()})
        blocks.append({"type": "text", "text": text})
        self.append_user(blocks)
        self.save()
        turn_cost, calls, retried_400, compactions = 0.0, 0, False, 0
        while True:
            if calls >= self.cfg.max_calls:
                warn("Reached %d model calls for this request; pausing. Type 'continue' to go on." % calls)
                break
            reason = self.needs_compaction()
            if reason:
                compactions += 1
                if compactions > 3:
                    error("The context is still too large after compacting; start a fresh conversation with /new.")
                    break
                self.compact(reason)
            printer = StreamPrinter(self.cfg.show_thinking)
            try:
                resp = self.client.create(self.system, TOOLS, self.messages, printer=printer)
            except KeyboardInterrupt:
                printer.finish()
                warn("Interrupted.")
                break
            except APIError as e:
                printer.finish()
                if e.status == 413 or "too long" in (e.message or "").lower():
                    compactions += 1
                    if compactions > 3:
                        error("The request is still too large after compacting; start fresh with /new.")
                        break
                    self.compact("request too large")
                    continue
                if e.status == 400 and not retried_400 and ("thinking" in e.message or "signature" in e.message):
                    retried_400 = True
                    warn("  The API rejected earlier thinking blocks; removing them and retrying.")
                    self.strip_thinking()
                    continue
                error("Model error: %s" % e)
                if e.status in (401, 403):
                    error("Check %s." % ("ANTHROPIC_API_KEY" if self.client.provider == "claude" else "OPENAI_API_KEY"))
                break
            calls += 1
            turn_cost += self.account(resp["usage"])
            content = resp["content"] or [{"type": "text", "text": "(no content)"}]
            self.messages.append({"role": "assistant", "content": content})
            self.save()
            uses = [b for b in content if b.get("type") == "tool_use"]
            if uses:
                results, interrupted = self.execute(uses, resp["invalid_tool_ids"])
                self.append_user(results)
                self.save()
                if interrupted:
                    warn("Stopped. Tell the agent how to continue.")
                    break
                continue
            if resp["stop_reason"] == "max_tokens":
                self.append_user([{"type": "text", "text": "(Your last response hit the output limit. Continue "
                                   "where you left off; split very long files into several edits.)"}])
                continue
            if resp["stop_reason"] == "refusal":
                warn("The model declined this request.")
            break
        info(self.usage_line(turn_cost))

    def execute(self, uses, invalid_ids):
        results, interrupted = [], False
        for b in uses:
            if interrupted:
                results.append({"type": "tool_result", "tool_use_id": b["id"], "is_error": True,
                                "content": "Not run: interrupted by the user."})
                continue
            name, inp = b["name"], b.get("input") or {}
            out(S.cyan + "  > " + describe_call(name, inp) + S.reset)
            if b["id"] in invalid_ids:
                content, is_err = ("This tool call was cut off (output limit or stream error). "
                                   "Send it again; split large content into smaller pieces."), True
            elif name not in self.handlers:
                content, is_err = "Unknown tool: %s" % name, True
            else:
                try:
                    content, is_err = self.handlers[name](inp)
                except KeyboardInterrupt:
                    interrupted = True
                    content, is_err = "Interrupted by the user (Ctrl-C) before it finished.", True
                except Exception as e:  # report tool bugs to the model instead of crashing
                    content, is_err = "Tool error: %s: %s" % (type(e).__name__, e), True
            preview(content, is_err)
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": content, "is_error": is_err})
        return results, interrupted

    # ---------------- tools ----------------

    def _path(self, p, write=False):
        rp = resolve_path(p.replace("$BASILISK", self.basilisk) if self.basilisk else p, self.work)
        ok = self.policy.can_write(rp) if write else self.policy.can_read(rp)
        if not ok:
            raise PermissionError("%s is outside the allowed folders (workspace %s%s). Ask the user to start the "
                                  "agent with --allow-%s if you need it." % (
                                      rp, self.work, (", Basilisk tree" if self.basilisk else ""),
                                      "write" if write else "read"))
        return rp

    def t_bash(self, a):
        cmd = a.get("command", "")
        if not cmd.strip():
            return "Empty command.", True
        cwd = self._path(a.get("cwd") or ".", write=True)
        if not cwd.is_dir():
            return "Directory does not exist: %s" % cwd, True
        timeout = min(max(int(a.get("timeout") or 600), 1), 7200)
        decision, reason = self.policy.check(cmd, cwd)
        if decision == "block":
            return "Blocked: this command matches a destructive pattern and will not be run.", True
        if decision == "ask" and not self.policy.confirm("run a shell command in %s" % cwd, cmd, reason):
            return ("Not run: the user did not approve (%s). Find another way, or explain to the user why "
                    "it is needed." % reason), True
        p = subprocess.Popen(["bash", "-c", cmd], cwd=str(cwd), env=clean_env(), stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        timed_out = False
        try:
            so, se = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(p.pid, signal.SIGKILL)
            so, se = p.communicate()
            timed_out = True
        except KeyboardInterrupt:
            kill_tree(p.pid, signal.SIGKILL)
            p.communicate()
            raise
        so = so.decode(errors="replace")
        se = se.decode(errors="replace")
        res = "exit code: %s%s\n" % (p.returncode, "  (killed after %ds timeout)" % timeout if timed_out else "")
        if so.strip():
            res += "--- stdout ---\n" + clip(so, self.max_out * 2 // 3) + "\n"
        if se.strip():
            res += "--- stderr ---\n" + clip(se, self.max_out // 3) + "\n"
        return res.rstrip(), (p.returncode != 0)

    def t_read_file(self, a):
        p = self._path(a["path"])
        if not p.is_file():
            return "Not a file: %s" % p, True
        with open(str(p), "rb") as f:
            head = f.read(8192)
        if b"\0" in head:
            return "%s looks binary (%s). Use bash tools (e.g. od, python) or view_image." % (
                p, human_bytes(p.stat().st_size)), True
        offset = max(int(a.get("offset") or 1), 1)
        limit = min(max(int(a.get("limit") or 400), 1), 5000)
        lines, total, chars = [], 0, 0
        with open(str(p), encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                total = n
                if n < offset or len(lines) >= limit or chars > self.max_out:
                    continue
                line = line.rstrip("\n")
                if len(line) > 2000:
                    line = line[:2000] + " [...]"
                lines.append("%6d  %s" % (n, line))
                chars += len(line)
        if not lines:
            return "%s has %d lines; nothing at offset %d." % (p, total, offset), False
        last = offset + len(lines) - 1
        hdr = "%s (lines %d-%d of %d)\n" % (p, offset, last, total)
        if last < total:
            hdr += "(more lines: call again with offset=%d)\n" % (last + 1)
        return hdr + "\n".join(lines), False

    def t_write_file(self, a):
        p = self._path(a["path"], write=True)
        content = a.get("content", "")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p), "w", encoding="utf-8") as f:
            f.write(content)
        return "Wrote %s (%d lines, %s)." % (p, content.count("\n") + (0 if content.endswith("\n") else 1),
                                             human_bytes(len(content.encode()))), False

    def t_edit_file(self, a):
        p = self._path(a["path"], write=True)
        if not p.is_file():
            return "File not found: %s" % p, True
        text = p.read_text(encoding="utf-8", errors="replace")
        old, new = a.get("old_text", ""), a.get("new_text", "")
        if not old:
            return "old_text is empty.", True
        n = text.count(old)
        if n == 0:
            return "old_text not found in %s (it must match exactly, including whitespace). Read the file first." % p, True
        if n > 1 and not a.get("replace_all"):
            return "old_text occurs %d times in %s; add surrounding lines to make it unique or set replace_all." % (n, p), True
        text = text.replace(old, new) if a.get("replace_all") else text.replace(old, new, 1)
        p.write_text(text, encoding="utf-8")
        line = text[:text.find(new)].count("\n") + 1 if new and new in text else "?"
        return "Edited %s (%d replacement%s, near line %s)." % (p, n if a.get("replace_all") else 1,
                                                               "s" if a.get("replace_all") and n > 1 else "", line), False

    def t_list_dir(self, a):
        p = self._path(a.get("path") or ".")
        if not p.is_dir():
            return "Not a directory: %s" % p, True
        pattern = a.get("pattern") or "*"
        it = p.rglob(pattern) if a.get("recursive") else p.glob(pattern)
        rows, count = [], 0
        for f in sorted(it):
            if any(part in (".git", "__pycache__", ".basilisk_agent") for part in f.relative_to(p).parts):
                continue
            count += 1
            if count > 300:
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            age = human_time(time.time() - st.st_mtime)
            if f.is_dir():
                rows.append("%s/  (dir, modified %s ago)" % (f.relative_to(p), age))
            else:
                rows.append("%s  (%s, modified %s ago)" % (f.relative_to(p), human_bytes(st.st_size), age))
        if count > 300:
            rows.append("... %d more entries" % (count - 300))
        return ("%s:\n" % p) + ("\n".join(rows) if rows else "(empty)"), False

    def t_search(self, a):
        target = a.get("path") or self.basilisk
        if not target:
            return "BASILISK is not set and no path was given.", True
        p = self._path(target)
        globs = [g.strip() for g in (a.get("glob") or "*.c,*.h").split(",") if g.strip()]
        cmd = ["grep", "-rnIE", "--color=never"]
        if a.get("files_only"):
            cmd = ["grep", "-rlIE", "--color=never"]
        ctx = min(max(int(a.get("context") or 0), 0), 10)
        if ctx and not a.get("files_only"):
            cmd.append("-C%d" % ctx)
        if p.is_dir():
            cmd += ["--include=%s" % g for g in globs]
        cmd += ["-e", a["pattern"], "--", str(p)]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        except subprocess.TimeoutExpired:
            return "Search timed out; narrow the path or pattern.", True
        lines = r.stdout.decode(errors="replace").splitlines()
        if r.returncode == 2:
            return "grep error: %s" % r.stderr.decode(errors="replace")[:500], True
        maxr = min(max(int(a.get("max_results") or 80), 1), 500)
        base = str(p) + "/" if p.is_dir() else ""
        shown = [l[len(base):] if base and l.startswith(base) else l for l in lines[:maxr]]
        res = "%d matching lines in %s%s\n" % (len(lines), p, " (showing %d)" % maxr if len(lines) > maxr else "")
        return res + clip("\n".join(shown), self.max_out), False

    def t_environment(self, a):
        env = self.environment(refresh=bool(a.get("refresh")))
        return environment_text(env), False

    def t_sim_start(self, a):
        return self.sims.start(a["command"], a.get("cwd") or ".", a.get("threads") or 1, a.get("label"),
                               bool(a.get("force")), a.get("max_hours"))

    def t_sim_status(self, a):
        return self.sims.status(a.get("run_id"))

    def t_sim_wait(self, a):
        return self.sims.wait(a["run_id"], a.get("max_minutes") or 30, a.get("until"), a.get("poll_seconds") or 30)

    def t_sim_stop(self, a):
        return self.sims.stop(a["run_id"])

    def t_view_image(self, a):
        paths = list(a.get("paths") or [])
        if a.get("path"):
            paths.insert(0, a["path"])
        paths = paths[:4]
        if not paths:
            return "Give path or paths.", True
        if not self.client.supports_vision:
            found = [str(self._path(p)) for p in paths if self._path(p).is_file()]
            return ("The current model (%s) cannot see images. Saved figure(s): %s. Check the results numerically "
                    "instead: print key values, extrema and trends from the data behind the figure, and tell the "
                    "user which files to open." % (self.client.model, ", ".join(found) or "none found")), False
        content = []
        for raw in paths:
            p = self._path(raw)
            if not p.is_file():
                content.append({"type": "text", "text": "Not found: %s" % p})
                continue
            try:
                mtype, data, size = load_image(p)
            except ValueError as e:
                content.append({"type": "text", "text": str(e)})
                continue
            label = "Image %s%s:" % (p, " (%dx%d px as sent)" % size if size else "")
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "source": {"type": "base64", "media_type": mtype, "data": data}})
        if not any(c["type"] == "image" for c in content):
            return "\n".join(c["text"] for c in content), True
        return content, False

    def t_fetch_url(self, a):
        url = a["url"]
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        allowed = self.cfg.fetch_domains
        if not any(host == d or host.endswith("." + d) for d in allowed):
            return ("Domain %s is not allowed (allowed: %s). The user can add domains with --fetch-domain."
                    % (host, ", ".join(allowed))), True
        req = urllib.request.Request(url, headers={"User-Agent": "basilisk-agent/%s" % VERSION})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read(4000000)
                ctype = r.headers.get_content_type()
                charset = r.headers.get_content_charset() or "utf-8"
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            return "Fetch failed: %s" % e, True
        text = raw.decode(charset, errors="replace")
        if "html" in ctype:
            text = html_to_text(text)
        off = max(int(a.get("offset") or 0), 0)
        n = min(max(int(a.get("max_chars") or 20000), 1000), 60000, self.max_out)
        chunk = text[off:off + n]
        more = len(text) - off - len(chunk)
        res = "URL: %s (%d characters)\n\n%s" % (url, len(text), chunk)
        if more > 0:
            res += "\n\n[%d more characters: call again with offset=%d]" % (more, off + n)
        return res, False

    def t_notes(self, a):
        path = self.notes_file if a.get("which", "project") == "project" else self.knowledge_file
        action = a.get("action")
        if action == "read":
            try:
                return path.read_text(encoding="utf-8") or "(empty)", False
            except OSError:
                return "(no notes yet: %s)" % path, False
        text = a.get("text") or ""
        if not text.strip():
            return "Nothing to write.", True
        if action == "append":
            title = "Project notebook" if path == self.notes_file else "Knowledge notes (machine and Basilisk lessons)"
            prefix = "" if path.exists() else "# %s\n" % title
            with open(str(path), "a", encoding="utf-8") as f:
                f.write("%s\n## %s\n%s\n" % (prefix, now_str(), text.strip()))
            return "Appended to %s." % path, False
        if action == "write":
            atomic_write(path, text if text.endswith("\n") else text + "\n")
            return "Wrote %s." % path, False
        return "Unknown action.", True

    def t_ask_user(self, a):
        q = a.get("question", "")
        if not self.cfg.interactive:
            return ("The user is not available (unattended run). Make the most reasonable choice, document it in "
                    "the project notes and the report, and continue."), False
        out("")
        out(S.magenta + S.bold + "Question from the agent:" + S.reset)
        out(S.magenta + textwrap.indent(q, "  ") + S.reset)
        try:
            ans = read_input(S.bold + "your answer > " + S.reset)
        except EOFError:
            ans = ""
        return ("User's answer: " + ans) if ans.strip() else "The user gave no answer; proceed with a documented assumption.", False


# --------------------------------------------------------------------------
# Terminal I/O helpers
# --------------------------------------------------------------------------

def describe_call(name, a):
    if name == "bash":
        cmd = a.get("command", "").strip().splitlines()
        first = cmd[0] if cmd else ""
        more = " ..." if len(cmd) > 1 else ""
        return "$ %s%s%s" % (first[:160], more, ("   [in %s]" % a["cwd"]) if a.get("cwd") else "")
    if name in ("read_file", "write_file", "edit_file", "list_dir"):
        return "%s %s" % (name, a.get("path", "."))
    if name == "view_image":
        return "view_image %s" % ", ".join(([a["path"]] if a.get("path") else []) + list(a.get("paths") or []))
    if name == "search":
        return "search /%s/ in %s" % (a.get("pattern"), a.get("path") or "$BASILISK")
    if name == "sim_start":
        return "sim_start [%s threads] %s   [in %s]" % (a.get("threads", 1), a.get("command"), a.get("cwd"))
    if name in ("sim_status", "sim_stop"):
        return "%s %s" % (name, a.get("run_id") or "(all)")
    if name == "sim_wait":
        return "sim_wait %s (up to %s min%s)" % (a.get("run_id"), a.get("max_minutes", 30),
                                                ", until /%s/" % a["until"] if a.get("until") else "")
    if name == "fetch_url":
        return "fetch %s" % a.get("url")
    if name == "notes":
        return "notes %s %s" % (a.get("action"), a.get("which"))
    if name == "environment":
        return "environment%s" % (" (refresh)" if a.get("refresh") else "")
    return name


def preview(content, is_err, n=6):
    if isinstance(content, list):
        txt = " ".join(c.get("text", "[image]") if c.get("type") == "text" else "[image]" for c in content)
    else:
        txt = str(content)
    lines = txt.strip().splitlines()
    color = S.red if is_err else S.dim
    for line in lines[:n]:
        out(color + "    " + line[:200] + S.reset)
    if len(lines) > n:
        out(S.dim + "    ... (%d more lines)" % (len(lines) - n) + S.reset)


def read_input(prompt):
    first = input(prompt)
    lines = [first]
    while lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(input("... "))
    return "\n".join(lines)


HELP = """Commands:
  /help              this help
  /env               show the detected environment (refreshes it)
  /runs              status of running and recent simulations
  /cost              token usage and estimated cost of this session
  /model NAME        switch model (opus, sonnet, fable, haiku or a full model id)
  /effort LEVEL      low | medium | high | xhigh | max
  /mode MODE         ask | auto | full (how much approval shell commands need)
  /compact           summarise the conversation to free context
  /new               start a fresh conversation (notes and runs are kept)
  /exit              quit (background simulations keep running)
End a line with \\ to type several lines.
Outside a session, `basilisk-agent sim --help` lists commands that need no AI."""


def interactive_loop(agent, first_task):
    try:
        import readline  # noqa: F401  (line editing and history)
    except ImportError:
        pass
    if first_task:
        agent.run_turn(first_task)
    last_interrupt = 0.0
    while True:
        try:
            text = read_input("\n" + S.bold + S.green + "you > " + S.reset)
        except KeyboardInterrupt:
            if time.time() - last_interrupt < 2:
                break
            last_interrupt = time.time()
            out("\n(Ctrl-C again to quit)")
            continue
        except EOFError:
            break
        text = text.strip()
        if not text:
            continue
        if text.startswith("/"):
            if handle_command(agent, text):
                break
            continue
        agent.run_turn(text)
    t = agent.totals
    out("")
    info("Session: %d model calls, %dk output tokens%s. Background runs keep going; check them with /runs next time."
         % (t["calls"], t["output"] // 1000, (", ~$%.2f" % t["cost"]) if agent.client.model in PRICES else ""))


def handle_command(agent, text):
    parts = text.split()
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
    if cmd in ("/exit", "/quit", "/q"):
        return True
    if cmd == "/help":
        out(HELP)
    elif cmd == "/env":
        out(environment_text(agent.environment(refresh=True)))
    elif cmd == "/runs":
        out(agent.sims.status()[0])
    elif cmd == "/cost":
        t = agent.totals
        out("calls %d | input %d | cache read %d | cache write %d | output %d | ~$%.2f" % (
            t["calls"], t["input"], t["cache_read"], t["cache_write"], t["output"], t["cost"]))
    elif cmd == "/model" and arg:
        new = MODEL_ALIASES.get(arg, arg) if agent.client.provider == "claude" else arg
        if agent.messages:
            agent.compact("switching model")
        agent.client.model = new
        if agent.client.provider == "ollama":
            ok, msg = agent.client.probe()
            (info if ok else warn)(msg)
        out("Model: %s" % new)
    elif cmd == "/effort" and arg in ("low", "medium", "high", "xhigh", "max"):
        if agent.client.provider != "claude":
            out("Effort applies to Claude models only.")
        else:
            agent.client.effort = arg
            out("Effort: %s" % arg)
    elif cmd == "/mode" and arg in ("ask", "auto", "full"):
        agent.cfg.mode = arg
        agent.policy.mode = arg
        out("Mode: %s" % arg)
    elif cmd == "/compact":
        if agent.messages:
            agent.compact("requested")
    elif cmd == "/new":
        agent.messages = []
        agent.save()
        out("Started a fresh conversation.")
    else:
        out(HELP)
    return False


# --------------------------------------------------------------------------
# No-AI simulation commands:  basilisk-agent sim ...
# --------------------------------------------------------------------------

def _cached_env():
    env = read_json(HOME_DIR / "env.json", None)
    if env is None:
        env = detect_environment(Path.cwd())
        HOME_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write(HOME_DIR / "env.json", json.dumps(env, indent=1))
    return env


def _basilisk_dir():
    bas = os.environ.get("BASILISK", "")
    q = shutil.which("qcc", path=clean_env().get("PATH"))
    if not bas and q:
        bas = os.path.dirname(os.path.realpath(q))
    return bas


def gl_link_flags():
    """Linker flags for programs that include view.h, depending on which Basilisk graphics libraries exist."""
    gl = Path(_basilisk_dir() or "/nonexistent") / "gl"

    def have(name):
        return (gl / name).exists()

    if have("libglutils.a") and have("libfb_tiny.a"):
        return ["-L%s" % gl, "-lglutils", "-lfb_tiny"]
    if have("libglutils.a") and have("libfb_osmesa.a"):
        return ["-L%s" % gl, "-lglutils", "-lfb_osmesa", "-lGLU", "-lOSMesa"]
    if have("libglutils.a") and have("libfb_glx.a"):
        return ["-L%s" % gl, "-lglutils", "-lfb_glx", "-lGLU", "-lGL", "-lX11"]
    warn("This code includes view.h but no Basilisk graphics library was found in %s; linking may fail." % gl)
    return []


def compile_case(src, flags=None):
    qcc = shutil.which("qcc", path=clean_env().get("PATH"))
    if not qcc:
        error("qcc not found: install Basilisk and export BASILISK and PATH (see README).")
        return None
    if flags is None:
        flags = "-O2 -Wall" + (" -fopenmp" if _cached_env().get("openmp") else "")
    cmd = [qcc] + shlex.split(flags) + [src.name, "-o", src.stem]
    try:
        text = src.read_text(errors="replace")
    except OSError:
        text = ""
    if re.search(r'#\s*include\s*"view\.h"', text):
        cmd += gl_link_flags()
    cmd.append("-lm")
    info("$ " + " ".join(shlex.quote(c) for c in cmd) + "   (in %s)" % src.parent)
    r = subprocess.run(cmd, cwd=str(src.parent), env=clean_env())
    if r.returncode != 0:
        error("Compilation failed (exit %d)." % r.returncode)
        return None
    return src.parent / src.stem


def _sim_manager(work):
    work = Path(work).resolve()
    pol = Policy(work, "full", True, [], [work], run_roots=[work])
    return SimManager(HOME_DIR, work, pol)


def _prepare_target(target, flags):
    """Return (cwd, command) for a .c file (compiled first), a program file, or a shell command."""
    p = Path(target).expanduser()
    if target.endswith(".c"):
        if not p.is_file():
            error("No such file: %s" % target)
            return None, None
        binary = compile_case(p.resolve(), flags)
        if not binary:
            return None, None
        return binary.parent, "./" + binary.name
    if p.is_file():
        return p.resolve().parent, "./" + p.name
    return Path.cwd(), target


def _run_id(msg):
    m = re.search(r"Started run id: (\S+)", msg)
    return m.group(1) if m else None


def sim_sweep(args):
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    if not values:
        error("No values given.")
        return 1
    cwd, prefix = _prepare_target(args.target, args.flags)
    if not cwd:
        return 1
    sims = _sim_manager(cwd)
    label = args.label or slug(Path(args.target).stem)
    pending, running, done = list(values), {}, []
    out("Sweep over %d values, up to %d at a time, %d thread(s) each." % (len(values), args.parallel, args.threads))
    try:
        while pending or running:
            for rid in list(running):
                v, r = running[rid]
                st, _ = sims.state(r)
                if st != "running":
                    done.append((v, rid, st))
                    del running[rid]
                    out("  finished  %-12s %s  [%s]" % (v, rid, st))
            while pending and len(running) < max(1, args.parallel):
                v = pending[0]
                command = prefix + " " + args.template.replace("{v}", v)
                msg, err = sims.start(command, cwd, args.threads, "%s-%s" % (label, slug(v, 12)), False, args.max_hours)
                rid = _run_id(msg)
                if not rid:
                    if "logical CPUs" in msg and running:
                        break                       # wait for a CPU to become free
                    error(msg)
                    done.append((v, None, "not started"))
                    pending.pop(0)
                    continue
                running[rid] = (v, sims._find(rid))
                pending.pop(0)
                out("  started   %-12s %s" % (v, rid))
            if pending or running:
                time.sleep(3)
    except KeyboardInterrupt:
        out("\nStopped scheduling. Runs already started keep going (see: basilisk-agent sim status).")
    out("\n%-14s %-36s %s" % ("value", "run id", "result"))
    for v, rid, st in done:
        out("%-14s %-36s %s" % (v, rid or "-", st))
    for rid, (v, _) in running.items():
        out("%-14s %-36s %s" % (v, rid, "running"))
    for v in pending:
        out("%-14s %-36s %s" % (v, "-", "not started"))
    return 0 if all(st == "finished" for _, _, st in done) and not pending and not running else 1


def sim_plot(args):
    sims = _sim_manager(Path.cwd())
    r = sims._find(args.source)
    if r:
        names = ("stdout.log", "stderr.log") if args.stdout else ("stderr.log", "stdout.log")
        logs = [Path(r["run_dir"]) / n for n in names]
        out_png = Path(args.out) if args.out else Path(r["cwd"]) / ("%s.png" % r["id"])
        title = "%s: %s" % (r["id"], r["command"])
    elif Path(args.source).is_file():
        logs = [Path(args.source)]
        out_png = Path(args.out) if args.out else Path(args.source).with_suffix(".png")
        title = Path(args.source).name
    else:
        error("%s is neither a run id nor a file." % args.source)
        return 1
    cols, data = [], []
    for log in logs:
        if log.is_file():
            cols, data = parse_table(log)
            if data:
                out("Data: %s" % log)
                break
    if not data:
        error("No numeric columns found in %s." % ", ".join(str(x) for x in logs))
        return 1
    out(table_summary(cols, data))
    ynames = [y.strip() for y in args.y.split(",")] if args.y else None
    try:
        ys, x = plot_table(cols, data, out_png, args.x, ynames, title, args.log)
    except ImportError:
        warn("matplotlib is not installed (sudo apt install python3-matplotlib): summary only.")
        return 0
    out(S.green + "Plot saved: %s  (%s vs %s)" % (out_png, ", ".join(ys), x) + S.reset)
    return 0


def sim_main(argv):
    ap = argparse.ArgumentParser(
        prog="basilisk-agent sim",
        description="Simulation commands that work without any AI and without an API key.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  basilisk-agent sim env\n"
               "  basilisk-agent sim run cavity.c 1000 --threads 4      compile with qcc, run in the background\n"
               "  basilisk-agent sim status\n"
               "  basilisk-agent sim wait <run id>\n"
               "  basilisk-agent sim plot <run id>                     plot the columns of the run's log\n"
               "  basilisk-agent sim sweep cavity.c --values 100,400,1000 --parallel 2 --threads 2\n"
               "Arguments after -- are passed to your program unchanged.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("env", help="show the machine environment (compilers, qcc, cores, RAM, GPU, Python)")
    p = sub.add_parser("run", help="compile a .c file with qcc (if given) and run it in the background")
    p.add_argument("target", help="case.c, a compiled program, or a quoted shell command")
    p.add_argument("--threads", type=int, default=1, help="OMP_NUM_THREADS (default 1)")
    p.add_argument("--label", help="label used in the run id")
    p.add_argument("--flags", help='qcc flags (default "-O2 -Wall", plus -fopenmp when available)')
    p.add_argument("--max-hours", type=float, help="stop the run after this many hours")
    p.add_argument("--force", action="store_true", help="start even if the CPUs are already busy")
    p.add_argument("--wait", action="store_true", help="stay and wait until the run finishes")
    s = sub.add_parser("status", help="status of running and recent runs")
    s.add_argument("run_id", nargs="?")
    w = sub.add_parser("wait", help="wait until a run finishes (or a pattern appears in its log)")
    w.add_argument("run_id")
    w.add_argument("--minutes", type=float, default=24 * 60)
    w.add_argument("--until", help="regular expression to wait for in the log")
    k = sub.add_parser("stop", help="stop a run")
    k.add_argument("run_id")
    pl = sub.add_parser("plot", help="plot the numeric columns of a run log or a data file")
    pl.add_argument("source", help="run id or data file")
    pl.add_argument("--x", help="column for the x axis (default t)")
    pl.add_argument("--y", help="comma-separated columns to plot (default: all)")
    pl.add_argument("--out", help="output PNG")
    pl.add_argument("--log", action="store_true", help="logarithmic y axis")
    pl.add_argument("--stdout", action="store_true", help="read the run's stdout log first")
    sw = sub.add_parser("sweep", help="run one case for several parameter values")
    sw.add_argument("target", help="case.c or a compiled program")
    sw.add_argument("--values", required=True, help="comma-separated values, e.g. 100,400,1000")
    sw.add_argument("--template", default="{v}", help='program arguments; {v} is replaced by each value (default "{v}")')
    sw.add_argument("--threads", type=int, default=1)
    sw.add_argument("--parallel", type=int, default=1, help="runs at the same time (default 1)")
    sw.add_argument("--flags")
    sw.add_argument("--label")
    sw.add_argument("--max-hours", type=float)
    prog_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, prog_args = argv[:i], argv[i + 1:]
    args, unknown = ap.parse_known_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 1
    if args.cmd != "run" and unknown:
        ap.error("unrecognized arguments: %s" % " ".join(unknown))
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "env":
        env = detect_environment(Path.cwd())
        atomic_write(HOME_DIR / "env.json", json.dumps(env, indent=1))
        out(environment_text(env))
        sugg = suggest_local_model(env)
        if sugg:
            out("Free local AI model that should fit this machine: " + sugg)
        return 0
    if args.cmd == "run":
        cwd, prefix = _prepare_target(args.target, args.flags)
        if not cwd:
            return 1
        extra = unknown + prog_args
        command = prefix + ("" if not extra else " " + " ".join(shlex.quote(a) for a in extra))
        sims = _sim_manager(cwd)
        msg, err = sims.start(command, cwd, args.threads, args.label, args.force, args.max_hours)
        (error if err else out)(msg)
        rid = _run_id(msg)
        if rid and args.wait:
            out(sims.wait(rid, 24 * 60, None, 15)[0])
        return 1 if err else 0
    if args.cmd == "plot":
        return sim_plot(args)
    if args.cmd == "sweep":
        return sim_sweep(args)
    sims = _sim_manager(Path.cwd())
    if args.cmd == "status":
        msg, err = sims.status(args.run_id)
    elif args.cmd == "wait":
        msg, err = sims.wait(args.run_id, args.minutes, args.until, 15)
    else:
        msg, err = sims.stop(args.run_id)
    (error if err else out)(msg)
    return 1 if err else 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def load_config():
    return read_json(HOME_DIR / "config.json", {})


def ollama_running(base, timeout=1.5):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, socket.timeout, OSError, ValueError):
        return False


def suggest_local_model(env):
    gpu = env.get("gpu_mem_gb") or 0
    ram = env.get("ram_total_gb") or 0
    for need, name, note in LOCAL_MODEL_SUGGESTIONS:
        if gpu >= need:
            return "%s (%s) on your %.0f GB GPU" % (name, note, gpu)
    for need, name, note in LOCAL_MODEL_SUGGESTIONS:
        if ram >= need + 8:
            return "%s (%s) on the CPU with %.0f GB RAM; it works, but slowly" % (name, note, ram)
    return None


def parse_args(argv=None):
    conf = load_config()
    ap = argparse.ArgumentParser(
        prog="basilisk-agent",
        description="Autonomous CFD agent for Basilisk. Run it from your project folder. The AI can be Claude, "
                    "a free local model through Ollama, or any OpenAI-compatible server. "
                    "`basilisk-agent sim --help` lists commands that need no AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  basilisk-agent                       interactive session in the current folder\n"
               "  basilisk-agent \"Rayleigh-Benard convection at Ra=1e5, Pr=0.71; Nusselt number vs theory\"\n"
               "  basilisk-agent --provider ollama \"...\"     free local model (Ollama)\n"
               "  basilisk-agent --resume              continue the last conversation in this folder\n"
               "  basilisk-agent --yes --mode full \"...\" unattended run (no questions, no approvals)\n"
               "  basilisk-agent --check               check the environment and the AI connection\n"
               "  basilisk-agent sim run cavity.c 1000 compile and run without any AI")
    ap.add_argument("task", nargs="*", help="what you want the agent to do (optional)")
    ap.add_argument("--provider", choices=["claude", "ollama", "openai"], default=conf.get("provider"),
                    help="which AI to use (default: claude if ANTHROPIC_API_KEY is set, otherwise a running Ollama)")
    ap.add_argument("--model", default=None,
                    help="Claude: opus, sonnet, fable, haiku or a model id (default %s). Ollama: a pulled model "
                         "(default %s). OpenAI-compatible: required." % (DEFAULT_MODEL, DEFAULT_LOCAL_MODEL))
    ap.add_argument("--base-url", default=conf.get("base_url"),
                    help="server URL for --provider ollama or openai (defaults: %s, $OPENAI_BASE_URL)" % OLLAMA_URL)
    ap.add_argument("--num-ctx", type=int, default=int(conf.get("num_ctx", DEFAULT_NUM_CTX)),
                    help="context window for Ollama models (default %(default)s; larger needs more memory)")
    ap.add_argument("--context", type=int, default=int(conf.get("context", 128000)),
                    help="context window of an OpenAI-compatible model (default %(default)s)")
    ap.add_argument("--vision", choices=["auto", "on", "off"], default=conf.get("vision", "auto"),
                    help="can a non-Claude model look at images? auto asks Ollama (default auto)")
    ap.add_argument("--effort", default=conf.get("effort", DEFAULT_EFFORT),
                    choices=["low", "medium", "high", "xhigh", "max"], help="Claude reasoning effort (default %(default)s)")
    ap.add_argument("--mode", default=conf.get("mode", "auto"), choices=["ask", "auto", "full"],
                    help="approvals for shell commands: ask=all, auto=risky only (default), full=none")
    ap.add_argument("--workdir", default=None, help="workspace folder (default: current folder)")
    ap.add_argument("--resume", action="store_true", help="continue the previous conversation in this workspace")
    ap.add_argument("--yes", action="store_true",
                    help="unattended: run the task and exit; no questions (risky commands are refused unless --mode full)")
    ap.add_argument("--allow-read", action="append", default=list(conf.get("allow_read", [])),
                    help="extra folder the agent may read (repeatable)")
    ap.add_argument("--allow-write", action="append", default=list(conf.get("allow_write", [])),
                    help="extra folder the agent may write (repeatable)")
    ap.add_argument("--fetch-domain", action="append", default=list(conf.get("fetch_domains", [])),
                    help="extra web domain fetch_url may read (basilisk.fr is always allowed)")
    ap.add_argument("--long-run-hours", type=float, default=float(conf.get("long_run_hours", 6)),
                    help="runs expected to take longer need your approval (default %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=int(conf.get("max_tokens", DEFAULT_MAX_TOKENS)),
                    help="Claude output limit per response")
    ap.add_argument("--max-calls", type=int, default=int(conf.get("max_calls", 250)),
                    help="model calls per request before pausing (default %(default)s)")
    ap.add_argument("--compact-at", type=int, default=int(conf.get("compact_at", DEFAULT_COMPACT_AT)),
                    help="compact the conversation above this many prompt tokens")
    ap.add_argument("--cache-ttl", choices=["5m", "1h"], default=conf.get("cache_ttl", "5m"),
                    help="Claude prompt cache lifetime (1h helps when you pause often)")
    ap.add_argument("--show-thinking", action="store_true", default=bool(conf.get("show_thinking", False)),
                    help="print the model's thinking (summaries for Claude)")
    ap.add_argument("--check", action="store_true", help="check the environment and the AI connection, then exit")
    ap.add_argument("--version", action="version", version="basilisk-agent " + VERSION)
    args = ap.parse_args(argv)
    args.api_key = os.environ.get("ANTHROPIC_API_KEY") or conf.get("api_key", "")
    if not args.provider:
        if args.api_key:
            args.provider = "claude"
        elif ollama_running(args.base_url or OLLAMA_URL):
            args.provider = "ollama"
        else:
            args.provider = "claude"
    if not args.model:
        args.model = {"claude": conf.get("claude_model") or conf.get("model") or DEFAULT_MODEL,
                      "ollama": conf.get("ollama_model") or DEFAULT_LOCAL_MODEL,
                      "openai": conf.get("openai_model") or ""}[args.provider]
    if args.provider == "claude":
        args.model = MODEL_ALIASES.get(args.model, args.model)
    args.local_max_tokens = int(conf.get("local_max_tokens", DEFAULT_LOCAL_MAX_TOKENS))
    args.fetch_domains = ["basilisk.fr"] + [d.lower() for d in args.fetch_domain]
    args.workdir = os.path.abspath(os.path.expanduser(args.workdir or os.getcwd()))
    args.interactive = (not args.yes) and sys.stdin.isatty()
    return args


NO_AI_HELP = ("No AI is configured. Choose one:\n"
              "  Claude (best results):  export ANTHROPIC_API_KEY=sk-ant-...\n"
              "  Free local model:       install Ollama, run `ollama pull %s`, then use --provider ollama\n"
              "  Without any AI:         basilisk-agent sim --help   (compile, run, monitor, plot, sweep)"
              % DEFAULT_LOCAL_MODEL)


def check(args):
    out(S.bold + "Basilisk Agent %s" % VERSION + S.reset)
    env = detect_environment(Path(args.workdir))
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(HOME_DIR / "env.json", json.dumps(env, indent=1))
    out(environment_text(env))
    sugg = suggest_local_model(env)
    if sugg:
        out("Free local AI model that should fit this machine: " + sugg)
    out("")
    out("AI provider: %s, model %s" % (args.provider, args.model or "(none)"))
    if args.provider == "claude" and not args.api_key:
        error(NO_AI_HELP)
        return 1
    if args.provider == "openai" and not args.model:
        error("--model is required with --provider openai.")
        return 1
    client = make_client(args)
    if args.provider == "claude":
        client.effort, client.max_tokens = "low", 2048
    if args.provider == "ollama":
        ok, msg = client.probe()
        (out if ok else error)(msg)
        if not ok:
            return 1
    try:
        r = client.create("Reply with the single word: ready", None, [{"role": "user", "content": "Are you ready?"}])
        txt = " ".join(b.get("text", "") for b in r["content"] if b.get("type") == "text").strip()
        out(S.green + "AI connection OK (%s): %s" % (args.model, txt[:60]) + S.reset)
        return 0
    except APIError as e:
        error("AI check failed: %s" % e)
        return 1


def main(argv=None):
    if os.name == "nt":
        print("Please run Basilisk Agent inside WSL (Ubuntu) or Linux, not in Windows Python.")
        return 1
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "sim":
        return sim_main(argv[1:])
    args = parse_args(argv)
    if args.check:
        return check(args)
    if args.provider == "claude" and not args.api_key:
        error(NO_AI_HELP)
        return 1
    if args.provider == "openai" and not args.model:
        error("--model is required with --provider openai.")
        return 1
    work = Path(args.workdir)
    if not work.is_dir():
        error("Workspace %s does not exist." % work)
        return 1
    if str(work) in ("/", str(Path.home())):
        warn("The workspace is %s. The agent can create and delete files anywhere inside it." % work)
        if args.interactive:
            if input("Use a project folder instead? Continue anyway? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
        else:
            error("Refusing to run unattended with %s as the workspace; use --workdir." % work)
            return 1
    if args.provider == "ollama" and args.num_ctx < 16384:
        warn("--num-ctx %d is too small for the agent's instructions and tools; use 32768 or more." % args.num_ctx)
    agent = Agent(args)
    if args.provider == "ollama":
        ok, msg = agent.client.probe()
        if not ok:
            error(msg)
            return 1
        info(msg)
    brain = ("%s, effort %s" % (args.model, args.effort)) if args.provider == "claude" else (
        "%s via %s" % (args.model, args.provider))
    out(S.bold + "Basilisk Agent %s" % VERSION + S.reset + S.dim +
        " | %s | mode %s | workspace %s" % (brain, args.mode, work) + S.reset)
    if args.resume:
        if agent.load():
            info("Resumed the previous conversation (%d messages)." % len(agent.messages))
        else:
            info("No previous conversation here; starting fresh.")
    elif agent.session_file.exists() and args.interactive:
        info("(A previous conversation exists here; start with --resume to continue it.)")
    if not agent.env_file.exists():
        info("Detecting the environment (first run)...")
    agent.environment()
    task = " ".join(args.task).strip()
    if not args.interactive:
        if not task:
            error("Unattended mode needs a task.")
            return 1
        agent.run_turn(task)
        return 0
    if not task:
        out(S.dim + "Describe a problem to simulate or analyse (type /help for commands)." + S.reset)
    interactive_loop(agent, task)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        out("")
        sys.exit(130)
