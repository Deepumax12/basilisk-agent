# Basilisk Agent

An AI agent that sets up, runs and analyses [Basilisk](http://basilisk.fr) simulations on your own
computer. Give it a problem in plain words, for any kind of flow (single or multiphase, embedded solids, heat
transfer, axisymmetric or 3D, shallow water, compressible, viscoelastic...). It then:

1. reads your machine: OS/WSL, cores, RAM, GPU, gcc, `qcc` and `$BASILISK`, OpenMP/MPI, Python packages, ffmpeg;
2. finds the closest Basilisk example in your source tree and checks the real headers, so it doesn't invent APIs;
3. writes the C code in a case folder, compiles it with `qcc`, and fixes errors;
4. runs a cheap smoke test, then estimates the cost of the production run (it asks you first if a run would exceed 6 h);
5. launches long runs in the background and monitors them, detecting NaN, crashes and memory problems;
6. post-processes with Python, looks at its own plots, and compares with theory or benchmarks;
7. writes `REPORT.md`, and keeps a project notebook (`NOTES.md`) and machine-specific lessons (`KNOWLEDGE.md`) between sessions.

It is a single Python file with no dependencies beyond the standard library.

## Three ways to use it

| Mode | What you need | What it can do |
|---|---|---|
| **Claude** (best) | an API key, pay per use | everything above, most reliably |
| **Free local AI** (Ollama) | a GPU or plenty of RAM; no key, no cost, data stays on your machine | everything above, but a local model makes more mistakes |
| **No AI** (`sim` commands) | nothing | compile, run in the background, monitor, plot, parameter sweeps, all for code you write |

---

## 1. Install (WSL / Linux)

```bash
git clone https://github.com/Deepumax12/basilisk-agent.git ~/tools/basilisk-agent
cd ~/tools/basilisk-agent
./install.sh
```

This links `basilisk-agent` into `~/.local/bin` and creates `~/.basilisk_agent/config.json`. To update later:
`cd ~/tools/basilisk-agent && git pull`.

Requirements:

- Linux, or **WSL2 (Ubuntu) on Windows**. Run the agent inside WSL, not in Windows Python.
- Python 3.8 or newer.
- Basilisk installed, with `qcc` on your `PATH` and `BASILISK` exported (pointing to `basilisk/src`).
- Recommended: `sudo apt install python3-numpy python3-matplotlib python3-scipy python3-pil ffmpeg`

Check the machine (no AI needed):

```bash
basilisk-agent sim env
```

It also suggests a local AI model that fits your GPU or RAM.

## 2. Choose the AI

### a) Claude (best results)

Get a key at <https://platform.claude.com>. API usage is billed separately from a claude.ai subscription.

```bash
echo 'export ANTHROPIC_API_KEY=sk-ant-...your key...' >> ~/.bashrc && source ~/.bashrc
basilisk-agent --check
```

The config created by `install.sh` selects Ollama. To make Claude your default, set `"provider": "claude"` in `~/.basilisk_agent/config.json`, or pass `--provider claude`.

The default is Claude Opus 5.5 at high effort. `--model sonnet` costs about half, and `--effort medium` costs less again. A complete task usually costs a few dollars. A running estimate is printed after each request.

### b) Free local AI with Ollama (no key, no cost)

[Ollama](https://ollama.com) runs open models on your own computer. Install it inside WSL:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > ~/ollama.log 2>&1 &     # only needed if it is not already running as a service
ollama pull qwen3-coder:30b            # pick a model that fits your machine (table below)
basilisk-agent --provider ollama --check
```

The config created by `install.sh` already uses Ollama, with `qwen3-coder:30b`. To use another model, change `"ollama_model"` there, or pass `--model gpt-oss:20b`. If you delete the `"provider"` line, the agent picks Claude when `ANTHROPIC_API_KEY` is set, and otherwise picks Ollama if it is running.

| Your machine | Suggested model | Size |
|---|---|---|
| GPU with 24 GB or more | `qwen3-coder:30b` (made for coding agents) | ~19 GB |
| GPU with 16 GB | `gpt-oss:20b` | ~14 GB |
| No suitable GPU, 24 GB+ RAM | `gpt-oss:20b` on the CPU | works, but slowly |
| GPU with 8 GB | `qwen2.5-coder:7b` | small; many mistakes |

These suggestions come from public benchmarks as of late 2026; newer models appear often at
<https://ollama.com/library>. Pick one that supports **tools**, which the agent needs.
Models that also support **vision** let the agent look at its own plots; with other models it checks results numerically instead.

Be realistic about local models:

- They are much weaker than Claude at writing correct Basilisk code from scratch and at long multi-step work. Expect more compile errors and more need for your guidance.
- They work well for running, monitoring and plotting, for adapting an existing case, and for standard problems.
- For new or complex physics, Claude gives far better results.
- A GPU makes a big difference. NVIDIA GPUs work inside WSL2 with the normal Windows driver; `nvidia-smi` should work in WSL.
- The context window is set with `--num-ctx` (default 32768). Use at least 16384; larger values need more memory.

### c) Any OpenAI-compatible server

This covers LM Studio, vLLM, a llama.cpp server, or a cloud provider.

```bash
export OPENAI_API_KEY=...        # if the server needs one
basilisk-agent --provider openai --base-url http://localhost:1234/v1 --model <model-name> "..."
```

Add `--vision on` if the model accepts images.

## 3. Use the agent

The agent works in the folder you start it from, which is its **workspace**.

```bash
mkdir -p ~/cfd/cavity && cd ~/cfd/cavity
basilisk-agent "2D lid-driven cavity at Re = 1000; validate the centreline velocities against Ghia et al. (1982)"
```

- Keep talking to it after it answers. Continue another day with `basilisk-agent --resume` in the same folder.
- For an unattended run, use `basilisk-agent --yes --mode full "..."`. It never asks you anything, documents its assumptions, then exits.

Example tasks:

- "Flow past a cylinder at Re = 100 with embed.h; measure the Strouhal number and the drag coefficient"
- "Axisymmetric rising bubble, Bo = 10, Mo = 1e-3, density ratio 1000; terminal Reynolds number vs Clift et al."
- "Capillary wave damping; compare the amplitude decay with Prosperetti's analytical solution"
- "Rayleigh-Taylor instability at At = 0.5; linear growth rate vs theory"
- "Read the dump files in ./run3 and plot the bubble shape and rise velocity versus time"
  (start it with `--allow-read /path/to/run3` if that folder is outside the workspace)

Commands inside a session:

| Command | What it does |
|---|---|
| `/runs` | status of running and recent simulations |
| `/env` | re-detect and show the environment |
| `/cost` | tokens used (and estimated cost for Claude) |
| `/model NAME` | switch model |
| `/effort LEVEL` | Claude only: `low`, `medium`, `high`, `xhigh`, `max` |
| `/mode ask` | approvals: `ask`, `auto`, `full` |
| `/compact` | summarise the conversation to free context |
| `/new` | fresh conversation (notes and runs are kept) |
| `/exit` | quit; background simulations keep running |

### Problem types and where the agent starts

The agent starts from the closest validated case in your Basilisk tree. These were checked against the Basilisk source of October 2026:

| Problem | Starting point in `$BASILISK` |
|---|---|
| Rising bubble | `test/rising.c` (Hysing benchmark, 2D and axisymmetric), `examples/bubble.c` (3D) |
| Drop | `test/oscillation.c` (shape oscillation), `test/sessile.c` (contact angle), `test/fall.c` (drop impact) |
| Waves | `test/capwave.c` (capillary wave vs Prosperetti), `test/dispersion_gravitocap.c`, `examples/breaking.c` |
| Pipe flow | `test/poiseuille-axi.c`, `test/pipe-axi-embed.c` (developing flow) |
| Inclined flow | `test/poiseuille45.c` (gravity-driven flow in a channel inclined at 45°) |
| Cylinder wake, cavity, turbulence | `examples/karman.c`, `test/lid.c`, `examples/isotropic.c` |
| **Taylor bubble, falling film** | no ready-made case: the agent combines the pieces above and validates against correlations (Taylor bubble) or the Nusselt film solution (falling film) |

For a Taylor bubble or another problem you have already solved, point the agent at your own validated code, for example
`basilisk-agent --allow-read ~/sims/taylor_bubble "Adapt my Taylor bubble code in ~/sims/taylor_bubble to Eo = 20 ..."`.
Adapting good code is far more reliable than writing it from scratch, especially with a local model.

To check that your Basilisk install handles these problem types, run Basilisk's own regression tests. Each one compares with reference results and can take several minutes:

```bash
cd $BASILISK/test && make rising.tst oscillation.tst capwave.tst poiseuille-axi.tst poiseuille45.tst
```

## 4. Without any AI: `sim` commands

```bash
basilisk-agent sim env                                    # machine report
basilisk-agent sim run cavity.c 1000 --threads 4          # compile with qcc, run in the background
basilisk-agent sim status                                 # all running and recent runs
basilisk-agent sim wait <run id>                          # wait until it finishes (or --until 'regex')
basilisk-agent sim plot <run id>                          # plot every column of the log vs t, print a summary
basilisk-agent sim plot data.txt --y u,v --log            # plot any whitespace-separated data file
basilisk-agent sim sweep cavity.c --values 100,400,1000 --parallel 2 --threads 2
basilisk-agent sim stop <run id>
```

- `run` compiles with `-O2 -Wall` (plus `-fopenmp` when available), and adds the view.h graphics libraries if your code includes `view.h`.
- Arguments after `--` go to your program unchanged.
- Logs are kept in `<case>/.runs/<run id>/`.
- `plot` understands Basilisk-style logs: a `# i t dt ...` header, whitespace-separated columns, and `key=value` tokens.

## 5. Safety

- **Files:** the agent writes only inside the workspace (plus `/tmp`). It reads the workspace, your Basilisk tree, `~/.basilisk_agent`, and folders you add with `--allow-read`.
- **Shell commands** depend on `--mode`:
  - `ask`: you approve every command. Use this for your first sessions.
  - `auto` (default): only risky commands need approval. These include sudo, package installs, deleting or writing outside the workspace, git push, ssh, editing `~/.bashrc`, and touching credentials.
  - `full`: no approvals.
  - Destructive commands such as `rm -rf /` are always blocked.
- API keys are never passed to the commands it runs. The Basilisk source tree is never modified.

## 6. Files it keeps

- In the workspace:
  - `NOTES.md`: project notebook.
  - `<case>/`: code, scripts, figures, `REPORT.md`.
  - `<case>/.runs/<run id>/`: logs of each background run.
  - `.basilisk_agent/session.json`: the conversation, used by `--resume`.
- In `~/.basilisk_agent/`:
  - `config.json`: your defaults (see `config.example.json`).
  - `KNOWLEDGE.md`: lessons about your machine; editable.
  - `USER.md` (optional, written by you): tells the agent about you. For example: "CFD researcher, multiphase flows; dimensionless variables; 300 dpi figures; keep 2 cores free".
  - `env.json`, `runs.json`: environment cache and the run registry.

## 7. ParaView (Windows side)

Ask the agent to export VTK files. ParaView on Windows can open them at `\\wsl$\Ubuntu\home\<user>\cfd\...`.
Keep workspaces under `~` rather than `/mnt/c`, because file I/O on the Windows drive is slow in WSL.

## 8. Tests and limits

- `tests/run_all.sh` runs 175 automated checks against mock Claude, Ollama and OpenAI-compatible servers and a stand-in solver. It needs no key, no Ollama and no Basilisk.
- It has not yet run against a real Basilisk install or a real model. Start with a benchmark (the cavity above) in `--mode ask` and watch each step.
- It runs real code on your computer and makes real modelling choices. Read its plan and `REPORT.md`; you remain the scientist.
