# LLM + MCP Meta-Optics Design Agent

A minimal research prototype that connects a Groq-hosted LLM to a
verified toolbox of meta-optics primitives backed by the vendored
`metabox3` RCWA solver. Inspired by the LLM + MCP + physics-backend
pattern of Huang et al. 2025.

```
User prompt
    │
    ▼
┌────────────────────┐        ┌───────────────────────┐
│ llm/ (Groq client) │──tool──▶│ mcp_server/ (tools)   │
│  agent.py loop     │◀─json──│  6 verified functions │
└────────────────────┘        └──────────┬────────────┘
                                         │
                                         ▼
                              ┌─────────────────────────┐
                              │ metaopticsai.physics    │
                              │   → vendor/metabox3     │
                              │     (RCWA / propagation)│
                              └─────────────────────────┘
```

The LLM is the **orchestrator**, not the physics engine. Every number
in the final answer traces back to a deterministic call into
`metabox3` (or the analytical fallback when TensorFlow is absent).

---

## Layout

```
config/              secrets (git-ignored)
  .env.example       template to copy to config/.env
llm/                 LLM-side code
  groq_client.py     tiny Groq SDK wrapper
  agent.py           tool-calling loop (max 5 iterations)
  __main__.py        `python -m llm "..."`
mcp_server/          verified tools & templates (no LLM knowledge)
  tools.py           6 tools + JSON-Schema descriptors
  templates.py       3 verified meta-atom starting points
  workflow.py        the canonical five-stage design workflow
  simulator.py       thin wrapper over metaopticsai.physics.rcwa
  README.md          (this file)
examples/
  demo.py            end-to-end runnable demo
src/metaopticsai/    core physics package
  vendor/metabox3/   RCWA backend
```

---

## Setup

```bash
# 1. Get a free Groq key from https://console.groq.com/keys
# 2. Copy the template and paste your key
cp config/.env.example config/.env
# edit config/.env, set GROQ_API_KEY=<your key>
```
---

## Environment variables

| Variable                | Default                     | Meaning                                |
|-------------------------|-----------------------------|----------------------------------------|
| `GROQ_API_KEY`          | *(required)*                | Your Groq key.                         |
| `GROQ_MODEL`            | `llama-3.3-70b-versatile`   | Any Groq tool-calling model.           |
| `GROQ_TEMPERATURE`      | `0.1`                       | Sampling temperature.                  |
| `GROQ_MAX_TOKENS`       | `1024`                      | Max output tokens per turn.            |
| `AGENT_MAX_ITERATIONS`  | `5`                         | Hard cap on tool-call round-trips (3–5).|

---

## MCP tools

| Tool                     | Purpose                                                   |
|--------------------------|-----------------------------------------------------------|
| `list_templates`         | Browse verified starting points.                          |
| `get_template`           | Full parameter dict for one template.                     |
| `get_workflow_guide`     | The canonical 5-stage design workflow (markdown).         |
| `validate_design`        | Re-check design params against `MetalensDesignParams`.    |
| `run_simulation`         | RCWA diameter sweep on metabox3 (or analytical fallback). |
| `get_optimization_tips`  | Human-readable suggestions from a sim result.             |

Each tool returns `{"ok": bool, ...}`. On error the LLM sees the
error string in the same turn and can retry with corrected arguments.

The tool-call loop is capped at **5 iterations** to prevent runaway
usage.

---

## Run the demo

```bash
# Dry-run: exercise the tools without touching the LLM (no key needed)
python examples/demo.py --no-llm

# Live: LLM drives the tools (requires GROQ_API_KEY in config/.env)
python examples/demo.py

# Custom prompt
python examples/demo.py --prompt "Design a silicon NIR metalens at 850 nm."

# Or via the module CLI
python -m llm "Design a UV metalens for 405 nm using GaN pillars."
```

Both modes print the tool-call trace (unless `--quiet`) and end with
the final agent summary.

---

## Design guarantees & limits

- **Physics is deterministic.** The LLM never generates numeric field
  values; every metric comes from `metabox3` or the analytical
  backend.
- **Templates are the ground truth.** The LLM starts from a verified
  template and only tweaks; it cannot invent unit-cell geometries.
- **Bounded loops.** Iteration cap 3–5, enforced in `llm/agent.py`.
- **No inverse design.** We wrap only what `metabox3` genuinely
  supports (forward simulation + parameter sweeps). We do not claim
  gradient-based inverse design in this prototype — for that path see
  `metaopticsai.optimization.gradient` (used outside the agent).
- **Single-agent, in-process.** No multi-agent orchestration, no
  separate stdio process. Tools are called via
  `mcp_server.tools.dispatch` — the same JSON-Schema shape the MCP
  protocol expects, so wiring in a real MCP server later is a
  ~30-line change.

---

## Extending

- **Add a template:** append to `TEMPLATES` in `mcp_server/templates.py`.
  Anything the Pydantic validator rejects will be caught before it
  reaches the LLM.
- **Add a tool:** implement `_tool_yourthing(args) -> {"ok": ..., ...}`
  in `mcp_server/tools.py`, add its JSON-Schema entry to `TOOLS`, and
  register it in `_HANDLERS`. No other file needs to change.
