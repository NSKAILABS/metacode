# MetaOpticsAI

An LLM-orchestrated meta-optics design agent. A Groq-hosted LLM plans and calls a
verified toolbox of physics primitives backed by the vendored `metabox3` RCWA solver
(Huang et al. 2025 pattern). The LLM never invents numbers — every metric traces back
to a deterministic call into `metabox3` or its analytical fallback.

```
prompt ─▶ llm/agent.py ──tool──▶ mcp_server/tools.py ──▶ metaopticsai.physics ──▶ metabox3
                       ◀─json───
```

## Layout

```
llm/                LLM client + tool-calling loop (Groq)
mcp_server/         Verified tools, templates, workflow, simulator wrapper
src/metaopticsai/   Core package
  physics/          RCWA, FDTD, propagation, materials
  optimization/     gradient / heuristic / surrogate
  analysis/         PSF, MTF, Strehl, Zernike, far-field
  fabrication/      GDS export
  vendor/metabox3/  RCWA backend
examples/demo.py    End-to-end runnable demo
config/.env         Secrets (git-ignored)
```

## Setup

```bash
pip install -r requirements.txt
cp config/.env.example config/.env    # then edit: GROQ_API_KEY=<key>
```

Get a free Groq key at https://console.groq.com/keys.

### Environment

| Variable               | Default                   | Purpose                          |
|------------------------|---------------------------|----------------------------------|
| `GROQ_API_KEY`         | *(required)*              | Groq key                         |
| `GROQ_MODEL`           | `llama-3.3-70b-versatile` | Any tool-calling model           |
| `GROQ_TEMPERATURE`     | `0.1`                     | Sampling temperature             |
| `GROQ_MAX_TOKENS`      | `1024`                    | Max output tokens per turn       |
| `AGENT_MAX_ITERATIONS` | `5`                       | Cap on tool-call round-trips     |

## Usage

```bash
python examples/demo.py --no-llm                          # exercise tools only
python examples/demo.py                                    # live agent run
python -m llm "Design a UV metalens for 405 nm with GaN." # module CLI
```

## MCP tools

| Tool                    | Purpose                                              |
|-------------------------|------------------------------------------------------|
| `list_templates`        | Browse verified starting points                      |
| `get_template`          | Full parameter dict for one template                 |
| `get_workflow_guide`    | Canonical 5-stage design workflow                    |
| `validate_design`       | Check params against `MetalensDesignParams`          |
| `run_simulation`        | RCWA diameter sweep (metabox3 or analytical)         |
| `get_optimization_tips` | Suggestions derived from a sim result                |

## Guarantees

- **Deterministic physics.** All numeric fields come from `metabox3`, not the LLM.
- **Verified templates.** The agent tweaks known-good starting points; it cannot invent unit-cell geometries.
- **Bounded loops.** Tool-call loop capped at 5 iterations.
- **Forward-only in-agent.** Inverse/gradient design lives in `metaopticsai.optimization` and runs outside the agent.

## Extending

- **Template:** append to `TEMPLATES` in `mcp_server/templates.py`; Pydantic validation blocks bad entries.
- **Tool:** add `_tool_x(args) -> {"ok": ...}` in `mcp_server/tools.py`, register in `TOOLS` and `_HANDLERS`.

## License

See [LICENSE](LICENSE).
