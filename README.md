# MetaOpticsAI

Autonomous AI-driven computational nanophotonics & metasurface design platform.

```
LangGraph Agents → MCP Tool Layer → Scientific Tool Servers → metabox3 RCWA backend
```

## Quick start

```bash
# 1. Install
pip install -e ".[tf,dev]"
cp .env.example .env

# 2. Vendor metabox3 into the tree
cp -r /path/to/metabox3 src/metaopticsai/vendor/metabox3/

# 3. Verify the install
python scripts/check_manifest.py    # confirms every file matches MANIFEST.txt
python scripts/verify_install.py    # imports every module + builds tool registry
pytest tests/                       # 10 tests, all passing

# 4. Run the all-in-one MCP server (stdio, for Claude Desktop)
python scripts/run_mcp_stdio.py

# 5. Or run HTTP/SSE for FastAPI / remote clients
python scripts/run_mcp_http.py --host 0.0.0.0 --port 8765

# 6. Run a headless Self-RAG metalens design
python scripts/run_workflow.py "Design a TiO2 metalens for 532 nm, f=2 mm, Ø=1 mm"
```

## Diagnostics

Two scripts catch the most common deployment problems:

- `scripts/check_manifest.py` — compares every `.py` file's SHA against the
  shipped `MANIFEST.txt`. Spots truncated, missing, or modified files in one
  pass. Run this whenever you've just unzipped or pulled changes.
- `scripts/verify_install.py` — imports every module, builds the tool
  registry, and prints which of the 15 expected tools are present. Run this
  whenever an import error surfaces during pytest or at MCP server startup.

On a healthy tree both scripts end with `✅ All checks passed.`

## Architecture

See `ARCHITECTURE.md` for the full layered design. Briefly:

- `domain/` — pure types (Pydantic / dataclasses), no I/O, no compute.
- `store/` — pluggable artifact handle store.
- `physics/backends/` — **only place** that imports `vendor/metabox3`.
- `tools/` — transport-agnostic scientific tools (Pydantic schemas + `BaseTool`).
- `mcp_servers/` — thin FastMCP adapters over `tools/`.
- `llm/` — provider abstraction over Ollama / Groq / Anthropic.
- `rag/` — Self-RAG knowledge base + grader.
- `workflows/` — LangGraph graph + nodes + edges.
- `orchestration/` — top-level controller; single composition root for non-MCP entry points.

## Project layout

```
src/metaopticsai/
├── domain/        ← types
├── store/         ← handles
├── physics/       ← simulation backends
├── phase/         ← phase-mask profiles
├── analysis/      ← PSF, MTF, Strehl, Zernike, EE, farfield
├── fabrication/   ← GDS export
├── optimization/  ← heuristic, gradient, surrogate optimizers
├── tools/         ← BaseTool implementations + Pydantic schemas
├── mcp_servers/   ← FastMCP transport shells
├── llm/           ← LLM provider abstraction
├── rag/           ← Self-RAG knowledge base
├── prompts/       ← centralized prompt templates
├── workflows/     ← LangGraph nodes, edges, graph builders
├── agents/        ← higher-level agent abstractions
├── orchestration/ ← MetaOpticsController, JobManager
├── configs/       ← Pydantic Settings
├── utils/         ← logging, images, json, async
└── vendor/
    └── metabox3/  ← vendored, untouched
```

## Backend strategy

`physics/backends/` exposes three implementations of `SimulationBackend`:

- `MetaboxBackend` — TF-backed RCWA, the primary workhorse.
- `AnalyticalBackend` — fast heuristic FOM estimator. No TF dependency.
- `S4Backend` — stub for the planned phoebe-p/S4 cross-validation.

A `BackendRegistry` resolves `"auto" | "metabox" | "analytical" | "s4"` to an instance.
Tools, workflows, and the optimizer never import `metabox3` directly.

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
