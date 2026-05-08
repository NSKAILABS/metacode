# MetaOpticsAI · MCP + Hybrid LLM Stack

This bundle wires the existing MetaOpticsAI codebase into a single **MCP server**
that's consumable by three different front-ends — Claude Desktop, a FastAPI
web UI, and a headless CLI — and a **hybrid LLM controller** that uses the
Anthropic Claude API for planning and your local Ollama for cheap reflection
loops.

## High-level architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ FRONT-ENDS (any of)                                                    │
│   • Claude Desktop  →  stdio                                           │
│   • FastAPI web UI  →  HTTP   (mounts the same FastMCP server)         │
│   • CLI / scripts   →  in-process import                               │
└─────────────────────────┬──────────────────────────────────────────────┘
                          │  Model Context Protocol  (tools/JSON-RPC)
                          ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MCP SERVER  (mcp_server/server.py — FastMCP 3.x)                       │
│   Tools exposed:                                                       │
│     • get_fdtd_library          — list materials/wavelengths            │
│     • run_rcwa_sweep            — phase-vs-diameter library             │
│     • generate_phase_mask       — FZL / axicon / SPP / image            │
│     • optimize_metalens         — gradient-based Adam loop              │
│     • analyze_psf               — PSF + Strehl + FWHM                   │
│     • analyze_mtf               — MTF curves                            │
│     • zernike_decompose         — Zernike aberration coefficients       │
│     • export_gds                — write fabrication-ready GDS           │
│     • search_knowledge          — Self-RAG over paper corpus            │
│     • submit_job / get_job      — long-running queue (in-process)       │
└─────────────────────────┬──────────────────────────────────────────────┘
                          │  Python imports (no extra serialization)
                          ▼
┌────────────────────────────────────────────────────────────────────────┐
│ METABOX + CORE                                                         │
│   metabox / core.rcwa_engine / core.phase_engine / core.gds_engine     │
│   analysis.psf / .mtf / .strehl / .zernike / .farfield                 │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ HYBRID LLM CONTROLLER  (controllers/hybrid_loop.py)                    │
│   uses ANTHROPIC_API_KEY to call Claude (planning) and                 │
│        the existing Ollama deepseek-r1:7b for reflection / cheap loops │
│   It calls MCP tools either via stdio (subprocess) or in-process       │
└────────────────────────────────────────────────────────────────────────┘
```

## Why this shape

| Concern | Decision |
|---|---|
| One backend, three front-ends | A single FastMCP server supports stdio AND HTTP simultaneously — no duplication |
| Ids over arrays | Tools that produce arrays (phase masks, simulation results) return UUID handles; arrays live in a session store. Keeps Claude's context window clean. |
| Claude for planning, Ollama for loops | Planning needs reasoning quality (Claude). Reflection runs many times per design (Ollama is free). |
| MCP server has no API key | The server is a tool provider only. API keys live in the controller, never in the server. Reusable and swappable. |
| Async by default | All tool funcs are `async def` so the FastAPI front-end doesn't block during long RCWA runs. |

## Files in this bundle

| Path | Purpose |
|---|---|
| `mcp_server/server.py` | FastMCP server — wraps your existing modules as 10 tools |
| `mcp_server/store.py` | In-process session store — UUID → numpy array / Path / metadata |
| `controllers/hybrid_loop.py` | Hybrid Claude + Ollama controller; consumes the MCP server |
| `clients/claude_desktop_config.json` | Drop into Claude Desktop config to add the metalens server |
| `clients/fastapi_app.py` | Web UI — mounts the same FastMCP server over HTTP |
| `clients/cli.py` | Headless CLI — `python clients/cli.py design "..."`  |
| `.env.example` | Template for `ANTHROPIC_API_KEY` etc. (copy → `.env`) |
| `transcript_example.md` | A worked example showing the full design flow |

## Install (one time)

```bash
# 1. add deps to your existing venv
pip install "fastmcp>=3.2.0" "anthropic>=0.99.0" python-dotenv "fastapi>=0.111" uvicorn

# 2. set up the API key
cp .env.example .env
# then edit .env and paste your Anthropic key

# 3. smoke-test the server with the inspector
fastmcp dev mcp_server/server.py
# opens http://localhost:6274/ — click "Tools" to see all 10 tools listed
```

## Walkthrough A — Claude Desktop chat

1. Edit `claude_desktop_config.json` (location in §clients/claude_desktop_config.json header)
   to point at `mcp_server/server.py`.
2. Restart Claude Desktop.
3. Hammer icon at the bottom of the chat shows "10 tools".
4. Try: *"Design me a 532 nm TiO₂ metalens, NA 0.4, diameter 100 µm.
   Run a phase sweep first, then generate the mask, optimize, and export the GDS."*
   Claude will choose the tools in the correct order.

## Walkthrough B — Headless hybrid pipeline

```bash
python controllers/hybrid_loop.py \
    --requirement "Design a 532 nm TiO2 metalens for NA 0.4, diameter 100 µm, focal length 110 µm. Optimize for Strehl > 0.8." \
    --max-iters 5
```

The controller:

1. **Plans** with Claude (one call) — turns the natural-language requirement
   into a structured `DesignSpec` and a sequence of MCP tool calls.
2. **Executes** the plan — calls the MCP server to do RCWA sweep → mask → optimize.
3. **Reflects** with Ollama (cheap, runs many times) — proposes parameter
   adjustments based on simulation results.
4. **Iterates** until target Strehl is met or `max-iters` reached.
5. **Reports** to Claude (one call) — generates a human-readable design summary.

You spend roughly 3–5 Claude API calls per design (planning + summary +
occasional escalation), and dozens of Ollama calls.

## Walkthrough C — FastAPI web UI

```bash
uvicorn clients.fastapi_app:app --reload --port 8080
# then open http://localhost:8080  → form-driven metalens design page
```

The same MCP server is mounted at `/mcp/` on the same port, so the
controller (or any external Claude API call) can drive the UI's backend
directly via the connector pattern.

## Security notes

- **Never commit `.env`.** Add it to `.gitignore`.
- **MCP tool inputs are LLM-generated.** Validate ranges (the FastMCP
  schemas + Pydantic models in `server.py` already do this, but extend
  them when you add new tools).
- **`export_gds` writes to disk.** It's pinned to a project-relative
  output directory; do not let an LLM specify absolute paths.
- **Claude API costs scale with conversation length.** The hybrid loop
  caps Claude calls and offloads the inner loop to Ollama precisely to
  keep cost bounded.

## What to do next

1. **Make the server work first.** Run `fastmcp dev mcp_server/server.py`
   and click through every tool in the Inspector before plugging in any LLM.
2. **Wire Claude Desktop** — easiest to verify the LLM-to-tool path.
3. **Build the hybrid controller** — copy the recipe, then expand the
   reflection prompts to whatever your AutoML loop needs.
4. **Productionise gradually** — add a real job queue (Celery/RQ) when
   the in-process queue stops scaling, swap the in-memory store for SQLite
   when you need persistence, add OAuth when you go remote.