# MetaOpticsAI — End-to-end Notebooks

Six progressive notebooks that walk through the platform end-to-end, from a
60-second "hello metalens" to the full Self-RAG AutoML loop driven by an LLM.

Open with `jupyter lab notebooks/` from the repo root. The notebooks use
`sys.path` injection so they work without `pip install -e .`.

| # | Notebook | What it teaches | Needs |
|---|---|---|---|
| 01 | `01_quickstart.ipynb`              | 60-second metalens: 5 stages in 8 code cells | analytical backend only |
| 02 | `02_metalens_design_pipeline.ipynb` | Full pipeline with viz at every stage          | analytical backend only |
| 03 | `03_advanced_analysis.ipynb`       | Aberrations, Zernike, through-focus scan        | analytical backend only |
| 04 | `04_tool_layer_for_agents.ipynb`   | The agent-facing `BaseTool` / registry API      | analytical backend only |
| 05 | `05_self_rag_automl.ipynb`         | Natural-language requirement → metalens         | LLM provider (Ollama/Groq/Anthropic) |
| 06 | `06_mcp_server_walkthrough.ipynb`  | Exposing the same tools over MCP                | nothing extra, but server runs separately |

## Recommended order

If you're **new to the codebase**: 01 → 02 → 04 → 05.

If you're **evaluating the platform** for a cofounder/customer demo: 01 → 05.

If you're **doing optics**: 01 → 02 → 03.

If you're **building agents on top**: 04 → 06 → 05.

## Backend requirements

Notebooks 01-04 default to the **analytical backend** — fast, TF-free, runs
without `metabox3` vendored. The numbers are heuristic, not Maxwell-rigorous;
treat them as design sanity checks.

To run real RCWA inside the notebooks, vendor metabox3 (`cp -r /your/metabox3
src/metaopticsai/vendor/metabox3/`), then add this near the top of any
notebook before `build_tool_registry`:

```python
from metaopticsai.physics.backends.metabox import MetaboxBackend
backends.register(MetaboxBackend())   # added alongside AnalyticalBackend
```

The `run_rcwa_sweep` tool defaults to backend="auto" which picks metabox if
available.

## Notebook 05 — LLM provider setup

Pick one in `.env`:

```dotenv
# Local Ollama (recommended for development)
LLM_DEFAULT_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b               # tool-calling capable
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434

# OR Groq (fast cloud Llama)
LLM_DEFAULT_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile

# OR Anthropic Claude
LLM_DEFAULT_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

## Notebook 06 — running the MCP server

Start in a separate terminal:

```bash
python scripts/run_mcp_stdio.py            # for Claude Desktop
python scripts/run_mcp_http.py --port 8765 # for browser / remote agents
```

Notebook 06 then walks through the tool catalog, resource URIs, and a sample
JSON-RPC round-trip without spawning the server inside the kernel.

## Rebuilding the notebooks

The notebooks are generated from `notebooks/_build.py`. Edit the source there,
then:

```bash
python notebooks/_build.py
```

Each notebook is regenerated from the cell tuples in the build script.
