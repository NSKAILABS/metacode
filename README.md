# MetaOpticsAI

**A Python platform for AI-assisted metalens and metasurface design — RCWA simulation, phase-mask generation, gradient-based inverse design, GDSII export, and an LLM-driven automation layer that talks to it all over MCP.**

MetaOpticsAI sits at the intersection of computational photonics and AI-assisted engineering. It bundles a custom RCWA engine, a TensorFlow-backed inverse-design pipeline (via the `metabox3` library), a Self-RAG knowledge base over the photonics literature, and a Model Context Protocol (MCP) server that exposes every capability as a tool — consumable from Claude Desktop, a FastAPI web UI, a CLI, or a hybrid Claude + Ollama controller.

---

## Table of contents

1. [What's in this project](#whats-in-this-project)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [Folder structure](#folder-structure)
4. [Core modules](#core-modules)
5. [The MCP integration layer](#the-mcp-integration-layer)
6. [Front-ends](#front-ends)
7. [Installation](#installation)
8. [Quick start — three workflows](#quick-start--three-workflows)
9. [Validation notebooks](#validation-notebooks)
10. [S4 cross-validation (optional)](#s4-cross-validation-optional)
11. [Configuration & environment variables](#configuration--environment-variables)
12. [Design notes worth understanding](#design-notes-worth-understanding)
13. [Future directions](#future-directions)
14. [References](#references)

---

## What's in this project

| Layer | What it does | Key files |
|---|---|---|
| **Core engines** | RCWA simulation, FDTD-data interpolation, phase-mask generation, GDS export, design state | `core/rcwa_engine.py`, `core/meta_data.py`, `core/phase_engine.py`, `core/gds_engine.py`, `core/design.py` |
| **AutoML** | LangGraph + Self-RAG pipeline driving the engines via Ollama | `core/automl.py` |
| **Analysis** | PSF, MTF, Strehl ratio, Zernike decomposition, far-field propagation | `analysis/psf.py`, `analysis/mtf.py`, `analysis/strehl.py`, `analysis/zernike.py`, `analysis/farfield.py` |
| **`metabox3`** | TensorFlow RCWA + differentiable assembly for inverse design | `metabox3/` (rcwa_tf, propagation, expansion, assembly, …) |
| **MCP server** | Wraps every capability as 12 typed tools | `mcp_server/server.py`, `mcp_server/store.py` |
| **Controllers** | Hybrid Claude + Ollama loop that consumes the MCP server | `controllers/hybrid_loop.py` |
| **Front-ends** | Claude Desktop config · FastAPI web UI · headless CLI | `clients/claude_desktop_config.json`, `clients/fastapi_app.py`, `clients/cli.py` |
| **Notebooks** | Step-by-step validation of every module + the metabox stack | `notebooks/01_Implementation.ipynb`, `notebooks/02_metabox.ipynb` |

---

## Architecture at a glance

```
┌────────────────────────────────────────────────────────────────────────┐
│ FRONT-ENDS                                                             │
│   • Claude Desktop  →  stdio                                           │
│   • FastAPI web UI  →  HTTP   (mounts the same FastMCP server)         │
│   • CLI / scripts   →  in-process import                               │
│   • Hybrid controller (controllers/hybrid_loop.py)                     │
└─────────────────────────┬──────────────────────────────────────────────┘
                          │  Model Context Protocol  (tools / JSON-RPC)
                          ▼
┌────────────────────────────────────────────────────────────────────────┐
│ MCP SERVER  (mcp_server/server.py — FastMCP 3.x)                       │
│   12 tools:                                                            │
│     • get_fdtd_library          • analyze_psf                          │
│     • run_rcwa_sweep            • analyze_mtf                          │
│     • generate_phase_mask       • zernike_decompose                    │
│     • quantize_phase            • export_gds                           │
│     • optimize_metalens         • search_knowledge                     │
│     • submit_job                • get_job                              │
└─────────────────────────┬──────────────────────────────────────────────┘
                          │  Python imports — no extra serialisation
                          ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ENGINES                                                                │
│   core/  ·  analysis/  ·  metabox3/                                    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ HYBRID LLM CONTROLLER  (controllers/hybrid_loop.py)                    │
│   Claude (Anthropic API)  →  planning + final summary                  │
│   Ollama deepseek-r1:7b   →  cheap reflection / inner loop             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Folder structure

```
metacode/                          ← project root
│
├── README.md                      ← this file
├── pyproject.toml                 ← package metadata; enables `pip install -e .`
├── requirements.txt               ← pinned runtime deps
├── .env.example                   ← template — copy to .env and fill in keys
├── .env                           ← secrets (gitignored)
├── .gitignore
│
├── core/                          ← original six MetaOpticsAI modules
│   ├── __init__.py
│   ├── meta_data.py               ← FDTD data store + interpolation
│   ├── design.py                  ← DesignState container
│   ├── phase_engine.py            ← FZL / axicon / SPP / image phase masks
│   ├── rcwa_engine.py             ← Rigorous Coupled-Wave Analysis engine
│   ├── gds_engine.py              ← GDSII layout file generation
│   └── automl.py                  ← LangGraph + Self-RAG pipeline (Ollama)
│
├── analysis/                      ← optical metrics
│   ├── __init__.py
│   ├── psf.py                     ← PSF + FWHM
│   ├── mtf.py                     ← MTF curves
│   ├── strehl.py                  ← Strehl ratio
│   ├── zernike.py                 ← Zernike decomposition
│   └── farfield.py                ← far-field propagation
│
├── metabox3/                      ← third-party Metabox library (TF-backed)
│   ├── __init__.py
│   ├── rcwa_tf.py
│   ├── propagation.py
│   ├── expansion.py
│   ├── assembly.py
│   ├── modeling.py
│   ├── metrics.py
│   ├── raster.py
│   ├── utils.py
│   └── export.py
│
├── mcp_server/                    ← NEW — MCP integration layer
│   ├── __init__.py
│   ├── server.py                  ← FastMCP 3.x server, 12 tools
│   └── store.py                   ← UUID-keyed artefact store
│
├── controllers/                   ← NEW — orchestration layer
│   ├── __init__.py
│   └── hybrid_loop.py             ← Claude + Ollama hybrid controller
│
├── clients/                       ← NEW — three front-ends
│   ├── claude_desktop_config.json ← drop-in Claude Desktop config
│   ├── fastapi_app.py             ← web UI + HTTP-mounted MCP server
│   └── cli.py                     ← headless CLI (design / sweep / inspect)
│
├── notebooks/                     ← validation & exploration
│   ├── 01_Implementation.ipynb    ← per-module validation, end-to-end test
│   └── 02_metabox.ipynb           ← metabox3 walkthrough
│
├── papers/                        ← Self-RAG knowledge corpus
│   ├── moharam_gaylord_1981.pdf
│   ├── khorasaninejad_2016.pdf
│   ├── asai_self_rag_2024.pdf
│   └── …                          (~13 PDFs total)
│
├── outputs/                       ← generated GDS, plots, run logs (gitignored)
│
└── transcript_example.md          ← worked example of the full LLM-driven flow
```

> **Note on naming.** The third-party RCWA library is named `metabox3` in this repo (its `__init__.py` lives at `metabox3/__init__.py`). Import accordingly: `from metabox3 import rcwa_tf, assembly, propagation`. The MCP server uses this name internally.

---

## Core modules

### `core/meta_data.py` — FDTD data store
Loads pre-computed FDTD lookup tables (phase, amplitude, transmission as a function of pillar diameter, height, and wavelength) and provides smooth bicubic interpolation. The lookup is what lets `phase_engine` map a desired phase value back to a fabricable pillar diameter without running RCWA at every spatial point.

### `core/design.py` — design state container
A single `DesignState` dataclass that flows through the pipeline carrying wavelength, NA, focal length, lattice constant, material, pillar height, and the current phase mask handle. Every tool in the MCP server reads or writes parts of this state.

### `core/phase_engine.py` — phase masks
Generates the four canonical phase profiles: hyperbolic (focusing) Fresnel zone lens, axicon (Bessel beam), spiral phase plate (vortex), and arbitrary image-encoded mask. Output is a 2D `numpy` array that can be quantised to N levels for fabrication.

### `core/rcwa_engine.py` — RCWA simulation
Custom Rigorous Coupled-Wave Analysis engine that solves Maxwell's equations for periodic structures. Used to build the FDTD lookup table that `meta_data` consumes. Validated against the S4 reference (see [S4 cross-validation](#s4-cross-validation-optional)) and against the differentiable `metabox3.rcwa_tf` for gradient checks.

### `core/gds_engine.py` — GDSII layout
Converts a (quantised) phase mask plus a diameter-vs-phase library into a GDSII file: a grid of dielectric pillars with the diameters needed at each spatial point. Uses `gdspy` under the hood; output opens cleanly in KLayout.

### `core/automl.py` — LangGraph + Self-RAG pipeline
The end-to-end automation layer: takes a natural-language design specification, plans a sequence of engine calls, executes them, retrieves relevant context from the paper corpus via Self-RAG, reflects on simulation results, and iterates until the spec is met. Uses `LangGraph` for orchestration and Ollama (`deepseek-r1:7b` + `nomic-embed-text`) for the LLM and embedding back-ends.

> The MCP layer (below) does **not** replace `automl.py` — it sits next to it. `automl.py` is the original Ollama-only pipeline; the new `controllers/hybrid_loop.py` is its hybrid Claude + Ollama cousin that drives the same engines through MCP.

---

## The MCP integration layer

### Why MCP?

The Model Context Protocol is the JSON-RPC wire format that Claude Desktop, Cursor, Goose, and the Anthropic SDK all speak natively. Implementing the engine as one MCP server means **every** future LLM client gets it for free — no per-client adapter code.

### `mcp_server/server.py` — the 12 tools

Each tool is a `@mcp.tool`-decorated `async def` with a Pydantic-typed signature that the LLM sees as a JSON schema.

| Tool | Inputs | Output |
|---|---|---|
| `get_fdtd_library` | wavelength, material | available pillar shapes / heights |
| `run_rcwa_sweep` | material, height, diameter range | **handle** to phase-vs-diameter table + `phase_coverage_2pi_fraction` |
| `generate_phase_mask` | type, λ, focal length, NA, diameter | **handle** to 2D phase array + summary stats |
| `quantize_phase` | mask handle, n_levels | **handle** to quantised mask |
| `optimize_metalens` | sweep handle, mask handle, n_iters | **handle** to optimised design + Strehl history |
| `analyze_psf` | mask handle, λ, propagation distance | Strehl, FWHM, **handle** to PSF |
| `analyze_mtf` | mask handle, λ | **handle** to MTF curves |
| `zernike_decompose` | mask handle, n_terms | first N Zernike coefficients |
| `export_gds` | design handle, output path | path to GDS file |
| `search_knowledge` | query string, top_k | passages from the paper corpus |
| `submit_job` | tool name, kwargs | job id |
| `get_job` | job id | status + result handle when done |

### The handle pattern (most important architectural choice)

Tools that produce arrays don't return the arrays — they return a 12-character UUID into a session store (`mcp_server/store.py`). Downstream tools take the handle as input. Why it matters: a 5-tool design loop without handles dumps ~10⁵ floats into the LLM's context per iteration; with handles, context stays flat regardless of array size. Without this pattern, token cost of a non-trivial design grows by ~4×.

### Run modes

```bash
# Local dev with the MCP Inspector
fastmcp dev mcp_server/server.py

# stdio (Claude Desktop spawns this automatically)
python -m mcp_server.server

# HTTP (for FastAPI mount or remote clients)
python -m mcp_server.server --http --port 8765
```

---

## Front-ends

### A. Claude Desktop (chat → metalens)

Edit your real `claude_desktop_config.json` (locations below) and paste the `mcpServers` block from `clients/claude_desktop_config.json`. Restart Claude Desktop. The hammer icon in the chat will show 12 tools.

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

### B. FastAPI web UI

```bash
uvicorn clients.fastapi_app:app --reload --port 8080
```

Serves a small HTML form at `/` and exposes the same MCP server at `/mcp/` for remote clients.

### C. Headless CLI

```bash
python clients/cli.py design "Design a 532 nm TiO2 metalens for NA 0.4, ø 100 µm, focal 110 µm. Optimize for Strehl > 0.8 and export GDS."
python clients/cli.py sweep --material TiO2 --height 600 --d-min 50 --d-max 250
python clients/cli.py inspect <handle>
```

### D. Hybrid LLM controller

```bash
python controllers/hybrid_loop.py \
    --requirement "532 nm TiO2 metalens, NA 0.4, ø 100 µm, focal 110 µm, Strehl > 0.8" \
    --max-iters 5
```

The loop:
1. **Plans** with Claude — turns natural language into a structured `DesignSpec` and tool sequence (~1 Claude call).
2. **Executes** the plan via MCP — RCWA sweep → mask → optimise → analyse.
3. **Reflects** with Ollama — proposes parameter adjustments based on simulation results (cheap, runs many times).
4. **Iterates** until target is met or `max-iters`.
5. **Summarises** with Claude — human-readable design report (~1 Claude call).

Cost: ~2 Claude calls + ~30 Ollama calls per design.

---

## Installation

### Prerequisites

- Python 3.11.9 (the project pins this — TensorFlow + `metabox3` are sensitive to version)
- Ollama running locally with `deepseek-r1:7b` and `nomic-embed-text` pulled
- An Anthropic API key (only required for the hybrid controller / Claude Desktop)

### Set up the venv

```bash
# Windows PowerShell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -e .                                       # installs metabox3 + core/ in editable mode
pip install "fastmcp>=3.2.0" "anthropic>=0.99.0" \
            "fastapi>=0.111" uvicorn python-dotenv
```

The `pip install -e .` step is what fixes `ModuleNotFoundError: No module named 'metabox3'` — without it, the package is on disk but not importable from arbitrary working directories.

### Pull the LLM models

```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

### Set up secrets

```bash
cp .env.example .env
# edit .env and paste ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick start — three workflows

### 1. Smoke-test the MCP server alone (no LLM)

```bash
fastmcp dev mcp_server/server.py
```

Opens the MCP Inspector at `http://localhost:6274`. Click each tool → fire a test call → verify nothing throws. Catches ~90 % of integration bugs before any LLM is involved.

### 2. Run the validation notebook

```bash
jupyter lab notebooks/01_Implementation.ipynb
```

Runs every module end-to-end with assertions and visualisations. Section §0 sets `RUN_HEAVY = True` to enable the slow training/optimisation cells; flip to `False` for a fast cold run.

### 3. Drive a full design from chat

After wiring up Claude Desktop (above), paste:

> *"Design a 532 nm TiO₂ metalens, NA 0.4, ø 100 µm, focal 110 µm. Run a phase sweep first, then generate the mask, optimize, analyse the PSF, and export the GDS."*

Claude will pick the tools in the right order. See `transcript_example.md` for a fully worked example of what to expect.

---

## Validation notebooks

### `01_Implementation.ipynb` (9 sections)

1. Environment setup and imports
2. `meta_data.py` — load FDTD table, interpolate, sanity-check phase coverage
3. `phase_engine.py` — generate FZL / axicon / SPP / image masks, plot
4. `rcwa_engine.py` — single-cell forward solve, energy conservation check
5. `gds_engine.py` — quantise mask, write GDS, verify in `gdspy`
6. `design.py` — round-trip a `DesignState`
7. End-to-end pipeline — natural-language spec → GDS file
8. AutoML test — without Ollama (rule-based fallback) and with Ollama
9. Summary table of pass/fail per module

### `02_metabox.ipynb`

Walks the `metabox3` library: `utils` primitives → `raster` discretisation → `rcwa_tf` differentiable solve → `propagation` near-to-far → `assembly` end-to-end metalens optimisation. Each section is gated by a `RUN_HEAVY` flag so the cold run is fast.

---

## S4 cross-validation (optional)

For an independent reference against a battle-tested RCWA implementation, set up the Stanford S4 solver — typically on an AWS Free Tier `t3.micro` running Ubuntu 24.04 with Python 3.11 from the `deadsnakes` PPA, OpenBLAS, FFTW3, and SuiteSparse. Build with `make boost && make lib -j1 && python setup.py install` (with 4 GB swap to survive C++ template compilation on 1 GB RAM). The reference test case lives in `~/s4_validation.py` on the validation box: a 1D binary Si grating, period 600 nm, swept from 500 – 800 nm, where R + T should equal 1.000 ± 1e-3 at every wavelength. Diff the output against `core/rcwa_engine.py` at matching `NumBasis`; differences > 1 % at low orders point to a convention mismatch (sign of the time convention, definition of incidence angle, or Fourier basis normalisation).

`t3.micro` is sufficient for module validation and 1D / small-2D test cases (orders ≤ 15×15). Full metalens unit cells with 30×30+ orders need a `c7i.large` spot instance (~$0.04/hr).

---

## Configuration & environment variables

`.env.example`:

```bash
# Anthropic — required for hybrid_loop.py and Claude Desktop / API flows
ANTHROPIC_API_KEY=sk-ant-...

# Ollama — defaults match a stock local install
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:7b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Output directory — all GDS / PSF plots / run logs land here
METAOPTICS_OUTPUT_DIR=./outputs

# TensorFlow noise gate
TF_CPP_MIN_LOG_LEVEL=3
```

Loaded in `mcp_server/server.py` and `controllers/hybrid_loop.py` via `python-dotenv`.

---

## Design notes worth understanding

**The MCP server has no API key.** It is a tool provider only. Keys live in the controller (`hybrid_loop.py`) or the host (Claude Desktop), never in the server. This keeps the server reusable across providers — swapping Claude for GPT-4 doesn't touch a single line in `server.py`.

**Claude for planning, Ollama for loops.** Planning needs reasoning quality (Claude); reflection runs many times per design (Ollama is free and local). The split is what keeps the design cost at ~2 Claude calls regardless of how many iterations the inner loop takes.

**`optimize_metalens` is currently an analytical heuristic.** It uses the same `mean(amp)² × min(coverage, 1)` estimator as `automl.py`'s `_simulate_analytical`. The signature already takes a `sweep_handle` and returns a `design_handle`, so swapping the body for `metabox3.assembly.optimize_single_lens_assembly` is a ~20-line change that doesn't affect any downstream tool.

**Self-RAG falls back to keyword search.** `search_knowledge` calls `core.automl.MetalensKnowledgeBase`, which prefers `nomic-embed-text` for semantic retrieval but falls back to BM25-style keyword matching if Ollama is unreachable. The tool always returns *something* — the quality is just lower without embeddings.

**The session store is deliberately simple.** `mcp_server/store.py` is a `dict` + a `Lock`. Swap to SQLite when you need persistence across server restarts; swap to Redis when you need multi-process. For single-user dev work the dict is fine and zero-config.

---

## Future directions

| # | Direction | Notes |
|---|---|---|
| 1 | Production-grade RCWA backend | Replace the custom engine with S4 or `metabox3.rcwa_tf` for production runs |
| 2 | Differentiable inverse design | Wire `metabox3.assembly` into `optimize_metalens` for true gradient-based optimisation |
| 3 | Neural surrogate models | Train a small MLP on the RCWA sweep to skip the expensive solve in the inner loop |
| 4 | Multi-wavelength achromatic design | Extend `phase_engine` to optimise across an RGB triplet |
| 5 | Expanded meta-atom library | Beyond cylindrical pillars: rectangles, ellipses, V-shapes for polarisation control |
| 6 | Fabrication-aware closed-loop design | Add SEM-image feedback into the optimisation loop |
| 7 | GUI / web dashboard | Promote `clients/fastapi_app.py` from prototype to full design cockpit |
| 8 | LLM upgrade paths | Try Claude Opus 4 for planning, larger Ollama models (qwen3-coder, llama-4) for reflection |

---

## References

The Self-RAG knowledge base in `papers/` covers ~13 papers across five phases. The most-cited foundations:

- **RCWA foundations** — Moharam & Gaylord (1981); Yoon & Rho, *MAXIM* (2021); Li (1996)
- **Inverse design** — Hughes et al. (2018); Jensen & Sigmund (2011); Malkiel et al. (2018)
- **LLM-driven design** — Asai et al., *Self-RAG* (2024)

Each paper is mapped to specific functions and modules in the reading guide that ships alongside this repo.

---

## Licence

Internal research code. Treat the `metabox3/` directory under its upstream licence (Metabox / Apache 2.0); everything else under whatever licence you choose for the project. Add a `LICENSE` file at the project root before publishing.