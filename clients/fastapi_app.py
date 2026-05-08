"""
clients.fastapi_app — FastAPI front-end that mounts the MetaOpticsAI MCP
server at /mcp/ AND serves a small HTML form for engineers to drive it.

Run:
    uvicorn clients.fastapi_app:app --reload --port 8080

Then:
    http://localhost:8080         → form-driven UI
    http://localhost:8080/mcp/    → MCP HTTP endpoint (Anthropic API can call this)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from mcp_server.server import mcp                          # noqa: E402
from controllers.hybrid_loop import run_design             # noqa: E402

# FastMCP exposes an ASGI sub-app via http_app(); mount it under /mcp/ so the
# whole thing serves on one origin/port.
mcp_asgi = mcp.http_app(path="/mcp/")

app = FastAPI(title="MetaOpticsAI", lifespan=mcp_asgi.lifespan)
app.mount("/mcp", mcp_asgi)


_HTML = """
<!doctype html>
<title>MetaOpticsAI — design a metalens</title>
<style>
  body { font-family: system-ui; max-width: 640px; margin: 4em auto; line-height: 1.5; }
  textarea { width: 100%; height: 5em; font-family: inherit; font-size: 1em; }
  button { padding: 0.5em 1.5em; font-size: 1em; }
  pre { background: #f4f4f4; padding: 1em; white-space: pre-wrap; }
</style>
<h1>🔬 MetaOpticsAI</h1>
<p>Describe the metalens you want. The hybrid Claude + Ollama controller will
plan, simulate, optimize and export GDS automatically.</p>
<form method="post" action="/design">
  <textarea name="requirement" required>Design a 532 nm TiO2 metalens for NA 0.4, diameter 100 µm, focal length 110 µm. Optimize for Strehl > 0.8.</textarea>
  <p><label>Max iterations: <input type="number" name="max_iters" value="3" min="1" max="10"></label></p>
  <p><button type="submit">Design</button></p>
</form>
<p style="color:#666">MCP endpoint also live at <code>/mcp/</code> for any external Claude API call.</p>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


@app.post("/design")
async def design(requirement: str = Form(...), max_iters: int = Form(3)) -> JSONResponse:
    """Run the full hybrid loop and return the design summary."""
    summary = await run_design(requirement, max_iters)
    return JSONResponse({"requirement": requirement, "summary": summary})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mcp_mounted_at": "/mcp/"}
