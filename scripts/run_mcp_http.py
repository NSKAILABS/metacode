"""Run the all-in-one MCP server over HTTP/SSE (for browser / curl testing).

    python scripts/run_mcp_http.py --port 8765
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaopticsai.mcp_servers.all_in_one.server import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--transport", "http", *sys.argv[1:]]
    main()