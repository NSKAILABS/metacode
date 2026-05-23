"""Run the all-in-one MCP server over stdio (for Claude Desktop, etc.).

    python scripts/run_mcp_stdio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaopticsai.mcp_servers.all_in_one.server import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--transport", "stdio", *sys.argv[1:]]
    main()