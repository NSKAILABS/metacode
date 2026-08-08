"""LLM orchestration layer for the meta-optics design agent.

The Groq client + tool-calling agent live here. Nothing in this package
knows about physics; it only routes LLM tool calls to `mcp_server.tools.dispatch`.
"""
