"""Prompts for the final-summary stage of the AutoML workflow."""
from __future__ import annotations

from langchain_core.prompts import PromptTemplate

SUMMARIZE_DESIGN_PROMPT = PromptTemplate.from_template(
    """Write a concise (≤ 250 words) summary of this completed metalens design.
It will be shown to a non-specialist project stakeholder.

User requirement:
{requirement}

Final design parameters (JSON):
{design}

Achieved metrics:
- Strehl ratio: {strehl}
- Mean transmission: {transmission}
- Phase coverage / 2π: {coverage}
- GDS file: {gds_path}

Cover, in order:
1. What was asked for (1 sentence).
2. What was delivered (key numbers).
3. Trade-offs and remaining limitations.
4. Suggested next steps (1–2 bullets).

Write plain English, no JSON, no markdown.
"""
)