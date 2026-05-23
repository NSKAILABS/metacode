"""Prompts for reflection / self-critique steps in the Self-RAG workflow."""
from __future__ import annotations

from langchain_core.prompts import PromptTemplate

GRADE_DOCUMENTS_PROMPT = PromptTemplate.from_template(
    """Grade the relevance of this retrieved document to the metalens-design
query.

Query: {query}

Document: {document}

Respond with ONLY: {{"score": <0.0-1.0>, "reason": "<short>"}}.
"""
)

REFLECT_PROMPT = PromptTemplate.from_template(
    """You are a senior optical engineer reviewing an in-progress metalens
design.

Current design state (JSON):
{state}

Achieved Strehl ratio: {strehl}
Target Strehl ratio: {target}
Phase coverage / 2π: {coverage}
Mean transmission: {transmission}

Decide the next action. Respond with ONLY this JSON:

{{
  "decision": <"accept" | "re_optimize" | "re_sweep" | "re_retrieve">,
  "rationale": <short, ≤ 300 chars>,
  "suggested_changes": {{
    "pillar_height_nm": <optional float>,
    "period_nm": <optional float>,
    "min_diameter_nm": <optional float>,
    "max_diameter_nm": <optional float>,
    "n_samples": <optional int>
  }}
}}

Use:
  - "accept"        if the design meets the target.
  - "re_optimize"   if Strehl is close but suboptimizer iterations remain.
  - "re_sweep"      if phase coverage < 0.9 or transmission < 0.5.
  - "re_retrieve"   if the knowledge base context seems insufficient.
"""
)