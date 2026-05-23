"""Centralized prompt templates."""
from metaopticsai.prompts.design import (
    PARSE_REQUIREMENTS_PROMPT,
    GENERATE_PARAMS_PROMPT,
)
from metaopticsai.prompts.reflection import (
    GRADE_DOCUMENTS_PROMPT,
    REFLECT_PROMPT,
)
from metaopticsai.prompts.summarize import SUMMARIZE_DESIGN_PROMPT

__all__ = [
    "PARSE_REQUIREMENTS_PROMPT", "GENERATE_PARAMS_PROMPT",
    "GRADE_DOCUMENTS_PROMPT", "REFLECT_PROMPT",
    "SUMMARIZE_DESIGN_PROMPT",
]