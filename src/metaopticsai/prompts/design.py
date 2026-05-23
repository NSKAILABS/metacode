"""Prompts for the design / generation stages of the AutoML workflow.

Centralizing prompts here makes them easy to A/B test and version separately
from workflow logic.
"""
from __future__ import annotations

from langchain_core.prompts import PromptTemplate

PARSE_REQUIREMENTS_PROMPT = PromptTemplate.from_template(
    """You are a metalens design engineer. Parse the user requirement below into
a structured JSON object.

User requirement:
{requirement}

Respond with ONLY a JSON object with these fields (all numeric values use SI
nanometers for wavelengths and micrometers for sizes):

{{
  "wavelength_nm": <int>,
  "focal_length_um": <float>,
  "diameter_um": <float>,
  "target_strehl": <float between 0 and 1>,
  "preferred_material": <one of "TiO2", "Si", "GaN", "SiN">,
  "operating_band": <"visible" | "nir" | "swir" | "mwir">,
  "polarization": <"unpolarized" | "linear" | "circular">,
  "notes": <free-text caveats, ≤ 200 chars>
}}

Do NOT include prose outside the JSON. If a field is unspecified, choose a
sensible default and note it in `notes`.
"""
)

GENERATE_PARAMS_PROMPT = PromptTemplate.from_template(
    """You are a metalens design engineer choosing meta-atom parameters.

Parsed user requirements: {parsed}

Retrieved background knowledge:
{context}

Choose meta-atom unit-cell parameters for an initial RCWA sweep. Respond with
ONLY this JSON object:

{{
  "pillar_material": <"TiO2" | "Si" | "GaN" | "SiN">,
  "pillar_height_nm": <float, 200–2000>,
  "period_nm": <float, 100–1500>,
  "min_diameter_nm": <float, 30–500>,
  "max_diameter_nm": <float, larger than min, ≤ period - 30 nm>,
  "n_samples": <int, 20–60>,
  "xy_harmonics": <int, 3–11>,
  "reasoning": <short, ≤ 300 chars>
}}

Pick values consistent with the retrieved knowledge and the operating
wavelength. No prose outside the JSON.
"""
)