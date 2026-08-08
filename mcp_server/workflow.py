"""The verified design workflow.

Returned by the `get_workflow_guide` tool. This is a static document; the
LLM should refer back to it whenever it is unsure what step comes next.
The steps map 1:1 to tools defined in `mcp_server/tools.py`.
"""

WORKFLOW_GUIDE = """\
# Meta-optics design workflow (verified)

Follow these five stages in order. Each stage has one dedicated tool.

## 1. Pick a starting template
Call `list_templates` to see verified recipes, then `get_template(name)` to
retrieve full parameters. Templates are pre-validated: their wavelength,
material, pillar height, and periodicity have been tested to give full
2*pi phase coverage on a diameter sweep.

## 2. Adapt parameters
Modify the template dict to match the user's request:
  - wavelength_nm     : design wavelength in nm  (200 < lambda < 20000)
  - focal_length_mm   : desired focal length in mm
  - diameter_mm       : aperture diameter in mm
  - pillar_material   : one of TiO2, Si, GaN, SiN
  - pillar_height_nm  : 50 < h < 3000
  - min/max diameter_nm : bracket of pillar diameters to sweep
  - periodicity_nm    : unit-cell pitch (must be > max_diameter + 20 nm)

## 3. Validate
Always call `validate_design` before running physics. It re-checks the
Pydantic constraints (min<max, pillar fits cell, etc.) and returns
{valid, errors, warnings}.  Fix all errors first.

## 4. Simulate
Call `run_simulation` with the validated params. It executes a diameter
sweep on the metabox3 RCWA backend (or the analytical fallback) and
returns {backend, phase_coverage, mean_transmission, sweep_values,
phases_rad, amplitudes}.

Good outputs: phase_coverage >= 0.9 AND mean_transmission >= 0.7.

## 5. Interpret and iterate
If the metrics are below target, call `get_optimization_tips` with the
sim result. It returns human-readable suggestions (e.g. "extend max
diameter", "reduce aspect ratio"). Apply, re-validate, re-simulate.

Cap yourself at 3-5 tool round-trips per user request. Report the final
result plus the design parameters you settled on.
"""


def get_workflow_guide() -> str:
    return WORKFLOW_GUIDE
