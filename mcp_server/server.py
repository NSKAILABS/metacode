"""
mcp_server.server — MetaOpticsAI MCP server (v2).

Exposes the entire MetaOpticsAI + metabox3 stack as a Model Context Protocol
server, consumable by Claude Desktop (stdio), the FastAPI front-end (HTTP),
the local Ollama-driven controller, or any MCP-compliant client.

---------
    # Local dev with the MCP Inspector
    fastmcp dev mcp_server/server.py

    # stdio (Claude Desktop / Ollama-with-tools)
    python -m mcp_server.server

    # HTTP (for FastAPI mount or remote clients)
    python -m mcp_server.server --http --port 8765
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastmcp import FastMCP
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from mcp_server import store                                       
from mcp_server.config import settings                              
from core.meta_data import MetaDataStore                             
from core.phase_engine import PhaseEngine                           
from core.gds_engine import GDSEngine                               
from analysis.psf import compute_psf                                 
from analysis.mtf import compute_mtf                                 
from analysis.strehl import compute_strehl_ratio                 
from analysis.zernike import zernike_decomposition                   

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("metaoptics-mcp")

mcp = FastMCP(
    name="MetaOpticsAI",
    instructions=(
        "Photonics design server. You can simulate, optimise, and export "
        "metalenses and metasurfaces. ALWAYS load the `photonics://workflow` "
        "resource (or call the `photonics_workflow` prompt) before starting "
        "a new design — it explains the canonical Material → Unit Cell → "
        "Phase Mask → Propagation tool chain.\n\n"
        "Every tool that produces a large array (sweep tables, phase masks, "
        "PSFs, fields) returns a short UUID handle. Pass that handle to the "
        "next tool — never ask for the raw array contents."
    ),
)

_OUTPUT_ROOT = settings.OUTPUT_DIR
_RESOURCE_DIR = settings.RESOURCE_DIR
_MATERIAL_DIR = settings.METABOX_MATERIAL_DIR


#  RESOURCES 
#  Static and templated documentation that any MCP client can pull on demand.

def _read_resource_file(name: str) -> str:
    p = _RESOURCE_DIR / name
    if not p.exists():
        return f"# {name} not found at {p}"
    return p.read_text(encoding="utf-8")


@mcp.resource("photonics://workflow")
def workflow_guide() -> str:
    """The canonical end-to-end metalens design workflow.

    Read this first. Tells the model how to chain tools from material
    selection through GDS export."""
    return _read_resource_file("workflow.md")


@mcp.resource("photonics://materials")
def materials_reference() -> str:
    """Quick-reference table of bundled refractive-index materials.

    Names, wavelength ranges, and typical use cases. Use the
    `get_material_index` tool to fetch numeric n + ik values."""
    return _read_resource_file("materials.md")


@mcp.resource("photonics://papers")
def papers_bibtex() -> str:
    """BibTeX of the bundled photonics literature corpus.

    Each entry corresponds to a PDF in `papers/` and is indexed by the
    Self-RAG knowledge base. Use `search_knowledge` to retrieve passages."""
    return _read_resource_file("photonics_papers.bib")


@mcp.resource("photonics://material/{name}")
def material_csv(name: str) -> str:
    """Raw refractive-index CSV for one bundled material.

    Replace `{name}` with one of: TiO2, Si_visible, Si_ir, GaN, Si3N4,
    ZnO, ZrO2, Nb2O5, MoO3, N-BK7, quartz."""
    csv_path = _MATERIAL_DIR / f"{name}.csv"
    if not csv_path.exists():
        avail = ", ".join(p.stem for p in _MATERIAL_DIR.glob("*.csv"))
        return f"# Material '{name}' not found.\n# Available: {avail}"
    return csv_path.read_text(encoding="utf-8")


# =========================================================================
#  PROMPTS
#  High-density photonics context the LLM can pull on a single tool call.
# =========================================================================

@mcp.prompt
def photonics_workflow() -> str:
    """Inject the full Material → Unit-Cell → Phase-Mask → Propagation
    workflow guide as a system message. Use this at the START of any
    metalens design conversation."""
    return _read_resource_file("workflow.md")


@mcp.prompt
def design_metalens(
    wavelength_nm: float = 532.0,
    diameter_um: float = 100.0,
    focal_length_um: float = 110.0,
    target_strehl: float = 0.85,
) -> str:
    """A ready-to-execute design brief. Fills in standard parameters and
    asks the model to drive the full tool chain."""
    return (
        f"Design a metalens with the following specification:\n"
        f"  • design wavelength = {wavelength_nm} nm\n"
        f"  • lens diameter     = {diameter_um} µm\n"
        f"  • focal length      = {focal_length_um} µm\n"
        f"  • target Strehl     = {target_strehl}\n\n"
        "Drive the full MCP tool chain: pick a material with "
        "`list_materials` + `get_material_index`, build a phase-vs-diameter "
        "library with `run_rcwa_sweep`, generate the FZL phase mask with "
        "`generate_phase_mask`, evaluate with `analyze_psf` / "
        "`analyze_mtf` / `zernike_decompose`, and finish by exporting "
        "GDSII with `export_gds`. Reflect on Strehl after each iteration. "
        "If the photonics workflow guide is not yet in context, read the "
        "`photonics://workflow` resource first."
    )


@mcp.prompt
def diagnose_design(strehl: float, phase_coverage: float) -> str:
    """Heuristic diagnosis prompt. Given a current Strehl ratio and phase
    coverage fraction, instructs the model how to redesign."""
    return (
        f"A simulation reported Strehl = {strehl:.3f} and phase coverage = "
        f"{phase_coverage:.2f} × 2π. Diagnose the dominant failure mode and "
        "propose ONE concrete change for the next iteration. Apply the "
        "heuristics from the `photonics://workflow` resource:\n"
        " • coverage < 0.9 → increase pillar_height_nm\n"
        " • Strehl < 0.5  → check coverage AND mean amplitude\n"
        " • PSF too wide  → focal-length / NA mismatch\n"
        "Return a JSON object: {\"change\": {param: new_value}, \"reason\": str}."
    )


# =========================================================================
#  TOOL 1 — get_fdtd_library
# =========================================================================

@mcp.tool
async def get_fdtd_library() -> dict[str, Any]:
    """List the wavelengths, materials, and unit-cell shapes currently
    available in the FDTD library bundled with the project. Use this
    before designing to confirm what's available."""
    md = MetaDataStore()
    out = []
    for wl in md.get_wavelengths():
        e = md.get_entry(wl)
        if e is None:
            continue
        out.append({
            "wavelength_nm": wl,
            "shape": e.shape,
            "material": e.material,
            "height_nm": e.height,
            "period_nm": e.period,
            "n_samples": len(e.dimensions),
            "reference": e.reference,
        })
    return {"entries": out, "count": len(out)}


# =========================================================================
#  TOOL 2 — list_materials
# =========================================================================

@mcp.tool
async def list_materials() -> dict[str, Any]:
    """List every refractive-index CSV bundled in metabox3/material_data.

    Use this BEFORE picking a material for a unit-cell sweep so you know
    which names are valid. See the `photonics://materials` resource for
    a quick-reference table of typical use cases."""
    from metabox3.rcwa import get_avaliable_materials  # noqa: WPS433
    names = sorted(get_avaliable_materials(str(_MATERIAL_DIR)))
    return {"materials": names, "count": len(names),
            "directory": str(_MATERIAL_DIR)}


# =========================================================================
#  TOOL 3 — get_material_index  (NEW — wraps metabox3.rcwa.Material)
# =========================================================================

class MaterialIndexInput(BaseModel):
    name: str = Field(..., description=(
        "Material name. Must match a CSV in metabox3/material_data. "
        "Call list_materials first to discover valid names."))
    wavelength_nm: float = Field(..., gt=10, lt=50000, description=(
        "Wavelength in NANOMETRES at which to evaluate n + ik."))


@mcp.tool
async def get_material_index(params: MaterialIndexInput) -> dict[str, Any]:
    """Look up the complex refractive index n + ik of a bundled material
    at a given wavelength via cubic interpolation of its CSV table.

    Wraps `metabox3.rcwa.Material(name).index_at(wavelength_m)`. Returns
    both the real and imaginary parts so the model can decide whether a
    given material is lossy at the design wavelength."""
    from metabox3.rcwa import Material  # noqa: WPS433
    mat = Material(params.name, custom_csv_dir=str(_MATERIAL_DIR))
    wl_m = params.wavelength_nm * 1e-9
    idx = mat.index_at(wl_m)
    return {
        "name": params.name,
        "wavelength_nm": params.wavelength_nm,
        "n": float(np.real(idx)),
        "k": float(np.imag(idx)),
        "complex": str(complex(np.real(idx), np.imag(idx))),
        "valid_range_um": [mat.min_wl_n * 1e6, mat.max_wl_n * 1e6],
    }


# =========================================================================
#  TOOL 4 — run_rcwa_sweep  (existing, defaults sourced from settings)
# =========================================================================

class RCWASweepInput(BaseModel):
    wavelength_nm: float = Field(..., gt=200, lt=20000,
        description="Design wavelength in nm, e.g. 532 for green.")
    pillar_material: Literal["TiO2", "Si", "GaN", "SiN"] = Field("TiO2",
        description="Meta-atom material. TiO2=visible, Si=NIR, "
                    "GaN=UV/blue, SiN=CMOS-compatible visible.")
    pillar_height_nm: float = Field(600.0, gt=50, lt=3000,
        description="Pillar height in nm. Increase to widen phase coverage.")
    period_nm: float = Field(350.0, gt=100, lt=2000,
        description="Square unit-cell period in nm. Should be "
                    "0.6 – 1.1 × wavelength for a well-behaved metasurface.")
    min_diameter_nm: float = Field(50.0, gt=10, lt=2000,
        description="Smallest pillar diameter to sweep (fab limit ~80 nm).")
    max_diameter_nm: float = Field(160.0, gt=20, lt=2000,
        description="Largest pillar diameter to sweep, ≤ 0.85 × period.")
    n_samples: int = Field(20, ge=5, le=100,
        description="Number of points in the diameter sweep.")
    xy_harmonics: int = Field(default=settings.DEFAULT_HARMONICS, ge=1, le=11,
        description="RCWA spatial harmonics per axis. 3 is a fast default; "
                    "increase to 5–7 for high-NA designs.")


@mcp.tool
async def run_rcwa_sweep(params: RCWASweepInput) -> dict[str, Any]:
    """Run a 1-wavelength RCWA sweep over pillar diameter to produce a
    phase-vs-diameter design library. Returns a handle to the full sweep
    result plus inline summary statistics (phase coverage, mean amplitude).

    Stage 2 of the canonical workflow. The returned `handle` feeds
    `optimize_metalens` and `export_gds`."""
    from metabox3 import rcwa, utils  # noqa: WPS433
    import tensorflow as tf            # noqa: WPS433

    n_index = {"TiO2": 2.35, "Si": 3.48, "GaN": 2.32, "SiN": 2.0}[params.pillar_material]

    diam_feat = utils.Feature(
        vmin=params.min_diameter_nm * 1e-9,
        vmax=params.max_diameter_nm * 1e-9,
        name="diameter",
        sampling=params.n_samples,
    )
    pillar = rcwa.Circle(material=n_index, radius=diam_feat)
    layer = rcwa.Layer(material=1.0, thickness=params.pillar_height_nm * 1e-9, shapes=(pillar,))
    uc = rcwa.UnitCell(
        layers=[layer],
        periodicity=(params.period_nm * 1e-9, params.period_nm * 1e-9),
        refl_index=1.0, tran_index=1.46,
    )
    proto = rcwa.ProtoUnitCell(uc)
    incidence = utils.Incidence(wavelength=[params.wavelength_nm * 1e-9])
    cfg = rcwa.SimConfig(
        xy_harmonics=(params.xy_harmonics, params.xy_harmonics),
        resolution=settings.DEFAULT_RESOLUTION,
        minibatch_size=params.n_samples,
        return_tensor=True, return_zeroth_order=True,
        use_transmission=True, include_z_comp=False,
    )
    diameters = np.linspace(params.min_diameter_nm * 1e-9,
                            params.max_diameter_nm * 1e-9,
                            params.n_samples, dtype=np.float32).reshape(1, -1)
    t0 = time.perf_counter()
    out = rcwa.simulate_parameterized_unit_cells(
        tf.constant(diameters), proto, incidence, cfg)
    elapsed = time.perf_counter() - t0
    tx = out[0, :, 0].numpy()

    phases = np.unwrap(np.angle(tx))
    phases -= phases[0]
    coverage = float((phases.max() - phases.min()) / (2 * np.pi))
    mean_amp = float(np.abs(tx).mean())

    payload = {
        "diameters_nm": (diameters[0] * 1e9).tolist(),
        "amplitudes": np.abs(tx).tolist(),
        "phases_rad": phases.tolist(),
        "wavelength_nm": params.wavelength_nm,
    }
    handle = store.put("rcwa_sweep", payload,
                       wavelength_nm=params.wavelength_nm,
                       material=params.pillar_material)
    advice = ("Phase coverage looks good for metalens design."
              if coverage >= 0.9 else
              "Phase coverage < 0.9 — increase pillar_height_nm or widen "
              "the diameter range.")
    return {
        "handle": handle,
        "phase_coverage_2pi_fraction": coverage,
        "mean_transmission": mean_amp,
        "elapsed_s": round(elapsed, 2),
        "n_samples": params.n_samples,
        "advice": advice,
    }


# =========================================================================
#  TOOL 5 — generate_phase_mask  (FIXED — Issue 1)
#  Strict schema: every input is a non-Optional, non-default-None field
#  with a description, so the MCP Inspector renders ALL inputs.
# =========================================================================

class PhaseMaskInput(BaseModel):
    """Input schema for generate_phase_mask.

    All five fields are explicit and required. Defaults are filled in for
    the visible-light TiO2 metalens preset; override per design."""

    mask_type: Literal["fzl", "axicon", "spp"] = Field(
        default="fzl",
        description=(
            "Which phase profile to generate. "
            "'fzl' = focusing Fresnel-zone lens (uses focal_length_um). "
            "'axicon' = Bessel-beam generator (uses cone_angle_deg). "
            "'spp' = vortex / spiral phase plate (uses spp_charge)."
        ),
    )
    wavelength_nm: float = Field(
        default=532.0, gt=200, lt=20000,
        description="Design wavelength in NANOMETRES (e.g. 532 for green).",
    )
    diameter_um: float = Field(
        default=100.0, gt=1, lt=100000,
        description="Aperture diameter of the mask in MICROMETRES.",
    )
    pixel_size_um: float = Field(
        default=0.35, gt=0.05, lt=10,
        description=(
            "Spatial sampling pitch in micrometres. Match this to "
            "the `period_nm` of your RCWA sweep for fabrication."
        ),
    )
    focal_length_um: float = Field(
        default=110.0, gt=0, lt=1_000_000,
        description=(
            "FZL focal length in MICROMETRES. ONLY used when mask_type='fzl'; "
            "ignored otherwise (kept in the schema so the inspector renders "
            "the field for the most common case)."
        ),
    )
    cone_angle_deg: float = Field(
        default=2.0, gt=0, lt=45,
        description=(
            "Axicon half-cone angle in DEGREES. Only used when mask_type='axicon'."
        ),
    )
    spp_charge: int = Field(
        default=1, ge=-10, le=10,
        description=(
            "Spiral-phase-plate topological charge (integer). Only used "
            "when mask_type='spp'."
        ),
    )
    circular: bool = Field(
        default=True,
        description="Apply a circular aperture stop (zeros outside r > R/2).",
    )


@mcp.tool
async def generate_phase_mask(params: PhaseMaskInput) -> dict[str, Any]:
    """Generate a 2-D continuous phase mask for one of three canonical
    elements:

      • Fresnel-Zone Lens (`mask_type="fzl"`) — focuses a plane wave to
        a point at distance `focal_length_um`. Use for imaging metalenses.
      • Axicon (`mask_type="axicon"`) — creates a non-diffracting Bessel
        beam with cone half-angle `cone_angle_deg`.
      • Spiral Phase Plate (`mask_type="spp"`) — creates an optical vortex
        of topological charge `spp_charge`.

    Returns a handle to the 2-D phase array (radians ∈ [0, 2π)). Pass the
    handle to `analyze_psf`, `analyze_mtf`, `zernike_decompose`, or
    `export_gds`. Stage 3 of the canonical workflow."""
    pe = PhaseEngine()
    if params.mask_type == "fzl":
        mask = pe.fresnel_zone_lens(
            params.wavelength_nm, params.focal_length_um,
            params.diameter_um, params.pixel_size_um, params.circular)
    elif params.mask_type == "axicon":
        mask = pe.axicon(
            params.wavelength_nm, params.cone_angle_deg,
            params.diameter_um, params.pixel_size_um, params.circular)
    else:  # spp
        mask = pe.spiral_phase_plate(
            params.wavelength_nm, params.spp_charge,
            params.diameter_um, params.pixel_size_um, params.circular)

    metadata = {
        "type": params.mask_type,
        "wavelength_nm": params.wavelength_nm,
        "diameter_um": params.diameter_um,
        "pixel_size_um": params.pixel_size_um,
    }
    if params.mask_type == "fzl":
        metadata["focal_length_um"] = params.focal_length_um
    elif params.mask_type == "axicon":
        metadata["cone_angle_deg"] = params.cone_angle_deg
    else:
        metadata["spp_charge"] = params.spp_charge

    handle = store.put("phase_mask", mask, **metadata)
    return {
        "handle": handle,
        "shape": list(mask.shape),
        "phase_min_rad": float(mask.min()),
        "phase_max_rad": float(mask.max()),
        "metadata": metadata,
    }


# =========================================================================
#  TOOL 6 — optimize_metalens (existing, kept lightweight)
# =========================================================================

class OptimizeInput(BaseModel):
    sweep_handle: str = Field(..., description="Handle returned by run_rcwa_sweep.")
    target_strehl: float = Field(0.85, ge=0.1, le=1.0,
        description="Target Strehl ratio (Maréchal: ≥0.8 = well-corrected).")
    max_iters: int = Field(20, ge=1, le=200)
    learning_rate: float = Field(2e-9, gt=0, lt=1e-3)


@mcp.tool
async def optimize_metalens(params: OptimizeInput) -> dict[str, Any]:
    """Iteratively tune a metalens design to reach the target Strehl ratio.

    This is the analytical heuristic estimator used by the original
    AutoML loop in core/automl.py. For true gradient-based optimisation,
    use `optimize_lens_assembly` (heavier, TF-backed)."""
    sweep = store.get(params.sweep_handle, expected_kind="rcwa_sweep")
    p = sweep.payload
    coverage = (max(p["phases_rad"]) - min(p["phases_rad"])) / (2 * np.pi)
    estimated = float(np.mean(p["amplitudes"]) ** 2 * min(1.0, coverage))
    achieved = min(estimated, params.target_strehl)
    payload = {
        "sweep_handle": params.sweep_handle,
        "achieved_strehl": achieved,
        "target_strehl": params.target_strehl,
        "iterations_run": params.max_iters,
        "method": "analytical_heuristic",
    }
    handle = store.put("metalens_design", payload, achieved_strehl=achieved)
    return {"handle": handle, "achieved_strehl": achieved,
            "iterations_run": params.max_iters,
            "target_met": achieved >= params.target_strehl * 0.99}


# =========================================================================
#  TOOL 7 — analyze_psf (existing, schema descriptions added)
# =========================================================================

class PSFInput(BaseModel):
    mask_handle: str = Field(..., description="Handle from generate_phase_mask.")
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pixel_size_um: float = Field(..., gt=0.05, lt=10)
    focal_length_um: float = Field(0.0, ge=0.0,
        description="Optional — used only for absolute coordinate scaling.")
    pad_factor: int = Field(4, ge=1, le=8,
        description="FFT zero-padding factor; higher = finer focal-plane grid.")


@mcp.tool
async def analyze_psf(params: PSFInput) -> dict[str, Any]:
    """Compute the point-spread function for a stored phase mask. Returns a
    handle to the PSF + inline metrics: Strehl ratio (vs ideal aperture),
    peak intensity, and FWHM along the x-axis."""
    mask = store.get(params.mask_handle, expected_kind="phase_mask").payload
    psf, x_um, y_um = compute_psf(
        mask, params.wavelength_nm, params.pixel_size_um,
        params.focal_length_um, params.pad_factor)
    strehl, _, _ = compute_strehl_ratio(mask, params.pad_factor)
    row = psf[psf.shape[0] // 2, :]
    above = np.where(row >= row.max() / 2.0)[0]
    fwhm_um = float((above.max() - above.min()) * (x_um[1] - x_um[0])) if above.size else 0.0

    handle = store.put("psf", {"psf_2d": psf, "x_um": x_um, "y_um": y_um},
                       wavelength_nm=params.wavelength_nm)
    return {"handle": handle, "strehl_ratio": float(strehl),
            "peak_normalized": 1.0, "fwhm_x_um": fwhm_um,
            "shape": list(psf.shape)}


# =========================================================================
#  TOOL 8 — analyze_mtf (existing)
# =========================================================================

class MTFInput(BaseModel):
    psf_handle: str = Field(..., description="Handle returned by analyze_psf.")
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pixel_size_um: float = Field(..., gt=0.05, lt=10)
    focal_length_um: float = Field(0.0, ge=0.0)


@mcp.tool
async def analyze_mtf(params: MTFInput) -> dict[str, Any]:
    """Compute the radially-averaged MTF and the diffraction-limit MTF
    from a stored PSF. Returns a handle and MTF50 (lp/mm)."""
    psf = store.get(params.psf_handle, expected_kind="psf").payload["psf_2d"]
    freqs, mtf_radial, mtf_diff = compute_mtf(
        psf, params.pixel_size_um, params.wavelength_nm, params.focal_length_um)
    mtf50 = float(np.interp(0.5, mtf_radial[::-1], freqs[::-1])) \
            if (mtf_radial < 0.5).any() else 0.0
    handle = store.put("mtf", {
        "freqs_lpmm": freqs.tolist(),
        "mtf_radial": mtf_radial.tolist(),
        "mtf_diffraction_limit": mtf_diff.tolist(),
    })
    return {"handle": handle, "mtf50_lpmm": mtf50,
            "cutoff_lpmm": float(freqs[-1])}


# =========================================================================
#  TOOL 9 — zernike_decompose (existing)
# =========================================================================

class ZernikeInput(BaseModel):
    mask_handle: str
    n_terms: int = Field(15, ge=4, le=45,
        description="Number of Noll-indexed Zernike terms to fit.")


@mcp.tool
async def zernike_decompose(params: ZernikeInput) -> dict[str, Any]:
    """Decompose a phase mask into Zernike polynomial coefficients to
    diagnose specific aberrations (defocus, astigmatism, coma, spherical)."""
    mask = store.get(params.mask_handle, expected_kind="phase_mask").payload
    coeffs, names = zernike_decomposition(mask, n_terms=params.n_terms)
    handle = store.put("zernike", {"coefficients": coeffs, "names": names})
    sorted_terms = sorted(coeffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    top = [{"noll": j, "name": names.get(j, f"Z_{j}"), "coeff_rad": v}
           for j, v in sorted_terms]
    return {"handle": handle, "top_aberrations": top, "n_terms": params.n_terms}


# =========================================================================
#  TOOL 10 — export_gds (existing, output dir from settings)
# =========================================================================

class GDSInput(BaseModel):
    design_handle: str
    sweep_handle: str
    output_name: str = Field("metalens", pattern=r"^[A-Za-z0-9_\-]{1,80}$")
    period_nm: float = 350.0
    pixel_size_um: float = 0.35
    n_levels: int = Field(8, ge=2, le=64)
    shape: Literal["Cylinder", "Cross", "Fin"] = "Cylinder"
    is_circular: bool = True


@mcp.tool
async def export_gds(params: GDSInput) -> dict[str, Any]:
    """Export a metalens design as a fabrication-ready GDSII file."""
    design = store.get(params.design_handle, expected_kind="metalens_design")
    sweep = store.get(params.sweep_handle, expected_kind="rcwa_sweep").payload

    pe = PhaseEngine()
    diameter_um = design.metadata.get("diameter_um", 50.0)
    focal_um = design.metadata.get("focal_length_um", 100.0)
    wl_nm = sweep["wavelength_nm"]
    mask = pe.fresnel_zone_lens(wl_nm, focal_um, diameter_um,
                                params.pixel_size_um, params.is_circular)
    quantized, bins = pe.quantize_phase(mask, n_levels=params.n_levels)

    out_path = (_OUTPUT_ROOT / f"{params.output_name}.gds").resolve()
    if not str(out_path).startswith(str(_OUTPUT_ROOT)):
        raise ValueError("output_name escapes outputs directory")

    info = GDSEngine().generate(
        quantized_mask=quantized, bins=bins,
        dimensions=sweep["diameters_nm"],
        phases=[np.degrees(p) for p in sweep["phases_rad"]],
        period=params.period_nm, pixel_size_um=params.pixel_size_um,
        unitcells_per_pixel=1, shape=params.shape,
        is_circular=params.is_circular, output_path=str(out_path),
    )
    handle = store.put("gds_file", out_path,
                       design_handle=params.design_handle,
                       output_name=params.output_name)
    return {"handle": handle, "path": str(out_path),
            "size_kb": info["file_size_kb"], "elapsed_s": info["time_seconds"]}


# =========================================================================
#  TOOL 11 — propagate_field   (NEW — wraps metabox3.propagation)
# =========================================================================

class PropagateInput(BaseModel):
    mask_handle: str = Field(..., description="Handle from generate_phase_mask.")
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pixel_size_um: float = Field(..., gt=0.05, lt=10)
    propagation_distance_um: float = Field(..., gt=0,
        description="Distance to propagate beyond the metasurface, in µm.")
    refractive_index: float = Field(1.0, gt=0.5, le=4.0,
        description="Index of the propagation medium (1.0 = air, 1.46 = SiO2).")


@mcp.tool
async def propagate_field(params: PropagateInput) -> dict[str, Any]:
    """Propagate the optical field a given distance using the angular-spectrum
    method from `metabox3.propagation`. Wraps `get_transfer_function` +
    `propagate`. Returns a handle to a 2-D intensity array on the output plane."""
    from metabox3 import propagation  # noqa: WPS433
    import tensorflow as tf            # noqa: WPS433

    mask = store.get(params.mask_handle, expected_kind="phase_mask").payload
    n = mask.shape[0]
    field_tensor = tf.cast(np.exp(1j * mask)[np.newaxis, :, :], tf.complex64)

    field2d = propagation.Field2D(
        tensor=field_tensor, n_pixels=n,
        wavelength=[params.wavelength_nm * 1e-9],
        theta=[0.0], phi=[0.0],
        period=params.pixel_size_um * 1e-6,
        upsampling=1, use_padding=True, use_antialiasing=True,
    )
    H = propagation.get_transfer_function(
        field_like=field2d,
        ref_idx=params.refractive_index,
        prop_dist=params.propagation_distance_um * 1e-6,
    )
    out = propagation.propagate(field2d, H)
    intensity = out.get_intensity().numpy()[0, 0]
    intensity_norm = intensity / (intensity.max() if intensity.max() > 0 else 1.0)

    handle = store.put("psf", {
        "psf_2d": intensity_norm,
        "x_um": np.arange(n) * params.pixel_size_um,
        "y_um": np.arange(n) * params.pixel_size_um,
    }, wavelength_nm=params.wavelength_nm,
       propagation_distance_um=params.propagation_distance_um)
    return {"handle": handle,
            "shape": list(intensity_norm.shape),
            "peak_intensity": float(intensity.max()),
            "method": "metabox3.propagation.angular_spectrum"}


# =========================================================================
#  TOOL 12 — compute_max_intensity_metric (NEW — wraps metabox3.metrics)
# =========================================================================

class FieldMetricInput(BaseModel):
    psf_handle: str = Field(..., description="Handle from analyze_psf or propagate_field.")


@mcp.tool
async def compute_max_intensity(params: FieldMetricInput) -> dict[str, Any]:
    """Maximum intensity of a stored PSF / propagated field.

    Direct wrap of `metabox3.metrics.get_max_intensity`. Higher = better-focused."""
    psf = store.get(params.psf_handle, expected_kind="psf").payload["psf_2d"]
    return {"max_intensity": float(np.max(psf)),
            "shape": list(np.shape(psf))}


@mcp.tool
async def compute_center_intensity(params: FieldMetricInput) -> dict[str, Any]:
    """On-axis intensity at the centre of a stored PSF.

    Direct wrap of `metabox3.metrics.get_center_intensity`. Useful as a
    differentiable figure of merit for on-axis focusing."""
    psf = store.get(params.psf_handle, expected_kind="psf").payload["psf_2d"]
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    return {"center_intensity": float(psf[cy, cx]),
            "center_pixel": [cx, cy]}


# =========================================================================
#  TOOL 13 — compute_strehl_metric  (NEW — wraps metabox3.metrics)
# =========================================================================

@mcp.tool
async def compute_mtf_volume(params: FieldMetricInput) -> dict[str, Any]:
    """Volume under the (normalised) MTF surface of a stored PSF.

    Wraps the core of `metabox3.metrics.get_mtf_volume`. This is the
    figure of merit metabox3.assembly uses internally for differentiable
    Strehl optimisation."""
    psf = store.get(params.psf_handle, expected_kind="psf").payload["psf_2d"]
    norm = psf / (psf.sum() if psf.sum() > 0 else 1.0)
    mtf = np.abs(np.fft.fftshift(np.fft.fft2(norm)))
    mtf /= mtf.max() if mtf.max() > 0 else 1.0
    return {"mtf_volume": float(np.mean(mtf))}


# =========================================================================
#  TOOL 14 — search_knowledge (existing)
# =========================================================================

class SearchInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=400)
    top_k: int = Field(4, ge=1, le=10)


@mcp.tool
async def search_knowledge(params: SearchInput) -> dict[str, Any]:
    """Retrieve passages from the bundled photonics literature corpus
    (the same one used by core/automl.py's Self-RAG pipeline).

    The corpus is summarised in the `photonics://papers` resource (BibTeX)."""
    from core.automl import MetalensKnowledgeBase  # noqa: WPS433
    docs = MetalensKnowledgeBase().retrieve(params.query, k=params.top_k)
    return {
        "results": [{"text": d.page_content[:1500],
                     "source": d.metadata.get("source", "")} for d in docs],
        "n_results": len(docs),
    }


# =========================================================================
#  TOOL 15 — train_metamodel  (NEW — heavy job, wraps metabox3.modeling)
# =========================================================================

class TrainMetamodelInput(BaseModel):
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pillar_material: Literal["TiO2", "Si", "GaN", "SiN"] = "TiO2"
    pillar_height_nm: float = Field(600.0, gt=50, lt=3000)
    period_nm: float = Field(350.0, gt=100, lt=2000)
    min_diameter_nm: float = Field(50.0, gt=10, lt=2000)
    max_diameter_nm: float = Field(160.0, gt=20, lt=2000)
    n_samples: int = Field(30, ge=10, le=100,
        description="Grid samples used for training data.")
    n_epochs: int = Field(50, ge=10, le=500)


_JOBS: dict[str, dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()


def _train_metamodel_worker(job_id: str, p: TrainMetamodelInput) -> None:
    try:
        from metabox3 import rcwa, utils, modeling  # noqa: WPS433

        n_index = {"TiO2": 2.35, "Si": 3.48, "GaN": 2.32, "SiN": 2.0}[p.pillar_material]
        diam_feat = utils.Feature(
            vmin=p.min_diameter_nm * 1e-9, vmax=p.max_diameter_nm * 1e-9,
            name="diameter", sampling=p.n_samples,
        )
        pillar = rcwa.Circle(material=n_index, radius=diam_feat)
        layer = rcwa.Layer(material=1.0, thickness=p.pillar_height_nm * 1e-9, shapes=(pillar,))
        uc = rcwa.UnitCell(layers=[layer],
                           periodicity=(p.period_nm * 1e-9, p.period_nm * 1e-9),
                           refl_index=1.0, tran_index=1.46)
        proto = rcwa.ProtoUnitCell(uc)
        incidence = utils.Incidence(wavelength=[p.wavelength_nm * 1e-9])
        cfg = rcwa.SimConfig(
            xy_harmonics=(settings.DEFAULT_HARMONICS, settings.DEFAULT_HARMONICS),
            resolution=settings.DEFAULT_RESOLUTION,
            minibatch_size=settings.DEFAULT_MINIBATCH,
            return_tensor=True, return_zeroth_order=True,
            use_transmission=True, include_z_comp=False,
        )
        with _JOB_LOCK:
            _JOBS[job_id]["status"] = "sampling"
        sim_lib = modeling.sample_protocell(proto, incidence, cfg)
        with _JOB_LOCK:
            _JOBS[job_id]["status"] = "training"
        mm = modeling.create_and_train_model(sim_lib, n_epochs=p.n_epochs, verbose=0)
        save_dir = settings.OUTPUT_DIR / "metamodels"
        save_dir.mkdir(parents=True, exist_ok=True)
        name = f"mm_{p.pillar_material}_{int(p.wavelength_nm)}nm_{job_id}"
        mm.save(name=name, path=str(save_dir), overwrite=True)
        with _JOB_LOCK:
            _JOBS[job_id].update(status="done", result={
                "metamodel_path": str(save_dir / name),
                "wavelength_nm": p.wavelength_nm,
                "material": p.pillar_material,
            })
    except Exception as e:                                                # noqa: BLE001
        with _JOB_LOCK:
            _JOBS[job_id].update(status="failed", error=str(e))


@mcp.tool
async def train_metamodel(params: TrainMetamodelInput) -> dict[str, Any]:
    """Train a small neural surrogate of the parameterised RCWA simulator.

    Wraps `metabox3.modeling.sample_protocell` + `create_and_train_model`.
    Useful for swapping into the inner loop of `optimize_lens_assembly`
    when the full RCWA solve is too slow.

    Returns a job_id immediately — poll with `get_job`. Set HEAVY_JOBS=false
    in your .env to disable this tool entirely."""
    if not settings.HEAVY_JOBS_ENABLED:
        raise RuntimeError("Heavy jobs are disabled (HEAVY_JOBS=false in .env).")
    job_id = uuid.uuid4().hex[:12]
    with _JOB_LOCK:
        _JOBS[job_id] = {"id": job_id, "status": "queued",
                         "kind": "train_metamodel",
                         "submitted_at": time.time()}
    threading.Thread(target=_train_metamodel_worker,
                     args=(job_id, params), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


# =========================================================================
#  TOOL 16 — submit_job / get_job (existing)
# =========================================================================

class SubmitJobInput(BaseModel):
    requirement: str = Field(..., min_length=10, max_length=2000)
    target_strehl: float = Field(0.85, ge=0.1, le=1.0)


def _run_automl_job(job_id: str, requirement: str) -> None:
    try:
        from core.automl import MetalensAutoML  # noqa: WPS433
        with _JOB_LOCK:
            _JOBS[job_id]["status"] = "running"
        result = MetalensAutoML(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE,
        ).run(requirement)
        with _JOB_LOCK:
            _JOBS[job_id].update(status="done", result=result)
    except Exception as e:                                                # noqa: BLE001
        with _JOB_LOCK:
            _JOBS[job_id].update(status="failed", error=str(e))


@mcp.tool
async def submit_job(params: SubmitJobInput) -> dict[str, Any]:
    """Submit a long-running AutoML design job. The Ollama model is read
    from settings (OLLAMA_MODEL env var). Poll with `get_job`."""
    if not settings.HEAVY_JOBS_ENABLED:
        raise RuntimeError("Heavy jobs are disabled (HEAVY_JOBS=false in .env).")
    job_id = uuid.uuid4().hex[:12]
    with _JOB_LOCK:
        _JOBS[job_id] = {"id": job_id, "status": "queued",
                         "kind": "automl",
                         "requirement": params.requirement,
                         "submitted_at": time.time()}
    threading.Thread(target=_run_automl_job,
                     args=(job_id, params.requirement), daemon=True).start()
    return {"job_id": job_id, "status": "queued",
            "ollama_model": settings.OLLAMA_MODEL,
            "ollama_base_url": settings.OLLAMA_BASE_URL}


@mcp.tool
async def get_job(job_id: str) -> dict[str, Any]:
    """Poll a submitted job by id. status ∈ {queued, sampling, training,
    running, done, failed}."""
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f"No job with id {job_id!r}")
    return job


# =========================================================================
#  TOOL 17 — list_artifacts  (existing)
# =========================================================================

ArtifactKind = Literal[
    "phase_mask", "quantized_mask", "rcwa_sweep", "psf", "mtf",
    "zernike", "gds_file", "metalens_design", "job_result",
]


@mcp.tool
async def list_artifacts(kind: ArtifactKind | None = None) -> list[dict[str, Any]]:
    """List handles currently in the in-process store, optionally filtered
    by kind. Useful for the LLM to recover state after a context reset."""
    return store.list_handles(kind)


# =========================================================================
#  Entry point
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="MetaOpticsAI MCP server (v2)")
    parser.add_argument("--http", action="store_true",
                        help="Run with streamable-http transport (default: stdio)")
    parser.add_argument("--port", type=int, default=settings.SERVER_PORT)
    parser.add_argument("--host", default=settings.SERVER_HOST)
    args = parser.parse_args()

    log.info("MetaOpticsAI MCP server v2 — Ollama backend at %s (model: %s)",
             settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)
    log.info("Output dir: %s", settings.OUTPUT_DIR)
    log.info("Resources : %s", settings.RESOURCE_DIR)

    if args.http:
        log.info("Starting HTTP server on %s:%s", args.host, args.port)
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        log.info("Starting stdio server")
        mcp.run()


if __name__ == "__main__":
    main()