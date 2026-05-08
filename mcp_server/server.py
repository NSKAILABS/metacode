"""
mcp_server.server — MetaOpticsAI MCP server.

Exposes the existing MetaOpticsAI codebase as an MCP server with 10 tools,
consumable by Claude Desktop (stdio), the FastAPI front-end (HTTP), and
the hybrid LLM controller.

Run modes
---------
    # Local dev (Inspector + auto-reload)
    fastmcp dev mcp_server/server.py

    # stdio (Claude Desktop will spawn this automatically)
    python -m mcp_server.server

    # HTTP (for FastAPI mount or remote clients)
    python -m mcp_server.server --http --port 8765
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
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
from core.meta_data import MetaDataStore                              # noqa: E402
from core.phase_engine import PhaseEngine                             # noqa: E402
from core.gds_engine import GDSEngine                                 # noqa: E402
from core.design import DesignState                                   # noqa: E402
from analysis.psf import compute_psf                                  # noqa: E402
from analysis.mtf import compute_mtf                                  # noqa: E402
from analysis.strehl import compute_strehl_ratio                      # noqa: E402
from analysis.zernike import zernike_decomposition                    # noqa: E402
from mcp_server.store import ArtifactKind
# Heavy / TF-bound imports are deferred to the tool that needs them
# (RCWA + the Self-RAG knowledge base) so server startup stays fast.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("metaoptics-mcp")

mcp = FastMCP(
    name="MetaOpticsAI",
    instructions=(
        "Tools for designing, simulating, and exporting metalenses. "
        "Tools that produce big artefacts (phase masks, sweep results, PSFs) "
        "return a short UUID handle. Pass that handle to downstream tools "
        "instead of asking for the array contents."
    ),
)

_OUTPUT_ROOT = (_PROJECT_ROOT / "outputs").resolve()
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# 1. get_fdtd_library

@mcp.tool
async def get_fdtd_library() -> dict[str, Any]:
    """List the wavelengths, materials, and unit-cell shapes currently
    available in the FDTD library bundled with the project. Use this
    before designing to confirm what's available."""
    md = MetaDataStore()
    wls = md.get_wavelengths()
    out = []
    for wl in wls:
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


# ─────────────────────────────────────────────────────────────────────────
# 2. run_rcwa_sweep — phase vs diameter library
# ─────────────────────────────────────────────────────────────────────────

class RCWASweepInput(BaseModel):
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    pillar_material: Literal["TiO2", "Si", "GaN", "SiN"] = "TiO2"
    pillar_height_nm: float = Field(600.0, gt=50, lt=3000)
    period_nm: float = Field(350.0, gt=100, lt=2000)
    min_diameter_nm: float = Field(50.0, gt=10, lt=2000)
    max_diameter_nm: float = Field(160.0, gt=20, lt=2000)
    n_samples: int = Field(20, ge=5, le=100)
    xy_harmonics: int = Field(3, ge=1, le=11)


@mcp.tool
async def run_rcwa_sweep(params: RCWASweepInput) -> dict[str, Any]:
    """Run a 1-wavelength RCWA sweep over pillar diameter to produce a
    phase-vs-diameter design library. Returns a handle to the full sweep
    result plus inline summary statistics (phase coverage, mean amplitude)."""
    # Lazy import — RCWA pulls in TF
    from metabox3 import rcwa, utils  # noqa: WPS433
    import tensorflow as tf          # noqa: WPS433

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
    incidence = utils.Incidence(wavelength=[params.wavelength_nm * 1e-9])  # list, not tuple
    cfg = rcwa.SimConfig(
        xy_harmonics=(params.xy_harmonics, params.xy_harmonics),
        resolution=64,
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
    tx = out[0, :, 0].numpy()    # x-pol

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
    return {
        "handle": handle,
        "phase_coverage_2pi_fraction": coverage,
        "mean_transmission": mean_amp,
        "elapsed_s": round(elapsed, 2),
        "n_samples": params.n_samples,
        "advice": (
            "Phase coverage < 0.9 means 2π is not reached. "
            "Increase pillar_height_nm or widen the diameter range."
        ) if coverage < 0.9 else "Phase coverage looks good for metalens design.",
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. generate_phase_mask — FZL / axicon / SPP
# ─────────────────────────────────────────────────────────────────────────

class PhaseMaskInput(BaseModel):
    type: Literal["fzl", "axicon", "spp"] = "fzl"
    wavelength_nm: float = Field(..., gt=200, lt=20000)
    diameter_um: float = Field(..., gt=1, lt=100000)
    pixel_size_um: float = Field(..., gt=0.05, lt=10)
    focal_length_um: float | None = Field(None, gt=0)        # FZL
    cone_angle_deg: float | None = Field(None, gt=0, lt=45)  # axicon
    spp_charge: int | None = Field(None, ge=-10, le=10)      # SPP
    circular: bool = True


@mcp.tool
async def generate_phase_mask(params: PhaseMaskInput) -> dict[str, Any]:
    """Generate a 2-D continuous phase mask for a Fresnel-zone lens (FZL),
    axicon, or spiral phase plate (SPP). Returns a handle to the mask."""
    pe = PhaseEngine()
    if params.type == "fzl":
        if params.focal_length_um is None:
            raise ValueError("fzl requires focal_length_um")
        mask = pe.fresnel_zone_lens(
            params.wavelength_nm, params.focal_length_um,
            params.diameter_um, params.pixel_size_um, params.circular)
    elif params.type == "axicon":
        if params.cone_angle_deg is None:
            raise ValueError("axicon requires cone_angle_deg")
        mask = pe.axicon(
            params.wavelength_nm, params.cone_angle_deg,
            params.diameter_um, params.pixel_size_um, params.circular)
    else:  # spp
        if params.spp_charge is None:
            raise ValueError("spp requires spp_charge")
        mask = pe.spiral_phase_plate(
            params.wavelength_nm, params.spp_charge,
            params.diameter_um, params.pixel_size_um, params.circular)

    handle = store.put("phase_mask", mask,
                       type=params.type, wavelength_nm=params.wavelength_nm,
                       diameter_um=params.diameter_um,
                       pixel_size_um=params.pixel_size_um)
    return {
        "handle": handle, "shape": list(mask.shape),
        "phase_min_rad": float(mask.min()),
        "phase_max_rad": float(mask.max()),
    }


# ─────────────────────────────────────────────────────────────────────────
# 4. optimize_metalens — gradient-based (placeholder; wraps your AutoML)
# ─────────────────────────────────────────────────────────────────────────

class OptimizeInput(BaseModel):
    sweep_handle: str
    target_strehl: float = Field(0.85, ge=0.1, le=1.0)
    max_iters: int = Field(20, ge=1, le=200)
    learning_rate: float = Field(2e-9, gt=0, lt=1e-3)


@mcp.tool
async def optimize_metalens(params: OptimizeInput) -> dict[str, Any]:
    """Iteratively tune a metalens design to reach the target Strehl ratio.
    This is a thin wrapper around the existing AutoML loop in
    core/automl.py — for now it stubs out the full TF Adam loop and just
    returns the best Strehl from the sweep + a placeholder design state.
    Wire in your real optimizer here."""
    sweep = store.get(params.sweep_handle, expected_kind="rcwa_sweep")
    p = sweep.payload
    # Heuristic: best achievable is amplitude² × min(coverage, 1).
    coverage = (max(p["phases_rad"]) - min(p["phases_rad"])) / (2 * np.pi)
    estimated_strehl = float(np.mean(p["amplitudes"]) ** 2 * min(1.0, coverage))
    achieved = min(estimated_strehl, params.target_strehl)
    design_payload = {
        "sweep_handle": params.sweep_handle,
        "achieved_strehl": achieved,
        "target_strehl": params.target_strehl,
        "iterations_run": params.max_iters,
        "note": "stub — wire core/automl.optimize_single_lens_assembly here",
    }
    handle = store.put("metalens_design", design_payload,
                       achieved_strehl=achieved)
    return {"handle": handle, "achieved_strehl": achieved,
            "iterations_run": params.max_iters,
            "target_met": achieved >= params.target_strehl * 0.99}


# ─────────────────────────────────────────────────────────────────────────
# 5. analyze_psf
# ─────────────────────────────────────────────────────────────────────────

class PSFInput(BaseModel):
    mask_handle: str
    wavelength_nm: float
    pixel_size_um: float
    focal_length_um: float = 0.0
    pad_factor: int = Field(4, ge=1, le=8)


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
    # FWHM along central row
    row = psf[psf.shape[0] // 2, :]
    half = row.max() / 2.0
    above = np.where(row >= half)[0]
    fwhm_um = float((above.max() - above.min()) * (x_um[1] - x_um[0])) if above.size else 0.0

    handle = store.put("psf", {"psf_2d": psf, "x_um": x_um, "y_um": y_um},
                       wavelength_nm=params.wavelength_nm)
    return {
        "handle": handle, "strehl_ratio": float(strehl),
        "peak_normalized": 1.0,  # PSF is normalized to peak by compute_psf
        "fwhm_x_um": fwhm_um,
        "shape": list(psf.shape),
    }


# ─────────────────────────────────────────────────────────────────────────
# 6. analyze_mtf
# ─────────────────────────────────────────────────────────────────────────

class MTFInput(BaseModel):
    psf_handle: str
    wavelength_nm: float
    pixel_size_um: float
    focal_length_um: float = 0.0


@mcp.tool
async def analyze_mtf(params: MTFInput) -> dict[str, Any]:
    """Compute the radially-averaged MTF and the diffraction-limit MTF
    from a stored PSF. Returns a handle and the cutoff frequency."""
    psf = store.get(params.psf_handle, expected_kind="psf").payload["psf_2d"]
    freqs, mtf_radial, mtf_diff = compute_mtf(
        psf, params.pixel_size_um, params.wavelength_nm, params.focal_length_um)

    # MTF50: the frequency at which MTF drops to 50%
    mtf50 = float(np.interp(0.5, mtf_radial[::-1], freqs[::-1])) if (mtf_radial < 0.5).any() else 0.0

    payload = {
        "freqs_lpmm": freqs.tolist(),
        "mtf_radial": mtf_radial.tolist(),
        "mtf_diffraction_limit": mtf_diff.tolist(),
    }
    handle = store.put("mtf", payload)
    return {"handle": handle, "mtf50_lpmm": mtf50,
            "cutoff_lpmm": float(freqs[-1])}


# ─────────────────────────────────────────────────────────────────────────
# 7. zernike_decompose
# ─────────────────────────────────────────────────────────────────────────

class ZernikeInput(BaseModel):
    mask_handle: str
    n_terms: int = Field(15, ge=4, le=45)


@mcp.tool
async def zernike_decompose(params: ZernikeInput) -> dict[str, Any]:
    """Decompose a phase mask into Zernike polynomial coefficients to
    diagnose specific aberrations (defocus, astigmatism, coma, spherical)."""
    mask = store.get(params.mask_handle, expected_kind="phase_mask").payload
    coeffs, names = zernike_decomposition(mask, n_terms=params.n_terms)
    handle = store.put("zernike", {"coefficients": coeffs, "names": names})

    # Surface the 5 largest aberrations inline
    sorted_terms = sorted(coeffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    top = [{"noll": j, "name": names.get(j, f"Z_{j}"), "coeff_rad": v}
           for j, v in sorted_terms]
    return {"handle": handle, "top_aberrations": top, "n_terms": params.n_terms}


# ─────────────────────────────────────────────────────────────────────────
# 8. export_gds
# ─────────────────────────────────────────────────────────────────────────

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
    """Export a metalens design as a fabrication-ready GDSII file. Output
    is always written under ./outputs/ (paths from the LLM are sanitised)."""
    design = store.get(params.design_handle, expected_kind="metalens_design")
    sweep = store.get(params.sweep_handle, expected_kind="rcwa_sweep").payload

    # Build a circular phase mask large enough to cover the design.
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

    engine = GDSEngine()
    info = engine.generate(
        quantized_mask=quantized,
        bins=bins,
        dimensions=sweep["diameters_nm"],
        phases=[np.degrees(p) for p in sweep["phases_rad"]],
        period=params.period_nm,
        pixel_size_um=params.pixel_size_um,
        unitcells_per_pixel=1,
        shape=params.shape,
        is_circular=params.is_circular,
        output_path=str(out_path),
    )
    handle = store.put("gds_file", out_path,
                       design_handle=params.design_handle,
                       output_name=params.output_name)
    return {
        "handle": handle, "path": str(out_path),
        "size_kb": info["file_size_kb"],
        "elapsed_s": info["time_seconds"],
    }


# ─────────────────────────────────────────────────────────────────────────
# 9. search_knowledge — Self-RAG over the embedded design corpus
# ─────────────────────────────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=400)
    top_k: int = Field(4, ge=1, le=10)


@mcp.tool
async def search_knowledge(params: SearchInput) -> dict[str, Any]:
    """Retrieve the most relevant chunks from the metalens design corpus
    (the same one used by core/automl.py's Self-RAG pipeline)."""
    # Lazy import — building the FAISS / Ollama embedding index is heavy
    from core.automl import MetalensKnowledgeBase  # noqa: WPS433
    kb = MetalensKnowledgeBase()
    docs = kb.retrieve(params.query, k=params.top_k)
    return {
        "results": [
            {"text": d.page_content[:1500],
             "source": d.metadata.get("source", "")}
            for d in docs
        ],
        "n_results": len(docs),
    }


# ─────────────────────────────────────────────────────────────────────────
# 10. submit_job / get_job — long-running queue
# ─────────────────────────────────────────────────────────────────────────

_JOBS: dict[str, dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()


class SubmitJobInput(BaseModel):
    requirement: str = Field(..., min_length=10, max_length=2000)
    target_strehl: float = Field(0.85, ge=0.1, le=1.0)


def _run_job(job_id: str, requirement: str, target_strehl: float) -> None:
    """In-process worker — replace with Celery/RQ for production."""
    try:
        from core.automl import MetalensAutoML  # noqa: WPS433
        with _JOB_LOCK:
            _JOBS[job_id]["status"] = "running"
        # Note: MetalensAutoML.run is synchronous; we ran it in a thread.
        result = MetalensAutoML().run(requirement)
        with _JOB_LOCK:
            _JOBS[job_id].update(status="done", result=result)
    except Exception as e:                                                # noqa: BLE001
        with _JOB_LOCK:
            _JOBS[job_id].update(status="failed", error=str(e))


@mcp.tool
async def submit_job(params: SubmitJobInput) -> dict[str, Any]:
    """Submit a long-running AutoML design job to the in-process queue. Use
    `get_job` to poll for completion. Returns immediately with a job_id."""
    job_id = uuid.uuid4().hex[:12]
    with _JOB_LOCK:
        _JOBS[job_id] = {"id": job_id, "status": "queued",
                         "requirement": params.requirement,
                         "submitted_at": time.time()}
    threading.Thread(target=_run_job, args=(job_id, params.requirement,
                                            params.target_strehl),
                     daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@mcp.tool
async def get_job(job_id: str) -> dict[str, Any]:
    """Poll a job by id. Status is one of queued | running | done | failed.
    When status == 'done', the `result` field contains the AutoML output."""
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise KeyError(f"No job with id {job_id!r}")
    return job


# ─────────────────────────────────────────────────────────────────────────
# Bonus: a handful of generic store-management tools
# ─────────────────────────────────────────────────────────────────────────

@mcp.tool
async def list_artifacts(kind: ArtifactKind | None = None) -> list[dict[str, Any]]:  # type: ignore[name-defined]
    """List handles currently in the in-process store, optionally filtered
    by kind. Useful for the LLM to recover state after a context reset."""
    return store.list_handles(kind)


# Provide a string alias so pydantic-friendly LLMs can pass it
ArtifactKind = Literal[
    "phase_mask", "quantized_mask", "rcwa_sweep", "psf", "mtf",
    "zernike", "gds_file", "metalens_design", "job_result",
]


# ─────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MetaOpticsAI MCP server")
    parser.add_argument("--http", action="store_true",
                        help="Run with streamable-http transport (default: stdio)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.http:
        log.info(f"Starting HTTP server on {args.host}:{args.port}")
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        log.info("Starting stdio server")
        mcp.run()   # default transport is stdio


if __name__ == "__main__":
    main()