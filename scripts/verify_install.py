"""Verify a local MetaOpticsAI install end-to-end without running the tests.

Reports:
  1. Which packages import cleanly.
  2. Which tools the registry exposes.
  3. Any names referenced in __init__.py files that fail to resolve at runtime.

Usage:
    python scripts/verify_install.py
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

# Allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


# ── 1. Try importing every package in the tree ────────────────────────────

PACKAGES = [
    "metaopticsai.configs.settings",
    "metaopticsai.domain.handles",
    "metaopticsai.domain.design",
    "metaopticsai.domain.results",
    "metaopticsai.store.memory",
    "metaopticsai.utils.json_io",
    "metaopticsai.utils.images",
    "metaopticsai.utils.logging",
    "metaopticsai.phase.profiles",
    "metaopticsai.phase.quantization",
    "metaopticsai.analysis",
    "metaopticsai.tools.base",
    "metaopticsai.tools.schemas",
    "metaopticsai.tools.rcwa_tools",
    "metaopticsai.tools.phase_tools",
    "metaopticsai.tools.analysis_tools",
    "metaopticsai.tools.materials_tools",
    "metaopticsai.tools.propagation_tools",
    "metaopticsai.tools.gds_tools",
    "metaopticsai.tools.optimization_tools",
    "metaopticsai.physics.backends.base",
    "metaopticsai.physics.backends.analytical",
    "metaopticsai.optimization.heuristic",
    "metaopticsai.orchestration.job_manager",
]

# ── 2. Specific symbols every tool module must export ────────────────────

EXPECTED_SYMBOLS = {
    "metaopticsai.tools.rcwa_tools": ["RunRCWASweepTool", "GetFDTDLibraryTool"],
    "metaopticsai.tools.phase_tools": ["GeneratePhaseMaskTool"],
    "metaopticsai.tools.analysis_tools": [
        "AnalyzePSFTool", "AnalyzeMTFTool", "ZernikeDecomposeTool",
    ],
    "metaopticsai.tools.materials_tools": [
        "ListMaterialsTool", "GetMaterialIndexTool",
    ],
    "metaopticsai.tools.propagation_tools": [
        "PropagateFieldTool", "ComputeMaxIntensityTool",
        "ComputeCenterIntensityTool", "ComputeMTFVolumeTool",
    ],
    "metaopticsai.tools.gds_tools": ["ExportGDSTool"],
    "metaopticsai.tools.optimization_tools": [
        "OptimizeMetalensTool", "TrainMetamodelTool",
        "SubmitAutoMLTool", "GetJobTool", "ListArtifactsTool",
    ],
}

EXPECTED_TOOL_NAMES = {
    "run_rcwa_sweep", "get_fdtd_library",
    "list_materials", "get_material_index",
    "generate_phase_mask",
    "analyze_psf", "analyze_mtf", "zernike_decompose",
    "propagate_field",
    "compute_max_intensity", "compute_center_intensity", "compute_mtf_volume",
    "optimize_metalens",
    "export_gds",
    "list_artifacts",
}


def step1_imports() -> list[str]:
    errors = []
    print("── 1. Module-level imports ─────────────────────────────────────────")
    for mod in PACKAGES:
        try:
            importlib.import_module(mod)
            print(f"  ✓ {mod}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {mod}: {type(e).__name__}: {e}")
            errors.append(f"{mod}: {e}")
    return errors


def step2_symbols() -> list[str]:
    errors = []
    print("\n── 2. Expected exported symbols ────────────────────────────────────")
    for mod, names in EXPECTED_SYMBOLS.items():
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ cannot import {mod}: {e}")
            errors.append(f"import {mod}: {e}")
            continue
        missing = [n for n in names if not hasattr(m, n)]
        if missing:
            print(f"  ✗ {mod} MISSING: {missing}")
            errors.append(f"{mod} missing {missing}")
        else:
            print(f"  ✓ {mod} ({len(names)} symbols)")
    return errors


def step3_registry() -> list[str]:
    errors = []
    print("\n── 3. Tool registry composition ────────────────────────────────────")
    try:
        from metaopticsai.physics.backends.analytical import AnalyticalBackend
        from metaopticsai.physics.backends.base import BackendRegistry
        from metaopticsai.store.memory import InMemoryArtifactStore
        from metaopticsai.tools import build_tool_registry

        store = InMemoryArtifactStore()
        backends = BackendRegistry()
        backends.register(AnalyticalBackend())
        registry = build_tool_registry(store, backends)
        names = set(registry.names())
        missing = EXPECTED_TOOL_NAMES - names
        extra = names - EXPECTED_TOOL_NAMES
        if missing:
            print(f"  ✗ Missing tools: {sorted(missing)}")
            errors.append(f"missing tools: {missing}")
        if extra:
            print(f"  ℹ Extra tools (informational): {sorted(extra)}")
        if not missing:
            print(f"  ✓ Registry has {len(names)} tools, all 15 expected present.")
    except Exception:  # noqa: BLE001
        print(f"  ✗ Registry build failed:")
        traceback.print_exc()
        errors.append("registry build failed")
    return errors


def main() -> int:
    all_errors = []
    all_errors.extend(step1_imports())
    all_errors.extend(step2_symbols())
    all_errors.extend(step3_registry())

    print("\n── Summary ──────────────────────────────────────────────────────────")
    if all_errors:
        print(f"❌ {len(all_errors)} issue(s) found:")
        for err in all_errors:
            print(f"   • {err}")
        return 1
    print("✅ All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())