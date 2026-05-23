"""Domain layer is the leaf — these tests must pass with zero infra setup."""
from __future__ import annotations

import math

import pytest

from metaopticsai.domain.design import MetalensDesignParams
from metaopticsai.domain.handles import Artifact


def test_metalens_params_computed_fields():
    p = MetalensDesignParams(
        wavelength_nm=532,
        focal_length_mm=0.1,   # 100 µm
        diameter_mm=0.05,      # 50 µm
    )
    # NA = sin(arctan(D / 2f))
    expected_na = math.sin(math.atan(p.diameter_mm / (2 * p.focal_length_mm)))
    assert p.numerical_aperture == pytest.approx(expected_na, rel=1e-6)
    assert p.f_number == pytest.approx(p.focal_length_mm / p.diameter_mm)


def test_metalens_params_rejects_max_diameter_above_period():
    """max_diameter must fit inside the unit cell."""
    with pytest.raises(Exception):
        MetalensDesignParams(
            periodicity_nm=200,
            min_diameter_nm=50,
            max_diameter_nm=500,   # > period - 20, invalid
        )


def test_metalens_params_rejects_min_ge_max():
    """min_diameter must be strictly less than max."""
    with pytest.raises(Exception):
        MetalensDesignParams(min_diameter_nm=200, max_diameter_nm=200)


def test_artifact_roundtrip():
    art = Artifact.new("phase_mask", {"x": 1}, k="v")
    assert art.kind == "phase_mask"
    assert art.metadata["k"] == "v"
    assert len(art.handle) == 12
    summary = art.summary()
    assert "handle" in summary
    assert "keys" in summary  # payload is a dict