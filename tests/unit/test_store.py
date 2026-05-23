"""ArtifactStore unit tests — pure stdlib, no external services."""
from __future__ import annotations

import pytest

from metaopticsai.store.memory import InMemoryArtifactStore


def test_put_get_roundtrip():
    s = InMemoryArtifactStore()
    h = s.put("phase_mask", payload={"k": 1}, wavelength_nm=532)
    a = s.get(h)
    assert a.payload == {"k": 1}
    assert a.metadata["wavelength_nm"] == 532


def test_expected_kind_mismatch_raises():
    s = InMemoryArtifactStore()
    h = s.put("phase_mask", payload={"k": 1})
    with pytest.raises(Exception):
        s.get(h, expected_kind="rcwa_sweep")


def test_list_filters_by_kind():
    s = InMemoryArtifactStore()
    s.put("phase_mask", payload=1)
    s.put("phase_mask", payload=2)
    s.put("rcwa_sweep", payload={"a": 1})
    pm = s.list("phase_mask")
    assert len(pm) == 2
    rs = s.list("rcwa_sweep")
    assert len(rs) == 1
    everything = s.list()
    assert len(everything) == 3