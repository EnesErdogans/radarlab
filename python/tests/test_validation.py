"""Validation matrix V1–V11 (fidelity_requirements.md §15).

Quick variants run in the default suite; paper-grade sizes are marked slow."""
import pytest

from radarlab import validation as V


def _check(r):
    assert r.passed, f"{r.id} failed: {r.rows}"


@pytest.mark.parametrize("vid", ["V1", "V1c", "V2", "V3", "V4", "V8", "V8f"])
def test_validation_quick(vid):
    _check(V.ALL[vid](quick=True))


@pytest.mark.parametrize("vid", ["V5", "V6", "V7", "V9", "V10", "V11"])
def test_validation_deterministic(vid, tmp_path):
    _check(V.ALL[vid](tmp_path) if vid == "V11" else V.ALL[vid]())


@pytest.mark.slow
@pytest.mark.parametrize("vid", ["V1", "V1c", "V2", "V3", "V4", "V5", "V8", "V8f"])
def test_validation_full(vid):
    _check(V.ALL[vid](quick=False))


def test_validation_kinds():
    theory = {"V1", "V1c", "V2", "V3", "V4", "V5", "V7", "V8"}
    consistency = {"V6", "V8f", "V9", "V10", "V11"}
    assert set(V.ALL) == theory | consistency
    assert V.v6_iq_imbalance().kind == "consistency" and V.v7_interference().kind == "theory"
