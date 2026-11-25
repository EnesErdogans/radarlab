"""golden/impairments.json (roadmap A4): up to date, and its stored inputs replay through the
Python impairment functions to the stored outputs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from radarlab import impairments as imp
from radarlab.golden import tiny_config

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "golden" / "impairments.json"


def _export_module():
    spec = importlib.util.spec_from_file_location("export_golden", ROOT / "python" / "scripts" / "export_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def stored():
    if not P.exists():
        pytest.skip("golden not exported")
    return json.loads(P.read_text())


def test_impairments_golden_up_to_date(stored):
    fresh = json.loads(json.dumps(_export_module().case_impairments()))
    assert fresh.keys() == stored.keys()
    for k in ("cube", "cubeNoPhaseNoise", "iqImbalance", "ifFilter"):
        np.testing.assert_allclose(fresh[k]["re"], stored[k]["re"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(fresh[k]["im"], stored[k]["im"], rtol=1e-9, atol=1e-9)
    assert fresh["adc"]["codesRe"] == stored["adc"]["codesRe"]
    assert fresh["adc"]["codesIm"] == stored["adc"]["codesIm"]
    assert fresh["phaseNoise"]["values"] == pytest.approx(stored["phaseNoise"]["values"], rel=1e-12, abs=1e-15)


def test_impairments_golden_replays_from_stored_inputs(stored):
    cfg = tiny_config()
    x = np.array(stored["cube"]["re"]) + 1j * np.array(stored["cube"]["im"])
    x = x.reshape(stored["cube"]["shape"])
    q = stored["iqImbalance"]
    y1 = imp.iq_imbalance(x, q["gainDb"], q["phaseDeg"])
    np.testing.assert_allclose(y1.real.ravel(), q["re"], rtol=1e-9, atol=1e-9)
    f = stored["ifFilter"]
    y2 = imp.if_filter(y1, cfg, f["hpfHz"], f["lpfHz"], f["order"])
    np.testing.assert_allclose(y2.imag.ravel(), f["im"], rtol=1e-9, atol=1e-9)
    a = stored["adc"]
    _, codes = imp.adc_quantize(y2, a["bits"], a["fullScale"], return_codes=True)
    assert a["clippedRails"] > 0
    assert np.mean(codes.real.ravel() != np.array(a["codesRe"])) == 0
    # phase noise is not negligible in this vector (otherwise the web test would be vacuous)
    c0 = np.array(stored["cubeNoPhaseNoise"]["re"]) + 1j * np.array(stored["cubeNoPhaseNoise"]["im"])
    assert np.max(np.abs(x.ravel() - c0)) > 0.05 * np.max(np.abs(c0))
