import json
from pathlib import Path

import numpy as np
import pytest

import radarlab as rl
from radarlab import golden
from radarlab.track import Tracker

GOLDEN = Path(__file__).resolve().parents[2] / "golden"


def test_pipeline_point_cloud_and_tracker():
    cfg = rl.automotive_77ghz()
    scene = rl.Scene([rl.PointTarget((5.0, 30.0, 0.0), (0.0, -6.0, 0.0), rcs=10),
                      rl.Car((-8.0, 50.0, 0.0), (0.0, 4.0, 0.0)),
                      rl.Pedestrian((3.0, 12.0, 0.0), (1.2, 0.0, 0.0))])
    sim = rl.Simulator(cfg, scene, rl.ProcessingParams(cfar_kind="os"), tracker=Tracker(), seed=3)
    out = sim.run(6)
    pts = out[-1].points
    t = sim.t - cfg.frame_period
    truth = scene.truth(t)
    # point target detected within one bin in range and 1.5 deg
    d = np.hypot(pts["x"] - truth[0, 0], pts["y"] - truth[0, 1])
    i = int(np.argmin(d))
    assert abs(pts["range"][i] - truth[0, 6]) < cfg.range_resolution
    assert abs(pts["velocity"][i] - truth[0, 8]) < cfg.velocity_resolution
    assert len(out[-1].tracks) >= 2


@pytest.mark.parametrize("name", list(golden.ALL_CASES))
def test_golden_files_up_to_date(name):
    p = GOLDEN / f"{name}.json"
    if not p.exists():
        pytest.skip("golden not exported")
    stored = json.loads(p.read_text())
    fresh = json.loads(json.dumps(golden.ALL_CASES[name]()))

    def cmp(a, b, path=""):
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                cmp(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            if a and isinstance(a[0], (int, float)) and not isinstance(a[0], bool):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, err_msg=path)
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    cmp(x, y, f"{path}[{i}]")
        elif isinstance(a, float):
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), path
        else:
            assert a == b, path

    cmp(stored, fresh)


SCENARIOS = ("tiny_cube", "automotive_single", "automotive_multi", "awr1843_elevated",
             "tiny_mti", "tiny_real_iq", "tiny_physical")


@pytest.mark.parametrize("name", SCENARIOS)
def test_golden_peaks_match_truth(name):
    case = golden.ALL_CASES[name]()
    tol = case["tolerances"]
    assert tol["truthAngleFftDegAbs"] <= 0.05 and tol["truthRangeBeatAbs"] <= 0.03 * case["derived"]["rangeResolution"]
    for pk in case["expected"]["peaks"]:
        t = pk["truth"]
        # observable truth: only interpolation bias allowed
        assert abs(pk["range"] - t["rangeBeat"]) < tol["truthRangeBeatAbs"]
        assert abs(pk["velocity"] - t["velocityEffective"]) < tol["truthVelocityEffectiveAbs"]
        assert abs(pk["angleDeg"] - t["angleDegFft"]) < tol["truthAngleFftDegAbs"]
        assert abs(pk["azimuthDeg"] - t["angleDeg"]) < tol["truthAzimuthDegAbs"]
        if name == "awr1843_elevated":
            assert abs(pk["elevationDeg"] - t["elevationDeg"]) < tol["truthElevationDegAbs"]
        # geometric truth with the documented deterministic offsets
        assert abs(pk["range"] - t["range"]) < tol["truthRangeAbs"]
        assert abs(pk["velocity"] - t["velocityAliased"]) < tol["truthVelocityAbs"]
        assert abs(pk["angleDeg"] - t["angleDeg"]) < tol["truthAngleDegAbs"]


def test_golden_mti_removes_static():
    case = golden.ALL_CASES["tiny_mti"]()
    rd = np.array(case["rdPowerDb"]["data"]).reshape(case["rdPowerDb"]["shape"])
    M = case["config"]["nChirps"]
    # the static scatterer (rcs 20, ~13 dB above the mover) is gone: only the mover's
    # Hann Doppler sidelobes (> 30 dB down) remain in the zero-Doppler column
    assert rd[:, M // 2].max() < rd.max() - 30
    assert len(case["expected"]["peaks"]) == 1


def test_golden_cfar_maps_reproducible_from_stored_input():
    case = golden.ALL_CASES["cfar_maps"]()
    shape = tuple(case["shape"])
    for m in case["maps"]:
        P = np.array(m["P"]).reshape(shape)
        for kind in ("ca", "go", "so", "os"):
            r = rl.dsp.cfar_2d(P, tuple(case["guard"]), tuple(case["train"]), case["pfa"], kind,
                               case["osRank"], n_channels=m["K"])
            np.testing.assert_allclose(r.threshold.ravel(), m[kind]["threshold"], rtol=1e-12)
            assert np.argwhere(r.mask).tolist() == m[kind]["detections"]
        # range edges and GO/SO fallback are exercised
        assert min(m["nTrain"]) < max(m["nTrain"])
        assert len(m["go"]["caFallbackCells"]) > 0
    # the K = 8 masks discriminate between CFAR kinds
    k8 = case["maps"][1]
    assert len({tuple(map(tuple, k8[k]["detections"])) for k in ("ca", "so", "os")}) == 3


def test_golden_clustering_and_tracker():
    cl = golden.ALL_CASES["clustering"]()["clusters"]
    sizes = sorted(c["size"] for c in cl)
    assert sizes == [1, 1, 2, 2, 4, 5]          # seam merge (4), L (5), diagonals (2, 2)
    tr = golden.ALL_CASES["tracker"]()
    last = tr["frames"][-1]["tracks"]
    assert [t["status"] for t in last] == ["confirmed", "confirmed"]   # clutter track deleted
