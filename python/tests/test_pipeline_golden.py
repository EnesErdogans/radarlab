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


def test_golden_peaks_match_truth():
    for name in ("tiny_cube", "automotive_single", "automotive_multi", "awr1843_elevated"):
        case = golden.ALL_CASES[name]()
        tol = case["tolerances"]
        for pk in case["expected"]["peaks"]:
            t = pk["truth"]
            assert abs(pk["range"] - t["range"]) < tol["truthRangeAbs"]
            assert abs(pk["velocity"] - t["velocityAliased"]) < tol["truthVelocityAbs"]
            assert abs(pk["angleDeg"] - t["angleDeg"]) < tol["truthAngleDegAbs"]
