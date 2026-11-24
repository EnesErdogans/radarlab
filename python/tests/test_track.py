import numpy as np
import pytest

import radarlab as rl
from radarlab.track import Tracker, TrackerParams, meas_fn, meas_jacobian


def test_jacobian_matches_finite_difference():
    x = np.array([3.0, 20.0, -1.5, 4.0])
    H = meas_jacobian(x)
    num = np.column_stack([(meas_fn(x + e) - meas_fn(x - e)) / 2e-6 for e in np.eye(4) * 1e-6])
    np.testing.assert_allclose(H, num, rtol=1e-6, atol=1e-8)


def test_tracker_rmse_below_measurement_noise(rng):
    p = TrackerParams(q=0.05, sigma_r=0.3, sigma_theta=np.deg2rad(2.0), sigma_vr=0.15)
    trk = Tracker(p)
    dt = 0.05
    truths = [np.array([-10.0, 30.0, 2.0, -1.0]), np.array([8.0, 15.0, -0.5, 3.0])]
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    err_trk, err_meas = [], []
    ids = set()
    for k in range(120):
        if k:
            truths = [F @ x for x in truths]
        Z = []
        for x in truths:
            z = meas_fn(x) + rng.standard_normal(3) * [p.sigma_r, p.sigma_theta, p.sigma_vr]
            Z.append(z)
        Z.append([rng.uniform(5, 40), rng.uniform(-1, 1), rng.uniform(-5, 5)])   # clutter
        conf = trk.step(np.array(Z), dt if k else 0.0)
        if k >= 30:
            for x, z in zip(truths, Z):
                pm = np.array([z[0] * np.sin(z[1]), z[0] * np.cos(z[1])])
                err_meas.append(np.sum((pm - x[:2]) ** 2))
                d = [np.sum((t.position - x[:2]) ** 2) for t in conf]
                err_trk.append(min(d))
                ids.add(conf[int(np.argmin(d))].id)
    rmse_t, rmse_m = np.sqrt(np.mean(err_trk)), np.sqrt(np.mean(err_meas))
    assert rmse_t < 0.7 * rmse_m
    assert len(ids) == 2                         # stable track IDs


def test_track_deleted_after_misses():
    trk = Tracker(TrackerParams(confirm_m=2, confirm_n=3, max_misses=2))
    z = np.array([[20.0, 0.1, 1.0]])
    for _ in range(3):
        trk.step(z, 0.05)
    assert len(trk.confirmed()) == 1
    for _ in range(2):
        trk.step(np.zeros((0, 3)), 0.05)
    assert trk.tracks == []
