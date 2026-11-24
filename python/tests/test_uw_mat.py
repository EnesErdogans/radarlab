"""Tests for the UW AWR1843 template .mat loader (synthetic files only)."""
import numpy as np
import pytest
import scipy.io

import radarlab as rl
from radarlab.config import C0
from radarlab.io.uw_mat import load_uw_mat, uw_awr1843_config, uw_decode, uw_encode


def _code(f, t, r, m, n):
    """Unique complex value per (frame, tx slot, rx, loop, sample)."""
    return (f * 10000 + t * 1000 + r * 100 + m * 10 + n) + 1j * (n - 2 * m + 3 * r)


def _file_layout(F, nT, nR, M, N):
    """data_frames exactly as the UW MATLAB scripts index it:
    column ((f·M + m)·nT + t)·N + n, row r (independent of the loader code)."""
    d = np.zeros((nR, F * M * nT * N), complex)
    for f in range(F):
        for m in range(M):
            for t in range(nT):
                for n in range(N):
                    col = ((f * M + m) * nT + t) * N + n
                    for r in range(nR):
                        d[r, col] = _code(f, t, r, m, n)
    return d


@pytest.fixture
def small_cfg():
    return uw_awr1843_config(n_chirps=3, n_samples=5)


def test_published_config_derived_values():
    cfg = uw_awr1843_config()
    b_eff = 21.0017e12 * 128 / 4e6
    assert cfg.range_resolution == pytest.approx(C0 / (2 * b_eff))
    assert cfg.range_resolution == pytest.approx(0.2230, abs=1e-4)
    assert cfg.te == pytest.approx(120e-6)
    assert cfg.max_velocity == pytest.approx(C0 / 77e9 / (4 * 120e-6))
    assert cfg.max_velocity == pytest.approx(8.11, abs=0.01)
    assert cfg.n_virtual == 8
    np.testing.assert_allclose(cfg.virtual_pos, [0, .5, 1, 1.5, 2, 2.5, 3, 3.5])
    assert cfg.frame_period == pytest.approx(1 / 30)
    assert cfg.validate() == []


def test_load_synthetic_mat_layout(tmp_path, small_cfg):
    F, nT, nR, M, N = 2, 2, 4, small_cfg.n_chirps, small_cfg.n_samples
    p = tmp_path / "x.mat"
    scipy.io.savemat(p, {"data_frames": _file_layout(F, nT, nR, M, N), "other": np.arange(3)})
    cubes, cfg = load_uw_mat(p, small_cfg)
    assert cfg is small_cfg
    assert cubes.shape == (F, nT * nR, M, N) and cubes.dtype == np.complex128
    for f in range(F):
        for t in range(nT):
            for r in range(nR):
                k = t * nR + r                       # brief layout k = t·nRx + r
                for m in range(M):
                    for n in range(N):
                        assert cubes[f, k, m, n] == _code(f, t, r, m, n)
    # each frame is a brief-layout (K, M, N) cube usable by the chain
    assert cubes[0].shape == (cfg.n_virtual, cfg.n_chirps, cfg.n_samples)


def test_tx_order_and_conj(small_cfg):
    F, nT, nR, M, N = 1, 2, 4, small_cfg.n_chirps, small_cfg.n_samples
    d = _file_layout(F, nT, nR, M, N)
    sw = uw_decode(d, small_cfg, tx_order=(1, 0))
    # the first chirp of each loop is now TX index 1 -> channels 4..7
    assert sw[0, 1 * nR + 2, 1, 3] == _code(0, 0, 2, 1, 3)
    assert sw[0, 0 * nR + 2, 1, 3] == _code(0, 1, 2, 1, 3)
    cj = uw_decode(d, small_cfg, conj=True)
    np.testing.assert_array_equal(cj, uw_decode(d, small_cfg).conj())
    with pytest.raises(ValueError):
        uw_decode(d, small_cfg, tx_order=(0, 0))


def test_encode_roundtrip_and_partial_frame(rng, small_cfg):
    c = rng.standard_normal((3, 8, 3, 5)) + 1j * rng.standard_normal((3, 8, 3, 5))
    for order in ((0, 1), (1, 0)):
        d = uw_encode(c, small_cfg, order)
        assert d.shape == (4, 3 * 3 * 2 * 5)
        np.testing.assert_array_equal(uw_decode(d, small_cfg, order), c)
    d = uw_encode(c, small_cfg)
    part = np.concatenate([d, d[:, :7]], axis=1)           # trailing incomplete frame
    assert uw_decode(part, small_cfg).shape == (3, 8, 3, 5)


def test_simulated_target_survives_file_roundtrip(tmp_path):
    """A simulated point target written in the UW layout is recovered by the chain."""
    cfg = uw_awr1843_config()
    tgt = rl.PointTarget.from_polar(7.0, -10.0, radial_velocity=-1.0)
    cube = rl.simulate_frame(cfg, tgt, rng=0)
    p = tmp_path / "sim.mat"
    scipy.io.savemat(p, {"data_frames": uw_encode(cube, cfg)})
    cubes, cfg2 = load_uw_mat(p)
    res = rl.process_frame(cubes[0], cfg2)
    b = res.points[np.argmax(res.points["power_db"])]
    assert b["range"] == pytest.approx(7.0, abs=cfg.range_resolution / 2)
    assert b["velocity"] == pytest.approx(-1.0, abs=cfg.velocity_resolution)
    assert np.rad2deg(b["azimuth"]) == pytest.approx(-10.0, abs=1.0)


def test_rejects_bad_files(tmp_path, small_cfg):
    p = tmp_path / "bad.mat"
    scipy.io.savemat(p, {"data_frames": np.zeros((3, 60))})           # 3 RX instead of 4
    with pytest.raises(ValueError):
        load_uw_mat(p, small_cfg)
    scipy.io.savemat(p, {"something": np.zeros((4, 60))})
    with pytest.raises(KeyError):
        load_uw_mat(p, small_cfg)
    scipy.io.savemat(p, {"data_frames": np.array(["abc"])})            # non-numeric
    with pytest.raises(ValueError):
        load_uw_mat(p, small_cfg)
    scipy.io.savemat(p, {"data_frames": np.zeros((4, 10))})            # shorter than a frame
    with pytest.raises(ValueError):
        load_uw_mat(p, small_cfg)
