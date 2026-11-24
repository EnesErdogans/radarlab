import numpy as np
import pytest

import radarlab as rl
from radarlab.signal import beat_sample


def test_vectorised_matches_scalar_formula(rng):
    cfg = rl.awr1843_like().replace(n_chirps=16, n_samples=64)
    tg = rl.PointTarget((3.0, 9.0, 1.5), (0.5, -2.0, 0.3), rcs=2.0, phase0=0.7)
    cube = rl.synthesize_cube(cfg, [tg], t_frame=0.013, noise=False)
    for _ in range(20):
        k, m, n = rng.integers(cfg.n_virtual), rng.integers(16), rng.integers(64)
        t = k // cfg.n_rx
        tm = cfg.chirp_start_times(0.013)[t, m]
        s = tg.scatterers([tm])
        p, v = s.pos[0, 0], s.vel[0, 0]
        R = np.linalg.norm(p)
        A = rl.amplitude(2.0, R, cfg)
        ref = beat_sample(cfg, R, p @ v / R, np.arctan2(p[0], p[1]), k, m, n, A=A,
                          phi=np.arctan2(p[2], np.hypot(p[0], p[1])), phase0=0.7,
                          posz=cfg.virtual_pos_z[k])
        assert cube[k, m, n] == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_noise_variance_and_real_mode(rng):
    cfg = rl.automotive_77ghz().replace(n_chirps=32)
    c = rl.synthesize_cube(cfg, [], rng=rng)
    assert np.var(c) == pytest.approx(1.0, rel=0.02)
    assert np.mean(c.real ** 2) == pytest.approx(0.5, rel=0.02)
    cr = rl.synthesize_cube(cfg.replace(iq_mode="real"), [], rng=np.random.default_rng(1234))
    c2 = rl.synthesize_cube(cfg, [], rng=np.random.default_rng(1234))
    assert np.isrealobj(cr)
    np.testing.assert_allclose(cr, c2.real)


def test_snr_amplitude_modes():
    cfg = rl.automotive_77ghz()
    assert rl.snr_per_sample_db(1.0, 10.0, cfg, "snr", 20.0, 10.0) == pytest.approx(20.0)
    assert rl.snr_per_sample_db(1.0, 20.0, cfg, "snr", 20.0, 10.0) == pytest.approx(20 - 40 * np.log10(2))
    # physical: radar equation by hand
    lam = cfg.wavelength
    pr = 10 ** ((12 - 30) / 10) * 10 ** 2 * lam ** 2 * 1.0 / ((4 * np.pi) ** 3 * 50.0 ** 4 * 10 ** 0.3)
    pn = 1.380649e-23 * 290 * 10 ** 1.5 * cfg.fs
    assert rl.snr_per_sample_db(1.0, 50.0, cfg, "physical") == pytest.approx(10 * np.log10(pr / pn))


def test_chunking_invariance():
    cfg = rl.automotive_77ghz().replace(n_chirps=8, n_samples=64)
    sc = rl.static_clutter(50, 5, 40, rng=3)
    a = rl.synthesize_cube(cfg, sc, noise=False)
    b = rl.synthesize_cube(cfg, sc, noise=False, max_chunk_elems=8 * 64 * 7)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def test_extended_targets_shapes():
    t = np.linspace(0, 0.1, 7)
    for tg in [rl.Car((0, 20, 0), (0, 10, 0), wheels=True), rl.Pedestrian((1, 10, 0)),
               rl.Drone((0, 30, 5)), rl.Bird((0, 50, 10)), rl.static_clutter(10, 5, 50)]:
        s = tg.scatterers(t)
        assert s.pos.shape == (s.n, 7, 3) and s.vel.shape == (s.n, 7, 3) and s.rcs.shape == (s.n, 7)
        # analytic velocity equals finite difference of position
        h = 1e-6
        p1, p0 = tg.scatterers(t + h).pos, tg.scatterers(t - h).pos
        np.testing.assert_allclose((p1 - p0) / (2 * h), s.vel, atol=2e-3 * max(1, np.abs(s.vel).max()))
