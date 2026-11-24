import numpy as np
import pytest

import radarlab as rl
from radarlab import dsp, impairments as imp


def _cfg(**kw):
    base = dict(fc=77e9, slope=20e12, fs=10e6, n_samples=128, chirp_period=20e-6, n_chirps=32,
                tx_pos=[0.0], rx_pos=[0.0, 0.5], adc_start_time=1e-6, idle_time=2e-6, frame_period=1e-3)
    base.update(kw)
    return rl.RadarConfig(**base)


def test_quantization_noise_and_clipping(rng):
    x = rng.uniform(-1, 1, 100000)
    y = imp.adc_quantize(x, bits=8, full_scale=1.0)
    lsb = 1 / 128
    assert np.var(y - x) == pytest.approx(lsb ** 2 / 12, rel=0.05)
    assert imp.adc_quantize(np.array([5.0]), 8, 1.0)[0] == pytest.approx(127 * lsb)


def test_iq_imbalance_image(rng):
    n = np.arange(256)
    x = np.exp(2j * np.pi * 20 * n / 256)
    y = imp.iq_imbalance(x, 1.0, 5.0)
    Y = np.abs(np.fft.fft(y)) ** 2
    assert 10 * np.log10(Y[20] / Y[-20]) == pytest.approx(imp.image_rejection_db(1.0, 5.0), abs=1e-6)


def test_hpf_attenuates_near_range():
    cfg = _cfg()
    near, far = rl.PointTarget((0, 1.0, 0)), rl.PointTarget((0, 40.0, 0))
    def pk(t):
        c = imp.if_filter(rl.synthesize_cube(cfg, [t], noise=False, snr_ref_db=0, ref_range=np.linalg.norm(t.p0)),
                          cfg, hpf_hz=500e3, order=2)
        return np.abs(dsp.range_fft(c)).max()
    assert pk(near) < 0.2 * pk(far)


def test_phase_noise_range_correlation(rng):
    cfg = _cfg()
    pn = imp.PhaseNoise(l0_dbc_hz=-70, loop_bw_hz=500e3)
    def skirt(R):
        t = rl.PointTarget((0, R, 0))
        c = rl.synthesize_cube(cfg, [t], noise=False, phase_noise=pn, rng=np.random.default_rng(0),
                               snr_ref_db=40, ref_range=R)
        p = np.abs(dsp.range_fft(c[0])) ** 2
        rb = int(round(R / cfg.range_resolution))
        mask = np.ones(p.shape[-1], bool)
        mask[rb - 3: rb + 4] = False
        return p[:, mask].mean() / p[:, rb].mean()
    assert skirt(60.0) > 3 * skirt(3.0)


def test_swerling_gain_statistics(rng):
    g1 = imp.swerling_gains(1, 20000, 1, 4, rng)
    assert np.mean(np.abs(g1) ** 2) == pytest.approx(1.0, rel=0.05)
    assert np.allclose(g1[:, :, 0], g1[:, :, 3])                 # per frame
    g3 = imp.swerling_gains(3, 20000, 1, 1, rng)
    assert np.var(np.abs(g3) ** 2) == pytest.approx(0.5, rel=0.1)   # chi2_4/4
    g2 = imp.swerling_gains(2, 10, 1, 4, rng)
    assert not np.allclose(g2[:, :, 0], g2[:, :, 1])               # per chirp


def test_leakage_and_two_ray():
    cfg = _cfg()
    c = imp.add_leakage(np.zeros((2, 32, 128), complex), cfg, distance=0.05, power_db=40)
    p = np.abs(dsp.range_fft(c[0, 0])) ** 2
    assert np.argmax(p) == 0
    tg = rl.PointTarget((0, 30.0, 1.0))
    f = imp.two_ray(tg, radar_height=0.5)
    s = f(np.array([0.0]))
    assert s.n == 3
    Rd, Ri = 30.0166, np.hypot(30, 2.0)
    assert np.linalg.norm(s.pos[1, 0]) == pytest.approx(0.5 * (np.linalg.norm([0, 30, 1]) + Ri))
    assert s.phase0[2] == pytest.approx(2 * np.pi)
    cube = rl.synthesize_cube(cfg, f, noise=False)
    assert np.isfinite(cube).all()


def test_fmcw_interference_burst():
    cfg = _cfg(n_chirps=64, frame_period=5e-3)
    intf = imp.FmcwInterferer(fc=77e9, slope=25e12, chirp_period=25e-6, ramp_time=20e-6, distance=20, inr_db=30)
    x = imp.fmcw_interference(cfg, intf)
    occ = np.mean(np.abs(x) > 0)
    assert 0 < occ < 0.5                          # short bursts, not continuous
    assert np.abs(x).max() == pytest.approx(10 ** 1.5)
