import numpy as np
import pytest

import radarlab as rl
from radarlab import metrics, dsp
from radarlab.config import C0


def test_crlb_unit_conversions():
    cfg = rl.automotive_77ghz()
    K, M, N = cfg.n_virtual, cfg.n_chirps, cfg.n_samples
    eta = 10.0
    s = np.sqrt(6 / (eta * K * M * N * (N ** 2 - 1)))
    assert metrics.crlb_range(cfg, 10.0) == pytest.approx(C0 * cfg.fs / (4 * np.pi * cfg.slope) * s)
    # ULA: general-array angle CRLB equals the 1-D frequency formula
    sa = np.sqrt(6 / (eta * K * M * N * (K ** 2 - 1)))
    assert metrics.crlb_angle(cfg, 10.0) == pytest.approx(sa / (2 * np.pi * 0.5))


def test_rmse_approaches_crlb_at_high_snr():
    cfg = rl.RadarConfig(fc=77e9, slope=20e12, fs=10e6, n_samples=32, chirp_period=20e-6, n_chirps=16,
                         tx_pos=[0.0], rx_pos=[0, .5, 1, 1.5], adc_start_time=1e-6, idle_time=2e-6,
                         frame_period=1e-3)
    R0, th_deg, v, snr_db = 5.3, 10.0, 1.0, 0.0
    tg = rl.PointTarget.from_polar(R0, th_deg, v)
    clean = rl.synthesize_cube(cfg, [tg], noise=False, snr_ref_db=snr_db, ref_range=R0)
    rng = np.random.default_rng(7)
    E = np.array([metrics.ml_tone_3d(clean + rl.complex_noise(clean.shape, rng)) for _ in range(250)])
    # model-exact frequencies at the frame / ADC-window centre
    Rm = R0 + v * 0.5 * (cfg.n_chirps - 1) * cfg.te
    tfc = cfg.adc_start_time + (cfg.n_samples - 1) / 2 / cfg.fs
    w_k = 2 * np.pi * 0.5 * np.sin(np.deg2rad(th_deg))
    w_m = 2 * np.pi * 2 * v * cfg.te * (cfg.fc + cfg.slope * tfc) / C0
    w_n = 2 * np.pi * (cfg.slope * 2 * Rm / C0 + 2 * v * cfg.fc / C0) / cfg.fs
    err = (E - [w_k, w_m, w_n] + np.pi) % (2 * np.pi) - np.pi
    # physical units
    rmse_angle = np.sqrt(np.mean((err[:, 0] / (2 * np.pi * 0.5 * np.cos(np.deg2rad(th_deg)))) ** 2))
    rmse_vel = np.sqrt(np.mean((err[:, 1] * cfg.wavelength / (4 * np.pi * cfg.te)) ** 2))
    rmse_rng = np.sqrt(np.mean((err[:, 2] * C0 * cfg.fs / (4 * np.pi * cfg.slope)) ** 2))
    snr_eff = rl.snr_per_sample_db(1.0, Rm, cfg, "snr", snr_db, R0)
    for rm, cr in [(rmse_rng, metrics.crlb_range(cfg, snr_eff)),
                   (rmse_vel, metrics.crlb_velocity(cfg, snr_eff)),
                   (rmse_angle, metrics.crlb_angle(cfg, snr_eff, np.deg2rad(th_deg)))]:
        assert abs(metrics.ratio_db(rm, cr)) < 1.5


def test_pd_closed_forms():
    snr = 10 ** (13 / 10)
    assert metrics.pd_swerling(1, snr, 1e-6, 1) == pytest.approx(1e-6 ** (1 / (1 + snr)))
    # integral form for n=1 equals closed form
    pdf = lambda s: np.exp(-s)
    assert metrics._pd_fluct(snr, 1e-6, 1, pdf) == pytest.approx(1e-6 ** (1 / (1 + snr)), rel=1e-6)
    assert metrics.pd_swerling0(0.0, 1e-3, 4) == pytest.approx(1e-3, rel=1e-6)


@pytest.mark.parametrize("model", [0, 1])
def test_pd_monte_carlo(model, rng):
    n, pfa, snr = 4, 1e-3, 10 ** (5 / 10)
    T = metrics.threshold_noncoherent(pfa, n)
    trials = 40000
    amp = np.sqrt(snr) * np.ones(trials)
    if model == 1:
        amp = amp * np.abs(rl.complex_noise((trials,), rng))
    x = amp[:, None] + rl.complex_noise((trials, n), rng)
    pd_mc = np.mean(np.sum(np.abs(x) ** 2, 1) > T)
    assert pd_mc == pytest.approx(float(metrics.pd_swerling(model, snr, pfa, n)), abs=0.015)


def test_empirical_helpers():
    m = np.zeros((10, 10), bool)
    m[1, 1] = m[2, 2] = True
    ex = np.zeros_like(m)
    ex[1, 1] = True
    assert metrics.empirical_pfa(m, ex) == pytest.approx(1 / 99)
    assert metrics.empirical_pd(np.stack([m, ~m]), [[1, 1]]) == 0.5
