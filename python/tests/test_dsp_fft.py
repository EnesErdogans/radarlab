"""Range/Doppler/angle processing against analytic expectations."""
import numpy as np
import pytest

import radarlab as rl
from radarlab import dsp


def _small(**kw):
    base = dict(fc=77e9, slope=20e12, fs=10e6, n_samples=128, chirp_period=20e-6, n_chirps=64,
                tx_pos=[0.0], rx_pos=[0.0, 0.5, 1.0, 1.5], adc_start_time=1e-6, idle_time=2e-6,
                frame_period=10e-3)
    base.update(kw)
    return rl.RadarConfig(**base)


def _peak(P):
    return np.unravel_index(int(np.argmax(P)), P.shape)


def test_on_grid_static_peak_amplitude():
    cfg = _small()
    r0 = 37
    tg = rl.PointTarget((0, r0 * cfg.range_resolution, 0))
    cube = rl.synthesize_cube(cfg, [tg], noise=False, snr_ref_db=10)
    rf, rd = dsp.range_doppler(cube)
    P = dsp.rd_power(rd[:1])
    r, d = _peak(P)
    assert (r, d) == (r0, cfg.n_chirps // 2)
    A = rl.amplitude(1.0, r0 * cfg.range_resolution, cfg, snr_ref_db=10)
    assert abs(rd[0, r, d]) == pytest.approx(A, rel=1e-6)     # Σw normalisation


@pytest.mark.parametrize("vbins", [-20, -3, 5, 17])
def test_doppler_bin_and_sign(vbins):
    cfg = _small()
    v = vbins * cfg.velocity_resolution
    tg = rl.PointTarget.from_polar(30.0, 0.0, v)
    rf, rd = dsp.range_doppler(rl.synthesize_cube(cfg, [tg], noise=False))
    P = dsp.rd_power(rd)
    r, d = _peak(P)
    assert d == cfg.n_chirps // 2 + vbins       # receding -> bin > M/2
    rr, dd = dsp.refine_peak(P, r, d)
    assert dsp.bin_to_velocity(dd[0], cfg) == pytest.approx(v, abs=0.05 * cfg.velocity_resolution)


@pytest.mark.parametrize("frac", [0.0, 0.17, 0.33, 0.5, 0.71])
def test_off_grid_parabolic_range(frac):
    cfg = _small()
    rb = 40 + frac
    tg = rl.PointTarget((0, rb * cfg.range_resolution, 0))
    rf, rd = dsp.range_doppler(rl.synthesize_cube(cfg, [tg], noise=False))
    P = dsp.rd_power(rd)
    r, d = _peak(P)
    rr, _ = dsp.refine_peak(P, r, d)
    # Hann + log-parabolic interpolation: bias well below 0.05 bin
    assert abs(rr[0] - rb) < 0.05


def test_velocity_aliasing():
    cfg = _small()
    v = cfg.max_velocity + 5 * cfg.velocity_resolution           # beyond vmax
    tg = rl.PointTarget.from_polar(20.0, 0.0, v)
    rf, rd = dsp.range_doppler(rl.synthesize_cube(cfg, [tg], noise=False))
    r, d = _peak(dsp.rd_power(rd))
    v_alias = v - 2 * cfg.max_velocity
    assert d == cfg.n_chirps // 2 + int(round(v_alias / cfg.velocity_resolution))
    assert dsp.bin_to_velocity(d, cfg) < 0


@pytest.mark.parametrize("theta", [-35.0, 0.0, 22.0])
def test_angle_with_tdm_compensation(theta):
    cfg = rl.automotive_77ghz()
    v = 9.3
    tg = rl.PointTarget.from_polar(30.0, theta, v)
    res = rl.pipeline.process_frame(rl.synthesize_cube(cfg, [tg], noise=False), cfg)
    p = res.points[0]
    assert np.rad2deg(p["azimuth"]) == pytest.approx(theta, abs=0.5)
    # without compensation the estimate is biased for v != 0
    x = res.rd[:, p["range_cell"], p["doppler_cell"]]
    _, _, th_raw = dsp.estimate_azimuth(x, cfg)
    assert abs(np.rad2deg(th_raw) - theta) > 2.0


def test_angle_sign_convention():
    cfg = _small()
    ula = dsp.azimuth_ula(cfg)
    x = dsp.steering_vector(cfg.virtual_pos, np.deg2rad(20.0))
    spec = np.abs(dsp.angle_fft(x[:, None], ula, 64)[:, 0])
    assert np.argmax(spec) > 32


def test_range_resolution_rect():
    cfg = _small()
    dR = cfg.range_resolution

    def n_peaks(sep):
        tgs = [rl.PointTarget((0, 30.0, 0)), rl.PointTarget((0, 30.0 + sep, 0), phase0=1.0)]
        cube = rl.synthesize_cube(cfg, tgs, noise=False)
        p = np.abs(dsp.range_fft(cube[0, 0], window=None)) ** 2
        seg = p[int(30 / dR) - 4: int((30 + sep) / dR) + 5]
        thr = seg.max() * 10 ** (-10 / 10)
        pk = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > thr]
        return len(pk)

    assert n_peaks(2 * dR) == 2
    assert n_peaks(0.5 * dR) == 1


def test_processing_gain(rng):
    cfg = _small()
    snr_db = -10.0
    tg = rl.PointTarget((0, 33 * cfg.range_resolution, 0), (0, 0, 0))
    r_tg = 33 * cfg.range_resolution
    pw_sig = 10 ** (rl.snr_per_sample_db(1.0, r_tg, cfg, "snr", snr_db, 10.0) / 10)
    cube_s = rl.synthesize_cube(cfg, [tg], noise=False, snr_ref_db=snr_db)
    _, rd_s = dsp.range_doppler(cube_s[:1])
    sig = np.abs(rd_s[0, 33, cfg.n_chirps // 2]) ** 2
    noise = []
    for _ in range(4):
        _, rd_n = dsp.range_doppler(rl.complex_noise(cube_s[:1].shape, rng))
        noise.append(np.mean(np.abs(rd_n) ** 2))
    gain = 10 * np.log10(sig / np.mean(noise)) - 10 * np.log10(pw_sig)
    w_n, w_m = dsp.get_window("hann", cfg.n_samples), dsp.get_window("hann", cfg.n_chirps)
    expected = 10 * np.log10(cfg.n_samples * cfg.n_chirps) - dsp.processing_loss_db(w_n) - dsp.processing_loss_db(w_m)
    assert dsp.processing_loss_db(w_n) == pytest.approx(1.76, abs=0.01)
    assert gain == pytest.approx(expected, abs=1.0)


def test_mti_removes_static():
    cfg = _small()
    tgs = [rl.PointTarget((0, 20, 0)), rl.PointTarget.from_polar(35.0, 0.0, 4.0)]
    rf = dsp.range_fft(rl.synthesize_cube(cfg, tgs, noise=False))
    rd = dsp.doppler_fft(dsp.mti(rf))
    P = dsp.rd_power(rd)
    r, d = _peak(P)
    assert abs(r * cfg.range_resolution - 35.0) < 1.0
    assert P[int(round(20 / cfg.range_resolution)), cfg.n_chirps // 2] < 1e-8 * P.max()


def test_real_iq_processing():
    cfg = _small(iq_mode="real")
    tg = rl.PointTarget.from_polar(10.0, 0.0, 2.0)
    rf, rd = dsp.range_doppler(rl.synthesize_cube(cfg, [tg], noise=False))
    assert rd.shape[1] == cfg.n_samples // 2
    r, d = _peak(dsp.rd_power(rd))
    assert abs(r * cfg.range_resolution - 10.0) < cfg.range_resolution


def test_range_angle_map():
    cfg = rl.automotive_77ghz()
    tg = rl.PointTarget.from_polar(25.0, -20.0, 5.0)
    _, rd = dsp.range_doppler(rl.synthesize_cube(cfg, [tg], noise=False))
    ra = dsp.range_angle_map(rd, cfg, 64)
    r, a = _peak(ra)
    assert abs(r * cfg.range_resolution - 25.0) < 1.0
    assert dsp.angle_bin_to_deg(a, 64, 0.5) == pytest.approx(-20.0, abs=2.5)
