import numpy as np
import pytest

import radarlab as rl
from radarlab import dsp


def _spec(target, n_frames=8, snr_ref_db=60.0, seed=0):
    cfg = rl.micro_doppler_24ghz()
    rng = np.random.default_rng(seed)
    frames = [dsp.range_fft(rl.synthesize_cube(cfg, target, t_frame=i * cfg.frame_period, rng=rng,
                                               snr_ref_db=snr_ref_db)) for i in range(n_frames)]
    R = float(np.linalg.norm(target.center([0.0])[0]))
    rb = int(round(R / cfg.range_resolution))
    y = dsp.slow_time(frames, rb, channels=0, cfg=cfg)
    return cfg, dsp.spectrogram(y, 1 / cfg.te, cfg.wavelength, nperseg=128)


def test_blade_tip_doppler_extent():
    """Max micro-Doppler = 2·v_tip/λ = 4π·f_rot·L/λ, within one spectrogram bin."""
    L, rpm = 0.12, 6000.0
    dr = rl.Drone((0, 15, 0), n_rotors=1, n_blades=2, rpm=rpm, blade_length=L,
                  rotor_offsets=[[0, 0, 0]], body_rcs=0.0, blade_rcs=0.05)
    cfg, sp = _spec(dr, snr_ref_db=80.0)
    f_rot = rpm / 60
    f_max = 4 * np.pi * f_rot * L / cfg.wavelength
    df = sp.freq[1] - sp.freq[0]
    vlo, vhi = dsp.doppler_extent(sp, threshold_db=-10.0)
    to_f = 2 / cfg.wavelength
    assert abs(vhi * to_f - f_max) <= df
    assert abs(-vlo * to_f - f_max) <= df


def test_drone_vs_bird_spread():
    cfg = rl.micro_doppler_24ghz()
    drone = rl.Drone((0, 20, 0), rpm=5000, body_rcs=0.05, blade_rcs=0.02)
    bird = rl.Bird((0, 20, 0), velocity=(0, 0.5, 0), flap_freq=8.0, body_rcs=0.02)
    _, sd = _spec(drone, 6)
    _, sb = _spec(bird, 6)
    wd = np.diff(dsp.doppler_extent(sd, -25))[0]
    wb = np.diff(dsp.doppler_extent(sb, -25))[0]
    assert wd > 5 * wb
    assert wb < 10.0


@pytest.mark.parametrize("n_blades", [2, 3, 4, 5])
def test_blade_flash_rate(n_blades):
    """Flash rate N_b·f_rot (even N_b) or 2·N_b·f_rot (odd N_b); fidelity §12.1."""
    from radarlab.validation import _rotor_slow_time, flash_times
    cfg = rl.micro_doppler_24ghz()
    dr = rl.Drone((0, 15, 0), n_rotors=1, n_blades=n_blades, rpm=2400.0, rotor_offsets=[[0, 0, 0]],
                  body_rcs=0.0, blade_rcs=0.05)
    rate = float(dr.flash_rate()[0])
    f_rot = 2400 / 60
    assert rate == pytest.approx(f_rot * n_blades * (1 if n_blades % 2 == 0 else 2))
    y = _rotor_slow_time(dr, cfg, 16, noise=False)
    tf = flash_times(y, cfg.te, 0.5 / rate)
    period = np.polyfit(np.arange(len(tf)), tf, 1)[0]
    assert 1 / period == pytest.approx(rate, rel=0.01)


def test_line_blade_matches_discretised_blade():
    """Analytic sinc line-blade (a) ≈ discretised point blade (b) for L ≪ ΔR."""
    cfg = rl.micro_doppler_24ghz().replace(n_chirps=64)
    kw = dict(n_rotors=1, n_blades=2, rpm=3000.0, rotor_offsets=[[0, 0, 0]], body_rcs=0.0, blade_rcs=0.05,
              random_blade_phase=False)
    a = rl.synthesize_cube(cfg, rl.Drone((0, 15, 0), blade_model="line", **kw), noise=False)
    b = rl.synthesize_cube(cfg, rl.Drone((0, 15, 0), blade_model="points", points_per_blade=200, **kw), noise=False)
    rb = int(round(15 / cfg.range_resolution))
    ya, yb = dsp.range_fft(a)[0, :, rb], dsp.range_fft(b)[0, :, rb]
    corr = np.abs(np.vdot(ya, yb)) / (np.linalg.norm(ya) * np.linalg.norm(yb))
    assert corr > 0.99
    assert np.linalg.norm(ya) == pytest.approx(np.linalg.norm(yb), rel=0.05)
