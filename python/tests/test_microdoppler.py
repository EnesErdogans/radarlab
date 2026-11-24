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
    y = dsp.slow_time(frames, slice(rb - 1, rb + 2), channels=0, cfg=cfg)
    return cfg, dsp.spectrogram(y, 1 / cfg.te, cfg.wavelength, nperseg=128)


def test_blade_tip_doppler_extent():
    """Max micro-Doppler = 2·v_tip/λ = 4π·f_rot·L/λ, within one spectrogram bin."""
    L, rpm = 0.12, 6000.0
    dr = rl.Drone((0, 15, 0), n_rotors=1, n_blades=2, rpm=rpm, blade_length=L,
                  rotor_offsets=[[0, 0, 0]], body_rcs=0.0, blade_rcs=0.05)
    cfg, sp = _spec(dr)
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
