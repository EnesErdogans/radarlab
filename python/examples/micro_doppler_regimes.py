"""Quadrotor micro-Doppler in three sampling regimes (after Rahman & Robertson 2018).

(a) 24 GHz, chirp rate 50 kHz: blade Doppler (±12 kHz) unaliased; flashes and
    the ±v_tip envelope are visible.
(b) K-band 24 GHz, chirp repetition 4.25 kHz (Rahman & Robertson's K-band
    CRF): blade Doppler aliases many times and folds into the ±2.1 kHz band.
    A long STFT window resolves the rotor (HERM-like) lines.
(c) W-band 94 GHz, 20 µs chirps: v_max ≈ 40 m/s < v_tip, so the flashes are
    still resolved in time but their Doppler extent wraps.
Four rotors with 3 % rpm spread (fidelity §12.1) give irregular flash spacing.
"""
import numpy as np

from _style import OUT, SEQ, plt
import radarlab as rl
from radarlab import dsp


def regime(cfg: rl.RadarConfig, duration: float, nperseg: int, drone: rl.Drone, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_fr = int(np.ceil(duration / cfg.frame_period))
    frames = [dsp.range_fft(rl.synthesize_cube(cfg, drone, i * cfg.frame_period, snr_ref_db=50, rng=rng))
              for i in range(n_fr)]
    rb = int(round(np.linalg.norm(drone.center([0.0])[0]) / cfg.range_resolution))
    y = dsp.slow_time(frames, rb, 0, cfg)
    return dsp.spectrogram(y, 1 / cfg.te, cfg.wavelength, nperseg=nperseg, noverlap=nperseg * 3 // 4)


def chirp_cfg(fc, tc, n, fs, m):
    slope = 250e6 / (n / fs)
    return rl.RadarConfig(fc=fc, slope=slope, fs=fs, n_samples=n, chirp_period=tc, n_chirps=m, tx_pos=[0.0],
                          rx_pos=[0.0], adc_start_time=1e-6, idle_time=tc - n / fs - 2e-6, frame_period=m * tc)


cases = [
    ("(a) 24 GHz, 50 kHz chirp rate (unaliased)", chirp_cfg(24e9, 20e-6, 64, 5e6, 256), 0.12, 128),
    ("(b) 24 GHz, CRF 4.25 kHz (aliased, long window)", chirp_cfg(24e9, 1 / 4250, 64, 1e6, 128), 0.6, 512),
    ("(c) 94 GHz, 20 µs chirps (flashes, wrapped)", chirp_cfg(94e9, 20e-6, 32, 5e6, 256), 0.12, 128),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
for ax, (title, cfg, dur, nps) in zip(axes, cases):
    drone = rl.Drone((0, 12, 0), rpm=5400, rpm_jitter=0.03, design_wavelength=cfg.wavelength, seed=1)
    s = regime(cfg, dur, nps, drone)
    db = s.power_db - s.power_db.max()
    im = ax.pcolormesh(s.times * 1e3, s.freq / 1e3, db, cmap=SEQ, vmin=-45, vmax=0, shading="auto")
    ax.grid(False)
    f_tip = 2 * drone.tip_speed.max() / cfg.wavelength / 1e3
    ax.set(title=title, xlabel="time [ms]", ylabel="Doppler [kHz]")
    ax.text(0.02, 0.97, f"f_tip = ±{f_tip:.1f} kHz\nPRF/2 = {0.5 / cfg.te / 1e3:.2f} kHz",
            transform=ax.transAxes, va="top", fontsize=8, color="#0b0b0b",
            bbox=dict(fc="#fcfcfb", ec="none", alpha=0.85))
fig.colorbar(im, ax=axes, label="relative power [dB]", shrink=0.9)
p = OUT / "micro_doppler_regimes.png"
fig.savefig(p)
print("wrote", p)
