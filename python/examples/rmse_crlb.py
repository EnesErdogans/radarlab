"""RMSE of ML range/velocity/angle estimates vs the Cramér-Rao bound."""
import numpy as np

from _style import SERIES, plt, save
import radarlab as rl
from radarlab import metrics
from radarlab.config import C0

cfg = rl.RadarConfig(fc=77e9, slope=20e12, fs=10e6, n_samples=32, chirp_period=20e-6, n_chirps=16,
                     tx_pos=[0.0], rx_pos=[0, .5, 1, 1.5], adc_start_time=1e-6, idle_time=2e-6, frame_period=1e-3)
R0, th, v = 5.3, np.deg2rad(10.0), 0.0
tg = rl.PointTarget.from_polar(R0, 10.0, v)
w_true = np.array([-np.pi * np.sin(th), 0.0, 2 * np.pi * cfg.slope * 2 * R0 / C0 / cfg.fs])
scale = np.array([1 / (np.pi * np.cos(th)), cfg.wavelength / (4 * np.pi * cfg.te), C0 * cfg.fs / (4 * np.pi * cfg.slope)])
snrs = np.arange(-25, 6, 3.0)
rng = np.random.default_rng(0)
rm = []
for s in snrs:
    clean = rl.synthesize_cube(cfg, [tg], noise=False, snr_ref_db=s, ref_range=R0)
    E = np.array([metrics.ml_tone_3d(clean + rl.complex_noise(clean.shape, rng)) for _ in range(150)])
    err = (E - w_true + np.pi) % (2 * np.pi) - np.pi
    rm.append(np.sqrt(np.mean((err * scale) ** 2, 0)))
rm = np.array(rm)
crb = np.array([[metrics.crlb_angle(cfg, s, th), metrics.crlb_velocity(cfg, s), metrics.crlb_range(cfg, s)] for s in snrs])
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for i, (name, unit, f) in enumerate([("angle", "deg", np.rad2deg(1)), ("velocity", "m/s", 1), ("range", "m", 1)]):
    ax[i].semilogy(snrs, rm[:, i] * f, "o", color=SERIES[0], label="ML RMSE")
    ax[i].semilogy(snrs, crb[:, i] * f, "-", color=SERIES[1], label="√CRLB")
    ax[i].set(xlabel="per-sample SNR [dB]", ylabel=f"{name} error [{unit}]", title=name.capitalize())
    ax[i].legend(frameon=False)
save(fig, "rmse_vs_crlb.png")
