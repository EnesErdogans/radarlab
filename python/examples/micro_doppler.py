"""Micro-Doppler spectrograms: quadrotor vs flapping bird (24 GHz fast-chirp)."""
import numpy as np

from _style import OUT, SEQ, plt
import radarlab as rl
from radarlab import dsp

cfg = rl.micro_doppler_24ghz()


def spec(target, n_frames=40, seed=0):
    rng = np.random.default_rng(seed)
    frames = [dsp.range_fft(rl.synthesize_cube(cfg, target, i * cfg.frame_period, snr_ref_db=45, rng=rng))
              for i in range(n_frames)]
    rb = int(round(np.linalg.norm(target.center([0.0])[0]) / cfg.range_resolution))
    y = dsp.slow_time(frames, rb, 0, cfg)
    return dsp.spectrogram(y, 1 / cfg.te, cfg.wavelength, nperseg=128, noverlap=112)


drone = rl.Drone((0, 20, 0), (0, 1.0, 0), rpm=[5200, 5600, 5400, 5800], body_rcs=0.05, blade_rcs=0.02)
bird = rl.Bird((6.0, 18.0, 8.0), (0, 1.0, 0), flap_freq=6.0, wingspan=0.7, body_rcs=0.03, wing_rcs=0.03)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
for a, (name, tg) in zip(ax, [("Quadrotor drone", drone), ("Bird (6 Hz flapping)", bird)]):
    s = spec(tg)
    db = s.power_db - s.power_db.max()
    im = a.pcolormesh(s.times * 1e3, s.velocity, db, cmap=SEQ, vmin=-50, vmax=0, shading="auto")
    a.grid(False)
    a.set(xlabel="time [ms]", title=name)
ax[0].set_ylabel("radial velocity [m/s]")
vt = drone.tip_speed.max()
ax[0].axhline(vt, color="#5f5e58", lw=0.8, ls="--")
ax[0].axhline(-vt, color="#5f5e58", lw=0.8, ls="--")
ax[0].text(s_t0 := 2, vt + 4, "±v_tip", color="#5f5e58")
fig.colorbar(im, ax=ax, label="relative power [dB]")
fig.savefig(OUT / "micro_doppler_drone_vs_bird.png")
print("wrote", OUT / "micro_doppler_drone_vs_bird.png")
