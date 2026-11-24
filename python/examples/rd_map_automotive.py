"""Range-Doppler map and point cloud of an automotive scene (77 GHz, 2TX x 4RX)."""
import numpy as np

from _style import SEQ, SERIES, plt, save
import radarlab as rl

cfg = rl.automotive_77ghz()
scene = rl.Scene([
    rl.Car((-3.5, 40.0, 0), (0, -8.0, 0), wheels=True),
    rl.Car((3.5, 65.0, 0), (0, 5.0, 0)),
    rl.Pedestrian((6.0, 15.0, 0), (-1.3, 0, 0)),
    rl.PointTarget((-8.0, 25.0, 0.5), rcs=5.0),
    rl.static_clutter(150, 5, 95, -50, 50, rcs_mean=0.3, rng=2),
])
cube = rl.simulate_frame(cfg, scene, rng=1)
res = rl.process_frame(cube, cfg, rl.ProcessingParams(cfar_kind="os"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
v, r = cfg.velocity_axis(), cfg.range_axis()
im = ax[0].pcolormesh(v, r, res.power_db, cmap=SEQ, shading="auto",
                      vmin=np.percentile(res.power_db, 50), vmax=res.power_db.max())
ax[0].grid(False)
fig.colorbar(im, ax=ax[0], label="power [dB]")
p = res.points
ax[0].plot(p["velocity"], p["range"], "o", mfc="none", mec=SERIES[1], mew=1.5, label="OS-CFAR clusters")
ax[0].set(xlabel="radial velocity [m/s]", ylabel="range [m]", title="Range-Doppler map")
ax[0].legend(loc="upper right", frameon=False)
ax[1].scatter(p["x"], p["y"], c=SERIES[0], s=24, label="point cloud")
truth = scene.truth(0.0)[:4]
ax[1].plot(truth[:, 0], truth[:, 1], "x", color=SERIES[1], mew=2, ms=9, label="target centres")
ax[1].set(xlabel="x [m]", ylabel="y [m]", title="Detections (bird's-eye)", xlim=(-30, 30), ylim=(0, 100))
ax[1].legend(frameon=False)
save(fig, "rd_map_automotive.png")
