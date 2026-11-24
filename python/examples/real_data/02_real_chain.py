"""Run the standard radarlab chain (process_frame: Hann range/Doppler FFT,
CA-CFAR Pfa 1e-5 with exact Gamma(8) α, connected clustering, TDM-compensated
FFT DoA, point cloud) on all 30 frames of the three real UW captures.

Writes results/02_real.json (noise statistics, empirical Pfa, dominant-target
tracks, RD summary) and fig/rd_real.png, fig/pointcloud_real.png.

    python -I python/examples/real_data/02_real_chain.py [--data DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir
from common import MUTED, SERIES, analyse, load, plt, process_all, rd_figure, save_json, savefig  # noqa: E402

def main() -> None:
    out, per_scene = {}, {}
    cfg = None
    for scene in ("pedestrian", "cyclist", "car"):
        cubes, cfg = load(scene)
        results = process_all(cubes, cfg)
        a = analyse(results, cfg, scene)
        a["dc_per_channel"] = cubes.mean(axis=(0, 2, 3))           # complex, file units
        a["dc_per_channel"] = np.stack([a["dc_per_channel"].real, a["dc_per_channel"].imag], -1)
        out[scene] = a
        per_scene[scene] = (results, a)
        s = a["track_summary"]
        print(f"{scene}: σ={np.sqrt(a['noise']['sigma2']):.2f} LSB  R={s['range_mean']:.2f} v={s['velocity_median']:+.2f} "
              f"az={s['azimuth_median']:+.1f}  peak={s['peak_db_median']:.1f} dB  KS p={a['noise']['ks_p']:.2g} "
              f"Pfa(1e-4)×{a['pfa']['0.0001']['ratio']:.2f}  det/frame={a['rd']['det_per_frame']:.1f}")
    save_json("02_real", out)
    rd_figure(per_scene, cfg, "rd_real.png", "real:")

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (scene, (results, a)) in zip(axs, per_scene.items()):
        pts = np.concatenate([r.points for r in results])
        pts = pts[pts["range_cell"] < cfg.n_range_bins // 2]
        mv = np.abs(pts["velocity"]) > 0.2
        ax.scatter(pts["x"][~mv], pts["y"][~mv], s=5, color=MUTED, alpha=0.4, label="|v| ≤ 0.2 m/s")
        ax.scatter(pts["x"][mv], pts["y"][mv], s=7, color=SERIES[0], alpha=0.7, label="|v| > 0.2 m/s")
        ax.set_xlim(-10, 10), ax.set_ylim(0, 14.5), ax.set_aspect("equal")
        ax.set_title(f"real {scene}: 30 frames"), ax.set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].legend(fontsize=8, loc="upper left")
    savefig(fig, "pointcloud_real.png")


if __name__ == "__main__":
    main()
