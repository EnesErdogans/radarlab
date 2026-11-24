"""Micro-Doppler (STFT over slow time across the 30 frames) for the real
pedestrian and cyclist and their simulated counterparts (ideal and with all
impairments), plus signature features.

Writes results/04_micro_doppler.json and fig/micro_doppler.png.
Requires results/02_real.json and results/03_fit.json.

    python -I python/examples/real_data/04_micro_doppler.py [--data DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir
from common import FIG, SEQ, load, load_json, plt, save_json  # noqa: E402
from mdfeat import md_features, md_spectrogram  # noqa: E402
from simbuild import MD_BAND, make_target, simulate  # noqa: E402

FEATS = ("torso_velocity_median", "torso_accel_fit", "extent_median", "extent_p95", "v_min_p5", "v_max_p95",
         "bandwidth_rms_median", "limb_envelope_period_s", "ac_peak")


def main() -> None:
    real = load_json("02_real")
    fit = load_json("03_fit")
    out: dict = {}
    panels = []
    for i, scene in enumerate(("pedestrian", "cyclist")):
        cubes, cfg = load(scene)
        cubes = cubes / np.sqrt(real[scene]["noise"]["sigma2"])
        srcs = {"real": cubes,
                "sim ideal": simulate(scene, cfg, fit, "ideal", seed=2000 + i),
                "sim all": simulate(scene, cfg, fit, "all", seed=2000 + i)}
        out[scene] = {}
        for lab, c in srcs.items():
            spec, tr, ok = md_spectrogram(c, cfg, MD_BAND[scene])
            f = md_features(spec, tr, ok)
            f.pop("series")
            out[scene][lab] = {k: f[k] for k in FEATS}
            panels.append((scene, lab, spec, tr, ok, f["noise_floor"]))
        tgt = make_target(scene, fit)
        if scene == "pedestrian":
            out[scene]["model"] = {"gait_cycle_hz_boulic": float(tgt.f_gait),
                                   "gait_cycle_period_s": float(1 / tgt.f_gait) if tgt.f_gait else None,
                                   "step_period_s": float(0.5 / tgt.f_gait) if tgt.f_gait else None}
        else:
            out[scene]["model"] = {"crank_hz_at_end": float((tgt.v_h + tgt.a_h * 1.0) / tgt.rw / tgt.gear_ratio / (2 * np.pi)),
                                   "wheel_top_velocity_at_end": float(-2 * (tgt.v_h + tgt.a_h * 1.0))}
        for lab in srcs:
            print(scene, lab, {k: round(v, 3) for k, v in out[scene][lab].items()})
    save_json("04_micro_doppler", out)

    fig, axs = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, (scene, lab, spec, tr, ok, nf) in zip(axs.ravel(), panels):
        img = np.where(ok[None], 10 * np.log10(spec.power / nf), np.nan)
        ax.pcolormesh(tr, spec.velocity, img, cmap=SEQ, vmin=0, vmax=35, shading="nearest", rasterized=True)
        ax.set_ylim(-3, 2)
        ax.set_title(f"{scene}: {lab}", fontsize=10)
    for ax in axs[:, 0]:
        ax.set_ylabel("radial velocity [m/s]")
    for ax in axs[1]:
        ax.set_xlabel("time [s]")
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 35))
    fig.colorbar(sm, ax=axs, label="power / noise floor [dB]", shrink=0.85)
    fig.savefig(FIG / "micro_doppler.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'micro_doppler.png'}")


if __name__ == "__main__":
    main()
