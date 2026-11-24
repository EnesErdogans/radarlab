"""Loader sanity checks on the real UW captures: I/Q sign, TX order and spacing,
fast-time transients, sensitivity to the unpublished adcStartTime, range profiles.

    python -I python/examples/real_data/01_loader_checks.py [--data DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir
from common import (FIG, PROC, array_coherence, SERIES, MUTED, dsp, load, plt, process_all, rl, save_json, savefig,
                    uw_awr1843_config)
from radarlab.io.uw_mat import uw_decode, uw_encode


def main() -> None:
    out: dict = {}
    profiles = {}
    for scene in ("pedestrian", "cyclist", "car"):
        cubes, cfg = load(scene)
        rf = dsp.range_fft(cubes, PROC.range_window)                 # (F, K, M, N)
        static = np.abs(rf.mean(axis=2)) ** 2                          # coherent slow-time mean
        moving = np.abs(rf - rf.mean(axis=2, keepdims=True)) ** 2
        p_static, p_moving = static.mean(axis=(0, 1)), moving.mean(axis=(0, 1, 2))
        profiles[scene] = (p_static, p_moving)
        half = cfg.n_samples // 2
        dyn = (np.abs(cubes - cubes.mean(axis=2, keepdims=True)) ** 2).mean(axis=(0, 1, 2))
        o = {
            "shape": list(cubes.shape),
            "max_abs": float(np.abs(cubes).max()),
            "dc_abs_per_channel": np.abs(cubes.mean(axis=(0, 2, 3))).round(2),
            "pos_half_fraction_total": float(p_static[:half].sum() / p_static.sum()),
            "pos_half_fraction_moving": float(p_moving[:half].sum() / p_moving.sum()),
            "dyn_power_first4_over_mid": float(dyn[:4].mean() / dyn[40:90].mean()),
            "dyn_power_last4_over_mid": float(dyn[-4:].mean() / dyn[40:90].mean()),
        }
        # TX order / spacing: strongest zero-Doppler return beyond 2 m (static, no TDM issue)
        rb = int(np.argmax(p_static[10:half])) + 10
        raw = uw_encode(cubes, cfg)                                    # back to file layout
        hyp = {}
        for label, order, txp in (("order(0,1) d=2", (0, 1), [0, 2]), ("order(1,0) d=2", (1, 0), [0, 2]),
                                  ("order(0,1) d=1", (0, 1), [0, 1]), ("order(0,1) d=3", (0, 1), [0, 3])):
            c2 = uw_awr1843_config(tx_pos=np.array(txp, float))
            cub = uw_decode(raw, c2, order)
            snaps = dsp.range_fft(cub, PROC.range_window)[..., rb].mean(axis=2)   # (F, K) zero-Doppler
            coh = [array_coherence(s, c2.virtual_pos) for s in snaps]
            hyp[label] = {"coherence_median": float(np.median([c[0] for c in coh])),
                          "theta_deg_median": float(np.median([c[1] for c in coh]))}
        o["static_ref_bin"], o["static_ref_range_m"] = rb, rb * cfg.range_resolution
        o["tx_hypotheses"] = hyp
        # per-channel amplitude/phase mismatch at the static reference (relative to a plane wave)
        snap = rf[..., rb].mean(axis=(0, 2))
        _, th = array_coherence(snap, cfg.virtual_pos)
        a = np.exp(-2j * np.pi * cfg.virtual_pos * np.sin(np.deg2rad(th)))
        e = snap / a
        e /= np.mean(np.abs(e)) * np.exp(1j * np.angle(e[0]))
        o["channel_gain_db"] = (20 * np.log10(np.abs(e))).round(2)
        o["channel_phase_err_deg"] = np.rad2deg(np.angle(e)).round(1)
        out[scene] = o

    # Doppler sign vs range rate (pedestrian: walking towards the radar)
    cubes, cfg = load("pedestrian")
    res = process_all(cubes, cfg)
    t, R, V = [], [], []
    for f, r in enumerate(res):
        p = r.points[(np.abs(r.points["velocity"]) > 0.25) & (r.points["range"] > 5) & (r.points["range"] < 9)]
        if len(p):
            b = p[np.argmax(p["power_db"])]
            t.append(f * cfg.frame_period), R.append(b["range"]), V.append(b["velocity"])
    slope = float(np.polyfit(t, R, 1)[0])
    out["doppler_sign_check"] = {"range_rate_fit_mps": slope, "median_doppler_velocity_mps": float(np.median(V)),
                                 "n_frames": len(t), "consistent": bool(np.sign(slope) == np.sign(np.median(V)))}

    # adcStartTime sensitivity (simulation, identical target, t0 = 0 / 6 / 12 µs)
    sens = {}
    for t0 in (0.0, 6e-6, 12e-6):
        c = uw_awr1843_config(adc_start_time=t0)
        tgt = rl.PointTarget.from_polar(7.0, -2.0, radial_velocity=-0.7)
        cube = rl.simulate_frame(c, tgt, sim=rl.SimParams(noise=False))
        r = rl.process_frame(cube, c, PROC)
        p = r.points[np.argmax(r.points["power_db"])]
        sens[f"{t0*1e6:.0f}us"] = {"range": p["range"], "velocity": p["velocity"],
                                   "azimuth_deg": np.rad2deg(p["azimuth"]), "power_db": p["power_db"]}
    out["adc_start_sensitivity"] = sens
    save_json("01_loader_checks", out)

    # figure: range profiles (fftshifted: negative beat frequencies to the left)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    N = cfg.n_samples
    rbin = np.arange(N) - N // 2
    for ax, (scene, (ps, pm)) in zip(axs, profiles.items()):
        ax.plot(rbin * cfg.range_resolution, 10 * np.log10(np.fft.fftshift(ps)), color=SERIES[0], lw=1.2,
                label="static (slow-time mean)")
        ax.plot(rbin * cfg.range_resolution, 10 * np.log10(np.fft.fftshift(pm)), color=SERIES[1], lw=1.2,
                label="moving (mean removed)")
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_title(f"{scene} ({out[scene]['pos_half_fraction_moving']*100:.0f}% of moving power at +f)")
        ax.set_xlabel("range [m]  (negative = negative beat frequency)")
    axs[0].set_ylabel("power per range bin [dB, file units]")
    axs[0].legend(fontsize=8, loc="lower left")
    savefig(fig, "range_profiles.png")


if __name__ == "__main__":
    main()
