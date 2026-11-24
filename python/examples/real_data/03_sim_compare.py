"""Fit the simulated counterparts to the real captures and compare them with
the identical processing chain (``analyse`` in common.py), for the ideal
simulation and for each impairment model switched on alone and all together.

Writes results/03_fit.json, results/03_compare.json and fig/noise_floor.png,
fig/noise_qq.png, fig/rd_sim.png, fig/tracks.png.

    python -I python/examples/real_data/03_sim_compare.py [--data DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir
from common import (MUTED, SERIES, analyse, load, load_json, plt, process_all, rd_figure,  # noqa: E402
                    save_json, savefig)
from simbuild import VARIANTS, fit_all, simulate  # noqa: E402

KEEP = ("noise", "noise_incl_zero_doppler", "pfa", "pfa_incl_zero_doppler", "track_summary",
        "array_coherence_median", "rd", "target_bin")


def slim(a: dict) -> dict:
    return {k: a[k] for k in KEEP}


def main() -> None:
    real = load_json("02_real")
    fit = fit_all(real)
    save_json("03_fit", fit)
    cfg = load("car")[1]
    comp: dict = {"real": {s: slim(real[s]) for s in ("pedestrian", "cyclist", "car")}}
    keep_full = {}
    for variant in VARIANTS:
        comp[variant] = {}
        for i, scene in enumerate(("pedestrian", "cyclist", "car")):
            cubes = simulate(scene, cfg, fit, variant, seed=1000 + i)
            res = process_all(cubes, cfg)
            a = analyse(res, cfg, scene)
            comp[variant][scene] = slim(a)
            comp[variant][scene]["noise_floor_profile_db"] = a["noise_floor_profile_db"]
            if variant in ("ideal", "all"):
                keep_full[(variant, scene)] = (res, a)
            ts = a["track_summary"]
            print(f"{variant:13s} {scene:10s} R={ts['range_mean']:.2f} v={ts['velocity_median']:+.2f} "
                  f"az={ts['azimuth_median']:+.1f} peak={ts['peak_db_median']:.1f} coh={a['array_coherence_median']:.3f} "
                  f"KS p={a['noise']['ks_p']:.2g} Pfa1e-4×{a['pfa']['0.0001']['ratio']:.2f} "
                  f"det/fr={a['rd']['det_per_frame']:.1f} zd={a['rd']['det_zero_doppler_per_frame']:.1f} "
                  f"dc={a['rd']['det_near_dc_per_frame']:.1f}")
    # noise-floor shape error per variant (vs real, same scene), excluding the DC bins |b| <= 2
    nb = cfg.n_range_bins
    use = np.minimum(np.arange(nb), nb - np.arange(nb)) > 2
    for variant in VARIANTS:
        for scene in ("pedestrian", "cyclist", "car"):
            d = np.asarray(comp[variant][scene]["noise_floor_profile_db"]) - np.asarray(real[scene]["noise_floor_profile_db"])
            comp[variant][scene]["noise_floor_rms_err_db"] = float(np.sqrt(np.mean(d[use] ** 2)))
    save_json("03_compare", comp)

    # ---- figures
    rd_figure({s: keep_full[("all", s)] for s in ("pedestrian", "cyclist", "car")}, cfg, "rd_sim.png",
              "sim all:")

    N = cfg.n_range_bins
    rr = (np.arange(N) - N // 2) * cfg.range_resolution
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    sh = np.fft.fftshift
    ax.plot(rr, sh(real["pedestrian"]["noise_floor_profile_db"]), color=INK_REAL, lw=1.6, label="real (pedestrian)")
    ax.plot(rr, sh(comp["ideal"]["pedestrian"]["noise_floor_profile_db"]), color=SERIES[0], lw=1.1, label="sim ideal")
    ax.plot(rr, sh(comp["+hpf"]["pedestrian"]["noise_floor_profile_db"]), color=SERIES[1], lw=1.1,
            label=f"sim + IF HPF ({fit['hpf_hz']/1e3:.0f} kHz, order {fit['hpf_order']})")
    ax.set_xlabel("range [m] (negative = negative beat frequency)")
    ax.set_ylabel("median noise floor [dB re mid-band]")
    ax.set_ylim(-14, 4)
    ax.legend(fontsize=8, loc="lower left")
    savefig(fig, "noise_floor.png")

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    q = real["car"]["qq"]
    th = np.asarray(q["theory"])
    ax.plot(th, th, color=MUTED, lw=1, label="Gamma(8) (theory)")
    ax.plot(th, q["emp"], ".", ms=3, color=SERIES[0], label=f"real, |v-bin|>3 (KS p={real['car']['noise']['ks_p']:.2f})")
    ax.plot(th, q["emp_incl_zd"], ".", ms=3, color=SERIES[1],
            label=f"real, incl. zero Doppler (p={real['car']['noise_incl_zero_doppler']['ks_p']:.1g})")
    ks = comp["ideal"]["car"]["noise"]
    ax.set_title(f"noise P/mean, car scene (sim ideal: KS p={ks['ks_p']:.2f})", fontsize=9)
    ax.set_xlabel("Gamma(8)/8 quantile"), ax.set_ylabel("empirical quantile")
    ax.legend(fontsize=7, loc="upper left")
    savefig(fig, "noise_qq.png")

    fig, axs = plt.subplots(3, 3, figsize=(11, 6.5), sharex=True)
    for j, scene in enumerate(("pedestrian", "cyclist", "car")):
        tr_r = real[scene]["track"]
        tr_s = keep_full[("ideal", scene)][1]["track"]
        tr_a = keep_full[("all", scene)][1]["track"]
        for i, (key, lab) in enumerate((("range", "range [m]"), ("velocity", "v [m/s]"), ("azimuth_deg", "azimuth [deg]"))):
            ax = axs[i, j]
            ax.plot(tr_r["t"], tr_r[key], "o", ms=4, color=INK_REAL, label="real")
            ax.plot(tr_s["t"], tr_s[key], "s", ms=3, color=SERIES[0], label="sim ideal")
            ax.plot(tr_a["t"], tr_a[key], "^", ms=3, color=SERIES[1], label="sim all")
            if i == 0:
                ax.set_title(f"{scene}: strongest detection")
            if j == 0:
                ax.set_ylabel(lab)
            if i == 2:
                ax.set_xlabel("time [s]")
    axs[0, 0].legend(fontsize=7)
    savefig(fig, "tracks.png")


INK_REAL = "#0b0b0b"

if __name__ == "__main__":
    main()
