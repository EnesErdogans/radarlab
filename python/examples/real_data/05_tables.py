"""Render the result tables of docs/real_data/report.md from results/*.json
(writes results/tables.md). No data access.

    python -I python/examples/real_data/05_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script dir
from common import RES, load_json  # noqa: E402

SC = ("pedestrian", "cyclist", "car")


def f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def table(head, rows) -> str:
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    return s + "".join("| " + " | ".join(r) + " |\n" for r in rows)


def main() -> None:
    real, fit, comp = load_json("02_real"), load_json("03_fit"), load_json("03_compare")
    md = load_json("04_micro_doppler")
    ck = load_json("01_loader_checks")
    out = []

    # --- noise and CFAR
    rows = []
    for s in SC:
        for lab, src in (("real", comp["real"][s]), ("sim ideal", comp["ideal"][s]), ("sim all", comp["all"][s])):
            n, nz = src["noise"], src["noise_incl_zero_doppler"]
            p = src["pfa"]
            rows.append([s if lab == "real" else "", lab,
                         f(np.sqrt(real[s]["noise"]["sigma2"]), 2) if lab == "real" else "1 (def.)",
                         f(n["gamma_shape_fit"]), f"{n['ks_D']:.4f} / {n['ks_p']:.2f}",
                         f"{f(nz['gamma_shape_fit'])} / {nz['ks_p']:.1g}",
                         *[f"{p[k]['ratio']:.2f} ({p[k]['count']})" for k in ("0.001", "0.0001", "1e-05")],
                         f(src["pfa_incl_zero_doppler"]["0.0001"]["ratio"]),
                         f(comp["ideal" if lab == "sim ideal" else "all"][s]["noise_floor_rms_err_db"], 1) if lab != "real" else "0"])
    out.append("### Noise floor, noise distribution, CFAR false alarms\n\n" + table(
        ["scene", "data", "σ [LSB]", "Gamma shape (K=8)", "KS D / p", "shape / KS p incl. 0-Doppler",
         "Pfa ratio @1e-3 (n)", "@1e-4 (n)", "@1e-5 (n)", "@1e-4 incl. 0-Doppler", "noise-floor shape rms err [dB] (|bin|>2)"], rows))
    ncell = comp["real"]["car"]["pfa"]["0.001"]["cells"]
    out.append(f"\nPfa ratio = empirical/nominal in the target-free region (CA-CFAR, Hann, exact Gamma(8) α; "
               f"{ncell} cells over 30 frames; n = false alarms). Hann-correlated training cells give ≈1.5× "
               f"in theory (validation V1c).\n")

    # --- dominant target
    rows = []
    for s in SC:
        for lab, src in (("real", comp["real"][s]), ("sim ideal", comp["ideal"][s]), ("sim all", comp["all"][s])):
            t = src["track_summary"]
            rows.append([s if lab == "real" else "", lab, f"{t['n_frames']}", f(t["range_median"]),
                         f(t["range_rate_theil_sen"]), f(t["velocity_median"]), f(t["velocity_iqr"]),
                         f(t["azimuth_median"], 1), f(t["azimuth_iqr"], 1),
                         f(t["peak_db_median"], 1), f(t["peak_db_iqr"], 1), f(src["array_coherence_median"], 3)])
    out.append("### Dominant target (strongest CFAR detection per frame)\n\n" + table(
        ["scene", "data", "frames found", "R median [m]", "dR/dt Theil–Sen [m/s]", "v median [m/s]", "v IQR",
         "az median [°]", "az IQR [°]", "peak [dB re noise] (fitted)", "peak IQR [dB]", "array coherence"], rows))

    # --- RD map statistics + ablation
    rows = []
    variants = ["real"] + [v for v in comp if v != "real"]
    for v in variants:
        r = [comp[v][s]["rd"] for s in SC]
        coh = [comp[v][s]["array_coherence_median"] for s in ("pedestrian", "cyclist")]
        nfe = "0" if v == "real" else f(np.mean([comp[v][s]["noise_floor_rms_err_db"] for s in SC]), 1)
        rows.append([v, f(np.mean([x["det_per_frame"] for x in r]), 1),
                     f(np.mean([x["det_zero_doppler_per_frame"] for x in r]), 1),
                     f(np.mean([x["det_near_dc_per_frame"] for x in r]), 1),
                     f(np.mean([x["zero_doppler_power_fraction"] for x in r]), 2),
                     f(np.mean([comp[v][s]["noise"]["ks_p"] for s in SC]), 2),
                     nfe, f(np.mean(coh), 3)])
    out.append("### RD-map statistics and impairment ablation (mean over the three scenes)\n\n" + table(
        ["data", "detections / frame", "at 0-Doppler (±2 bins)", "near DC (range bin ≤2 or ≥126)",
         "power fraction at 0-Doppler", "noise KS p (excl. 0-Doppler)",
         "noise-floor shape rms err [dB] (|bin|>2)", "array coherence (ped., cyc.)"], rows))

    # --- micro-Doppler
    rows = []
    for s in ("pedestrian", "cyclist"):
        for lab in ("real", "sim ideal", "sim all"):
            m = md[s][lab]
            rows.append([s if lab == "real" else "", lab, f(m["torso_velocity_median"]), f(m["torso_accel_fit"]),
                         f(m["extent_median"]), f(m["extent_p95"]), f(m["v_min_p5"]), f(m["v_max_p95"]),
                         f(m["bandwidth_rms_median"], 3), f(m["limb_envelope_period_s"]), f(m["ac_peak"])])
    out.append("### Micro-Doppler signature features\n\n" + table(
        ["scene", "data", "torso v [m/s]", "torso accel [m/s²]", "extent median [m/s]", "extent p95 [m/s]",
         "v min p5 [m/s]", "v max p95 [m/s]", "rms bandwidth [m/s]", "limb-envelope period [s]",
         "autocorr. peak"], rows))

    # --- fitted constants
    rows = [[s, f(fit[s]["sigma2_lsb2"], 1), f(fit[s]["range0"]), f(fit[s]["v0"]), f(fit[s]["accel"]),
             f(fit[s]["azimuth_deg"], 1), f(fit[s]["snr_ref_db"], 1), f(fit[s]["implied_rcs_dbsm"], 1),
             f(fit["clutter_scale"][s] * 1e4, 2)] for s in SC]
    out.append("### Fitted constants (all fitted to the real data)\n\n" + table(
        ["scene", "σ² [LSB²]", "R(t=0) [m]", "v0 [m/s]", "a [m/s²]", "az [°]", "snr_ref_db (1 m² @10 m)",
         "implied RCS [dBsm]", "clutter scale ×1e4"], rows))
    out.append(f"\nGlobal: IF HPF {fit['hpf_hz']/1e3:.0f} kHz order {fit['hpf_order']} (noise-floor rms error "
               f"{fit['no_hpf_rms_err_db']:.2f} → {fit['hpf_rms_err_db']:.2f} dB); image rejection "
               f"{fit['irr_db_measured']:.1f} dB → IQ phase {fit['iq_phase_deg']:.2f}°; channel gain "
               f"[{', '.join(f'{x:+.1f}' for x in fit['mismatch_gain_db'])}] dB, phase "
               f"[{', '.join(f'{x:+.0f}' for x in fit['mismatch_phase_deg'])}]°.\n")

    # --- loader checks
    h = ck["car"]["tx_hypotheses"]
    out.append("### Loader / ordering checks (parked car, θ ≈ −25°)\n\n" + table(
        ["hypothesis", "array coherence (median of 30 frames)", "θ̂ [°]"],
        [[k, f(v["coherence_median"], 3), f(v["theta_deg_median"], 1)] for k, v in h.items()]))
    d = ck["doppler_sign_check"]
    out.append(f"\nDoppler sign: range-rate fit {d['range_rate_fit_mps']:+.2f} m/s vs median Doppler velocity "
               f"{d['median_doppler_velocity_mps']:+.2f} m/s ({d['n_frames']} frames) → consistent: {d['consistent']}. "
               f"Positive-beat fraction of moving power: "
               + ", ".join(f"{s} {ck[s]['pos_half_fraction_moving']*100:.0f} %" for s in SC) + ".\n")
    a = ck["adc_start_sensitivity"]
    out.append("adcStartTime 0 / 6 / 12 µs (simulated target, 7 m, −0.7 m/s): range "
               + " / ".join(f"{a[k]['range']:.5f}" for k in a) + " m, velocity "
               + " / ".join(f"{a[k]['velocity']:.4f}" for k in a) + " m/s, azimuth "
               + " / ".join(f"{a[k]['azimuth_deg']:.4f}" for k in a) + "°.\n")
    (RES / "tables.md").write_text("\n".join(out))
    print(f"wrote {RES / 'tables.md'}")


if __name__ == "__main__":
    main()
