"""Run the validation matrix V1–V11 (theory validations and consistency checks) and write python/examples/out/validation_report.md.

Usage: python validation_report.py [--quick]
The default is the paper-grade Monte-Carlo size (a few minutes); --quick uses the unit-test sizes.
"""
import argparse
import time

import numpy as np

from _style import INK, MUTED, OUT, SERIES, plt, save
from radarlab import metrics, validation as V


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return esc(str(v))


def esc(s):
    """Escape pipes so criteria such as |log10(…)| do not break Markdown tables."""
    return str(s).replace("|", "\\|")


def table(rows):
    """Markdown table over the *union* of the row keys (first-seen order), so rows
    with a different schema (e.g. V7 ghost, V8 wing-beat) are not blanked."""
    if not rows:
        return ""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    out += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in rows]
    return "\n".join(out)


def fig_v2(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, K in zip(axes, (1, 8)):
        for col, m in zip(SERIES, (0, 1, 3)):
            c = r.curves[f"K{K}_SW{m}"]
            sd = np.linspace(min(c["snr_db"]), max(c["snr_db"]), 60)
            ax.plot(sd, metrics.pd_swerling(m, 10 ** (sd / 10), 1e-4, K), "-", color=col, label=f"SW{m} theory")
            ax.plot(c["snr_db"], c["sim"], "o", color=col, mfc="none", label=f"SW{m} sim")
        ax.set(title=f"K = {K}, Pfa = 1e-4", xlabel="per-channel SNR χ [dB]")
    axes[0].set_ylabel("Pd")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    return save(fig, "val_v2_pd.png")


def fig_v3(r):
    s = np.array(r.curves["snr_db"])
    rm, cr = np.array(r.curves["rmse"]), np.array(r.curves["crlb"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for i, (ax, name, unit, f) in enumerate(zip(axes, ("range", "velocity", "angle"), ("m", "m/s", "deg"),
                                                (1, 1, np.rad2deg(1)))):
        ax.semilogy(s, cr[:, i] * f, "-", color=SERIES[1], label="√CRLB")
        ax.semilogy(s, rm[:, i] * f, "o", color=SERIES[0], label="ML RMSE")
        ax.set(title=name.capitalize(), xlabel="per-sample SNR [dB]", ylabel=f"error [{unit}]")
        ax.legend(frameon=False)
    return save(fig, "val_v3_crlb.png")


def fig_v4(r):
    f = np.array(r.curves["f"]) / 1e6
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(f, 10 * np.log10(r.curves["L"]), color=MUTED, lw=1, label="L(f)")
    ax.plot(f, 10 * np.log10(r.curves["psd_theory"]), color=SERIES[1], label="L(f)·4sin²(πfτ) (aliased)")
    ax.plot(f, 10 * np.log10(r.curves["psd_sim"]), "o", ms=3, color=SERIES[0], label="simulated residual")
    ax.set(xlabel="frequency [MHz]", ylabel="phase PSD [dB rad²/Hz]", title="V4: range-correlated phase noise")
    ax.legend(frameon=False, fontsize=8)
    return save(fig, "val_v4_phase_noise.png")


def fig_v5(r):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    d = np.array(r.curves["d"])
    ax.plot(d, 20 * np.log10(np.maximum(r.curves["theory"], 1e-3)), color=SERIES[1], label="4·sin²(2πh_r h_t/(λd))")
    ax.plot(d, 20 * np.log10(np.maximum(r.curves["sim"], 1e-3)), color=SERIES[0], lw=1.2, label="simulated")
    ax.set(xlabel="range d [m]", ylabel="two-ray / free-space amplitude [dB]", ylim=(-40, 15),
           title="V5: two-ray multipath (h_r = 0.5 m, h_t = 0.3 m)")
    ax.legend(frameon=False)
    return save(fig, "val_v5_multipath.png")


def fig_v8(r, rf):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for col, nb in zip(SERIES, (2, 3)):
        c = rf.curves[f"flash_nb{nb}"]
        axes[0].plot(np.array(c["t"]) * 1e3, c["env"], color=col, lw=1, label=f"N_b = {nb}")
    axes[0].set(xlim=(0, 40), xlabel="time [ms]", ylabel="|slow-time signal|", title="Blade flashes (3000 rpm, V8f)")
    axes[0].legend(frameon=False)
    b = r.curves["bird"]
    axes[1].plot(b["t"], b["centroid"], color=SERIES[0])
    axes[1].set(xlabel="time [s]", ylabel="Doppler centroid [m/s]", title="Bird wing-beat (5 Hz)")
    return save(fig, "val_v8_micro_doppler.png")


FIGS = {"V2": fig_v2, "V3": fig_v3, "V4": fig_v4, "V5": fig_v5}
QUICK_ARG = ("V1", "V1c", "V2", "V3", "V4", "V5", "V8", "V8f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    results, times = {}, {}
    for vid, fn in V.ALL.items():
        t0 = time.time()
        r = fn(quick=quick) if vid in QUICK_ARG else fn()
        times[vid] = time.time() - t0
        results[vid] = r
        print(f"{vid}: {'PASS' if r.passed else 'FAIL'} ({times[vid]:.1f} s)")
    theory = [r for r in results.values() if r.kind == "theory"]
    cons = [r for r in results.values() if r.kind == "consistency"]
    n_tp, n_cp = sum(r.passed for r in theory), sum(r.passed for r in cons)

    def summary(rs):
        out = ["| ID | Check | Criterion | Result | Time |", "|---|---|---|---|---|"]
        out += [f"| {r.id} | {esc(r.title)} | {esc(r.criterion)} | {'PASS' if r.passed else '**FAIL**'} | "
                f"{times[r.id]:.0f} s |" for r in rs]
        return out

    md = ["# RadarLab validation report (Python reference)", "",
          f"Mode: {'quick (unit-test sizes)' if quick else 'full (paper-grade Monte-Carlo sizes)'}. "
          "Generated by `python/examples/validation_report.py`; specifications in "
          "`docs/literature/fidelity_requirements.md` §15. Convention version 2.", "",
          f"**{n_tp} of {len(theory)} theory validations and {n_cp} of {len(cons)} consistency checks pass.**", "",
          "## Validation against independent closed-form theory", "",
          "The simulator and processing output are compared with closed-form results derived independently "
          "of the code (detection theory, estimation bounds, phase-noise spectra, propagation geometry, "
          "kinematics).", ""] + summary(theory) + [
          "", "## Implementation consistency checks", "",
          "Here the reference is produced by the same model or code: V6 is an algebraic identity of the IQ "
          "model, V8f holds by construction of a strictly periodic rotor, V9 uses the same t·Tc timing in the "
          "synthesiser and the compensator, V10 compares stored golden files with a fresh export by the same "
          "code, and V11 uses a DCA1000 writer and reader built on the same byte-layout assumption. These "
          "checks catch regressions and wiring errors, but they cannot detect a shared modelling error.", ""
          ] + summary(cons) + [
          "", "V10 checks the Python side only (stored golden files equal a fresh export). The TypeScript side "
          "is checked by the web test suite against the same files. For V11, public raw captures were not "
          "available in the sandbox. The closed loop .cfg → sim → int16 .bin → read → process is checked "
          "instead.", ""]
    body = []
    for vid, r in results.items():
        sec = [f"## {r.id}: {r.title}", "",
               f"Type: {'theory validation' if r.kind == 'theory' else 'consistency check'}. "
               f"Criterion: {r.criterion}. Result: **{'PASS' if r.passed else 'FAIL'}**.", ""]
        if r.note:
            sec += [r.note, ""]
        sec += [table(r.rows), ""]
        if vid in FIGS and r.curves:
            p = FIGS[vid](r)
            sec += [f"![{vid}]({p.name})", ""]
        if vid == "V8" and r.curves and results.get("V8f") and results["V8f"].curves:
            p = fig_v8(r, results["V8f"])
            sec += [f"![{vid}]({p.name})", ""]
        body += sec
    (OUT / "validation_report.md").write_text("\n".join(md + body) + "\n")
    print("wrote", OUT / "validation_report.md")


if __name__ == "__main__":
    main()
