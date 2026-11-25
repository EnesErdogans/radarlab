"""SIU 2027 Fig. 4: theory validations V2 (Pd vs SNR, K = 8, Swerling 0/1/3,
fixed threshold) and V3 (RMSE / √CRLB for range, velocity, angle).

Calls radarlab.validation.v2_pd / v3_crlb in full (paper-grade) mode with
their fixed seeds, so the numbers are identical to
python/examples/out/validation_report.md.

Run: python -I python/examples/paper/fig_validation.py   (≈1.5 min)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from _paper_style import COL_W, MUTED, SERIES, dump, plt, save  # noqa: E402
from radarlab import metrics, validation as V  # noqa: E402


def main():
    r2 = V.v2_pd(quick=False)
    r3 = V.v3_crlb(quick=False)
    dump({"V2": {"passed": r2.passed, "note": r2.note, "rows": r2.rows},
          "V3": {"passed": r3.passed, "rows": r3.rows}}, "fig_validation.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W, 1.75), gridspec_kw={"wspace": 0.42})
    K = 8
    for c, mk, m in zip(SERIES, ("o", "s", "^"), (0, 1, 3)):
        cv = r2.curves[f"K{K}_SW{m}"]
        sd = np.linspace(min(cv["snr_db"]), max(cv["snr_db"]), 60)
        ax1.plot(sd, metrics.pd_swerling(m, 10 ** (sd / 10), 1e-4, K), "-", color=c, lw=1.0)
        ax1.plot(cv["snr_db"], cv["sim"], mk, color=c, mfc="none", label=f"SW{m}")
    ax1.set(xlabel="Kanal başına SNR $\\chi$ [dB]", ylabel="$P_d$", ylim=(0, 1.02))
    ax1.set_title("(a) V2: $K$=8, $P_{fa}=10^{-4}$", fontsize=7.5)
    ax1.legend(loc="lower right", handlelength=1.0)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))

    s = np.array(r3.curves["snr_db"])
    rm, cr = np.array(r3.curves["rmse"]), np.array(r3.curves["crlb"])
    for i, (c, mk, lab) in enumerate(zip(SERIES, ("o", "s", "^"), ("menzil", "hız", "açı"))):
        ax2.semilogy(s, rm[:, i] / cr[:, i], mk + "-", color=c, mfc="none", lw=0.9, label=lab)
    ax2.axhline(1, color=MUTED, lw=0.7, ls=":")
    ax2.set(xlabel="Örnek başına SNR [dB]", ylabel="RMSE / $\\sqrt{\\mathrm{CRLB}}$", ylim=(0.8, 100))
    ax2.set_title("(b) V3: ML kestirici", fontsize=7.5)
    ax2.legend(loc="upper right", handlelength=1.4)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    save(fig, "fig_validation.pdf")


if __name__ == "__main__":
    main()
