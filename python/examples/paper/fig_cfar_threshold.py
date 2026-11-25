"""SIU 2027 Fig. 2: cost of applying the single-channel CA-CFAR factor to a
K-channel power sum, and Monte-Carlo false-alarm rates through the real
FFT noise path (rect vs Hann).

(a) threshold error 10·log10(α_1/α_K) vs K for N = 16, 24, 32 (Pfa = 1e-4),
    from radarlab.dsp.ca_alpha (closed form, Beta(K, NK)).
(b) realised vs design Pfa, K = 8, 2-D CA-CFAR guard (1,1) train (2,2)
    (N = 40): exact Gamma(K) α with rect and Hann windows, and the naive
    single-channel α (Monte-Carlo + analytic 1 − I_{α/(N+α)}(K, NK)).

Run: python -I python/examples/paper/fig_cfar_threshold.py   (≈1–2 min)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from _paper_style import COL_W, INK, MUTED, SERIES, dump, plt, save  # noqa: E402
from radarlab import dsp, metrics  # noqa: E402
from radarlab.validation import _noise_rd  # noqa: E402
from _cfar_theory import ca_pfa_with_alpha, pd_ca, required_snr_db  # noqa: E402


def main():
    pfa = 1e-4
    Ks = np.arange(1, 17)
    Ns = (16, 24, 32)
    err = {n: [10 * np.log10(dsp.ca_alpha(n, pfa, 1) / dsp.ca_alpha(n, pfa, int(K))) for K in Ks] for n in Ns}
    real = {n: [ca_pfa_with_alpha(dsp.ca_alpha(n, pfa, 1), n, int(K)) for K in Ks] for n in Ns}
    table = []
    for K in (4, 8):
        for n in Ns:
            a1, aK = dsp.ca_alpha(n, pfa, 1), dsp.ca_alpha(n, pfa, K)
            row = {"K": K, "N": n, "alpha_exact": aK, "alpha_naive": a1,
                   "threshold_error_db": 10 * np.log10(a1 / aK),
                   "realised_pfa_naive": ca_pfa_with_alpha(a1, n, K)}
            for model in (0, 1):
                for pdt in (0.5, 0.9):
                    s_ex = required_snr_db(lambda s: pd_ca(s, aK, n, K, model), pdt)
                    s_nv = required_snr_db(lambda s: pd_ca(s, a1, n, K, model), pdt)
                    row[f"req_snr_exact_db_sw{model}_pd{pdt}"] = s_ex
                    row[f"snr_loss_db_sw{model}_pd{pdt}"] = s_nv - s_ex
            table.append(row)
            print(row)

    # ---------------- Monte-Carlo through the FFT noise path
    rng = np.random.default_rng(2027)
    K, M, Nf = 8, 64, 128
    guard, train = (1, 1), (2, 2)
    n_tr = (2 * 3 + 1) ** 2 - 9
    frames = 300
    design = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    mc = {"rect_exact": [], "hann_exact": [], "rect_naive": []}
    for win, keys in ((None, ("rect_exact", "rect_naive")), ("hann", ("hann_exact",))):
        maps = [_noise_rd((K, M, Nf), rng, win) for _ in range(frames)]
        cells = frames * M * Nf
        for key in keys:
            for p in design:
                if key.endswith("exact"):
                    fa = sum(int(dsp.cfar_2d(P, guard, train, p, "ca", n_channels=K).mask.sum()) for P in maps)
                else:
                    # same detector, α forced to the single-channel value
                    fa = 0
                    for P in maps:
                        r = dsp.cfar_2d(P, guard, train, p, "ca", n_channels=K)
                        a1 = {int(n): dsp.ca_alpha(int(n), p, 1) for n in np.unique(r.n_train)}
                        thr = np.vectorize(a1.get, otypes=[float])(r.n_train) * r.noise
                        fa += int((P > thr).sum())
                mc[key].append({"pfa_design": p, "pfa_sim": fa / cells, "false_alarms": fa, "cells": cells})
        print(win, {k: [round(x["pfa_sim"] / x["pfa_design"], 3) for x in mc[k]] for k in keys})
    pd_grid = np.logspace(-1, -5.3, 40)
    naive_analytic = [ca_pfa_with_alpha(dsp.ca_alpha(n_tr, p, 1), n_tr, K) for p in pd_grid]

    dump({"pfa": pfa, "K": Ks, "threshold_error_db": err, "realised_pfa_naive": real,
          "table_K4_K8": table,
          "mc": {"K": K, "M": M, "N_fft": Nf, "guard": guard, "train": train, "n_train": n_tr,
                 "frames": frames, "seed": 2027, **mc},
          "naive_analytic": {"pfa_design": pd_grid, "pfa_realised": naive_analytic}},
         "fig_cfar_threshold.json")

    # ---------------- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_W, 1.75), gridspec_kw={"wspace": 0.42})
    for c, mk, n in zip(SERIES, ("o", "s", "^"), Ns):
        ax1.plot(Ks, err[n], "-", marker=mk, color=c, ms=2.5, label=f"$N$={n}")
    ax1.set(xlabel="Kanal sayısı $K$", ylabel="Eşik hatası [dB]", xlim=(1, 16), ylim=(0, None))
    ax1.set_xticks([1, 4, 8, 12, 16])
    ax1.legend(loc="lower right", handlelength=1.4)
    ax1.set_title("(a) $\\alpha_1$ ile eşik hatası, $P_{fa}=10^{-4}$", fontsize=7.5)

    for c, mk, key, lab in ((SERIES[0], "o", "rect_exact", "dikdörtgen"), (SERIES[1], "s", "hann_exact", "Hann")):
        dp = np.array([x["pfa_design"] for x in mc[key]])
        ps = np.array([x["pfa_sim"] for x in mc[key]])
        ci = np.array([metrics.binomial_ci(x["pfa_sim"], x["cells"]) for x in mc[key]])
        ax2.errorbar(dp, ps / dp, yerr=[(ps - ci[:, 0]) / dp, (ci[:, 1] - ps) / dp], fmt=mk + "-", color=c,
                     mfc="none", ms=3, lw=0.9, capsize=1.5, elinewidth=0.7, label=lab)
    ax2.axhline(1, color=MUTED, lw=0.7, ls=":")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set(xlabel="Tasarım $P_{fa}$", ylabel="Ölçülen / tasarım $P_{fa}$", xlim=(5e-6, 2e-1), ylim=(0.6, 4))
    ax2.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax2.set_yticks([0.7, 1, 1.5, 2, 3])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax2.legend(loc="upper right", handlelength=1.6)
    ax2.set_title("(b) kesin $\\alpha_K$, $K$=8, $N$=40 (MC)", fontsize=7.5)
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax1.yaxis.set_major_formatter(fmt)
    save(fig, "fig_cfar_threshold.pdf")


if __name__ == "__main__":
    main()
