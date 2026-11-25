"""SIU 2027 Fig. 3: CA-CFAR detection on a K-channel power sum (exact Gamma(K)
α and the naive single-channel α) vs. beamforming first (K-point angle FFT,
orthogonal beams) and a single-channel CA-CFAR per beam.

Monte-Carlo at the level of one range–Doppler cell under test plus N training
cells (each a K-vector of unit-variance complex noise), ULA with d = λ/2.
Per-beam noise after the unitary DFT is CN(0,1) and independent across beams,
so the single-channel (exponential) α is exact per beam; the per-beam design
Pfa is 1 − (1 − Pfa)^(1/K) so that the "any beam" false-alarm rate per cell
equals the design Pfa of the power-sum detector.

Target direction: on a beam centre, half-way between two beams (worst-case
straddle), or sin θ ~ U(−1, 1). Targets: Swerling 0 and Swerling 1 (one
complex amplitude shared by all channels).

Run: python -I python/examples/paper/fig_beamforming_vs_sum.py   (≈1–3 min)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from _cfar_theory import ca_pfa_with_alpha, pd_ca, required_snr_db  # noqa: E402
from _paper_style import COL_W, MUTED, SERIES, dump, plt, save  # noqa: E402
from radarlab import dsp  # noqa: E402

K, N, PFA = 8, 32, 1e-4
PFA_B = 1 - (1 - PFA) ** (1 / K)
A_K, A_1, A_B = dsp.ca_alpha(N, PFA, K), dsp.ca_alpha(N, PFA, 1), dsp.ca_alpha(N, PFA_B, 1)
CHUNK = 20000


def cnoise(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def decide(cut, train):
    """cut (T, K), train (T, N, K) -> dict of bool decisions (T,)."""
    P = np.sum(np.abs(cut) ** 2, -1)
    S = np.mean(np.sum(np.abs(train) ** 2, -1), -1)
    Yc = np.fft.fft(cut, axis=-1) / np.sqrt(K)
    Yt = np.fft.fft(train, axis=-1) / np.sqrt(K)
    Sb = np.mean(np.abs(Yt) ** 2, 1)                                 # (T, K) per-beam mean
    return {"sum_exact": P > A_K * S, "sum_naive": P > A_1 * S,
            "beam": np.any(np.abs(Yc) ** 2 > A_B * Sb, -1)}


def run_pd(rng, snr_db, mode, model, trials):
    hits = {"sum_exact": 0, "sum_naive": 0, "beam": 0}
    k = np.arange(K)
    done = 0
    while done < trials:
        T = min(CHUNK, trials - done)
        if mode == "grid":
            u = np.full(T, 2 * rng.integers(0, K, T) / K)
        elif mode == "half":
            u = 2 * (rng.integers(0, K, T) + 0.5) / K
        else:
            u = rng.uniform(-1, 1, T)
        amp = np.sqrt(10 ** (snr_db / 10)) * np.exp(1j * rng.uniform(0, 2 * np.pi, T))
        if model == 1:
            amp = amp * np.abs(cnoise(rng, T))      # Rayleigh amplitude, mean power χ, shared by channels
        sig = amp[:, None] * np.exp(-1j * np.pi * k[None, :] * u[:, None])
        d = decide(sig + cnoise(rng, (T, K)), cnoise(rng, (T, N, K)))
        for key in hits:
            hits[key] += int(d[key].sum())
        done += T
    return {key: v / trials for key, v in hits.items()}


def main():
    rng = np.random.default_rng(20270)
    # ---- false-alarm check (noise only)
    n_fa = 2_000_000
    fa = {"sum_exact": 0, "sum_naive": 0, "beam": 0}
    for _ in range(n_fa // CHUNK):
        d = decide(cnoise(rng, (CHUNK, K)), cnoise(rng, (CHUNK, N, K)))
        for key in fa:
            fa[key] += int(d[key].sum())
    pfa_mc = {key: v / n_fa for key, v in fa.items()}
    print("Pfa MC", pfa_mc)

    snrs = np.arange(-8.0, 12.01, 1.0)
    trials = 20000
    curves = {}
    for model in (0, 1):
        for mode in ("grid", "half", "random"):
            res = [run_pd(rng, s, mode, model, trials) for s in snrs]
            curves[f"sw{model}_{mode}"] = {key: [r[key] for r in res] for key in res[0]}
            print(model, mode, "done")

    def req(y, pdt):
        y = np.asarray(y)
        i = int(np.argmax(y >= pdt))
        if y[i] < pdt or i == 0:
            return None
        return float(snrs[i - 1] + (pdt - y[i - 1]) / (y[i] - y[i - 1]) * (snrs[i] - snrs[i - 1]))

    summary = []
    for model in (0, 1):
        for pdt in (0.5, 0.9):
            row = {"swerling": model, "pd": pdt,
                   "sum_exact_analytic": required_snr_db(lambda s: pd_ca(s, A_K, N, K, model), pdt),
                   "sum_naive_analytic": required_snr_db(lambda s: pd_ca(s, A_1, N, K, model), pdt),
                   "beam_grid_analytic": required_snr_db(
                       lambda s: pd_ca(s + 10 * np.log10(K), A_B, N, 1, model), pdt)}
            for mode in ("grid", "half", "random"):
                for key in ("sum_exact", "sum_naive", "beam"):
                    row[f"{key}_{mode}_mc"] = req(curves[f"sw{model}_{mode}"][key], pdt)
            summary.append(row)
            print(row)

    sd = np.linspace(snrs[0], snrs[-1], 91)
    th = {"sum_exact": [pd_ca(s, A_K, N, K) for s in sd], "sum_naive": [pd_ca(s, A_1, N, K) for s in sd],
          "beam_grid": [pd_ca(s + 10 * np.log10(K), A_B, N, 1) for s in sd]}
    dump({"K": K, "N": N, "pfa": PFA, "pfa_per_beam": PFA_B, "alpha_K": A_K, "alpha_1": A_1, "alpha_beam": A_B,
          "pfa_mc": pfa_mc, "pfa_mc_trials": n_fa, "pfa_analytic_sum_exact": ca_pfa_with_alpha(A_K, N, K),
          "pfa_analytic_beam_any": 1 - (1 - ca_pfa_with_alpha(A_B, N, 1)) ** K,
          "snr_db": snrs, "trials": trials, "seed": 20270, "curves": curves,
          "required_snr_db": summary, "theory_sw0": {"snr_db": sd, **th}}, "fig_beamforming_vs_sum.json")

    # ---- figure (Swerling 0)
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    c0 = curves["sw0_random"]
    ax.plot(sd, th["sum_exact"], "-", color=SERIES[0], lw=1.0)
    ax.plot(sd, th["sum_naive"], "-", color=SERIES[3], lw=1.0)
    ax.plot(sd, th["beam_grid"], "-", color=SERIES[1], lw=1.0)
    ax.plot(snrs, curves["sw0_grid"]["sum_exact"], "o", color=SERIES[0], mfc="none",
            label="güç toplamı + kesin $\\alpha_K$")
    ax.plot(snrs, curves["sw0_grid"]["sum_naive"], "v", color=SERIES[3], mfc="none",
            label="güç toplamı + tek kanal $\\alpha_1$")
    ax.plot(snrs, curves["sw0_grid"]["beam"], "s", color=SERIES[1], mfc="none",
            label="hüzme (FFT), ızgara üstü")
    ax.plot(snrs, c0["beam"], "--", marker="D", color=SERIES[2], ms=2.5, lw=0.9,
            label="hüzme (FFT), rastgele $\\sin\\theta$")
    ax.plot(snrs, curves["sw0_half"]["beam"], ":", marker="^", color=MUTED, ms=2.5, lw=0.9,
            label="hüzme (FFT), iki hüzme arası")
    ax.set(xlabel="Kanal başına SNR $\\chi$ [dB]", ylabel="Tespit olasılığı $P_d$", xlim=(snrs[0], snrs[-1]),
           ylim=(0, 1.0))
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.set_xticks(np.arange(-8, 13, 2))
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.6)
    save(fig, "fig_beamforming_vs_sum.pdf")


if __name__ == "__main__":
    main()
