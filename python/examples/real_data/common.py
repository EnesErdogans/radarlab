"""Shared helpers for the real-data (UW AWR1843) validation scripts.

Data location: ``--data DIR`` or env ``RADARLAB_UW_DATA`` (default
``/home/claude/data/uw_radar``); files ``{pms,bms,cms}1000_30fs.mat`` from
github.com/Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification
(MIT). The raw data is never stored in the repository.

Outputs: figures in ``docs/real_data/fig/``, numbers in ``docs/real_data/results/*.json``.
Run with ``python -I`` (the captures are untrusted input; ``-I`` keeps the
current directory off ``sys.path``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # for _style (examples/)
import _style  # noqa: E402  (sets the Agg backend and the RadarLab look)
from _style import plt, SEQ, SERIES, INK, MUTED  # noqa: E402,F401

import radarlab as rl  # noqa: E402
from radarlab import dsp  # noqa: E402
from radarlab.dsp.windows import get_window  # noqa: E402
from radarlab.io.uw_mat import load_uw_mat, uw_awr1843_config  # noqa: E402

REPO = HERE.parents[2]
DOC = REPO / "docs" / "real_data"
FIG = DOC / "fig"
RES = DOC / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

SCENES = {"pedestrian": "pms1000", "cyclist": "bms1000", "car": "cms1000"}
DEFAULT_DATA = "/home/claude/data/uw_radar"

# Processing used everywhere (identical for real and simulated data).
PROC = rl.ProcessingParams()          # Hann/Hann, CA-CFAR Pfa 1e-5, guard (2,2), train (8,4), K = 8 exact α
# Target-free region for noise statistics: negative beat frequencies (range
# bins 66..106 of the complex FFT, i.e. -13.8 m … -4.9 m), away from the DC /
# high-pass-filter notch (bins ≥ 107) and from the IQ images of the targets
# (mirror bins ±4). Zero Doppler (±ZD_EXCL bins) holds static image/alias
# content and is excluded unless stated.
NEG_LO, NEG_HI = 66, 106
ZD_EXCL = 3


def data_dir() -> Path:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data", default=os.environ.get("RADARLAB_UW_DATA", DEFAULT_DATA))
    a, _ = ap.parse_known_args()
    return Path(a.data)


def load(scene: str):
    """(cubes (F, K, M, N), cfg) of a scene name ('pedestrian', 'cyclist', 'car')."""
    return load_uw_mat(data_dir() / f"{SCENES[scene]}_30fs.mat")


def window_noise_gain(cfg, proc=PROC) -> float:
    """Noise power gain of the Σw-normalised range and Doppler FFTs:
    E|X|² = σ²·g, g = Σw_r²/(Σw_r)² · Σw_d²/(Σw_d)²."""
    wr = np.asarray(get_window(proc.range_window, cfg.n_samples))
    wd = np.asarray(get_window(proc.doppler_window, cfg.n_chirps))
    return float((wr @ wr) / wr.sum() ** 2 * (wd @ wd) / wd.sum() ** 2)


def noise_region_mask(cfg, target_bins=(), exclude_zero_doppler: int = 0) -> np.ndarray:
    """Boolean (Nr, M) mask of target-free RD cells (negative beat half, see NEG_LO/HI)."""
    Nr, M = cfg.n_range_bins, cfg.n_chirps
    m = np.zeros((Nr, M), bool)
    m[NEG_LO:NEG_HI + 1] = True
    for b in target_bins:
        img = (Nr - int(b)) % Nr
        m[max(img - 4, 0):img + 5] = False
    if exclude_zero_doppler:
        c = M // 2
        m[:, c - exclude_zero_doppler:c + exclude_zero_doppler + 1] = False
    return m


def noise_sigma2(rd_powers: np.ndarray, cfg, mask: np.ndarray) -> float:
    """Per-complex-sample noise variance σ² from RD powers (F, Nr, M) (sum over K)."""
    return float(rd_powers[:, mask].mean() / (cfg.n_virtual * window_noise_gain(cfg)))


def process_all(cubes: np.ndarray, cfg, proc=PROC) -> list:
    return [rl.process_frame(c, cfg, proc, t=f * cfg.frame_period) for f, c in enumerate(cubes)]


def save_json(name: str, obj) -> Path:
    p = RES / f"{name}.json"

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    p.write_text(json.dumps(obj, indent=1, default=conv))
    print(f"wrote {p}")
    return p


def load_json(name: str):
    return json.loads((RES / f"{name}.json").read_text())


def savefig(fig, name: str) -> Path:
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, dpi=90)
    plt.close(fig)
    print(f"wrote {p}")
    return p


# ------------------------------------------------------------------ statistics
import scipy.stats as _st  # noqa: E402


def stack_power(results) -> np.ndarray:
    return np.stack([r.power for r in results])                 # (F, Nr, M)


def noise_stats(Pstack: np.ndarray, cfg, mask: np.ndarray) -> dict:
    """Noise level and distribution in the target-free cells ``mask`` (Nr, M).

    P/mean is compared with Gamma(K, 1/K) (K = 8 channels summed). Cells are
    subsampled 2×2 so Hann-correlated neighbours (power correlation 0.44 at
    lag 1, 0.03 at lag 2) do not bias the KS p-value."""
    K = cfg.n_virtual
    sub = np.zeros_like(mask)
    sub[::2, ::2] = True
    z = Pstack[:, mask & sub]
    mu = z.mean()
    z = (z / mu).ravel()
    shape = _st.gamma.fit(z, floc=0)[0]
    ks = _st.kstest(z, _st.gamma(K, scale=1 / K).cdf)
    q = np.linspace(0.001, 0.999, 200)
    return {"sigma2": float(mu / (K * window_noise_gain(cfg))), "n_cells": int(z.size),
            "gamma_shape_fit": float(shape), "var_over_gammaK_var": float(z.var() / (1 / K)),
            "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
            "qq_theory": _st.gamma(K, scale=1 / K).ppf(q), "qq_emp": np.quantile(z, q)}


def empirical_pfa(Pstack: np.ndarray, cfg, mask: np.ndarray, pfas=(1e-3, 1e-4, 1e-5), proc=PROC) -> dict:
    """Fraction of CFAR detections inside the target-free mask (same CFAR as the chain)."""
    out = {}
    n = int(mask.sum()) * len(Pstack)
    for pfa in pfas:
        k = 0
        for P in Pstack:
            c = dsp.cfar_2d(P, proc.cfar_guard, proc.cfar_train, pfa, proc.cfar_kind, proc.cfar_os_rank,
                            n_channels=cfg.n_virtual)
            k += int(c.mask[mask].sum())
        lo, hi = rl.metrics.binomial_ci(k / n, n)
        out[f"{pfa:g}"] = {"count": k, "cells": n, "pfa_emp": k / n, "ratio": k / n / pfa,
                           "ci95": [max(lo, 0.0), hi]}
    return out


def noise_floor_profile(Pstack: np.ndarray, cfg, sigma2: float) -> np.ndarray:
    """Median RD power per range bin over |Doppler| > 25 bins, in units of the
    mid-band noise mean (dB)."""
    M = cfg.n_chirps
    d = np.r_[0:M // 2 - 25, M // 2 + 26:M]
    prof = np.median(Pstack[:, :, d], axis=(0, 2))
    ref = sigma2 * cfg.n_virtual * window_noise_gain(cfg) * _st.gamma(cfg.n_virtual, scale=1 / cfg.n_virtual).median()
    return 10 * np.log10(prof / ref)


def dominant_track(results, cfg, r_lo: float, r_hi: float, moving: bool = True, vmin: float = 0.2) -> dict:
    """Per-frame strongest detection in [r_lo, r_hi] m (|v| > vmin if moving,
    |v| <= 2Δv otherwise)."""
    rows = []
    for f, r in enumerate(results):
        p = r.points
        sel = (p["range"] >= r_lo) & (p["range"] <= r_hi)
        sel &= (np.abs(p["velocity"]) > vmin) if moving else (np.abs(p["velocity"]) <= 2 * cfg.velocity_resolution)
        p = p[sel]
        if len(p):
            b = p[np.argmax(p["power_db"])]
            rows.append([f * cfg.frame_period, b["range"], b["velocity"], np.rad2deg(b["azimuth"]),
                         b["snr_db"], b["power_db"], b["range_cell"], b["doppler_cell"]])
    a = np.array(rows).reshape(-1, 8)
    return {"t": a[:, 0], "range": a[:, 1], "velocity": a[:, 2], "azimuth_deg": a[:, 3],
            "snr_db": a[:, 4], "power_db": a[:, 5], "range_cell": a[:, 6], "doppler_cell": a[:, 7]}


def track_summary(tr: dict, noise_db: float = 0.0) -> dict:
    t, R = tr["t"], tr["range"]
    fit = np.polyfit(t, R, 1) if len(t) > 2 else [np.nan, np.nan]
    ts = _st.theilslopes(R, t)[0] if len(t) > 2 else np.nan
    iqr = lambda x: float(np.subtract(*np.percentile(x, [75, 25])))  # noqa: E731
    return {"n_frames": int(len(t)), "range_mean": float(R.mean()), "range_std": float(R.std()),
            "range_median": float(np.median(R)), "range_rate_theil_sen": float(ts),
            "velocity_iqr": iqr(tr["velocity"]), "azimuth_iqr": iqr(tr["azimuth_deg"]),
            "peak_db_iqr": iqr(tr["power_db"]),
            "range_rate_fit": float(fit[0]), "velocity_median": float(np.median(tr["velocity"])),
            "velocity_std": float(np.std(tr["velocity"])),
            "azimuth_median": float(np.median(tr["azimuth_deg"])), "azimuth_std": float(np.std(tr["azimuth_deg"])),
            "peak_db_median": float(np.median(tr["power_db"]) - noise_db),
            "peak_db_std": float(np.std(tr["power_db"])), "snr_db_median": float(np.median(tr["snr_db"]))}


def rd_summary(results, cfg) -> dict:
    """Detection statistics of the whole RD map (per frame averages)."""
    M, Nr = cfg.n_chirps, cfg.n_range_bins
    nd, nzd, npos, nr0 = [], [], [], []
    for r in results:
        p = r.points
        nd.append(len(p))
        nzd.append(int(np.sum(np.abs(p["doppler_cell"] - M // 2) <= 2)))
        npos.append(int(np.sum(p["range_cell"] < Nr // 2)))
        nr0.append(int(np.sum((p["range_cell"] <= 2) | (p["range_cell"] >= Nr - 2))))
    P = stack_power(results)
    zd = P[:, :, M // 2 - 2:M // 2 + 3].sum() / P.sum()
    return {"det_per_frame": float(np.mean(nd)), "det_zero_doppler_per_frame": float(np.mean(nzd)),
            "det_near_dc_per_frame": float(np.mean(nr0)),
            "det_pos_half_fraction": float(np.sum(npos) / max(np.sum(nd), 1)),
            "zero_doppler_power_fraction": float(zd)}


def array_coherence(x: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    """max_θ |a(θ)ᴴx|²/(K·|x|²) (1 = perfect plane wave) and θ̂ [deg]."""
    th = np.deg2rad(np.linspace(-90, 90, 3601))
    A = np.exp(-2j * np.pi * pos[:, None] * np.sin(th)[None])
    c = np.abs(A.conj().T @ x) ** 2 / (len(x) * np.vdot(x, x).real)
    i = int(np.argmax(c))
    return float(c[i]), float(np.rad2deg(th[i]))


# ------------------------------------------------------------------ shared analysis
# range windows [m] in which the dominant target is searched (from a first look at the data)
TARGET_WINDOW = {"pedestrian": (5.5, 8.5, True), "cyclist": (9.0, 12.0, True), "car": (9.5, 12.0, False)}


def analyse(results, cfg, scene: str) -> dict:
    """Everything that is computed identically for real and simulated data."""
    lo, hi, moving = TARGET_WINDOW[scene]
    tr = dominant_track(results, cfg, lo, hi, moving)
    tbin = int(np.median(tr["range_cell"]))
    P = stack_power(results)
    m_nz = noise_region_mask(cfg, [tbin], exclude_zero_doppler=ZD_EXCL)
    m_all = noise_region_mask(cfg, [tbin], exclude_zero_doppler=0)
    ns = noise_stats(P, cfg, m_nz)
    ns_all = noise_stats(P, cfg, m_all)
    noise_db = 10 * np.log10(ns["sigma2"] * cfg.n_virtual * window_noise_gain(cfg))
    coh = []
    for f, r in enumerate(results):
        sel = np.where(tr["t"] == f * cfg.frame_period)[0]
        if len(sel):
            i = sel[0]
            x = dsp.tdm_compensate(r.rd[:, int(tr["range_cell"][i]), int(tr["doppler_cell"][i])], cfg,
                                   tr["velocity"][i])
            coh.append(array_coherence(x, cfg.virtual_pos)[0])
    return {
        "target_bin": tbin,
        "noise": {k: v for k, v in ns.items() if not k.startswith("qq")},
        "noise_incl_zero_doppler": {k: v for k, v in ns_all.items() if not k.startswith("qq")},
        "qq": {"theory": ns["qq_theory"], "emp": ns["qq_emp"], "emp_incl_zd": ns_all["qq_emp"]},
        "pfa": empirical_pfa(P, cfg, m_nz),
        "pfa_incl_zero_doppler": empirical_pfa(P, cfg, m_all, pfas=(1e-4,)),
        "noise_floor_profile_db": noise_floor_profile(P, cfg, ns["sigma2"]),
        "track": {k: v for k, v in tr.items()},
        "track_summary": track_summary(tr, noise_db),
        "array_coherence_median": float(np.median(coh)) if coh else float("nan"),
        "rd": rd_summary(results, cfg),
        "noise_db": noise_db,
    }


def rd_figure(per_scene: dict, cfg, name: str, title_prefix: str, frame: int = 15) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    v = cfg.velocity_axis()
    R = cfg.range_axis()
    half = cfg.n_range_bins // 2
    for ax, (scene, (results, a)) in zip(axs, per_scene.items()):
        r = results[frame]
        img = 10 * np.log10(r.power[:half] / 10 ** (a["noise_db"] / 10))
        ax.pcolormesh(v, R[:half], img, cmap=SEQ, vmin=-5, vmax=45, shading="auto", rasterized=True)
        p = r.points[r.points["range_cell"] < half]
        ax.scatter(p["velocity"], p["range"], s=14, facecolors="none", edgecolors=SERIES[1], lw=0.9,
                   label="CFAR detections")
        ax.set_title(f"{title_prefix} {scene}, frame {frame}", fontsize=9)
        ax.set_xlabel("radial velocity [m/s]")
    axs[0].set_ylabel("range [m]")
    axs[0].legend(fontsize=8, loc="upper left")
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(-5, 45))
    fig.colorbar(sm, ax=axs, label="power / mean noise [dB]", shrink=0.9)
    fig.savefig(FIG / name, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / name}")


