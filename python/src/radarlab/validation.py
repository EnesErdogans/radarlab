"""Validation matrix V1–V11 (``docs/literature/fidelity_requirements.md`` §15).

Two kinds of entries (``ValidationResult.kind``):

* ``theory`` — the simulator/processing output is compared with an
  *independent* closed-form result (V1, V1c, V2, V3, V4, V5, V7, V8);
* ``consistency`` — implementation self-checks in which the reference is
  produced by the same model or code (V6, V8f, V9, V10, V11). They catch
  regressions and wiring errors but cannot detect a shared modelling error.

Each ``vN_*`` function runs one validation against its analytic reference and
returns a :class:`ValidationResult`: the numbers, the pass/fail verdict, and
optional curve data for figures. ``quick=True`` gives a reduced Monte-Carlo
size for unit tests. ``quick=False`` is the paper-grade run used by
``examples/validation_report.py``. All runs are seeded, so they are
deterministic.

Unless a test says otherwise, analytic detection tests use **rectangular**
windows: Hann makes adjacent noise cells correlated (−2/3), which breaks the
i.i.d. assumption behind the CFAR formulas (fidelity §0.3). The Hann
deviation is measured and reported separately (V1c).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.signal

from . import dsp, impairments as imp, metrics
from .config import C0, RadarConfig, automotive_77ghz, micro_doppler_24ghz
from .scene import Bird, Drone, PointTarget
from .signal import complex_noise, synthesize_cube


@dataclass
class ValidationResult:
    id: str
    title: str
    criterion: str
    passed: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    curves: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    kind: str = "theory"     # 'theory': vs independent closed form; 'consistency': implementation self-check


def _cfg(n_tx: int = 1, n_rx: int = 1, M: int = 16, N: int = 32, **kw: Any) -> RadarConfig:
    base = dict(fc=77e9, slope=20e12, fs=10e6, n_samples=N, chirp_period=max(20e-6, N / 10e6 + 4e-6),
                n_chirps=M, tx_pos=np.arange(n_tx) * n_rx * 0.5, rx_pos=np.arange(n_rx) * 0.5,
                adc_start_time=1e-6, idle_time=2e-6, frame_period=1.0, name="validation")
    base.update(kw)
    return RadarConfig(**base)


# ----------------------------------------------------------------------- V1
def _noise_rd(cfg_shape: tuple[int, int, int], rng: np.random.Generator, window: str | None) -> np.ndarray:
    K, M, N = cfg_shape
    x = complex_noise((K, M, N), rng)
    rf = dsp.range_fft(x, window)
    rd = dsp.doppler_fft(rf, window)
    P = dsp.rd_power(rd)
    # normalise to unit per-channel noise power per cell: σ²·Σw²/(Σw)² per dimension
    wn, wm = dsp.get_window(window, N), dsp.get_window(window, M)
    var = np.sum(wn ** 2) / np.sum(wn) ** 2 * np.sum(wm ** 2) / np.sum(wm) ** 2
    return P / var


def v1_pfa(quick: bool = True, seed: int = 1) -> ValidationResult:
    """V1: fixed-threshold and CA/OS/GO/SO Pfa vs analytic, K = 1, 4 and 8, rect windows.

    The noise path is the real one: complex white noise → rect range FFT →
    rect Doppler FFT → P = Σ_k|X_k|² → 2-D CFAR (Doppler circular, range
    clipped, exact Gamma(K) α). Criterion |log10(Pfa_sim/Pfa)| < 0.1."""
    rng = np.random.default_rng(seed)
    pfa = 1e-3 if quick else 1e-4
    n_cells_target = 2.5e5 if quick else 3.0e6
    M, N = 64, 128
    guard, train = (1, 1), (4, 3)
    rows = []
    ok = True
    for K in (1, 4, 8):
        frames = int(np.ceil(n_cells_target / (M * N)))
        maps = [_noise_rd((K, M, N), rng, None) for _ in range(frames)]
        T = metrics.threshold_noncoherent(pfa, K)
        cells = sum(P.size for P in maps)
        fa = sum(int((P > T).sum()) for P in maps)
        res = [("fixed", fa / cells)]
        for kind in ("ca", "os", "go", "so"):
            fa = sum(int(dsp.cfar_2d(P, guard, train, pfa, kind, n_channels=K).mask.sum()) for P in maps)
            res.append((kind, fa / cells))
        for name, p in res:
            err = abs(np.log10(max(p, 1e-12) / pfa))
            ok &= err < 0.1
            rows.append({"K": K, "detector": name, "pfa_design": pfa, "pfa_sim": p,
                         "log10_ratio": float(np.log10(max(p, 1e-12) / pfa)), "cells": cells})
    return ValidationResult("V1", "Noise + CFAR false-alarm rate (rect windows)",
                            "|log10(Pfa_sim/Pfa)| < 0.1", bool(ok), rows)


def v1c_hann_deviation(quick: bool = True, seed: int = 2) -> ValidationResult:
    """V1c (documentation): CFAR Pfa with periodic-Hann windows deviates from
    the i.i.d. formulas because adjacent noise cells are correlated (−2/3)."""
    rng = np.random.default_rng(seed)
    pfa = 1e-3
    M, N = 64, 128
    frames = 30 if quick else 120
    rows = []
    for win in (None, "hann"):
        maps = [_noise_rd((1, M, N), rng, win) for _ in range(frames)]
        # measured correlation of adjacent range cells of the complex RD output
        x = complex_noise((M, N), rng)
        X = np.fft.fft(x * dsp.get_window(win, N), axis=-1)
        rho = np.real(np.mean(X[:, 1:] * X[:, :-1].conj())) / np.mean(np.abs(X) ** 2)
        for kind in ("ca", "os"):
            fa = sum(int(dsp.cfar_2d(P, (1, 1), (4, 3), pfa, kind).mask.sum()) for P in maps)
            p = fa / (frames * M * N)
            rows.append({"window": win or "rect", "detector": kind, "pfa_design": pfa, "pfa_sim": p,
                         "ratio": p / pfa, "adjacent_corr": float(rho)})
    hann = [r for r in rows if r["window"] == "hann"]
    passed = all(abs(r["adjacent_corr"] + 2 / 3) < 0.05 for r in hann)
    return ValidationResult("V1c", "CFAR Pfa deviation with Hann windows (documented effect)",
                            "adjacent-cell correlation = −2/3 (±0.05); Pfa ratio reported",
                            passed, rows)


# ----------------------------------------------------------------------- V2
def v2_pd(quick: bool = True, seed: int = 3) -> ValidationResult:
    """V2: Pd vs SNR for SW0 / SW1 (correlated) / SW3 at K = 1 and 8 via the
    full synthesiser (on-grid static target, rect windows, fixed threshold).
    Criterion: analytic Pd inside the 95 % Wilson interval of the MC estimate
    (99.9 % intervals are also reported)."""
    rng = np.random.default_rng(seed)
    pfa = 1e-4
    trials = 400 if quick else 3000
    snr_db = [6.0, 12.0] if quick else [0.0, 4.0, 8.0, 12.0, 16.0]
    rows, ok = [], True
    curves: dict[str, Any] = {}
    for K, (nt, nr) in ((1, (1, 1)), (8, (2, 4))):
        cfg = _cfg(nt, nr, M=8, N=16)
        r0 = 5
        R = r0 * cfg.range_resolution
        T = metrics.threshold_noncoherent(pfa, K)
        cell_var = 1.0 / (cfg.n_samples * cfg.n_chirps)
        for model in (0, 1, 3):
            pds = []
            for s in snr_db:
                # per-channel post-integration SNR χ = A²·N·M  ->  per-sample SNR
                snr_sample = s - 10 * np.log10(cfg.n_samples * cfg.n_chirps)
                tg = PointTarget((0, R, 0))
                det = 0
                for _ in range(trials):
                    c = synthesize_cube(cfg, [tg], snr_ref_db=snr_sample, ref_range=R, rng=rng, swerling=model)
                    rd = dsp.doppler_fft(dsp.range_fft(c, None), None)
                    det += dsp.rd_power(rd)[r0, cfg.n_chirps // 2] / cell_var > T
                p = det / trials
                th = float(metrics.pd_swerling(model, 10 ** (s / 10), pfa, K))
                lo, hi = metrics.binomial_ci(p, trials)
                lo3, hi3 = metrics.binomial_ci(p, trials, 3.29)
                ok &= lo3 <= th <= hi3
                rows.append({"K": K, "swerling": model, "snr_db": s, "pd_sim": p, "pd_theory": th,
                             "ci95": (lo, hi), "in_ci95": lo <= th <= hi, "trials": trials})
                pds.append((p, th))
            curves[f"K{K}_SW{model}"] = {"snr_db": snr_db, "sim": [a for a, _ in pds], "theory": [b for _, b in pds]}
    n_in = sum(r["in_ci95"] for r in rows)
    return ValidationResult("V2", "Pd vs SNR (synthesiser, Swerling 0/1/3, K = 1, 8)",
                            "theory inside the 99.9 % Wilson CI at every point (Bonferroni-style bound "
                            "for 30 comparisons; 95 % CI reported)", bool(ok), rows, curves,
                            note=f"{n_in}/{len(rows)} points inside the 95 % CI")


# ----------------------------------------------------------------------- V3
def v3_crlb(quick: bool = True, seed: int = 4) -> ValidationResult:
    """V3: ML (unwindowed, zero-padded FFT + refinement) RMSE vs CRLB for range,
    velocity, angle with random phase / sub-bin offsets. Efficiency = CRLB/MSE."""
    rng = np.random.default_rng(seed)
    cfg = _cfg(1, 4, M=16, N=32)
    trials = 250 if quick else 2000
    snrs = [0.0] if quick else [-24.0, -20.0, -16.0, -12.0, -6.0, 0.0, 6.0]
    rows, ok = [], True
    curves: dict[str, list] = {"snr_db": snrs, "rmse": [], "crlb": []}
    for s in snrs:
        err = []
        crl = []
        for _ in range(trials):
            R = rng.uniform(4.0, 6.0)
            th = np.deg2rad(rng.uniform(-30, 30))
            v = rng.uniform(-3, 3)
            tg = PointTarget.from_polar(R, np.rad2deg(th), v, phase0=rng.uniform(0, 2 * np.pi))
            c = synthesize_cube(cfg, [tg], snr_ref_db=s, ref_range=R, rng=rng)
            w = metrics.ml_tone_3d(c)
            Rm = R + v * 0.5 * (cfg.n_chirps - 1) * cfg.te
            tfc = cfg.adc_start_time + (cfg.n_samples - 1) / 2 / cfg.fs
            wk = -np.pi * np.sin(th)
            wm = 2 * np.pi * 2 * v * cfg.te * (cfg.fc + cfg.slope * tfc) / C0
            wn = 2 * np.pi * (cfg.slope * 2 * Rm / C0 + 2 * v * cfg.fc / C0) / cfg.fs
            e = (w - [wk, wm, wn] + np.pi) % (2 * np.pi) - np.pi
            err.append([e[2] * C0 * cfg.fs / (4 * np.pi * cfg.slope),
                        e[1] * cfg.wavelength / (4 * np.pi * cfg.te),
                        e[0] / (np.pi * np.cos(th))])
            snr_eff = s + 20 * np.log10((R / Rm) ** 2)
            crl.append([metrics.crlb_range(cfg, snr_eff) ** 2, metrics.crlb_velocity(cfg, snr_eff) ** 2,
                        metrics.crlb_angle(cfg, snr_eff, th) ** 2])
        mse = np.mean(np.square(err), 0)
        crb = np.mean(crl, 0)
        eff = crb / mse
        curves["rmse"].append(np.sqrt(mse).tolist())
        curves["crlb"].append(np.sqrt(crb).tolist())
        above = s >= -12.0
        for name, i in (("range", 0), ("velocity", 1), ("angle", 2)):
            rows.append({"snr_db": s, "param": name, "rmse": float(np.sqrt(mse[i])),
                         "sqrt_crlb": float(np.sqrt(crb[i])), "efficiency": float(eff[i]),
                         "above_threshold": above})
            if above:
                ok &= eff[i] >= (0.8 if quick else 0.9)
    return ValidationResult("V3", "Estimation: RMSE vs CRLB (range, velocity, angle)",
                            "efficiency CRLB/MSE ≥ 0.9 above threshold (≥ 0.8 in quick mode)",
                            bool(ok), rows, curves)


# ----------------------------------------------------------------------- V4
def v4_phase_noise(quick: bool = True, seed: int = 5) -> ValidationResult:
    """V4: phase-noise residual through the synthesiser.

    Residual phase angle(x_pn / x_clean) of a scatterer at delay τ:
    variance vs ∫L(f)·4sin²(πfτ) df (±2 %), PSD vs L(f)·4sin²(πfτ) (aliased to
    the ADC rate) within ±0.5 dB around fτ = ½."""
    rng = np.random.default_rng(seed)
    pn = imp.PhaseNoise()
    cfg = _cfg(1, 1, M=64, N=256, fs=20e6)
    tau = 0.68e-6
    R = C0 * tau / 2
    tg = PointTarget((0, R, 0))
    clean = synthesize_cube(cfg, [tg], noise=False)[0]
    frames = 20 if quick else 80
    res = []
    for f in range(frames):
        x = synthesize_cube(cfg, [tg], t_frame=f * 1e-3, noise=False, phase_noise=pn, rng=rng)[0]
        res.append(np.angle(x / clean))
    res = np.concatenate(res)                     # (frames·M, N)
    var_sim = float(np.var(res))
    var_th = pn.residual_variance(tau)
    fw, Pw = scipy.signal.welch(res, fs=cfg.fs, nperseg=cfg.n_samples, axis=-1, return_onesided=False,
                                detrend="constant", scaling="density")
    Pw = Pw.mean(0)
    # theory aliased to the ADC rate
    k = np.arange(-int(pn.fs_gen / cfg.fs / 2), int(pn.fs_gen / cfg.fs / 2) + 1)
    fa = fw[:, None] + k[None, :] * cfg.fs
    psd_th = np.sum(np.where(np.abs(fa) <= pn.fs_gen / 2,
                             pn.psd(fa) * 4 * np.sin(np.pi * fa * tau) ** 2, 0.0), axis=1)
    band = (np.abs(np.abs(fw) - 1 / (2 * tau)) < 0.25 / tau)
    dev_db = float(10 * np.log10(np.mean(Pw[band]) / np.mean(psd_th[band])))
    gain_db = float(10 * np.log10(4 * np.sin(np.pi * 0.5) ** 2))
    rows = [{"tau_us": tau * 1e6, "var_sim": var_sim, "var_theory": var_th,
             "var_rel_err": var_sim / var_th - 1, "psd_dev_db_at_ftau_half": dev_db,
             "skirt_gain_at_ftau_half_db": gain_db}]
    ok = abs(var_sim / var_th - 1) < (0.04 if quick else 0.02) and abs(dev_db) < 0.5
    order = np.argsort(fw)
    return ValidationResult("V4", "Phase noise: range-correlated residual",
                            "var within ±2 % (±4 % quick); PSD within ±0.5 dB at fτ = ½", bool(ok), rows,
                            {"f": fw[order].tolist(), "psd_sim": Pw[order].tolist(), "psd_theory": psd_th[order].tolist(),
                             "L": pn.psd(fw[order]).tolist()})


# ----------------------------------------------------------------------- V5
def v5_multipath(quick: bool = True) -> ValidationResult:
    """V5: two-ray ground multipath. Peak amplitude ratio (two-ray / free space)
    vs d against |F|² = 4·sin²(2π·h_r·h_t/(λ_c·d)) (Γ = −1, two-way amplitude
    factor; power factor is its square). Null positions vs d_n = 2h_r h_t/(nλ_c).

    λ_c = c/(fc + S·t_mid) is the wavelength at the centre of the ADC window
    (t_mid = adcStartTime + N/(2fs)): ``fc`` is the chirp *start* frequency, and
    the beat-tone phase that forms the interference pattern is sampled across
    the window, so the effective carrier is the window-centre frequency (0.27 %
    above fc here). Null positions are refined below the sweep step by a
    parabola through ratio² at the three samples around each minimum."""
    cfg = _cfg(1, 1, M=2, N=256, slope=15e12)
    hr, ht = 0.5, 0.3
    step = 0.1 if quick else 0.02
    d = np.arange(15.0, 95.0, step)
    ratio = np.empty_like(d)
    zt = ht - hr                       # radar at z = 0, ground at z = −h_r
    for i, di in enumerate(d):
        tg = PointTarget((0, di, zt))
        c0 = synthesize_cube(cfg, [tg], noise=False)[0, 0]
        c1 = synthesize_cube(cfg, imp.two_ray(tg, hr, -1.0), noise=False)[0, 0]
        X0, X1 = np.abs(np.fft.fft(c0)), np.abs(np.fft.fft(c1))
        ratio[i] = X1.max() / X0.max()
    f_c = cfg.fc + cfg.slope * (cfg.adc_start_time + 0.5 * cfg.n_samples / cfg.fs)
    lam = C0 / f_c
    theory = 4 * np.sin(2 * np.pi * hr * ht / (lam * d)) ** 2
    nulls_th = np.array([2 * hr * ht / (n * lam) for n in range(1, 10)])
    nulls_th = nulls_th[(nulls_th > d[0]) & (nulls_th < d[-1])]
    # one minimum per contiguous valley below 0.25 (fine sweeps ripple around shoulders)
    below = ratio < 0.25
    edges = np.flatnonzero(np.diff(np.r_[0, below.astype(int), 0]))
    mins = np.array([a + int(np.argmin(ratio[a:b])) for a, b in zip(edges[::2], edges[1::2])
                     if a > 0 and b < len(ratio)], int)
    rows = []
    ok = len(mins) == len(nulls_th)
    for nt in nulls_th:
        j = int(mins[np.argmin(np.abs(d[mins] - nt))]) if len(mins) else 1
        y = ratio[j - 1:j + 2] ** 2
        dn = float(d[j] + step * dsp.parabolic_offset(-y[0], -y[1], -y[2]))   # minimum of y = peak of −y
        err = dn - float(nt)
        ok &= abs(err) / nt < 1e-3
        rows.append({"null_theory_m": float(nt), "null_sim_m": dn, "error_m": err, "rel_err": err / float(nt),
                     "error_fc_start_m": dn - float(nt) * f_c / cfg.fc, "range_bin_m": cfg.range_resolution})
    # amplitude: compare away from the nulls (|F|² ≥ 1, i.e. ≥ 0 dB), where dB is well conditioned
    m = theory >= 1.0
    dev_db = 20 * np.log10(ratio[m] / theory[m])
    amp_max = float(np.max(np.abs(dev_db)))
    ok &= amp_max < 1.0
    rms_all = float(np.sqrt(np.mean((20 * np.log10(np.maximum(ratio, 1e-3) / np.maximum(theory, 1e-3))) ** 2)))
    return ValidationResult("V5", "Two-ray ground multipath",
                            "null positions within 0.1 % (λ at the ADC-window centre); amplitude within "
                            "±1 dB where 4sin² ≥ 1",
                            bool(ok), rows, {"d": d.tolist(), "sim": ratio.tolist(), "theory": theory.tolist()},
                            note=f"Amplitude where 4sin² ≥ 1: max |deviation| {amp_max:.2f} dB, rms "
                                 f"{float(np.sqrt(np.mean(dev_db ** 2))):.2f} dB. Over the whole sweep (floor 1e-3, "
                                 f"dominated by the nulls) the rms is {rms_all:.2f} dB. With λ at the chirp start "
                                 f"frequency fc the nulls would be predicted {100 * (f_c / cfg.fc - 1):.2f} % too "
                                 "close (column error_fc_start_m).")


# ----------------------------------------------------------------------- V6
def v6_iq_imbalance() -> ValidationResult:
    """V6: IQ imbalance image at −IMRR dBc (g = 1.05, φ_e = 3°)."""
    cfg = _cfg(1, 1, M=1, N=256)
    r0 = 40
    tg = PointTarget((0, r0 * cfg.range_resolution, 0))
    c = synthesize_cube(cfg, [tg], noise=False)[0, 0]
    gdb = 20 * np.log10(1.05)
    y = imp.iq_imbalance(c, gdb, 3.0)
    Y = np.abs(np.fft.fft(y)) ** 2
    img_db = float(10 * np.log10(Y[r0] / Y[-r0]))
    th = imp.image_rejection_db(gdb, 3.0)
    return ValidationResult("V6", "IQ imbalance image", "image level = −IMRR ± 0.5 dB",
                            abs(img_db - th) < 0.5, [{"imrr_theory_db": th, "image_sim_dbc": -img_db,
                                                      "error_db": img_db - th}],
                            note="Consistency check: the model is y = μx + νx*, so image/signal = |ν|²/|μ|² "
                                 "holds by algebra; this verifies the implementation, not the physics.",
                            kind="consistency")


# ----------------------------------------------------------------------- V7
def v7_interference() -> ValidationResult:
    """V7: FMCW interference: burst length 2·f_LP/|ΔS| (±1 sample); equal-slope
    ghost at c·Δf_IF/(2S)."""
    cfg = _cfg(1, 1, M=4, N=256, slope=20e12, chirp_period=30e-6)
    f_lp = 0.45 * cfg.fs
    rows, ok = [], True
    intf = imp.FmcwInterferer(fc=cfg.fc - 0.2e9, slope=35e12, chirp_period=30e-6, ramp_time=28e-6,
                              t0=-3e-6, distance=15.0, inr_db=30.0)
    x = imp.fmcw_interference(cfg, intf, if_bandwidth=f_lp)
    lens = [int(np.count_nonzero(x[0, m])) for m in range(cfg.n_chirps)]
    lens = [L for L in lens if 0 < L < cfg.n_samples]
    exp_len = 2 * f_lp / abs(cfg.slope - intf.slope) * cfg.fs
    for L in lens:
        ok &= abs(L - exp_len) <= 1.0
    rows.append({"case": "crossing ramps", "burst_samples_sim": lens, "burst_samples_theory": exp_len})
    df = 1.2e6
    d = 30.0
    intf2 = imp.FmcwInterferer(fc=cfg.fc - df, slope=cfg.slope, chirp_period=cfg.chirp_period,
                               ramp_time=cfg.ramp_time, t0=-d / C0, distance=d, inr_db=20.0)
    x2 = imp.fmcw_interference(cfg, intf2, if_bandwidth=f_lp)
    P = np.abs(np.fft.fft(x2[0, 0])) ** 2
    rb = int(np.argmax(P))
    rng_ghost = rb * cfg.range_resolution
    rng_th = C0 * df / (2 * cfg.slope)
    ok &= abs(rng_ghost - rng_th) <= cfg.range_resolution
    rows.append({"case": "equal slopes (ghost)", "ghost_range_sim_m": rng_ghost, "ghost_range_theory_m": rng_th,
                 "range_bin_m": cfg.range_resolution})
    return ValidationResult("V7", "FMCW mutual interference", "burst ±1 sample; ghost ±1 range bin",
                            bool(ok), rows)


# ----------------------------------------------------------------------- V8
def _rotor_slow_time(drone: Drone, cfg: RadarConfig, n_frames: int, snr_ref_db: float = 80.0,
                     noise: bool = True, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = float(np.linalg.norm(drone.center([0.0])[0]))
    rb = int(round(R / cfg.range_resolution))
    frames = [dsp.range_fft(synthesize_cube(cfg, drone, i * cfg.frame_period, snr_ref_db=snr_ref_db,
                                            noise=noise, rng=rng))
              for i in range(n_frames)]
    return dsp.slow_time(frames, rb, 0, cfg)


def flash_times(y: np.ndarray, te: float, min_sep: float, rel: float = 0.5) -> np.ndarray:
    """Times of blade flashes: local maxima of |y| above ``rel``·max separated by ``min_sep``."""
    a = np.abs(y)
    pk, _ = scipy.signal.find_peaks(a, height=rel * a.max(), distance=max(1, int(min_sep / te)))
    return pk * te


def v8_micro_doppler(quick: bool = True) -> ValidationResult:
    """V8: rotor max micro-Doppler 2LΩcosβ/λ (±1 bin) and bird wing-beat
    frequency (±2 %), both against closed-form kinematics. The blade-flash rate
    is a separate consistency check (:func:`v8f_flash_rate`)."""
    cfg = micro_doppler_24ghz()
    rows, ok = [], True
    curves: dict[str, Any] = {}
    # max Doppler
    L, rpm = 0.12, 6000.0
    dr = Drone((0, 15, 0), n_rotors=1, n_blades=2, rpm=rpm, blade_length=L, rotor_offsets=[[0, 0, 0]],
               body_rcs=0.0, blade_rcs=0.05)
    y = _rotor_slow_time(dr, cfg, 8)
    sp = dsp.spectrogram(y, 1 / cfg.te, cfg.wavelength, nperseg=128)
    vlo, vhi = dsp.doppler_extent(sp, -10.0)
    f_max = 2 * L * (2 * np.pi * rpm / 60) / cfg.wavelength
    df = sp.freq[1] - sp.freq[0]
    e1, e2 = vhi * 2 / cfg.wavelength - f_max, -vlo * 2 / cfg.wavelength - f_max
    ok &= abs(e1) <= df and abs(e2) <= df
    rows.append({"check": "max micro-Doppler", "f_max_theory_hz": f_max, "f_pos_sim_hz": f_max + e1,
                 "f_neg_sim_hz": -(f_max + e2), "bin_hz": df})
    # bird wing-beat: periodicity of the micro-Doppler centroid
    fw = 5.0
    bcfg = _cfg(1, 1, M=256, N=32, fc=24e9, slope=250e6 / 6.4e-6, fs=5e6, chirp_period=40e-6)
    bcfg = bcfg.replace(frame_period=bcfg.frame_active_time)
    bird = Bird((0, 8, 8), velocity=(0, 0.0, 0), flap_freq=fw, body_rcs=0.005, wing_rcs=0.01)
    n_fr = int(np.ceil((1.2 if quick else 3.0) / bcfg.frame_period))
    rng = np.random.default_rng(0)
    frames = [dsp.range_fft(synthesize_cube(bcfg, bird, i * bcfg.frame_period, snr_ref_db=70, rng=rng))
              for i in range(n_fr)]
    rb = int(round(np.hypot(8, 8) / bcfg.range_resolution))
    yb = dsp.slow_time(frames, rb, 0)
    spb = dsp.spectrogram(yb, 1 / bcfg.te, bcfg.wavelength, nperseg=256, noverlap=192)
    pw = spb.power * (spb.power > spb.power.max(0, keepdims=True) * 1e-3)
    cen = (spb.velocity[:, None] * pw).sum(0) / pw.sum(0)
    cen = cen - cen.mean()
    dt = spb.times[1] - spb.times[0]
    ac = np.correlate(cen, cen, "full")[len(cen) - 1:]
    lag0 = int(round(1 / fw / dt))
    w = slice(max(1, lag0 // 2), min(len(ac) - 1, 3 * lag0 // 2))
    j = w.start + int(np.argmax(ac[w]))
    jf = dsp.refine_peak_1d(ac - ac.min() + 1e-12, j, log=False)
    f_sim = 1 / (jf * dt)
    ok &= abs(f_sim / fw - 1) < 0.02
    rows.append({"check": "bird wing-beat", "f_w_theory_hz": fw, "f_w_sim_hz": f_sim, "rel_err": f_sim / fw - 1})
    curves["bird"] = {"t": spb.times.tolist(), "centroid": cen.tolist()}
    return ValidationResult("V8", "Micro-Doppler (rotor extent, bird wing-beat)",
                            "f_mD,max ±1 bin; wing-beat ±2 %", bool(ok), rows, curves)


def v8f_flash_rate(quick: bool = True) -> ValidationResult:
    """V8f (consistency): blade-flash rate N_bΩ/2π (even N_b) / 2N_bΩ/2π (odd)
    within ±1 %. The rotor model is strictly periodic by construction, so this
    only checks that the synthesis and the flash-timing code agree with the
    model's own rate; it is not an independent validation."""
    cfg = micro_doppler_24ghz()
    rows, ok = [], True
    curves: dict[str, Any] = {}
    # flash rate
    for nb in (2, 3):
        drn = Drone((0, 15, 0), n_rotors=1, n_blades=nb, rpm=3000.0, rotor_offsets=[[0, 0, 0]], body_rcs=0.0,
                    blade_rcs=0.05)
        rate_th = float(drn.flash_rate()[0])
        yy = _rotor_slow_time(drn, cfg, 20 if quick else 60, noise=False)
        tf = flash_times(yy, cfg.te, 0.5 / rate_th)
        period = float(np.polyfit(np.arange(len(tf)), tf, 1)[0])
        rel = 1 / period / rate_th - 1
        ok &= abs(rel) < 0.01
        rows.append({"check": f"flash rate N_b={nb}", "rate_theory_hz": rate_th, "rate_sim_hz": 1 / period,
                     "rel_err": rel, "n_flashes": len(tf)})
        curves[f"flash_nb{nb}"] = {"t": (np.arange(len(yy)) * cfg.te).tolist()[:4000],
                                   "env": np.abs(yy).tolist()[:4000]}
    return ValidationResult("V8f", "Blade-flash rate (rotor model self-consistency)", "flash rate ±1 %",
                            bool(ok), rows, curves, kind="consistency")


# ----------------------------------------------------------------------- V9
def v9_tdm() -> ValidationResult:
    """V9: TDM angle bias at v = 0.8·vmax without / with compensation (< 0.1 bin)."""
    cfg = automotive_77ghz()
    v = 0.8 * cfg.max_velocity
    rows, ok = [], True
    na = 64
    for th in (-30.0, 0.0, 20.0):
        tg = PointTarget.from_polar(30.0, th, v)
        rf, rd = dsp.range_doppler(synthesize_cube(cfg, [tg], noise=False))
        P = dsp.rd_power(rd)
        r, d = np.unravel_index(int(np.argmax(P)), P.shape)
        _, df = dsp.refine_peak(P, r, d)
        vel = float(dsp.bin_to_velocity(df[0], cfg))
        x = rd[:, r, d]
        a_c, _, _ = dsp.estimate_azimuth(dsp.tdm_compensate(x, cfg, vel), cfg, na)
        a_u, _, _ = dsp.estimate_azimuth(x, cfg, na)
        truth = float(dsp.sin_to_angle_bin(np.sin(np.deg2rad(th)), na, 0.5))
        ok &= abs(a_c - truth) < 0.1
        rows.append({"theta_deg": th, "v_over_vmax": 0.8, "bias_uncomp_bins": a_u - truth,
                     "bias_comp_bins": a_c - truth})
    return ValidationResult("V9", "TDM-MIMO Doppler phase compensation", "bias < 0.1 angle bin (Na = 64)",
                            bool(ok), rows, kind="consistency",
                            note="Consistency check: synthesiser and compensator use the same t·Tc timing, so a "
                                 "shared timing error would not be detected. An independent bistatic "
                                 "first-principles synthesiser (review round 1) confirmed the physics.")


# ----------------------------------------------------------------------- V10
def _max_rel_dev(a, b) -> float:
    """Largest relative deviation between two JSON trees (inf on structural mismatch)."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return float("inf")
        return max([_max_rel_dev(a[k], b[k]) for k in a], default=0.0)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return float("inf")
        return max([_max_rel_dev(x, y) for x, y in zip(a, b)], default=0.0)
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
        return abs(a - b) / max(abs(b), 1e-12) if a != b else 0.0
    return 0.0 if a == b else float("inf")


def v10_golden() -> ValidationResult:
    """V10 (Python side): golden vectors are deterministic and the stored files
    match a fresh regeneration (relative deviation ≤ 1e-9 per value). The TS side is
    checked by the web test-suite against the same files."""
    import json
    from pathlib import Path

    from . import golden
    gdir = Path(__file__).resolve().parents[3] / "golden"
    rows, ok = [], True
    for name, fn in golden.ALL_CASES.items():
        p = gdir / f"{name}.json"
        if not p.exists():
            rows.append({"file": name, "status": "missing"})
            ok = False
            continue
        stored = json.loads(p.read_text())
        fresh = json.loads(json.dumps(fn()))
        dev = _max_rel_dev(stored, fresh)
        good = (stored.get("conventionVersion") == fresh.get("conventionVersion")
                and stored.get("goldenSet") == fresh.get("goldenSet") and dev <= 1e-9)
        ok &= good
        rows.append({"file": name, "max_rel_dev": dev, "goldenSet": stored.get("goldenSet"),
                     "conventionVersion": stored.get("conventionVersion"), "ok": good})
    return ValidationResult("V10", "Cross-language golden vectors (Python side)",
                            "stored == regenerated (rel ≤ 1e-9); TS checks max|Δ| < 1e-9 of peak", bool(ok), rows,
                            kind="consistency")


# ----------------------------------------------------------------------- V11
def v11_dca1000_roundtrip(tmpdir: str | None = None) -> ValidationResult:
    """V11: .cfg → simulate → int16 DCA1000 .bin → read → process: exact cube
    round trip and identical detections (public raw captures: not available
    in the sandbox)."""
    import tempfile
    from pathlib import Path

    from .io import parse_ti_cfg, read_dca1000, write_dca1000
    from .pipeline import ProcessingParams, process_frame
    tcfg = parse_ti_cfg(SAMPLE_TI_CFG)
    cfg = tcfg.config.replace(n_chirps=32)
    targets = [PointTarget.from_polar(6.0, -20.0, 1.5, rcs=5), PointTarget.from_polar(11.0, 15.0, -2.0, rcs=5)]
    cube = synthesize_cube(cfg, targets, rng=11, snr_ref_db=25, ref_range=10)
    q = imp.adc_quantize(cube, bits=16, noise_rms_lsb=64.0)
    lsb = 1 / 64.0 * np.sqrt(0.5)
    codes = np.round(q / lsb)
    d = Path(tmpdir or tempfile.mkdtemp())
    path = d / "adc_data.bin"
    write_dca1000(path, codes, cfg)
    back = read_dca1000(path, cfg)[0]
    exact = bool(np.array_equal(back, codes))
    pp = ProcessingParams(cfar_pfa=1e-6)
    r_float = process_frame(cube, cfg, pp)
    r_bin = process_frame(back * lsb, cfg, pp)
    key = lambda pts: sorted((int(a), int(b)) for a, b in zip(pts["range_cell"], pts["doppler_cell"]))
    strong = lambda pts: pts[pts["snr_db"] > 20]
    same = key(strong(r_float.points)) == key(strong(r_bin.points))
    errs = []
    for t in targets:
        tr = t.p0
        R = np.linalg.norm(tr)
        p = r_bin.points
        i = int(np.argmin(np.abs(p["range"] - R) + 10 * np.abs(p["velocity"] - t.v0 @ tr / R)))
        errs.append(abs(p["range"][i] - R))
    ok = exact and same and max(errs) < cfg.range_resolution / 4
    return ValidationResult("V11", "DCA1000 closed loop (.cfg → sim → int16 .bin → read → process)",
                            "exact round trip; same strong detections; range error < ΔR/4", bool(ok),
                            [{"exact_roundtrip": exact, "same_detections": same,
                              "n_detections": int(len(strong(r_bin.points))),
                              "max_range_err_m": float(max(errs)), "range_bin_m": cfg.range_resolution}],
                            kind="consistency",
                            note="Consistency check: writer and reader follow the same byte-layout assumption and "
                                 "no real capture is used; a check against TI's readDCA1000.m or a public "
                                 "capture is still open.")


SAMPLE_TI_CFG = """
% xWR18xx SDK profile (3TX TDM), complex 1x, 16 bit
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 77 7 4 40 0 0 30 1 256 10000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 64 0 50 1 0
sensorStart
"""

ALL: dict[str, Callable[..., ValidationResult]] = {
    "V1": v1_pfa, "V1c": v1c_hann_deviation, "V2": v2_pd, "V3": v3_crlb, "V4": v4_phase_noise,
    "V5": v5_multipath, "V6": v6_iq_imbalance, "V7": v7_interference, "V8": v8_micro_doppler,
    "V8f": v8f_flash_rate,
    "V9": v9_tdm, "V10": v10_golden, "V11": v11_dca1000_roundtrip,
}
