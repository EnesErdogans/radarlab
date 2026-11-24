r"""2-D CFAR detectors on a square-law range-Doppler power map.

Geometry: a rectangular window of half-sizes (gr+tr, gd+td) around the cell
under test (CUT); the inner (gr, gd) guard rectangle (incl. CUT) is excluded.
Doppler is circular; range is clipped — edge cells use only the available
training cells and α is recomputed for the actual count (brief).

Threshold multipliers (noise exponential, i.e. K = 1 channel; Richards,
*Fundamentals of Radar Signal Processing*, 2nd ed., ch. 7; Rohling 1983;
Gandhi & Kassam 1988)::

    CA:  T = α·mean(train),  α = N·(Pfa^(−1/N) − 1)
    OS:  T = α·x_(k),  k = round(rank·N),  Pfa = Π_{i=0}^{k−1} (N−i)/(N−i+α)   (bisection)
    GO/SO: T = α·max/min(mean(lead), mean(lag)) with n = N/2 cells per half,
         β = α/n,  Pfa_SO(β) = 2·Σ_{j=0}^{n−1} C(n−1+j, j)·(2+β)^{−(n+j)},
                   Pfa_GO(β) = 2·(1+β)^{−n} − Pfa_SO(β)

Leading/lagging halves are split by the sign of the range offset (cells with
zero range offset are split by the sign of the Doppler offset). At range edges
the halves are unequal; α then uses n = min(n_lead, n_lag) and, if a half has
fewer than a quarter of the cells, CA is used for that CUT (documented
approximation).

Non-coherent channel integration (brief REVISION 1): with P = Σ_k |X_k|² over
K channels the noise is Gamma(K), not exponential, and α is computed *exactly*
for that case (pass ``n_channels=K``; the pipeline does so by default):
  CA:    X/(X+S) ~ Beta(K, N·K) → α = N·u/(1−u), u = I⁻¹(K, N·K; 1−Pfa)
  OS:    Pfa = ∫₀¹ Q(K, α·F⁻¹(u))·Beta(u; k, N−k+1) du       (quadrature)
  GO/SO: Pfa = ∫₀¹ Q(K, β·G⁻¹(u))·{2u | 2(1−u)} du, β = α/n   (quadrature)
with Q the regularised upper incomplete gamma, F/G the Gamma(K)/Gamma(nK)
cdfs. All formulas assume i.i.d. training cells — true for rectangular FFT
windows, *not* for Hann (adjacent cells correlated by −2/3, fidelity §0.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import scipy.ndimage as ndi
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats

CfarKind = Literal["ca", "go", "so", "os"]


# ------------------------------------------------------------------ α formulas
@lru_cache(maxsize=4096)
def ca_alpha(n: int, pfa: float, n_channels: int = 1) -> float:
    """CA-CFAR multiplier for N training cells (threshold = α · mean).

    K = 1: α = N·(Pfa^(−1/N) − 1). K > 1 (Gamma(K) noise, exact):
    α = N·u/(1−u), u = I⁻¹(K, N·K; 1 − Pfa) (``betaincinv``)."""
    if n <= 0:
        return np.inf
    if n_channels == 1:
        return float(n * (pfa ** (-1.0 / n) - 1.0))
    K = n_channels
    u = float(scipy.special.betaincinv(K, n * K, 1.0 - pfa))
    return float(n * u / (1.0 - u))


def os_pfa(alpha: float, n: int, k: int, n_channels: int = 1) -> float:
    """OS-CFAR Pfa for threshold α·X_(k) (k-th smallest of N cells).

    K = 1: Π_{i=0}^{k−1} (N−i)/(N−i+α) (Rohling 1983). K > 1: exact
    Pfa = ∫₀¹ Q(K, α·F⁻¹(u)) · Beta(u; k, N−k+1) du with F the Gamma(K) cdf
    (X_(k) = F⁻¹(U_(k)), U_(k) ~ Beta(k, N−k+1)); adaptive quadrature."""
    if n_channels == 1:
        i = np.arange(k)
        return float(np.exp(np.sum(np.log(n - i) - np.log(n - i + alpha))))
    K = n_channels
    bd = scipy.stats.beta(k, n - k + 1)
    f = lambda u: scipy.special.gammaincc(K, alpha * scipy.special.gammaincinv(K, u)) * bd.pdf(u)
    lo, hi = bd.ppf(1e-14), bd.ppf(1 - 1e-14)
    return float(scipy.integrate.quad(f, lo, hi, epsabs=0, epsrel=1e-10, limit=400,
                                      points=[bd.mean()])[0])


def _solve_alpha(pfa_fn, pfa: float) -> float:
    """Solve pfa_fn(α) = pfa (pfa_fn decreasing) on a log scale."""
    hi = 1.0
    while pfa_fn(hi) > pfa:
        hi *= 2.0
    lo = 0.0 if hi == 1.0 else hi / 2
    g = lambda a: np.log(max(pfa_fn(a), 1e-300)) - np.log(pfa)
    return float(scipy.optimize.brentq(g, lo, hi, xtol=1e-13, rtol=1e-12))


@lru_cache(maxsize=4096)
def os_alpha(n: int, pfa: float, k: int, n_channels: int = 1) -> float:
    """OS-CFAR multiplier (threshold = α · k-th smallest of N); exact for any K."""
    if n <= 0 or k <= 0:
        return np.inf
    k = min(k, n)
    return _solve_alpha(lambda a: os_pfa(a, n, k, n_channels), pfa)


def os_rank_index(n: int, rank: float) -> int:
    """k = round(rank·N), clipped to [1, N] (1-based order statistic).

    Rounding is numpy's round-half-to-even (e.g. 0.75·6 = 4.5 → 4)."""
    return int(min(max(int(np.round(rank * n)), 1), n))


def so_pfa_beta(beta: float, n: int) -> float:
    """SO-CFAR Pfa for threshold β·min(S1, S2), S_i sums of n exponential cells."""
    j = np.arange(n)
    logc = scipy.special.gammaln(n + j) - scipy.special.gammaln(j + 1) - scipy.special.gammaln(n)
    return float(2 * np.sum(np.exp(logc - (n + j) * np.log(2 + beta))))


def go_pfa_beta(beta: float, n: int) -> float:
    """GO-CFAR Pfa for threshold β·max(S1, S2)."""
    return float(2 * (1 + beta) ** (-n) - so_pfa_beta(beta, n))


def gs_pfa_beta(beta: float, n: int, kind: str, n_channels: int = 1) -> float:
    """GO/SO Pfa for threshold β·max/min(S1, S2) with Gamma(K) cells.

    K = 1: closed forms above. K > 1: exact quadrature
    Pfa = ∫₀¹ Q(K, β·G⁻¹(u))·w(u) du, G the Gamma(nK) cdf of a half sum,
    w(u) = 2u (GO, max) or 2(1−u) (SO, min)."""
    if n_channels == 1:
        return go_pfa_beta(beta, n) if kind == "go" else so_pfa_beta(beta, n)
    K = n_channels
    w = (lambda u: 2 * u) if kind == "go" else (lambda u: 2 * (1 - u))
    f = lambda u: scipy.special.gammaincc(K, beta * scipy.special.gammaincinv(n * K, u)) * w(u)
    # split at quantiles where the integrand changes scale; tolerance ≪ Pfa
    edges = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
    return float(sum(scipy.integrate.quad(f, a, b, epsabs=1e-16, epsrel=1e-9, limit=200)[0]
                     for a, b in zip(edges[:-1], edges[1:])))


@lru_cache(maxsize=4096)
def gs_alpha(n_half: int, pfa: float, kind: str, n_channels: int = 1) -> float:
    """GO/SO multiplier on the *mean* of a half (α = n·β); exact for any K."""
    if n_half <= 0:
        return np.inf
    beta = _solve_alpha(lambda b: gs_pfa_beta(b, n_half, kind, n_channels), pfa)
    return float(beta * n_half)


def cfar_alpha(kind: CfarKind, n: int, pfa: float, os_rank: float = 0.75, n_channels: int = 1) -> float:
    """Multiplier α for a CFAR of the given kind, N training cells, K channels."""
    if kind == "ca":
        return ca_alpha(n, pfa, n_channels)
    if kind == "os":
        return os_alpha(n, pfa, os_rank_index(n, os_rank), n_channels)
    if kind in ("go", "so"):
        return gs_alpha(n // 2, pfa, kind, n_channels)
    raise ValueError(kind)


# --------------------------------------------------------------- 2-D detector
@dataclass
class CfarResult:
    """mask: detections (Nr, Md) bool; threshold: (Nr, Md); noise: noise estimate
    (the statistic multiplied by α); alpha: per-cell α; n_train: per-cell count."""

    mask: np.ndarray
    threshold: np.ndarray
    noise: np.ndarray
    alpha: np.ndarray
    n_train: np.ndarray


def _kernels(guard: tuple[int, int], train: tuple[int, int]):
    gr, gd = guard
    tr, td = train
    hr, hd = gr + tr, gd + td
    dr, dd = np.meshgrid(np.arange(-hr, hr + 1), np.arange(-hd, hd + 1), indexing="ij")
    trainm = ~((np.abs(dr) <= gr) & (np.abs(dd) <= gd))
    lead = trainm & ((dr < 0) | ((dr == 0) & (dd < 0)))
    lag = trainm & ((dr > 0) | ((dr == 0) & (dd > 0)))
    return hr, hd, trainm, lead, lag


def _pad(P: np.ndarray, hr: int, hd: int, fill: float) -> np.ndarray:
    Pw = np.concatenate([P[:, -hd:], P, P[:, :hd]], axis=1) if hd > 0 else P
    return np.pad(Pw, ((hr, hr), (0, 0)), constant_values=fill)


def _box(Pp: np.ndarray, kern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = ndi.correlate(Pp, kern.astype(float), mode="constant", cval=0.0)
    hr, hd = kern.shape[0] // 2, kern.shape[1] // 2
    return out[hr:hr + shape[0], hd:hd + shape[1]]


def cfar_2d(P: np.ndarray, guard: tuple[int, int] = (2, 2), train: tuple[int, int] = (8, 4),
            pfa: float = 1e-5, kind: CfarKind = "ca", os_rank: float = 0.75,
            n_channels: int = 1, row_chunk: int = 64) -> CfarResult:
    """2-D CFAR on a power map P[range, Doppler] (Doppler circular, range clipped).

    Parameters
    ----------
    P : (Nr, Md) square-law power.
    guard, train : (range, Doppler) half-widths of the guard and training bands.
    pfa : design false-alarm probability.
    kind : 'ca' | 'go' | 'so' | 'os'.
    os_rank : OS rank fraction (k = round(rank·N)).
    n_channels : K, channels summed non-coherently in P (exact Gamma(K) α, all kinds).
    """
    P = np.asarray(P, float)
    Nr, Md = P.shape
    hr, hd, trainm, lead, lag = _kernels(guard, train)
    ones = np.ones_like(P)
    Op = _pad(ones, hr, hd, 0.0)
    n_tr = np.rint(_box(Op, trainm, P.shape)).astype(int)
    if kind in ("ca", "go", "so"):
        Pp = _pad(P, hr, hd, 0.0)
        if kind == "ca":
            stat = _box(Pp, trainm, P.shape) / np.maximum(n_tr, 1)
            alpha_of = {n: ca_alpha(int(n), pfa, n_channels) for n in np.unique(n_tr)}
            alpha = np.vectorize(alpha_of.get, otypes=[float])(n_tr)
        else:
            n1 = np.rint(_box(Op, lead, P.shape)).astype(int)
            n2 = np.rint(_box(Op, lag, P.shape)).astype(int)
            m1 = _box(Pp, lead, P.shape) / np.maximum(n1, 1)
            m2 = _box(Pp, lag, P.shape) / np.maximum(n2, 1)
            stat = np.maximum(m1, m2) if kind == "go" else np.minimum(m1, m2)
            nmin = np.minimum(n1, n2)
            fallback = nmin < n_tr / 4
            ca_stat = _box(Pp, trainm, P.shape) / np.maximum(n_tr, 1)
            stat = np.where(fallback, ca_stat, stat)
            keys = set(zip(nmin.ravel().tolist(), fallback.ravel().tolist(), n_tr.ravel().tolist()))
            amap = {key: (ca_alpha(key[2], pfa, n_channels) if key[1] else gs_alpha(key[0], pfa, kind, n_channels))
                    for key in keys}
            alpha = np.array([amap[key] for key in zip(nmin.ravel().tolist(), fallback.ravel().tolist(),
                                                       n_tr.ravel().tolist())]).reshape(P.shape)
    elif kind == "os":
        Pp = _pad(P, hr, hd, np.nan)
        win = np.lib.stride_tricks.sliding_window_view(Pp, trainm.shape)   # (Nr, Md, Wr, Wd)
        sel = trainm.ravel()
        stat = np.empty_like(P)
        for r0 in range(0, Nr, row_chunk):
            r1 = min(Nr, r0 + row_chunk)
            vals = win[r0:r1].reshape(r1 - r0, Md, -1)[..., sel]
            vals = np.sort(vals, axis=-1)                                    # NaN last
            k = np.array([os_rank_index(int(n), os_rank) for n in n_tr[r0:r1, 0]])
            stat[r0:r1] = np.take_along_axis(vals, (k - 1)[:, None, None].repeat(Md, 1), -1)[..., 0]
        uniq = {int(n): os_alpha(int(n), pfa, os_rank_index(int(n), os_rank), n_channels) for n in np.unique(n_tr)}
        alpha = np.vectorize(uniq.get, otypes=[float])(n_tr)
    else:
        raise ValueError(f"unknown CFAR kind {kind!r}")
    thr = alpha * stat
    return CfarResult(P > thr, thr, stat, alpha, n_tr)


def cfar_1d(p: np.ndarray, guard: int = 2, train: int = 8, pfa: float = 1e-5,
            kind: CfarKind = "ca", os_rank: float = 0.75, circular: bool = False) -> CfarResult:
    """1-D CFAR (convenience wrapper around :func:`cfar_2d`). With
    ``circular=False`` the axis is clipped, otherwise wrapped."""
    p = np.asarray(p, float)
    if circular:
        r = cfar_2d(p[None, :], (0, guard), (0, train), pfa, kind, os_rank)
        return CfarResult(r.mask[0], r.threshold[0], r.noise[0], r.alpha[0], r.n_train[0])
    r = cfar_2d(p[:, None], (guard, 0), (train, 0), pfa, kind, os_rank)
    return CfarResult(r.mask[:, 0], r.threshold[:, 0], r.noise[:, 0], r.alpha[:, 0], r.n_train[:, 0])
