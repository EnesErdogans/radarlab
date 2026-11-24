r"""Performance bounds and detection theory.

SNR definition (consistent with ``signal.py``): η = A²/σ² is the **per complex
ADC sample** SNR of the target (σ² = 1), i.e. ``10^(snr_db/10)``.

Cramér–Rao bounds for a single complex exponential in AWGN observed on a
separable K×M×N grid (L = K·M·N samples, unknown amplitude and phase; Rife &
Boorstyn, IEEE T-IT 1974; separable multidimensional extension, e.g.
Kay, *Fundamentals of Statistical Signal Processing* vol. I, §3.11)::

    var(ω_i) ≥ 6 / (η · L · (N_i² − 1))      [rad²/sample²]  for the dimension of length N_i

Mapped to physical units:
    range    ω_r = 4π·S·R/(c·fs)         → σ_R = c·fs/(4π·S) · σ_ω
    velocity ω_d = 4π·v·Te/λ             → σ_v = λ/(4π·Te) · σ_ω
    angle    (general array, u = sinθ)   var(u) ≥ 1 / (2η·M·N·(2π)²·Σ_k (x_k − x̄)²)
             σ_θ = σ_u / cosθ   (x_k in wavelengths; ULA reduces to the 1-D formula)

Detection (square-law, n non-coherently integrated samples, unit noise power;
Richards, *Fundamentals of Radar Signal Processing*, ch. 6)::

    threshold T: Pfa = Q_Γ(n, T)                      (regularised upper incomplete Γ)
    Swerling 0:  Pd = Q_{χ'²}(2T; 2n, 2n·snr)          (Marcum Q via scipy.stats.ncx2.sf)
    Swerling 1:  Pd = ∫₀^∞ Pd_SW0(snr·s)·e^{−s} ds     (n = 1: Pd = Pfa^{1/(1+snr)})
    Swerling 2:  Pd = Q_Γ(n, T/(1+snr))
    Swerling 3:  Pd = ∫₀^∞ Pd_SW0(snr·s)·4s·e^{−2s} ds
"""
from __future__ import annotations

import numpy as np
import scipy.integrate
import scipy.special
import scipy.stats

from .config import C0, RadarConfig


# ---------------------------------------------------------------------- CRLB
def crlb_frequency_var(snr_lin: float, n_dim: int, n_total: int | None = None) -> float:
    """CRLB var(ω) [rad²/sample²] for the dimension of length ``n_dim`` of a
    separable tone with ``n_total`` samples (default n_dim)."""
    L = n_dim if n_total is None else n_total
    return 6.0 / (snr_lin * L * (n_dim ** 2 - 1))


def _dims(cfg: RadarConfig, dims: tuple[int, int, int] | None) -> tuple[int, int, int]:
    return dims if dims is not None else (cfg.n_virtual, cfg.n_chirps, cfg.n_samples)


def crlb_range(cfg: RadarConfig, snr_db: float, dims: tuple[int, int, int] | None = None) -> float:
    """CRLB standard deviation of range [m] using K×M×N samples (dims=(K,M,N))."""
    K, M, N = _dims(cfg, dims)
    s = np.sqrt(crlb_frequency_var(10 ** (snr_db / 10), N, K * M * N))
    return float(C0 * cfg.fs / (4 * np.pi * cfg.slope) * s)


def crlb_velocity(cfg: RadarConfig, snr_db: float, dims: tuple[int, int, int] | None = None) -> float:
    """CRLB standard deviation of radial velocity [m/s]."""
    K, M, N = _dims(cfg, dims)
    s = np.sqrt(crlb_frequency_var(10 ** (snr_db / 10), M, K * M * N))
    return float(cfg.wavelength / (4 * np.pi * cfg.te) * s)


def crlb_angle(cfg: RadarConfig, snr_db: float, theta: float = 0.0,
               dims: tuple[int, int, int] | None = None, positions: np.ndarray | None = None) -> float:
    """CRLB standard deviation of azimuth [rad] for an arbitrary linear array
    (positions in wavelengths, default the virtual array)."""
    K, M, N = _dims(cfg, dims)
    x = cfg.virtual_pos if positions is None else np.asarray(positions, float)
    eta = 10 ** (snr_db / 10)
    var_u = 1.0 / (2 * eta * M * N * (2 * np.pi) ** 2 * np.sum((x - x.mean()) ** 2))
    return float(np.sqrt(var_u) / np.cos(theta))


# --------------------------------------------------------------------- errors
def rmse(est: np.ndarray, truth: np.ndarray | float) -> float:
    """Root-mean-square error."""
    e = np.asarray(est, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(e ** 2)))


def ratio_db(a: float, b: float) -> float:
    """20·log10(a/b) (e.g. RMSE over CRLB std)."""
    return float(20 * np.log10(a / b))


# ------------------------------------------------------------------ detection
def threshold_noncoherent(pfa: float, n: int = 1) -> float:
    """T with P(Σ_{i<n}|w_i|² > T) = pfa for unit-variance complex noise."""
    return float(scipy.special.gammainccinv(n, pfa))


def pd_swerling0(snr: np.ndarray | float, pfa: float, n: int = 1) -> np.ndarray:
    """Pd for a non-fluctuating target (per-sample SNR linear)."""
    T = threshold_noncoherent(pfa, n)
    return scipy.stats.ncx2.sf(2 * T, 2 * n, 2 * n * np.asarray(snr, float))


def _pd_fluct(snr: float, pfa: float, n: int, pdf) -> float:
    f = lambda s: float(pd_swerling0(snr * s, pfa, n)) * pdf(s)
    return float(scipy.integrate.quad(f, 0, np.inf, limit=200)[0])


def pd_swerling(model: int, snr: np.ndarray | float, pfa: float, n: int = 1) -> np.ndarray:
    """Pd for Swerling 0–3 (``snr`` = mean per-sample SNR, linear)."""
    snr = np.asarray(snr, float)
    if model == 0:
        return pd_swerling0(snr, pfa, n)
    T = threshold_noncoherent(pfa, n)
    if model == 2:
        return scipy.special.gammaincc(n, T / (1 + snr))
    if model == 1:
        if n == 1:
            return pfa ** (1 / (1 + snr))
        pdf = lambda s: np.exp(-s)
    elif model == 3:
        pdf = lambda s: 4 * s * np.exp(-2 * s)
    else:
        raise ValueError("Swerling model must be 0..3")
    return np.vectorize(lambda x: _pd_fluct(x, pfa, n, pdf))(snr)


def empirical_pfa(mask: np.ndarray, exclude: np.ndarray | None = None) -> float:
    """Fraction of detections among cells not in ``exclude`` (target cells)."""
    m = np.asarray(mask, bool)
    if exclude is None:
        return float(m.mean())
    keep = ~np.asarray(exclude, bool)
    return float(m[keep].mean())


def empirical_pd(masks: np.ndarray, cells: np.ndarray) -> float:
    """Detection rate at target cells: masks (trials, Nr, Md), cells (n, 2)."""
    m = np.asarray(masks, bool)
    c = np.asarray(cells, int).reshape(-1, 2)
    return float(m[:, c[:, 0], c[:, 1]].mean())


# ------------------------------------------------------------ ML estimator
def ml_tone_3d(x: np.ndarray, pad: int = 4, n_iter: int = 3) -> np.ndarray:
    """Maximum-likelihood frequencies (ω_k, ω_m, ω_n) [rad/sample] of a single
    separable complex exponential in white noise, x shape (K, M, N).

    Coarse search on a ``pad``-times zero-padded 3-D FFT, then cyclic
    coordinate ascent of the periodogram |Σ x·e^{−j(ω_k k + ω_m m + ω_n n)}|²
    with bounded scalar optimisation (±1 padded bin) per axis. Unbiased and
    efficient (attains the CRLB) above the threshold SNR.
    """
    import scipy.optimize as so

    K, M, N = x.shape
    F = np.abs(np.fft.fftn(x, s=(pad * K, pad * M, pad * N))) ** 2
    i = np.unravel_index(int(np.argmax(F)), F.shape)
    w = np.array([2 * np.pi * i[a] / F.shape[a] for a in range(3)])
    w = (w + np.pi) % (2 * np.pi) - np.pi
    idx = [np.arange(K), np.arange(M), np.arange(N)]

    def cost(wv: np.ndarray) -> float:
        e = [np.exp(-1j * wv[a] * idx[a]) for a in range(3)]
        return -float(np.abs(np.einsum("kmn,k,m,n->", x, e[0], e[1], e[2])) ** 2)

    for _ in range(n_iter):
        for a in range(3):
            step = 2 * np.pi / F.shape[a]

            def f(v: float, a: int = a) -> float:
                wv = w.copy()
                wv[a] = v
                return cost(wv)

            r = so.minimize_scalar(f, bounds=(w[a] - step, w[a] + step), method="bounded",
                                   options={"xatol": 1e-10})
            w[a] = r.x
    return w
