"""Closed-form / quadrature CA-CFAR results for a K-channel power sum
(Gamma(K) noise), used by the SIU 2027 figure scripts.

Notation: unit-variance complex noise per channel, P = Σ_k |x_k|²,
CA-CFAR threshold α·S/N with S the sum of N training cells (S ~ Gamma(NK)),
per-channel SNR χ (linear) in the cell under test.
"""
import numpy as np
import scipy.special as sp
import scipy.stats as st
from scipy.optimize import brentq

_U = (np.arange(4000) + 0.5) / 4000           # midpoint grid in probability


def ca_pfa_with_alpha(alpha: float, n: int, K: int) -> float:
    """Exact Pfa = 1 − I_u(K, NK), u = α/(N+α)  (X/(X+S) ~ Beta(K, NK))."""
    u = alpha / (n + alpha)
    return float(sp.betainc(n * K, K, 1.0 - u))


def _s_grid(n: int, K: int) -> np.ndarray:
    return st.gamma(n * K).ppf(_U)


def _pd_fixed_sw0(T, chi, K):
    return st.ncx2.sf(2 * T, 2 * K, 2 * K * chi)


def _pd_fixed_sw1(T, chi, K):
    """Swerling 1 shared by the K channels, fixed threshold T (Richards ch. 6)."""
    if K == 1:
        return np.exp(-T / (1 + chi))
    g = 1 + 1 / (K * chi)
    return (1 - sp.gammainc(K - 1, T) + g ** (K - 1) * sp.gammainc(K - 1, T / g) * np.exp(-T / (1 + K * chi)))


def pd_ca(snr_db: float, alpha: float, n: int, K: int, model: int = 0) -> float:
    """CA-CFAR Pd = E_S[Pd_fixed(T = α·S/N)], expectation on a quantile grid."""
    chi = 10 ** (snr_db / 10)
    T = alpha * _s_grid(n, K) / n
    f = _pd_fixed_sw0 if model == 0 else _pd_fixed_sw1
    return float(np.mean(f(T, chi, K)))


def required_snr_db(fn, pd_target: float, lo: float = -25.0, hi: float = 35.0) -> float:
    return float(brentq(lambda s: fn(s) - pd_target, lo, hi, xtol=1e-4))
