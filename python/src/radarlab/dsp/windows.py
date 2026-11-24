"""Window functions (periodic / DFT-even, as in the brief).

Periodic Hann: ``w[n] = 0.5 − 0.5·cos(2πn/N)``, n = 0…N−1 (Harris, 1978).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.signal


@lru_cache(maxsize=64)
def _cached(name: str, n: int) -> np.ndarray:
    k = np.arange(n)
    if name in ("rect", "rectangular", "boxcar", "none"):
        w = np.ones(n)
    elif name == "hann":
        w = 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(2 * np.pi * k / n)
    elif name == "blackman":
        w = 0.42 - 0.5 * np.cos(2 * np.pi * k / n) + 0.08 * np.cos(4 * np.pi * k / n)
    elif name.startswith("cheb"):
        at = float(name.split(":")[1]) if ":" in name else 60.0
        w = scipy.signal.windows.chebwin(n, at, sym=False)
    elif name.startswith("taylor"):
        w = scipy.signal.windows.taylor(n, sym=False)
    else:
        raise ValueError(f"unknown window {name!r}")
    w.setflags(write=False)
    return w


def get_window(name: str | None, n: int) -> np.ndarray:
    """Periodic window of length n ('hann' | 'hamming' | 'blackman' | 'rect' |
    'cheb[:dB]' | 'taylor'). ``None`` means rectangular. Read-only array."""
    return _cached((name or "rect").lower(), int(n))


def coherent_gain(w: np.ndarray) -> float:
    """Σw / N."""
    return float(np.sum(w) / len(w))


def processing_loss_db(w: np.ndarray) -> float:
    """SNR loss of a window relative to rectangular: 10·log10(N·Σw² / (Σw)²) [dB]
    (1.76 dB for Hann)."""
    return float(10 * np.log10(len(w) * np.sum(w ** 2) / np.sum(w) ** 2))


def enbw_bins(w: np.ndarray) -> float:
    """Equivalent noise bandwidth N·Σw²/(Σw)² [bins] (1.5 for Hann)."""
    return float(len(w) * np.sum(w ** 2) / np.sum(w) ** 2)
