"""Range / Doppler FFT, MTI and range-Doppler power maps.

Conventions (brief): forward FFT kernel exp(−j2πkn/N); windows are normalised
by their coherent gain (divide by Σw) so a noiseless on-grid tone of amplitude
A yields |X| = A at its bin. Range-FFT output layout (K, M, Nr); RD cube
layout (K, Nr, M) with Doppler fftshifted (bin M/2 ↔ 0 m/s).

Noise after processing: complex white noise of variance σ² becomes
σ²·Σw²/(Σw)² per FFT dimension, i.e. the Hann processing loss is 1.76 dB per
dimension relative to the ideal coherent gain N.
"""
from __future__ import annotations

import numpy as np

from ..config import RadarConfig
from .windows import get_window


def range_fft(cube: np.ndarray, window: str | None = "hann", n_fft: int | None = None,
              real_input: bool | None = None) -> np.ndarray:
    """Windowed, Σw-normalised FFT along fast time (last axis).

    Parameters
    ----------
    cube : (..., N) ADC data. For real input only the first ``n_fft//2`` bins
        (non-negative beat frequencies) are returned.
    window : window name (see :func:`get_window`); ``None`` = rectangular.
    n_fft : FFT length (zero-padding), default N.

    Returns (..., Nr) complex.
    """
    N = cube.shape[-1]
    w = get_window(window, N)
    nf = n_fft or N
    real = np.isrealobj(cube) if real_input is None else real_input
    X = np.fft.fft(cube * w, n=nf, axis=-1) / np.sum(w)
    return X[..., : nf // 2] if real else X


def doppler_fft(rfft: np.ndarray, window: str | None = "hann", n_fft: int | None = None) -> np.ndarray:
    """Doppler FFT over chirps of a range-FFT cube (K, M, Nr) → RD cube (K, Nr, Md).

    Σw-normalised, fftshifted along Doppler (bin Md/2 = 0 m/s)."""
    M = rfft.shape[-2]
    w = get_window(window, M)
    nf = n_fft or M
    X = np.fft.fft(rfft * w[:, None], n=nf, axis=-2) / np.sum(w)
    X = np.fft.fftshift(X, axes=-2)
    return np.swapaxes(X, -1, -2)


def mti(x: np.ndarray, axis: int = -2, kind: str = "mean") -> np.ndarray:
    """Moving-target indication along slow time.

    ``kind='mean'``: subtract the slow-time mean (removes the zero-Doppler bin
    exactly). ``kind='two_pulse'``: x[m] − x[m−1] (first chirp set to 0).
    """
    if kind == "mean":
        return x - x.mean(axis=axis, keepdims=True)
    if kind == "two_pulse":
        d = np.diff(x, axis=axis)
        pad = [(0, 0)] * x.ndim
        pad[axis] = (1, 0)
        return np.pad(d, pad)
    raise ValueError(kind)


def rd_power(rd: np.ndarray) -> np.ndarray:
    """Square-law RD power map P[r, d] = Σ_k |X[k, r, d]|² (non-coherent over channels)."""
    return np.sum(np.abs(rd) ** 2, axis=0) if rd.ndim == 3 else np.abs(rd) ** 2


def to_db(p: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """10·log10(max(p, floor))."""
    return 10 * np.log10(np.maximum(p, floor))


def range_doppler(cube: np.ndarray, cfg: RadarConfig | None = None, range_window: str | None = "hann",
                  doppler_window: str | None = "hann", mti_kind: str | None = None,
                  n_range_fft: int | None = None, n_doppler_fft: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: (range-FFT cube (K, M, Nr), RD cube (K, Nr, Md))."""
    rf = range_fft(cube, range_window, n_range_fft)
    if mti_kind:
        rf = mti(rf, axis=-2, kind=mti_kind)
    return rf, doppler_fft(rf, doppler_window, n_doppler_fft)


def bin_to_range(r: np.ndarray | float, cfg: RadarConfig, n_fft: int | None = None) -> np.ndarray:
    """Range [m] of (fractional) range bin r for an n_fft-point range FFT."""
    return np.asarray(r) * cfg.range_resolution * cfg.n_samples / (n_fft or cfg.n_samples)


def bin_to_velocity(d: np.ndarray | float, cfg: RadarConfig, n_fft: int | None = None) -> np.ndarray:
    """Velocity [m/s] of (fractional, fftshifted) Doppler bin d."""
    nf = n_fft or cfg.n_chirps
    return (np.asarray(d) - nf // 2) * cfg.velocity_resolution * cfg.n_chirps / nf
