r"""Micro-Doppler: slow-time extraction and STFT spectrograms.

For a range bin (or band) the slow-time signal y[m] = Σ_r X_k[m, r] of
channel(s) k is taken from consecutive frames (contiguous when
``frame_period = M·Te``). Its short-time Fourier transform

    S(t, f) = |Σ_m y[m]·w[m − t]·exp(−j2πfm·Te)|²

is mapped to radial velocity v = f·λ/2 (positive = receding). A rotating
blade of length L at f_rot Hz has maximum micro-Doppler
f_max = 2·v_tip/λ = 4π·f_rot·L/λ (Chen, *The Micro-Doppler Effect in Radar*, 2019).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal

from ..config import RadarConfig
from .windows import get_window


def slow_time(rfft_frames: np.ndarray | list[np.ndarray], range_bins: int | slice | np.ndarray,
              channels: int | slice | np.ndarray | None = 0, cfg: RadarConfig | None = None,
              tx: int = 0) -> np.ndarray:
    """Concatenate the slow-time signal of selected range bins over frames.

    rfft_frames : list of range-FFT cubes (K, M, Nr) or a single one.
    range_bins : bin index, slice or array (coherently summed).
    channels : channel(s) to coherently sum (None = all). With a multi-TX
        ``cfg`` only channels of TX ``tx`` are used so samples are spaced Te.
    """
    frames = [rfft_frames] if isinstance(rfft_frames, np.ndarray) and rfft_frames.ndim == 3 else list(rfft_frames)
    y = []
    for X in frames:
        if cfg is not None and cfg.n_tx > 1:
            X = X[tx * cfg.n_rx:(tx + 1) * cfg.n_rx]
        Xc = X if channels is None else X[np.atleast_1d(np.arange(X.shape[0])[channels])]
        rb = np.atleast_1d(np.arange(X.shape[-1])[range_bins])
        y.append(Xc[..., rb].sum(axis=(0, 2)))
    return np.concatenate(y)


@dataclass
class Spectrogram:
    """times [s] (centre of each STFT segment), velocity axis [m/s],
    frequency axis [Hz], power (n_freq, n_time) linear."""

    times: np.ndarray
    velocity: np.ndarray
    freq: np.ndarray
    power: np.ndarray

    @property
    def power_db(self) -> np.ndarray:
        return 10 * np.log10(np.maximum(self.power, 1e-300))


def spectrogram(y: np.ndarray, prf: float, wavelength: float, nperseg: int = 128,
                noverlap: int | None = None, window: str = "hann", nfft: int | None = None) -> Spectrogram:
    """Two-sided STFT spectrogram of a complex slow-time signal.

    prf = 1/Te [Hz]; frequency axis fftshifted; velocity v = f·λ/2."""
    noverlap = nperseg * 7 // 8 if noverlap is None else noverlap
    w = np.asarray(get_window(window, nperseg))
    f, t, Z = scipy.signal.stft(y, fs=prf, window=w, nperseg=nperseg, noverlap=noverlap,
                                nfft=nfft, return_onesided=False, boundary=None, padded=False)
    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)
    return Spectrogram(t, f * wavelength / 2, f, np.abs(Z) ** 2)


def doppler_extent(spec: Spectrogram, threshold_db: float = -25.0, noise_floor_db: float | None = None
                   ) -> tuple[float, float]:
    """(min, max) velocity whose time-maximum power exceeds
    ``max(peak + threshold_db, noise_floor + 6 dB)`` (micro-Doppler envelope)."""
    p = spec.power_db.max(axis=1)
    thr = p.max() + threshold_db
    if noise_floor_db is not None:
        thr = max(thr, noise_floor_db + 6)
    idx = np.where(p >= thr)[0]
    return float(spec.velocity[idx.min()]), float(spec.velocity[idx.max()])
