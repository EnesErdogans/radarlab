"""Detections → point cloud (range, velocity, azimuth, elevation, x/y/z, SNR).

For each detection (integer RD cell + fractional refinement):
    range    = r_frac · ΔR            (ΔR scaled by N/Nfft for zero-padded FFTs)
    velocity = (d_frac − Md/2) · Δv
    azimuth  : TDM-compensate X[:, r, d] with ``velocity``, zero-padded angle FFT
               over the azimuth ULA, parabolic refinement, θ = asin(sinθ)
    elevation: from same-x / different-z virtual pairs when available, else 0
    snr_db   = 10·log10(P[r, d] / noise[r, d])  (noise = CFAR statistic)
"""
from __future__ import annotations

import numpy as np

from ..config import RadarConfig
from .detect import refine_peak, refine_peak_1d
from .doa import (angle_bin_to_sin, angle_fft, azimuth_ula, estimate_elevation, tdm_compensate)
from .fft import bin_to_range, bin_to_velocity

POINT_DTYPE = np.dtype([
    ("x", "f8"), ("y", "f8"), ("z", "f8"), ("range", "f8"), ("velocity", "f8"),
    ("azimuth", "f8"), ("elevation", "f8"), ("snr_db", "f8"), ("power_db", "f8"),
    ("range_bin", "f8"), ("doppler_bin", "f8"), ("angle_bin", "f8"),
    ("range_cell", "i4"), ("doppler_cell", "i4"), ("angle_cell", "i4"),
])


def estimate_azimuth(x: np.ndarray, cfg: RadarConfig, n_fft: int = 64,
                     window: str | None = None) -> tuple[float, int, float]:
    """(fractional angle bin, integer bin, θ [rad]) from one virtual-array
    snapshot x (K,) that is already TDM-compensated."""
    ula = azimuth_ula(cfg)
    spec = np.abs(angle_fft(x[:, None], ula, n_fft, window)[:, 0]) ** 2
    a = int(np.argmax(spec))
    af = refine_peak_1d(spec, a, log=True, circular=True)
    s = angle_bin_to_sin(af, n_fft, ula.d)
    if s > 1:          # wrapped past the edge of visible space
        s -= 1 / ula.d
    elif s < -1:
        s += 1 / ula.d
    return af, a, float(np.arcsin(np.clip(s, -1, 1)))


def point_cloud(rd: np.ndarray, P: np.ndarray, cells: np.ndarray, cfg: RadarConfig,
                noise: np.ndarray | None = None, n_angle_fft: int = 64,
                angle_window: str | None = None, refine: bool = True,
                n_range_fft: int | None = None) -> np.ndarray:
    """Build a structured point cloud (dtype :data:`POINT_DTYPE`).

    rd : RD cube (K, Nr, Md); P : power map (Nr, Md); cells : (n, 2) integer
    (range, Doppler) cells (e.g. cluster peaks); noise : CFAR noise estimate.
    """
    cells = np.asarray(cells, int).reshape(-1, 2)
    out = np.zeros(len(cells), POINT_DTYPE)
    if len(cells) == 0:
        return out
    Md = P.shape[1]
    if refine:
        rf, df = refine_peak(P, cells[:, 0], cells[:, 1])
    else:
        rf, df = cells[:, 0].astype(float), cells[:, 1].astype(float)
    rng = bin_to_range(rf, cfg, n_range_fft)
    vel = bin_to_velocity(df, cfg, Md)
    for i, (r, d) in enumerate(cells):
        x = tdm_compensate(rd[:, r, d], cfg, vel[i])
        af, a, th = estimate_azimuth(x, cfg, n_angle_fft, angle_window)
        el = estimate_elevation(x, cfg) if np.any(cfg.virtual_pos_z != cfg.virtual_pos_z[0]) else 0.0
        el = 0.0 if np.isnan(el) else el
        out[i]["azimuth"], out[i]["elevation"] = th, el
        out[i]["angle_bin"], out[i]["angle_cell"] = af, a
    out["range"], out["velocity"] = rng, vel
    out["range_bin"], out["doppler_bin"] = rf, df
    out["range_cell"], out["doppler_cell"] = cells[:, 0], cells[:, 1]
    ce = np.cos(out["elevation"])
    out["x"] = rng * np.sin(out["azimuth"]) * ce
    out["y"] = rng * np.cos(out["azimuth"]) * ce
    out["z"] = rng * np.sin(out["elevation"])
    pw = P[cells[:, 0], cells[:, 1]]
    out["power_db"] = 10 * np.log10(np.maximum(pw, 1e-300))
    if noise is not None:
        out["snr_db"] = 10 * np.log10(np.maximum(pw, 1e-300) / np.maximum(noise[cells[:, 0], cells[:, 1]], 1e-300))
    return out
