r"""Direction-of-arrival estimation and TDM-MIMO phase compensation.

Steering vector (brief convention version 2 / REVISION 1; positions in wavelengths)::

    a_k(θ, φ) = exp( −j·2π·(x_k·sinθ·cosφ + z_k·sinφ) )

(an element at +x is closer to a target at θ > 0). The angle spectrum uses the
matched **+j kernel** A[a] = Σ_k x_k·exp(+j2πka/Na) (Na·IFFT) followed by
fftshift, so a target at θ > 0 still lands in a bin > Na/2 and bin a maps to
``sinθ = (a − Na/2)/(Na·d)``.

TDM compensation (brief): TX t fires t·Tc after TX 0, so a target with radial
velocity v acquires an extra phase 2π·(2v/λ)·t·Tc on TX-t channels; multiply
those channels by exp(−j·2π·(2v/λ)·t·Tc) using the (aliased) estimate of v.

Spectral estimators (Van Trees, *Optimum Array Processing*, 2002):
    MVDR/Capon: P(θ) = 1 / (aᴴ R⁻¹ a)
    MUSIC:      P(θ) = 1 / ‖E_nᴴ a‖²   (E_n: noise-subspace eigenvectors)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RadarConfig
from .windows import get_window


def tdm_phase(cfg: RadarConfig, v: np.ndarray | float) -> np.ndarray:
    """Compensation factors exp(−j·2π·(2v/λ)·t·Tc) per virtual channel.

    Returns shape (K,) for scalar v, else (K, *v.shape)."""
    v = np.asarray(v, float)
    t = np.repeat(np.arange(cfg.n_tx), cfg.n_rx).astype(float)
    t = t.reshape((-1,) + (1,) * v.ndim)
    return np.exp(-2j * np.pi * (2 * v / cfg.wavelength) * t * cfg.chirp_period)


def tdm_compensate(x: np.ndarray, cfg: RadarConfig, v: np.ndarray | float) -> np.ndarray:
    """Apply TDM phase compensation to x with the channel axis first.

    ``v`` may be scalar or broadcastable to ``x.shape[1:]`` (e.g. one velocity
    per Doppler bin of an RD cube (K, Nr, Md): pass ``cfg.velocity_axis()``)."""
    ph = tdm_phase(cfg, v)
    ph = ph.reshape(ph.shape + (1,) * (x.ndim - ph.ndim))
    return x * ph


def steering_vector(pos_x: np.ndarray, theta: np.ndarray | float, phi: np.ndarray | float = 0.0,
                    pos_z: np.ndarray | None = None) -> np.ndarray:
    """a(θ, φ), shape (K, *θ.shape)."""
    th = np.asarray(theta, float)
    ph = np.asarray(phi, float)
    px = np.asarray(pos_x, float).reshape((-1,) + (1,) * th.ndim)
    arg = px * (np.sin(th) * np.cos(ph))
    if pos_z is not None:
        arg = arg + np.asarray(pos_z, float).reshape((-1,) + (1,) * th.ndim) * np.sin(ph)
    return np.exp(-2j * np.pi * arg)


@dataclass
class UlaSelection:
    """Channels forming a uniform linear azimuth array.

    ``index`` are virtual-channel indices ordered by x; ``slot`` their position
    in units of ``d`` (wavelengths) from the first element."""

    index: np.ndarray
    slot: np.ndarray
    d: float

    @property
    def n_slots(self) -> int:
        return int(self.slot.max()) + 1


def azimuth_ula(cfg: RadarConfig, d: float | None = None) -> UlaSelection:
    """Select the z = 0 virtual channels with unique x positions and place them
    on a ULA grid of spacing d (default: smallest x spacing). Gaps (sparse
    arrays) are zero-filled by :func:`angle_fft`."""
    xz = cfg.virtual_pos_z
    xs = cfg.virtual_pos
    cand = np.where(np.abs(xz - xz.min()) < 1e-9)[0]
    order = cand[np.argsort(xs[cand], kind="stable")]
    _, first = np.unique(np.round(xs[order], 9), return_index=True)
    idx = order[np.sort(first)]
    x = xs[idx]
    if d is None:
        dx = np.diff(x)
        d = float(dx[dx > 1e-9].min()) if len(x) > 1 else 0.5
    slot = np.rint((x - x[0]) / d).astype(int)
    if not np.allclose(slot * d, x - x[0], atol=1e-6):
        raise ValueError("virtual azimuth positions are not on a uniform grid")
    return UlaSelection(idx, slot, d)


def angle_fft(x: np.ndarray, ula: UlaSelection, n_fft: int = 64, window: str | None = None) -> np.ndarray:
    """Zero-padded, fftshifted angle spectrum over the ULA channels (axis 0).

    Uses the **+j kernel** A[a] = Σ_k g_k·exp(+j2π·k·a/Na) (= Na·IFFT), matched
    to the steering vector exp(−j2π·x·sinθ), so θ > 0 maps to a bin > Na/2 and
    a ↔ sinθ = (a − Na/2)/(Na·d). Normalised by Σw. Output (n_fft, ...)."""
    shp = (ula.n_slots,) + x.shape[1:]
    g = np.zeros(shp, dtype=complex)
    w = get_window(window, ula.n_slots)[ula.slot]
    g[ula.slot] = x[ula.index] * w.reshape((-1,) + (1,) * (x.ndim - 1))
    # +j kernel (matched to the −j steering vector): A[a] = Σ_k g_k·exp(+j2πka/Na)
    X = np.fft.ifft(g, n=n_fft, axis=0) * (n_fft / np.sum(w))
    return np.fft.fftshift(X, axes=0)


def angle_bin_to_sin(a: np.ndarray | float, n_fft: int, d: float) -> np.ndarray:
    """sinθ of (fractional, fftshifted) angle bin a: (a − Na/2)/(Na·d)."""
    return (np.asarray(a, float) - n_fft // 2) / (n_fft * d)


def angle_bin_to_deg(a: np.ndarray | float, n_fft: int, d: float) -> np.ndarray:
    """θ [deg] of angle bin a (|sinθ| clipped to 1)."""
    return np.rad2deg(np.arcsin(np.clip(angle_bin_to_sin(a, n_fft, d), -1, 1)))


def sin_to_angle_bin(sin_theta: np.ndarray | float, n_fft: int, d: float) -> np.ndarray:
    """Fractional angle bin of sinθ (inverse of :func:`angle_bin_to_sin`)."""
    return np.asarray(sin_theta, float) * n_fft * d + n_fft // 2


def covariance(snap: np.ndarray, forward_backward: bool = False, loading: float = 0.0) -> np.ndarray:
    """Sample covariance R = X·Xᴴ / L of snapshots X (K, L), optional
    forward–backward averaging (ULA only) and diagonal loading ``loading·tr(R)/K``."""
    X = np.asarray(snap)
    if X.ndim == 1:
        X = X[:, None]
    R = X @ X.conj().T / X.shape[1]
    if forward_backward:
        J = np.eye(R.shape[0])[::-1]
        R = 0.5 * (R + J @ R.conj() @ J)
    if loading:
        R = R + loading * np.real(np.trace(R)) / R.shape[0] * np.eye(R.shape[0])
    return R


def mvdr_spectrum(R: np.ndarray, A: np.ndarray) -> np.ndarray:
    """MVDR/Capon pseudo-spectrum 1/(aᴴR⁻¹a) for steering matrix A (K, G)."""
    Ri_A = np.linalg.solve(R, A)
    return 1.0 / np.maximum(np.real(np.sum(A.conj() * Ri_A, axis=0)), 1e-300)


def music_spectrum(R: np.ndarray, A: np.ndarray, n_sources: int) -> np.ndarray:
    """MUSIC pseudo-spectrum 1/‖E_nᴴ a‖² (Schmidt, 1986)."""
    w, V = np.linalg.eigh(R)                        # ascending
    En = V[:, : R.shape[0] - n_sources]
    return 1.0 / np.maximum(np.sum(np.abs(En.conj().T @ A) ** 2, axis=0), 1e-300)


def bartlett_spectrum(R: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Conventional beamformer aᴴRa / K²."""
    K = A.shape[0]
    return np.real(np.sum(A.conj() * (R @ A), axis=0)) / K ** 2


def find_peaks_1d(p: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n largest local maxima of a 1-D spectrum."""
    pk = np.where((p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
    return pk[np.argsort(p[pk])[::-1][:n]]


def doa_spectrum(snap: np.ndarray, pos_x: np.ndarray, grid_deg: np.ndarray, method: str = "mvdr",
                 n_sources: int = 1, loading: float = 1e-3, forward_backward: bool = False,
                 pos_z: np.ndarray | None = None) -> np.ndarray:
    """Angular pseudo-spectrum on an azimuth grid [deg] ('bartlett'|'mvdr'|'music')."""
    R = covariance(snap, forward_backward, loading)
    A = steering_vector(pos_x, np.deg2rad(grid_deg), 0.0, pos_z)
    if method == "mvdr":
        return mvdr_spectrum(R, A)
    if method == "music":
        return music_spectrum(R, A, n_sources)
    if method == "bartlett":
        return bartlett_spectrum(R, A)
    raise ValueError(method)


def estimate_elevation(x: np.ndarray, cfg: RadarConfig) -> float:
    """Elevation φ [rad] from virtual pairs with identical x but different z:
    φ = asin( −arg(Σ conj(x_low)·x_high) / (2π·Δz) ). NaN if no such pairs.
    (Assumes cosφ-scaled azimuth phase cancels within a pair.)"""
    xs, zs = cfg.virtual_pos, cfg.virtual_pos_z
    acc, dz = 0j, None
    for i in range(len(xs)):
        for j in range(len(xs)):
            if abs(xs[i] - xs[j]) < 1e-9 and zs[j] - zs[i] > 1e-9:
                acc += np.conj(x[i]) * x[j]
                dz = zs[j] - zs[i]
    if dz is None:
        return float("nan")
    return float(np.arcsin(np.clip(-np.angle(acc) / (2 * np.pi * dz), -1, 1)))


def range_angle_map(rd: np.ndarray, cfg: RadarConfig, n_fft: int = 64, window: str | None = "hann",
                    exclude_zero_doppler: bool = False) -> np.ndarray:
    """Range–azimuth power map (Nr, n_fft): Σ_d |AngleFFT(TDMcomp(X[:, r, d]))|².

    Each Doppler bin d is TDM-compensated with its own velocity v_d."""
    ula = azimuth_ula(cfg)
    Md = rd.shape[-1]
    v = (np.arange(Md) - Md // 2) * cfg.velocity_resolution * cfg.n_chirps / Md
    X = tdm_compensate(rd, cfg, v[None, :])
    A = angle_fft(X, ula, n_fft, window)                       # (Na, Nr, Md)
    pw = np.abs(A) ** 2
    if exclude_zero_doppler:
        pw[..., Md // 2] = 0
    return pw.sum(-1).T
