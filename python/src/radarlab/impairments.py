r"""Hardware and propagation impairments (Phase 2).

* :func:`swerling_gains` — Swerling 1–4 complex RCS fluctuation (1/3 per frame,
  2/4 per chirp; Swerling, 1960). Power gain |g|² is χ²₂/2 (SW1/2) or χ²₄/4
  (SW3/4), mean 1; phase uniform.
* :func:`adc_quantize` — mid-tread uniform quantiser with clipping per I/Q rail.
* :func:`if_filter` — Butterworth IF high-/low-pass along fast time (causal,
  zero initial state → realistic start-of-chirp transient).
* :class:`PhaseNoise` — PLL-like phase noise, L(f) = L0/(1+(f/f_bw)²) + floor;
  the beat signal acquires φ(t) − φ(t−τ) (range-correlation effect; Budge &
  Burt, 1993), applied inside :func:`radarlab.signal.synthesize_cube`.
* :func:`add_leakage` — TX→RX coupling as a static near-range scatterer.
* :func:`iq_imbalance` — y = I + j·g·(Q·cosδ − I·sinδ) ≡ μx + νx*,
  μ = (1 + g·e^{−jδ})/2, ν = (1 − g·e^{jδ})/2; IRR = |μ|²/|ν|².
* :func:`two_ray` — flat-ground multipath (direct, 2× mixed, image paths).
* :func:`fmcw_interference` — another FMCW radar received one-way; IF content
  is kept only where the instantaneous difference frequency lies inside the
  IF bandwidth (ideal LPF approximation), producing the classic chirp bursts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.signal

from .config import C0, RadarConfig
from .scene import Scatterers, Scene, PointTarget
from .signal import SceneLike, _scatter_fn, synthesize_cube


# ------------------------------------------------------------ RCS fluctuation
def swerling_gains(model: int, n_scat: int, n_tx: int, n_chirps: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Complex amplitude gains (S, nTx, M) for Swerling model 0–4."""
    shape = (n_scat, n_tx, n_chirps)
    if model == 0:
        return np.ones(shape, complex)
    per_chirp = model in (2, 4)
    dof = 2 if model in (1, 2) else 4
    s = shape if per_chirp else (n_scat, 1, 1)
    power = rng.chisquare(dof, s) / dof
    g = np.sqrt(power) * np.exp(2j * np.pi * rng.random(s))
    return np.broadcast_to(g, shape).copy()


# ------------------------------------------------------------------- ADC
def adc_quantize(x: np.ndarray, bits: int = 12, full_scale: float | None = None,
                 return_codes: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Uniform quantisation with clipping of each real rail to [−FS, FS).

    LSB = FS/2^(bits−1); codes are integers in [−2^(bits−1), 2^(bits−1)−1].
    ``full_scale`` defaults to 4× the rms of one rail. Returns dequantised
    data (and integer codes if ``return_codes``)."""
    xr = np.asarray(x)
    if full_scale is None:
        rails = np.concatenate([xr.real.ravel(), xr.imag.ravel()]) if np.iscomplexobj(xr) else xr.ravel()
        full_scale = 4 * float(np.sqrt(np.mean(rails ** 2)) or 1.0)
    lsb = full_scale / 2 ** (bits - 1)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

    def q(v: np.ndarray) -> np.ndarray:
        return np.clip(np.round(v / lsb), lo, hi)

    if np.iscomplexobj(xr):
        ci, cq = q(xr.real), q(xr.imag)
        codes = ci + 1j * cq
    else:
        codes = q(xr)
    out = codes * lsb
    return (out, codes) if return_codes else out


# ------------------------------------------------------------- IF filtering
def if_filter(cube: np.ndarray, cfg: RadarConfig, hpf_hz: float | None = 150e3,
              lpf_hz: float | None = None, order: int = 1) -> np.ndarray:
    """Butterworth IF filtering along fast time (last axis), per chirp, causal."""
    y = cube
    nyq = cfg.fs / 2
    if hpf_hz:
        b, a = scipy.signal.butter(order, hpf_hz / nyq, btype="highpass")
        y = scipy.signal.lfilter(b, a, y, axis=-1)
    if lpf_hz and lpf_hz < nyq:
        b, a = scipy.signal.butter(order, lpf_hz / nyq, btype="lowpass")
        y = scipy.signal.lfilter(b, a, y, axis=-1)
    return y


# --------------------------------------------------------------- phase noise
@dataclass
class PhaseNoise:
    """Phase-noise process with two-sided PSD S_φ(f) = L(f) [rad²/Hz],
    L(f) = 10^(L0/10)/(1 + (f/f_bw)²) + 10^(floor/10).

    ``realize`` draws φ(t) on a grid of rate ``fs_gen`` (FFT filtering of white
    noise, |H(f)|² = S_φ(f)·fs_gen) and returns a linear interpolator."""

    l0_dbc_hz: float = -85.0
    loop_bw_hz: float = 1e6
    floor_dbc_hz: float = -140.0
    fs_gen: float = 50e6

    def psd(self, f: np.ndarray) -> np.ndarray:
        return 10 ** (self.l0_dbc_hz / 10) / (1 + (f / self.loop_bw_hz) ** 2) + 10 ** (self.floor_dbc_hz / 10)

    def realize(self, t0: float, t1: float, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        n = int(np.ceil((t1 - t0) * self.fs_gen)) + 2
        nfft = 1 << int(np.ceil(np.log2(n)))
        w = rng.standard_normal(nfft)
        f = np.fft.rfftfreq(nfft, 1 / self.fs_gen)
        phi = np.fft.irfft(np.fft.rfft(w) * np.sqrt(self.psd(f) * self.fs_gen), nfft)[:n]
        tg = t0 + np.arange(n) / self.fs_gen
        return lambda t: np.interp(t, tg, phi)


# ------------------------------------------------------------------ leakage
def add_leakage(cube: np.ndarray, cfg: RadarConfig, distance: float = 0.05, power_db: float = 40.0,
                phase0: float = 0.0, t_frame: float = 0.0) -> np.ndarray:
    """Add TX→RX leakage: a static boresight scatterer at ``distance`` with
    per-sample power ``power_db`` relative to the unit noise variance."""
    leak = synthesize_cube(cfg, [PointTarget((0, distance, 0), rcs=1.0, phase0=phase0)], t_frame,
                           snr_ref_db=power_db, ref_range=distance, noise=False)
    return cube + leak


# ------------------------------------------------------------- IQ imbalance
def iq_imbalance(x: np.ndarray, gain_db: float = 0.5, phase_deg: float = 3.0) -> np.ndarray:
    """Receiver IQ gain/phase imbalance: y = I + j·g·(Q·cosδ − I·sinδ)."""
    g = 10 ** (gain_db / 20)
    d = np.deg2rad(phase_deg)
    return x.real + 1j * g * (x.imag * np.cos(d) - x.real * np.sin(d))


def image_rejection_db(gain_db: float, phase_deg: float) -> float:
    """IRR = |μ|²/|ν|² in dB for :func:`iq_imbalance`."""
    g = 10 ** (gain_db / 20)
    d = np.deg2rad(phase_deg)
    mu = (1 + g * np.exp(-1j * d)) / 2
    nu = (1 - g * np.exp(1j * d)) / 2
    return float(20 * np.log10(abs(mu) / max(abs(nu), 1e-300)))


# --------------------------------------------------------------- multipath
def two_ray(scene: SceneLike, radar_height: float, reflection_coeff: complex = -1.0
            ) -> Callable[[np.ndarray], Scatterers]:
    """Flat-ground two-ray model (ground at z = −radar_height).

    For each scatterer at p with image p' = (x, y, −2h − z):
      * direct:  round-trip 2·R_d, amplitude 1
      * mixed:   round-trip R_d + R_i (two paths, amplitude 2Γ) — represented by
        an equivalent scatterer on the direct line of sight at range
        (R_d+R_i)/2 with radial velocity (vr_d+vr_i)/2 (approximation: the
        angle of arrival is the direct one)
      * image:   round-trip 2·R_i at the image position, amplitude Γ²
    Returns a callable usable as ``scene`` in :func:`synthesize_cube`.
    """
    fn = _scatter_fn(scene)
    G = complex(reflection_coeff)

    def f(t: np.ndarray) -> Scatterers:
        s = fn(t)
        h = radar_height
        pi = s.pos.copy()
        pi[..., 2] = -2 * h - s.pos[..., 2]
        vi = s.vel.copy()
        vi[..., 2] = -s.vel[..., 2]
        Rd = np.linalg.norm(s.pos, axis=-1)
        Ri = np.linalg.norm(pi, axis=-1)
        ud = s.pos / np.maximum(Rd, 1e-12)[..., None]
        vrd = np.sum(s.pos * s.vel, -1) / np.maximum(Rd, 1e-12)
        vri = np.sum(pi * vi, -1) / np.maximum(Ri, 1e-12)
        Rm = 0.5 * (Rd + Ri)
        pm = ud * Rm[..., None]
        vm = s.vel + ((0.5 * (vrd + vri) - vrd)[..., None]) * ud
        # amplitude law in synthesize_cube uses R^-2 at the equivalent range:
        # scale rcs so that sqrt(rcs_eq)/Rm^2 = |2G|·sqrt(rcs)/(Rd·Ri)
        rcs_m = s.rcs * abs(2 * G) ** 2 * (Rm ** 2 / (Rd * Ri)) ** 2
        rcs_i = s.rcs * abs(G) ** 4
        return Scatterers(np.concatenate([s.pos, pm, pi]), np.concatenate([s.vel, vm, vi]),
                          np.concatenate([s.rcs, rcs_m, rcs_i]),
                          np.concatenate([s.phase0, s.phase0 + np.angle(G), s.phase0 + 2 * np.angle(G)]),
                          np.concatenate([s.label, s.label, s.label]))

    return f


# ------------------------------------------------------------- interference
@dataclass
class FmcwInterferer:
    """Interfering FMCW radar (one-way path).

    fc, slope, chirp_period, ramp_time: its waveform; t0: start of its first
    chirp [s] (absolute); distance: one-way path length [m]; inr_db: per-sample
    interference-to-noise ratio before IF filtering; theta_deg: azimuth."""

    fc: float = 77.05e9
    slope: float = 20e12
    chirp_period: float = 40e-6
    ramp_time: float = 35e-6
    t0: float = 0.0
    distance: float = 30.0
    inr_db: float = 30.0
    theta_deg: float = 0.0


def fmcw_interference(cfg: RadarConfig, intf: FmcwInterferer, t_frame: float = 0.0,
                      if_bandwidth: float | None = None, rng: np.random.Generator | int | None = 0
                      ) -> np.ndarray:
    """IF-domain interference cube (K, M, N) from one FMCW interferer.

    Victim LO phase φ_v(t_f) = 2π(fc·t_f + ½·S·t_f²); received interferer phase
    φ_i(u) = 2π(fc_i·u + ½·S_i·u²) with u its local chirp time after the
    one-way delay. IF = A·exp(j(φ_v − φ_i) + jψ_j) (ψ_j random per interferer
    chirp), kept only where |f_v − f_i| < if_bandwidth (default fs/2)."""
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    bw = cfg.fs / 2 if if_bandwidth is None else if_bandwidth
    tm = cfg.chirp_start_times(t_frame)                      # (nT, M)
    tf = cfg.fast_times()
    ta = tm[:, :, None] + tf[None, None, :]                  # absolute time
    u_all = ta - intf.distance / C0 - intf.t0
    j = np.floor(u_all / intf.chirp_period)
    u = u_all - j * intf.chirp_period
    active = (u >= 0) & (u < intf.ramp_time) & (u_all >= 0)
    fv = cfg.fc + cfg.slope * tf[None, None, :]
    fi = intf.fc + intf.slope * u
    keep = active & (np.abs(fv - fi) < bw)
    jmax = int(j.max()) + 2 if j.size else 1
    jmin = int(j.min())
    psi = 2 * np.pi * g.random(jmax - jmin + 1)
    # phase difference with large constant terms removed modulo 2π
    dphi = 2 * np.pi * ((cfg.fc - intf.fc) * u + cfg.fc * (tf[None, None, :] - u)
                        + 0.5 * cfg.slope * tf[None, None, :] ** 2 - 0.5 * intf.slope * u ** 2)
    A = 10 ** (intf.inr_db / 20)
    sig = np.where(keep, A * np.exp(1j * (dphi + psi[(j - jmin).astype(int)])), 0)   # (nT, M, N)
    th = np.deg2rad(intf.theta_deg)
    # one-way: only the RX array phase (TX of the victim is irrelevant)
    steer = np.exp(2j * np.pi * cfg.rx_pos * np.sin(th))
    out = sig[:, None, :, :] * steer[None, :, None, None]
    return out.reshape(cfg.n_virtual, cfg.n_chirps, cfg.n_samples)
