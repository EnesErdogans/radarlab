r"""Exact FMCW TDM-MIMO beat-signal synthesis.

Model (``docs/PROJECT_BRIEF.md``, binding). For scatterer s, TX t, chirp m and
ADC sample n::

    t_{m,t} = t_frame + m·Te + t·Tc                 chirp start
    t_f     = adcStartTime + n / fs                 fast time from chirp start
    R(t)    = R_m + vr_m·(t − t_{m,t})              (R_m, vr_m at t_{m,t})
    τ       = 2·R(t_{m,t} + t_f) / c
    x[k,m,n] = Σ_s A_s · exp( j·[ 2π(fc·τ + S·τ·t_f − ½·S·τ²)
                                  + 2π·(pos_k·sinθ·cosφ + posz_k·sinφ) + φ0 ] )

with k = t·nRx + r, pos_k = txPos[t] + rxPos[r] (wavelengths). θ, φ are the
scatterer's azimuth/elevation at t_{m,t} (constant within a chirp). The z-term
(``posz_k·sinφ``) is an extension for arrays with elevated elements; it is zero
for planar-x arrays and therefore consistent with the brief.

Amplitude ``A_s`` (evaluated at t_{m,t})::

    'snr'      A = sqrt(σ)·(R_ref/R)²·10^(snrRefDb/20)
    'physical' A = sqrt(Pr/Pn),  Pr = Pt·Gt·Gr·λ²·σ/((4π)³·R⁴·L),  Pn = k·T0·F·fs

Noise is circular complex white Gaussian with variance 1 per complex sample.
For ``iq_mode='real'`` the output is the real part of the complex cube
including noise (so real noise variance is ½).

Computation is vectorised: per TX and per chunk of scatterers, the fast-time
factor E1[s,m,n] and the spatial factor E2[s,m,r] are formed and contracted
over s with a batched matrix product (M, R, S) @ (M, S, N).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Literal

import numpy as np

from .config import C0, RadarConfig
from .scene import Scatterers, Scene, Target

AmplitudeMode = Literal["snr", "physical"]
SceneLike = Scene | Target | Iterable[Target] | Callable[[np.ndarray], Scatterers]


def _scatter_fn(scene: SceneLike) -> Callable[[np.ndarray], Scatterers]:
    if isinstance(scene, (Scene, Target)):
        return scene.scatterers
    if callable(scene):
        return scene
    return Scene(list(scene)).scatterers


def amplitude(rcs: np.ndarray, rng_m: np.ndarray, cfg: RadarConfig,
              mode: AmplitudeMode = "snr", snr_ref_db: float = 20.0,
              ref_range: float = 10.0) -> np.ndarray:
    """Per-sample signal amplitude A (noise variance 1), see module docstring."""
    rcs = np.asarray(rcs, float)
    R = np.maximum(np.asarray(rng_m, float), 1e-6)
    if mode == "snr":
        return np.sqrt(rcs) * (ref_range / R) ** 2 * 10 ** (snr_ref_db / 20)
    if mode == "physical":
        lam = cfg.wavelength
        pt = 10 ** ((cfg.tx_power_dbm - 30) / 10)
        g = 10 ** ((cfg.tx_gain_dbi + cfg.rx_gain_dbi) / 10)
        loss = 10 ** (cfg.system_loss_db / 10)
        pr = pt * g * lam ** 2 * rcs / ((4 * np.pi) ** 3 * R ** 4 * loss)
        return np.sqrt(pr / cfg.noise_power_w())
    raise ValueError(f"unknown amplitude mode {mode!r}")


def snr_per_sample_db(rcs: float, rng_m: float, cfg: RadarConfig, mode: AmplitudeMode = "snr",
                      snr_ref_db: float = 20.0, ref_range: float = 10.0) -> float:
    """Per-complex-sample SNR A²/σ² [dB] of a scatterer."""
    a = amplitude(np.array(rcs), np.array(rng_m), cfg, mode, snr_ref_db, ref_range)
    return float(20 * np.log10(a))


def complex_noise(shape: tuple[int, ...], rng: np.random.Generator, var: float = 1.0) -> np.ndarray:
    """Circular complex white Gaussian noise, E|w|² = var."""
    s = np.sqrt(var / 2)
    return s * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def synthesize_cube(cfg: RadarConfig, scene: SceneLike, t_frame: float = 0.0,
                    amplitude_mode: AmplitudeMode = "snr", snr_ref_db: float = 20.0,
                    ref_range: float = 10.0, noise: bool = True,
                    rng: np.random.Generator | int | None = None, swerling: int = 0,
                    phase_noise: Any = None, max_chunk_elems: int = 1 << 22,
                    noise_var: float = 1.0) -> np.ndarray:
    """Synthesise one frame of IF data.

    Parameters
    ----------
    cfg : radar configuration.
    scene : :class:`Scene`, a :class:`Target`, a list of targets or a callable
        ``t -> Scatterers``.
    t_frame : frame start time [s].
    amplitude_mode, snr_ref_db, ref_range : amplitude model (module docstring).
    noise : add complex white Gaussian noise of variance ``noise_var``.
    rng : ``numpy.random.Generator`` or seed (deterministic).
    swerling : RCS fluctuation model 0–4 (1/3 per frame, 2/4 per chirp), see
        :func:`radarlab.impairments.swerling_gains`.
    phase_noise : optional :class:`radarlab.impairments.PhaseNoise`; the beat
        signal acquires ``φ(t) − φ(t − τ)`` (range-correlation effect).

    Returns
    -------
    cube : complex128 array (K, M, N) (float64 for ``iq_mode='real'``).
    """
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    fn = _scatter_fn(scene)
    nT, nR, M, N = cfg.n_tx, cfg.n_rx, cfg.n_chirps, cfg.n_samples
    tm = cfg.chirp_start_times(t_frame)                       # (nT, M)
    tf = cfg.fast_times()                                     # (N,)
    sc = fn(tm.reshape(-1))
    S = sc.n
    cube = np.zeros((nT, nR, M, N), dtype=np.complex128)
    if S:
        pos = sc.pos.reshape(S, nT, M, 3)
        R = np.linalg.norm(pos, axis=-1)                      # (S, nT, M)
        vr = np.einsum("stmj,stmj->stm", pos, sc.vel.reshape(S, nT, M, 3)) / np.maximum(R, 1e-12)
        th = np.arctan2(pos[..., 0], pos[..., 1])
        ph = np.arctan2(pos[..., 2], np.hypot(pos[..., 0], pos[..., 1]))
        u_x = np.sin(th) * np.cos(ph)
        u_z = np.sin(ph)
        A = amplitude(sc.rcs.reshape(S, nT, M), R, cfg, amplitude_mode, snr_ref_db, ref_range)
        Ac = A.astype(np.complex128) * np.exp(1j * sc.phase0)[:, None, None]
        if swerling:
            from .impairments import swerling_gains
            Ac = Ac * swerling_gains(swerling, S, nT, M, g)
        rxz = cfg.rx_pos_z if cfg.rx_pos_z is not None else np.zeros(nR)
        txz = cfg.tx_pos_z if cfg.tx_pos_z is not None else np.zeros(nT)
        pn_fn = None
        if phase_noise is not None:
            pn_fn = phase_noise.realize(t_frame - 1e-5, t_frame + M * cfg.te + cfg.chirp_period, g)
        chunk = max(1, int(max_chunk_elems // (M * N)))
        two_pi = 2 * np.pi
        for t in range(nT):
            for s0 in range(0, S, chunk):
                s1 = min(S, s0 + chunk)
                tau = 2.0 * (R[s0:s1, t, :, None] + vr[s0:s1, t, :, None] * tf[None, None, :]) / C0
                psi = two_pi * (cfg.fc * tau + cfg.slope * tau * tf - 0.5 * cfg.slope * tau ** 2)
                if pn_fn is not None:
                    ta = tm[t][:, None] + tf[None, :]
                    psi = psi + pn_fn(ta)[None] - pn_fn(ta[None] - tau)
                e1 = Ac[s0:s1, t, :, None] * np.exp(1j * psi)             # (s, M, N)
                spat = (cfg.tx_pos[t] + cfg.rx_pos)[None, None, :] * u_x[s0:s1, t, :, None] \
                    + (txz[t] + rxz)[None, None, :] * u_z[s0:s1, t, :, None]
                e2 = np.exp(1j * two_pi * spat)                              # (s, M, nR)
                cube[t] += np.matmul(e2.transpose(1, 2, 0), e1.transpose(1, 0, 2)).transpose(1, 0, 2)
    cube = cube.reshape(nT * nR, M, N)
    if noise:
        cube += complex_noise(cube.shape, g, noise_var)
    if cfg.iq_mode == "real":
        return cube.real.copy()
    return cube


def beat_sample(cfg: RadarConfig, R: float, vr: float, theta: float, k: int, m: int, n: int,
                A: float = 1.0, phi: float = 0.0, phase0: float = 0.0, posz: float = 0.0) -> complex:
    """Direct (scalar) evaluation of the brief's beat formula for one sample.

    ``R``/``vr`` are the scatterer's range/radial velocity at the chirp start
    t_{m,t}. Used for cross-checking the vectorised synthesiser.
    """
    t_f = cfg.adc_start_time + n / cfg.fs
    tau = 2 * (R + vr * t_f) / C0
    pos_k = cfg.virtual_pos[k]
    psi = 2 * np.pi * (cfg.fc * tau + cfg.slope * tau * t_f - 0.5 * cfg.slope * tau ** 2)
    psi += 2 * np.pi * (pos_k * np.sin(theta) * np.cos(phi) + posz * np.sin(phi)) + phase0
    return complex(A * np.exp(1j * psi))
