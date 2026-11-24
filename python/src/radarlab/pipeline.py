"""One-call simulation and processing chain.

``simulate_frame`` → ADC cube (K, M, N); ``process_frame`` → :class:`FrameResult`
(range FFT, RD cube, power map, CFAR, clusters, point cloud); ``Simulator``
steps frames through time, optionally feeding a :class:`~radarlab.track.Tracker`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import dsp
from .config import RadarConfig
from .signal import AmplitudeMode, SceneLike, synthesize_cube
from .track import Tracker, Track


@dataclass
class ProcessingParams:
    """Signal-processing parameters (defaults follow the brief)."""

    range_window: str | None = "hann"
    doppler_window: str | None = "hann"
    angle_window: str | None = None
    n_angle_fft: int = 64
    mti: str | None = None                # None | 'mean' | 'two_pulse'
    cfar_kind: dsp.cfar.CfarKind = "ca"
    cfar_guard: tuple[int, int] = (2, 2)
    cfar_train: tuple[int, int] = (8, 4)
    cfar_pfa: float = 1e-5
    cfar_os_rank: float = 0.75
    cfar_gamma_exact: bool = False         # CA α for Gamma(K) noise
    clustering: str = "connected"          # 'connected' | 'dbscan' | 'peaks'
    dbscan_eps: float = 1.5
    max_detections: int = 256


@dataclass
class SimParams:
    """Synthesis parameters."""

    amplitude_mode: AmplitudeMode = "snr"
    snr_ref_db: float = 20.0
    ref_range: float = 10.0
    noise: bool = True
    swerling: int = 0
    phase_noise: Any = None


@dataclass
class FrameResult:
    range_fft: np.ndarray            # (K, M, Nr)
    rd: np.ndarray                   # (K, Nr, Md)
    power: np.ndarray                # (Nr, Md)
    cfar: dsp.CfarResult
    clusters: list[dsp.Cluster]
    points: np.ndarray               # structured, dsp.POINT_DTYPE
    t: float = 0.0
    tracks: list[Track] = field(default_factory=list)

    @property
    def power_db(self) -> np.ndarray:
        return dsp.to_db(self.power)


def simulate_frame(cfg: RadarConfig, scene: SceneLike, t_frame: float = 0.0,
                   sim: SimParams | None = None, rng: np.random.Generator | int | None = None
                   ) -> np.ndarray:
    """Synthesise one ADC frame (see :func:`radarlab.signal.synthesize_cube`)."""
    s = sim or SimParams()
    return synthesize_cube(cfg, scene, t_frame, s.amplitude_mode, s.snr_ref_db, s.ref_range,
                           s.noise, rng, s.swerling, s.phase_noise)


def process_frame(cube: np.ndarray, cfg: RadarConfig, params: ProcessingParams | None = None,
                  t: float = 0.0) -> FrameResult:
    """Run the full chain on one ADC cube (K, M, N)."""
    p = params or ProcessingParams()
    rf = dsp.range_fft(cube, p.range_window)
    if p.mti:
        rf = dsp.mti(rf, axis=-2, kind=p.mti)
    rd = dsp.doppler_fft(rf, p.doppler_window)
    P = dsp.rd_power(rd)
    cf = dsp.cfar_2d(P, p.cfar_guard, p.cfar_train, p.cfar_pfa, p.cfar_kind, p.cfar_os_rank,
                     n_channels=rd.shape[0] if p.cfar_gamma_exact else 1)
    if p.clustering == "dbscan":
        cl = dsp.cluster_dbscan(cf.mask, P, p.dbscan_eps)
    elif p.clustering == "peaks":
        cells = dsp.local_maxima(P, cf.mask)
        cl = [dsp.Cluster(int(r), int(d), float(P[r, d]), 1, np.array([[r, d]])) for r, d in cells]
        cl.sort(key=lambda c: -c.power)
    else:
        cl = dsp.cluster_connected(cf.mask, P)
    cl = cl[: p.max_detections]
    cells = np.array([[c.range_bin, c.doppler_bin] for c in cl], int).reshape(-1, 2)
    pts = dsp.point_cloud(rd, P, cells, cfg, cf.noise, p.n_angle_fft, p.angle_window)
    return FrameResult(rf, rd, P, cf, cl, pts, t)


def measurements_from_points(points: np.ndarray) -> np.ndarray:
    """(n, 3) [R, θ, vr] from a point cloud."""
    return np.column_stack([points["range"], points["azimuth"], points["velocity"]])


class Simulator:
    """Frame-stepping simulator: t_k = t0 + k·framePeriod.

    Each :meth:`step` synthesises a frame of ``scene`` at the current time,
    processes it and (optionally) updates the tracker with the point cloud.
    """

    def __init__(self, cfg: RadarConfig, scene: SceneLike, processing: ProcessingParams | None = None,
                 sim: SimParams | None = None, tracker: Tracker | None = None, seed: int | None = 0,
                 t0: float = 0.0) -> None:
        self.cfg, self.scene = cfg, scene
        self.processing = processing or ProcessingParams()
        self.sim = sim or SimParams()
        self.tracker = tracker
        self.rng = np.random.default_rng(seed)
        self.t = t0
        self.frame = 0

    def step(self) -> FrameResult:
        cube = simulate_frame(self.cfg, self.scene, self.t, self.sim, self.rng)
        res = process_frame(cube, self.cfg, self.processing, self.t)
        if self.tracker is not None:
            dt = self.cfg.frame_period if self.frame > 0 else 0.0
            res.tracks = self.tracker.step(measurements_from_points(res.points), dt)
        self.t += self.cfg.frame_period
        self.frame += 1
        return res

    def run(self, n_frames: int) -> list[FrameResult]:
        return [self.step() for _ in range(n_frames)]
