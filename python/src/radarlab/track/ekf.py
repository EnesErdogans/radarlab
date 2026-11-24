r"""Multi-target tracker: EKF (constant velocity) + GNN association + M/N logic.

State x = [px, py, vx, vy]ᵀ (ground plane), white-noise-acceleration CV model
(Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and
Navigation*, 2001, §6.3.2)::

    F = [[I, T·I], [0, I]],   Q = q·[[T³/3·I, T²/2·I], [T²/2·I, T·I]]

Measurement z = [R, θ, vr]ᵀ (polar + radial velocity)::

    R = √(px²+py²),  θ = atan2(px, py),  vr = (px·vx + py·vy)/R

with Jacobian H = ∂h/∂x (see :func:`meas_jacobian`). Angles are wrapped in
the innovation. Association: global nearest neighbour (Hungarian algorithm)
on the Mahalanobis distance d² = νᵀS⁻¹ν with χ²₃ gate. Track management:
tentative → confirmed after M hits in the last N scans; deleted after
``max_misses`` consecutive misses (confirmed) or failing M/N (tentative).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
import scipy.stats


def cv_model(T: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """(F, Q) of the 2-D CV model, sample time T, accel PSD q [m²/s³]."""
    I = np.eye(2)
    F = np.block([[I, T * I], [np.zeros((2, 2)), I]])
    Q = q * np.block([[T ** 3 / 3 * I, T ** 2 / 2 * I], [T ** 2 / 2 * I, T * I]])
    return F, Q


def meas_fn(x: np.ndarray) -> np.ndarray:
    """h(x) = [R, θ, vr]."""
    px, py, vx, vy = x
    R = np.hypot(px, py)
    return np.array([R, np.arctan2(px, py), (px * vx + py * vy) / R])


def meas_jacobian(x: np.ndarray) -> np.ndarray:
    """H = ∂h/∂x (3×4)."""
    px, py, vx, vy = x
    R2 = px ** 2 + py ** 2
    R = np.sqrt(R2)
    vr = (px * vx + py * vy) / R
    return np.array([
        [px / R, py / R, 0, 0],
        [py / R2, -px / R2, 0, 0],
        [vx / R - vr * px / R2, vy / R - vr * py / R2, px / R, py / R],
    ])


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Track:
    id: int
    x: np.ndarray
    P: np.ndarray
    status: str = "tentative"          # tentative | confirmed | deleted
    hits: list[bool] = field(default_factory=list)
    misses: int = 0
    age: int = 0

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]


@dataclass
class TrackerParams:
    """q: accel PSD [m²/s³]; sigma_*: measurement std (R [m], θ [rad], vr [m/s]);
    gate_prob: χ²₃ gate probability; confirm (M, N); max_misses; init_vt_std:
    initial tangential-velocity std [m/s]."""

    q: float = 1.0
    sigma_r: float = 0.2
    sigma_theta: float = np.deg2rad(1.5)
    sigma_vr: float = 0.1
    gate_prob: float = 0.999
    confirm_m: int = 3
    confirm_n: int = 5
    max_misses: int = 3
    init_vt_std: float = 5.0


class Tracker:
    """EKF-CV multi-target tracker with GNN association."""

    def __init__(self, params: TrackerParams | None = None) -> None:
        self.p = params or TrackerParams()
        self.tracks: list[Track] = []
        self._next_id = 1
        self.R = np.diag([self.p.sigma_r ** 2, self.p.sigma_theta ** 2, self.p.sigma_vr ** 2])
        self.gate = float(scipy.stats.chi2.ppf(self.p.gate_prob, 3))

    def _init_track(self, z: np.ndarray) -> Track:
        R, th, vr = z
        u = np.array([np.sin(th), np.cos(th)])
        t = np.array([np.cos(th), -np.sin(th)])
        x = np.array([R * u[0], R * u[1], vr * u[0], vr * u[1]])
        # position covariance from polar errors, velocity: radial known, tangential unknown
        J = np.column_stack([u, R * t])
        Pp = J @ np.diag([self.p.sigma_r ** 2, self.p.sigma_theta ** 2]) @ J.T
        Pv = np.outer(u, u) * self.p.sigma_vr ** 2 + np.outer(t, t) * self.p.init_vt_std ** 2
        P = np.zeros((4, 4))
        P[:2, :2], P[2:, 2:] = Pp, Pv
        tr = Track(self._next_id, x, P, hits=[True])
        self._next_id += 1
        return tr

    def predict(self, dt: float) -> None:
        F, Q = cv_model(dt, self.p.q)
        for tr in self.tracks:
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + Q

    def _innov(self, tr: Track, z: np.ndarray):
        H = meas_jacobian(tr.x)
        nu = z - meas_fn(tr.x)
        nu[1] = _wrap(nu[1])
        S = H @ tr.P @ H.T + self.R
        return nu, S, H

    def update(self, measurements: np.ndarray) -> None:
        """Associate measurements (n, 3) = [R, θ, vr] and update / manage tracks."""
        Z = np.asarray(measurements, float).reshape(-1, 3)
        nT, nZ = len(self.tracks), len(Z)
        assigned_t: dict[int, int] = {}
        if nT and nZ:
            big = 1e9
            cost = np.full((nT, nZ), big)
            for i, tr in enumerate(self.tracks):
                for j in range(nZ):
                    nu, S, _ = self._innov(tr, Z[j])
                    d2 = float(nu @ np.linalg.solve(S, nu))
                    if d2 <= self.gate:
                        cost[i, j] = d2 + np.log(np.linalg.det(S))
            ri, ci = scipy.optimize.linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < big:
                    assigned_t[i] = j
        used = set(assigned_t.values())
        for i, tr in enumerate(self.tracks):
            tr.age += 1
            if i in assigned_t:
                nu, S, H = self._innov(tr, Z[assigned_t[i]])
                K = tr.P @ H.T @ np.linalg.inv(S)
                tr.x = tr.x + K @ nu
                I_KH = np.eye(4) - K @ H
                tr.P = I_KH @ tr.P @ I_KH.T + K @ self.R @ K.T      # Joseph form
                tr.hits.append(True)
                tr.misses = 0
            else:
                tr.hits.append(False)
                tr.misses += 1
            tr.hits = tr.hits[-self.p.confirm_n:]
            if tr.status == "tentative":
                if sum(tr.hits) >= self.p.confirm_m:
                    tr.status = "confirmed"
                elif len(tr.hits) >= self.p.confirm_n or tr.misses >= self.p.max_misses:
                    tr.status = "deleted"
            elif tr.misses >= self.p.max_misses:
                tr.status = "deleted"
        self.tracks = [t for t in self.tracks if t.status != "deleted"]
        for j in range(nZ):
            if j not in used:
                self.tracks.append(self._init_track(Z[j]))

    def step(self, measurements: np.ndarray, dt: float) -> list[Track]:
        """Predict by dt, then update. Returns confirmed tracks."""
        self.predict(dt)
        self.update(measurements)
        return self.confirmed()

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.status == "confirmed"]
