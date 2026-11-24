"""Targets and scenes: time-varying point-scatterer models.

Every target exposes :meth:`Target.scatterers`, which evaluates all of its
point scatterers at an arbitrary 1-D array of times ``t`` (vectorised) and
returns positions and *analytic* velocities (needed for the radial velocity
``vr = p·v/|p|`` in the beat model; see ``signal.py``).

Coordinate frame (brief): radar at the origin, boresight +y, x right, z up.
Extended targets are described in a body frame (bx right, by forward, bz up)
that is rotated about z by the heading ``h = atan2(vx, vy)``::

    world = p(t) + Rz(h) · body,   Rz(h) = [[cos h, sin h, 0], [−sin h, cos h, 0], [0, 0, 1]]

Models
------
* :class:`PointTarget` — single scatterer, constant acceleration trajectory.
* :class:`Car` — body corners/bumpers plus optional rotating wheel-rim points.
* :class:`Pedestrian` — torso bob + pendulum arms/legs; gait period from the
  Boulic (1990) relation ``T = 1.346/sqrt(v/H_t)`` as used by V.C. Chen,
  *The Micro-Doppler Effect in Radar* (2nd ed., 2019), ch. 4 (simplified).
* :class:`Drone` — multirotor; each blade is a rigid rod discretised into
  point scatterers. Summing their returns is a midpoint-rule evaluation of the
  rotating-blade integral of Martin & Mulgrew (1990), reproducing the
  sinc-shaped "blade flashes" when a blade is normal to the line of sight
  and a micro-Doppler extent of ±2·v_tip/λ, v_tip = Ω·L.
* :class:`Bird` — body plus two flapping wings (rods, sinusoidal flap angle).
* :class:`ClutterField` / :func:`static_clutter` — many static scatterers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import C0

ArrayLike = np.ndarray | Sequence[float] | float


@dataclass
class Scatterers:
    """Point scatterers sampled at T time instants.

    Attributes
    ----------
    pos : (S, T, 3) positions [m]
    vel : (S, T, 3) velocities [m/s]
    rcs : (S, T) RCS [m²]
    phase0 : (S,) scatterer phase φ0 [rad]
    label : (S,) integer target index inside a :class:`Scene` (or −1)
    """

    pos: np.ndarray
    vel: np.ndarray
    rcs: np.ndarray
    phase0: np.ndarray
    label: np.ndarray

    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

    def range(self) -> np.ndarray:
        """|p|, shape (S, T)."""
        return np.linalg.norm(self.pos, axis=-1)

    def radial_velocity(self) -> np.ndarray:
        """vr = p·v / |p| (positive = receding), shape (S, T)."""
        r = np.maximum(self.range(), 1e-12)
        return np.einsum("stj,stj->st", self.pos, self.vel) / r

    def azimuth(self) -> np.ndarray:
        """θ = atan2(x, y), shape (S, T)."""
        return np.arctan2(self.pos[..., 0], self.pos[..., 1])

    def elevation(self) -> np.ndarray:
        """φ = atan2(z, hypot(x, y)), shape (S, T)."""
        return np.arctan2(self.pos[..., 2], np.hypot(self.pos[..., 0], self.pos[..., 1]))

    @staticmethod
    def concatenate(items: Iterable["Scatterers"]) -> "Scatterers":
        items = list(items)
        if not items:
            return Scatterers(np.zeros((0, 1, 3)), np.zeros((0, 1, 3)), np.zeros((0, 1)),
                              np.zeros(0), np.zeros(0, dtype=int))
        return Scatterers(
            np.concatenate([i.pos for i in items]), np.concatenate([i.vel for i in items]),
            np.concatenate([i.rcs for i in items]), np.concatenate([i.phase0 for i in items]),
            np.concatenate([i.label for i in items]))

    def subset(self, idx: slice | np.ndarray) -> "Scatterers":
        return Scatterers(self.pos[idx], self.vel[idx], self.rcs[idx], self.phase0[idx],
                          self.label[idx])


def _t1d(t: ArrayLike) -> np.ndarray:
    return np.atleast_1d(np.asarray(t, dtype=float)).reshape(-1)


def _vec3(v: ArrayLike) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 2:
        a = np.array([a[0], a[1], 0.0])
    if a.size != 3:
        raise ValueError("expected a 2- or 3-vector")
    return a


class Target(ABC):
    """Abstract target: anything that yields point scatterers at times t."""

    @abstractmethod
    def scatterers(self, t: ArrayLike) -> Scatterers:
        """Evaluate scatterers at times ``t`` (1-D array, seconds)."""

    def center(self, t: ArrayLike) -> np.ndarray:
        """Reference (body-centre) position, shape (T, 3)."""
        return self.scatterers(t).pos.mean(axis=0)


class BodyTarget(Target):
    """Rigid-body trajectory ``p(t) = p0 + v0·t + ½·a·t²`` with constant heading.

    Subclasses implement :meth:`_body` returning body-frame positions and
    velocities of their scatterers relative to the body centre.
    """

    def __init__(self, position: ArrayLike, velocity: ArrayLike = (0, 0, 0),
                 accel: ArrayLike = (0, 0, 0), heading: float | None = None,
                 seed: int | None = 0) -> None:
        self.p0 = _vec3(position)
        self.v0 = _vec3(velocity)
        self.a = _vec3(accel)
        if heading is None:
            heading = float(np.arctan2(self.v0[0], self.v0[1])) if np.hypot(*self.v0[:2]) > 0 else 0.0
        self.heading = float(heading)
        self.rng = np.random.default_rng(seed)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v0))

    def _rot(self) -> np.ndarray:
        h = self.heading
        c, s = np.cos(h), np.sin(h)
        # columns: images of body x (right), body y (forward), body z (up)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def trajectory(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.p0 + self.v0 * t[:, None] + 0.5 * self.a * t[:, None] ** 2
        v = self.v0 + self.a * t[:, None]
        return p, v

    def center(self, t: ArrayLike) -> np.ndarray:
        return self.trajectory(_t1d(t))[0]

    @abstractmethod
    def _body(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (pos_body (S,T,3), vel_body (S,T,3), rcs (S,) or (S,T), phase0 (S,))."""

    def scatterers(self, t: ArrayLike) -> Scatterers:
        t = _t1d(t)
        pc, vc = self.trajectory(t)
        pb, vb, rcs, ph = self._body(t)
        rot = self._rot()
        pos = pc[None] + pb @ rot.T
        vel = vc[None] + vb @ rot.T
        rcs = np.broadcast_to(np.asarray(rcs, float).reshape(len(ph), -1), pos.shape[:2]).copy()
        return Scatterers(pos, vel, rcs, np.asarray(ph, float), -np.ones(len(ph), dtype=int))


class PointTarget(BodyTarget):
    """Single point scatterer.

    Parameters
    ----------
    position, velocity, accel : 3-vectors [m], [m/s], [m/s²] at t = 0.
    rcs : RCS σ [m²]. phase0 : scatterer phase φ0 [rad].
    """

    def __init__(self, position: ArrayLike, velocity: ArrayLike = (0, 0, 0),
                 rcs: float = 1.0, phase0: float = 0.0, accel: ArrayLike = (0, 0, 0)) -> None:
        super().__init__(position, velocity, accel, heading=0.0, seed=None)
        self.rcs = float(rcs)
        self.phase0 = float(phase0)

    @classmethod
    def from_polar(cls, rng: float, azimuth_deg: float, radial_velocity: float = 0.0,
                   elevation_deg: float = 0.0, rcs: float = 1.0, phase0: float = 0.0) -> "PointTarget":
        """Create a target at (R, θ, φ) moving purely radially with ``vr``."""
        th, ph = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
        u = np.array([np.sin(th) * np.cos(ph), np.cos(th) * np.cos(ph), np.sin(ph)])
        return cls(rng * u, radial_velocity * u, rcs, phase0)

    def _body(self, t: np.ndarray):
        z = np.zeros((1, t.size, 3))
        return z, z.copy(), np.array([self.rcs]), np.array([self.phase0])


class Car(BodyTarget):
    """Passenger car as a set of dominant scatterers.

    Body scatterers: rear/front bumper centres, four corners and the roof
    (fractions of ``rcs_total``). With ``wheels=True`` each wheel adds one
    rim point rolling without slip (angular rate ``|v|/r_wheel``), producing
    the characteristic wheel micro-Doppler between 0 and 2|v|.
    """

    def __init__(self, position: ArrayLike, velocity: ArrayLike = (0, 0, 0),
                 length: float = 4.5, width: float = 1.8, height: float = 1.5,
                 rcs_total: float = 10.0, wheels: bool = False, wheel_radius: float = 0.32,
                 accel: ArrayLike = (0, 0, 0), heading: float | None = None,
                 seed: int | None = 0) -> None:
        super().__init__(position, velocity, accel, heading, seed)
        L, W, H = length, width, height
        pts = [(0, -L / 2, 0.5, 0.30), (0, L / 2, 0.5, 0.20),
               (-W / 2, -L / 2, 0.6, 0.10), (W / 2, -L / 2, 0.6, 0.10),
               (-W / 2, L / 2, 0.6, 0.10), (W / 2, L / 2, 0.6, 0.10),
               (0, 0, H, 0.10)]
        self.body_pts = np.array([p[:3] for p in pts], dtype=float)
        self.body_rcs = rcs_total * np.array([p[3] for p in pts])
        self.wheels = wheels
        self.wheel_radius = wheel_radius
        wb = 0.6 * L
        self.wheel_centres = np.array([[-W / 2, wb / 2, wheel_radius], [W / 2, wb / 2, wheel_radius],
                                       [-W / 2, -wb / 2, wheel_radius], [W / 2, -wb / 2, wheel_radius]])
        self.wheel_rcs = 0.05 * rcs_total
        self.wheel_phase = self.rng.uniform(0, 2 * np.pi, 4)
        n = len(self.body_rcs) + (4 if wheels else 0)
        self.phase0 = self.rng.uniform(0, 2 * np.pi, n)

    def _body(self, t: np.ndarray):
        T = t.size
        pb = np.repeat(self.body_pts[:, None, :], T, axis=1)
        vb = np.zeros_like(pb)
        rcs = list(self.body_rcs)
        if self.wheels:
            r = self.wheel_radius
            w = self.speed / r
            beta = self.wheel_phase[:, None] + w * t[None, :]               # (4, T)
            off = np.stack([np.zeros_like(beta), r * np.sin(beta), r * np.cos(beta)], -1)
            vel = np.stack([np.zeros_like(beta), r * w * np.cos(beta), -r * w * np.sin(beta)], -1)
            pb = np.concatenate([pb, self.wheel_centres[:, None, :] + off])
            vb = np.concatenate([vb, vel])
            rcs += [self.wheel_rcs] * 4
        return pb, vb, np.array(rcs), self.phase0


def _rod_points(pivot: np.ndarray, length: np.ndarray, ang: np.ndarray, angdot: np.ndarray,
                plane: str) -> tuple[np.ndarray, np.ndarray]:
    """Point at distance ``length`` from ``pivot`` on a rod rotated by ``ang``.

    plane 'yz' (sagittal swing, ang=0 pointing down):  d = l·(0, sin a, −cos a)
    Returns positions/velocities (P, T, 3) given pivot (P,3), length (P,),
    ang/angdot (P, T).
    """
    l = length[:, None]
    if plane == "yz":
        d = np.stack([np.zeros_like(ang), l * np.sin(ang), -l * np.cos(ang)], -1)
        dd = np.stack([np.zeros_like(ang), l * np.cos(ang) * angdot, l * np.sin(ang) * angdot], -1)
    else:
        raise ValueError(plane)
    return pivot[:, None, :] + d, dd


class Pedestrian(BodyTarget):
    """Walking human (simplified Boulic / V.C. Chen kinematics).

    Gait cycle period ``T_c = 1.346 / sqrt(v / H_t)`` with ``H_t`` the
    leg (hip) height (Boulic et al., 1990). Torso bobs vertically at twice the
    gait frequency; arms and legs are pendulums swinging in the sagittal plane
    with opposite phases (left arm with right leg). Scatterers: head, torso,
    2×(elbow, hand), 2×(knee, foot).
    """

    def __init__(self, position: ArrayLike, velocity: ArrayLike = (0, 1.4, 0),
                 height: float = 1.8, rcs_total: float = 1.0,
                 arm_swing: float = 0.5, leg_swing: float = 0.45,
                 accel: ArrayLike = (0, 0, 0), heading: float | None = None,
                 seed: int | None = 0) -> None:
        super().__init__(position, velocity, accel, heading, seed)
        self.height = height
        self.ht = 0.53 * height
        v_rel = self.speed / self.ht
        self.f_gait = 0.0 if v_rel <= 0 else 1.0 / (1.346 / np.sqrt(v_rel))
        self.phi_g = self.rng.uniform(0, 2 * np.pi)
        s = 1.0 if self.speed > 0 else 0.0
        self.arm_swing, self.leg_swing = arm_swing * s, leg_swing * s
        frac = np.array([0.05, 0.45, 0.04, 0.03, 0.04, 0.03, 0.10, 0.08, 0.10, 0.08])
        self.rcs = rcs_total * frac / frac.sum()
        self.phase0 = self.rng.uniform(0, 2 * np.pi, 10)

    def _body(self, t: np.ndarray):
        H, T = self.height, t.size
        w = 2 * np.pi * self.f_gait
        g = w * t + self.phi_g
        bob = 0.02 * H * np.sin(2 * g)
        bobd = 0.02 * H * 2 * w * np.cos(2 * g)
        z_torso, z_head = 0.62 * H, 0.93 * H
        shoulder, hip = 0.82 * H, self.ht
        torso = np.stack([np.zeros(T), np.zeros(T), z_torso + bob], -1)[None]
        head = np.stack([np.zeros(T), np.zeros(T), z_head + bob], -1)[None]
        vt = np.stack([np.zeros(T), np.zeros(T), bobd], -1)[None]
        # limbs: [left arm, right arm, left leg, right leg]; left arm ~ right leg
        sgn = np.array([1.0, -1.0, -1.0, 1.0])
        amp = np.array([self.arm_swing] * 2 + [self.leg_swing] * 2)
        ang = (amp * sgn)[:, None] * np.sin(g)[None, :]
        angd = (amp * sgn)[:, None] * w * np.cos(g)[None, :]
        piv = np.array([[-0.2 * H / 1.8, 0, shoulder], [0.2 * H / 1.8, 0, shoulder],
                        [-0.1 * H / 1.8, 0, hip], [0.1 * H / 1.8, 0, hip]])
        llen = np.array([0.36 * H] * 2 + [self.ht] * 2)
        pivT = piv[:, None, :] + np.stack([np.zeros(T), np.zeros(T), bob], -1)[None]
        pos, vel = [torso, head], [vt, vt]
        for fr in (0.5, 1.0):
            p, v = _rod_points(piv, fr * llen, ang, angd, "yz")
            pos.append(p - piv[:, None, :] + pivT)
            vel.append(v + vt)
        pb = np.concatenate(pos)    # torso, head, mid(4 limbs), end(4 limbs)
        vb = np.concatenate(vel)
        # reorder rcs: head .05, torso .45, then elbows/knees mid, hands/feet end
        rcs = np.array([self.rcs[1], self.rcs[0], self.rcs[2], self.rcs[4], self.rcs[6],
                        self.rcs[8], self.rcs[3], self.rcs[5], self.rcs[7], self.rcs[9]])
        return pb, vb, rcs, self.phase0


class Drone(BodyTarget):
    """Multirotor UAV with rotating blades (discretised Martin & Mulgrew model).

    Each blade of length ``blade_length`` (from ``blade_root`` to the tip)
    rotates in the horizontal body plane at Ω = 2π·rpm/60 about its rotor hub.
    The blade is represented by P point scatterers at the midpoint radii
    ``r_p = r0 + (p + ½)(L − r0)/P``. The coherent sum

        s(t) = Σ_p a·exp(j·4π/λ·(R0 + r_p·cos φ·cos(Ωt + ψ_k)))

    is the midpoint-rule approximation of the Martin & Mulgrew (1990) blade
    integral ``∫ exp(j4πr cos φ cos(Ωt+ψ_k)/λ) dr`` whose closed form is the
    sinc "blade flash". P defaults to ``ceil(8·(L−r0)/λ_design)`` (two-way phase
    step ≤ π/2 between points). Per-point RCS is ``σ_blade/P²`` so that the
    fully coherent flash has RCS ``σ_blade``.

    Max radial blade-tip speed v_tip = Ω·L, max micro-Doppler 2·v_tip/λ =
    4π·f_rot·L/λ (for a radar in the rotor plane).
    """

    def __init__(self, position: ArrayLike, velocity: ArrayLike = (0, 0, 0),
                 n_rotors: int = 4, n_blades: int = 2, rpm: float | Sequence[float] = 6000.0,
                 blade_length: float = 0.12, blade_root: float = 0.01,
                 rotor_offsets: np.ndarray | None = None, arm_length: float = 0.25,
                 body_rcs: float = 0.05, blade_rcs: float = 0.005,
                 points_per_blade: int | None = None,
                 design_wavelength: float = C0 / 24e9,
                 accel: ArrayLike = (0, 0, 0), heading: float | None = None,
                 random_blade_phase: bool = True, seed: int | None = 0) -> None:
        super().__init__(position, velocity, accel, heading, seed)
        self.n_rotors, self.n_blades = int(n_rotors), int(n_blades)
        self.rpm = np.broadcast_to(np.asarray(rpm, float), (self.n_rotors,)).copy()
        self.blade_length, self.blade_root = blade_length, blade_root
        if rotor_offsets is None:
            ang = np.pi / 4 + 2 * np.pi * np.arange(self.n_rotors) / self.n_rotors
            rotor_offsets = np.stack([arm_length * np.cos(ang), arm_length * np.sin(ang),
                                      np.full(self.n_rotors, 0.05)], -1)
        self.rotor_offsets = np.asarray(rotor_offsets, float).reshape(self.n_rotors, 3)
        self.direction = np.where(np.arange(self.n_rotors) % 2 == 0, 1.0, -1.0)
        if points_per_blade is None:
            points_per_blade = int(max(4, np.ceil(8 * (blade_length - blade_root) / design_wavelength)))
        self.P = int(points_per_blade)
        self.body_rcs, self.blade_rcs = body_rcs, blade_rcs
        self.psi0 = (self.rng.uniform(0, 2 * np.pi, self.n_rotors) if random_blade_phase
                     else np.zeros(self.n_rotors))
        self.radii = blade_root + (np.arange(self.P) + 0.5) * (blade_length - blade_root) / self.P

    @property
    def omega(self) -> np.ndarray:
        """Rotor angular rates Ω [rad/s], shape (n_rotors,)."""
        return 2 * np.pi * self.rpm / 60.0

    @property
    def tip_speed(self) -> np.ndarray:
        """v_tip = Ω·L [m/s]."""
        return self.omega * self.blade_length

    def _body(self, t: np.ndarray):
        T = t.size
        R, B, P = self.n_rotors, self.n_blades, self.P
        om = (self.omega * self.direction)[:, None, None]                     # (R,1,1)
        psi = (self.psi0[:, None, None] + 2 * np.pi * np.arange(B)[None, :, None] / B
               + om * t[None, None, :])                                       # (R,B,T)
        c, s = np.cos(psi), np.sin(psi)
        r = self.radii[None, None, :, None]                                   # (1,1,P,1)
        px = self.rotor_offsets[:, 0, None, None, None] + r * c[:, :, None, :]
        py = self.rotor_offsets[:, 1, None, None, None] + r * s[:, :, None, :]
        pz = np.broadcast_to(self.rotor_offsets[:, 2, None, None, None], px.shape)
        vx = -r * s[:, :, None, :] * om[..., None]
        vy = r * c[:, :, None, :] * om[..., None]
        blades_p = np.stack([px, py, pz], -1).reshape(R * B * P, T, 3)
        blades_v = np.stack([vx, vy, np.zeros_like(vx)], -1).reshape(R * B * P, T, 3)
        pb = np.concatenate([np.zeros((1, T, 3)), blades_p])
        vb = np.concatenate([np.zeros((1, T, 3)), blades_v])
        rcs = np.concatenate([[self.body_rcs], np.full(R * B * P, self.blade_rcs / P ** 2)])
        ph = np.zeros(1 + R * B * P)
        return pb, vb, rcs, ph


class Bird(BodyTarget):
    """Flapping bird: body + two wings as rods with flap angle A·sin(2πf t).

    Wings extend laterally (body x) from shoulder pivots and rotate in the
    body x-z plane; each wing is discretised into ``points_per_wing`` points.
    """

    def __init__(self, position: ArrayLike, velocity: ArrayLike = (0, 10, 0),
                 wingspan: float = 0.6, flap_freq: float = 5.0, flap_amplitude: float = 0.6,
                 body_rcs: float = 0.01, wing_rcs: float = 0.004, points_per_wing: int = 6,
                 accel: ArrayLike = (0, 0, 0), heading: float | None = None,
                 seed: int | None = 0) -> None:
        super().__init__(position, velocity, accel, heading, seed)
        self.half_span = wingspan / 2
        self.flap_freq, self.flap_amplitude = flap_freq, flap_amplitude
        self.body_rcs, self.wing_rcs = body_rcs, wing_rcs
        self.P = int(points_per_wing)
        self.flap_phase = self.rng.uniform(0, 2 * np.pi)
        self.phase0 = self.rng.uniform(0, 2 * np.pi, 1 + 2 * self.P)

    def _body(self, t: np.ndarray):
        T, P = t.size, self.P
        w = 2 * np.pi * self.flap_freq
        a = self.flap_amplitude * np.sin(w * t + self.flap_phase)            # (T,)
        ad = self.flap_amplitude * w * np.cos(w * t + self.flap_phase)
        root = 0.03
        s = root + (np.arange(P) + 0.5) * (self.half_span - root) / P         # (P,)
        pos, vel = [np.zeros((1, T, 3))], [np.zeros((1, T, 3))]
        for side in (-1.0, 1.0):
            x = side * (root + s[:, None] * np.cos(a)[None])
            z = s[:, None] * np.sin(a)[None]
            vx = -side * s[:, None] * np.sin(a)[None] * ad[None]
            vz = s[:, None] * np.cos(a)[None] * ad[None]
            pos.append(np.stack([x, np.zeros_like(x), z], -1))
            vel.append(np.stack([vx, np.zeros_like(vx), vz], -1))
        rcs = np.concatenate([[self.body_rcs], np.full(2 * P, self.wing_rcs / P)])
        return np.concatenate(pos), np.concatenate(vel), rcs, self.phase0


class ClutterField(Target):
    """Static point scatterers given by arrays (positions (S,3), rcs (S,), phase0 (S,))."""

    def __init__(self, positions: np.ndarray, rcs: np.ndarray, phase0: np.ndarray | None = None) -> None:
        self.positions = np.asarray(positions, float).reshape(-1, 3)
        self.rcs = np.asarray(rcs, float).reshape(-1)
        self.phase0 = np.zeros(len(self.rcs)) if phase0 is None else np.asarray(phase0, float)

    def scatterers(self, t: ArrayLike) -> Scatterers:
        t = _t1d(t)
        S, T = len(self.rcs), t.size
        pos = np.broadcast_to(self.positions[:, None, :], (S, T, 3)).copy()
        return Scatterers(pos, np.zeros_like(pos), np.repeat(self.rcs[:, None], T, 1),
                          self.phase0.copy(), -np.ones(S, dtype=int))


def static_clutter(n: int, r_min: float, r_max: float, az_min_deg: float = -60.0,
                   az_max_deg: float = 60.0, z: float = 0.0, rcs_mean: float = 1.0,
                   rng: np.random.Generator | int | None = 0) -> ClutterField:
    """Random static clutter: uniform in area over an annular sector,
    exponentially distributed RCS (Swerling-like) with mean ``rcs_mean`` and
    uniform random phase."""
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    r = np.sqrt(g.uniform(r_min ** 2, r_max ** 2, n))
    az = np.deg2rad(g.uniform(az_min_deg, az_max_deg, n))
    pos = np.stack([r * np.sin(az), r * np.cos(az), np.full(n, z)], -1)
    return ClutterField(pos, g.exponential(rcs_mean, n), g.uniform(0, 2 * np.pi, n))


class Scene:
    """Collection of targets. ``scatterers(t)`` concatenates all of them and
    sets ``label`` to the target index."""

    def __init__(self, targets: Iterable[Target] = ()) -> None:
        self.targets: list[Target] = list(targets)

    def add(self, target: Target) -> "Scene":
        self.targets.append(target)
        return self

    def __len__(self) -> int:
        return len(self.targets)

    def scatterers(self, t: ArrayLike) -> Scatterers:
        parts = []
        for i, tg in enumerate(self.targets):
            s = tg.scatterers(t)
            s.label = np.full(s.n, i, dtype=int)
            parts.append(s)
        return Scatterers.concatenate(parts)

    def truth(self, t: float) -> np.ndarray:
        """Ground truth per target at time t: rows (x, y, z, vx, vy, vz, R, az, vr)."""
        rows = []
        for tg in self.targets:
            if isinstance(tg, BodyTarget):
                p, v = tg.trajectory(np.array([t]))
                p, v = p[0], v[0]
            else:
                s = tg.scatterers([t])
                p, v = s.pos[:, 0].mean(0), s.vel[:, 0].mean(0)
            R = np.linalg.norm(p)
            rows.append([*p, *v, R, np.arctan2(p[0], p[1]), p @ v / max(R, 1e-12)])
        return np.array(rows).reshape(-1, 9)
