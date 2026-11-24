"""Extra target models used only for the real-data comparison.

``Cyclist`` is a simple kinematic model built on :class:`radarlab.scene.BodyTarget`
(same body-frame conventions as ``Car``/``Pedestrian``): rigid rider + frame,
two wheels whose rim points roll without slip (velocity 0 … 2|v| along the
heading, the classic wheel micro-Doppler), and two feet/knees on a crank
turning at the pedalling cadence. All numbers are generic adult/bicycle
dimensions, not fitted, except ``cadence_hz`` when given.
"""
from __future__ import annotations

import numpy as np

from radarlab.scene import BodyTarget


class Cyclist(BodyTarget):
    def __init__(self, position, velocity=(0, 4.0, 0), rcs_total: float = 2.0,
                 wheel_radius: float = 0.34, crank: float = 0.17, gear_ratio: float = 2.6,
                 cadence_hz: float | None = None, n_rim: int = 6, seed: int | None = 0, heading=None,
                 accel=(0, 0, 0)):
        super().__init__(position, velocity, accel, heading=heading, seed=seed)
        fwd = self._rot()[:, 1]
        self.a_h = float(np.dot(self.a, fwd))            # acceleration along the heading
        self.v_h = float(np.dot(self.v0, fwd))           # speed along the heading at t = 0
        self.gear_ratio = gear_ratio
        self.fixed_cadence = cadence_hz
        self.rw, self.rc = wheel_radius, crank
        self.cadence = (self.speed / (2 * np.pi * wheel_radius) / gear_ratio
                        if cadence_hz is None else cadence_hz)
        self.n_rim = n_rim
        # static body points (body frame: x right, y forward, z up; origin on the ground)
        self.body_pts = np.array([[0, -0.15, 1.25],    # rider torso
                                  [0, 0.05, 1.65],     # head
                                  [0, 0.0, 0.75],      # frame / down tube
                                  [0, 0.45, 1.00]])    # handlebar + hands
        self.body_rcs = rcs_total * np.array([0.40, 0.05, 0.20, 0.08])
        self.hubs = np.array([[0, 0.52, wheel_radius], [0, -0.52, wheel_radius]])
        self.bb = np.array([0, 0.0, 0.30])             # bottom bracket (crank axle)
        self.rim_rcs = rcs_total * 0.10 / (2 * n_rim)   # 10 % in the wheels (spokes/rims)
        self.foot_rcs = rcs_total * 0.06                # 2 feet + 2 knees: 17 %
        self.knee_rcs = rcs_total * 0.025
        n = 4 + 2 * n_rim + 4
        self.phase0 = self.rng.uniform(0, 2 * np.pi, n)
        self.rim_ang0 = self.rng.uniform(0, 2 * np.pi, 2 * n_rim)
        self.crank0 = self.rng.uniform(0, 2 * np.pi)

    def _body(self, t: np.ndarray):
        T = t.size
        pos = [np.repeat(self.body_pts[:, None, :], T, 1)]
        vel = [np.zeros((4, T, 3))]
        rcs = list(self.body_rcs)
        dist = self.v_h * t + 0.5 * self.a_h * t ** 2   # distance rolled along the heading
        w = ((self.v_h + self.a_h * t) / self.rw)[None]  # wheel spin rate (rolling, no slip)
        for h in range(2):
            ang = self.rim_ang0[h * self.n_rim:(h + 1) * self.n_rim, None] + dist[None] / self.rw
            off = np.stack([np.zeros_like(ang), self.rw * np.sin(ang), self.rw * np.cos(ang)], -1)
            # body-frame velocity of a rim point relative to the hub (forward roll: top moves +y)
            v = np.stack([np.zeros_like(ang), self.rw * w * np.cos(ang), -self.rw * w * np.sin(ang)], -1)
            pos.append(self.hubs[h][None, None] + off)
            vel.append(v)
            rcs += [self.rim_rcs] * self.n_rim
        if self.fixed_cadence is None:                  # crank follows the wheel through the gear
            crank_ang = dist / self.rw / self.gear_ratio
            wc = (self.v_h + self.a_h * t) / self.rw / self.gear_ratio
        else:
            wc = 2 * np.pi * self.fixed_cadence * np.ones_like(t)
            crank_ang = wc * t
        for side in (0.0, np.pi):                       # left/right pedals 180° apart
            a = self.crank0 + side + crank_ang
            foot = self.bb[None] + np.stack([np.full(T, 0.12 * np.cos(side)), self.rc * np.sin(a),
                                            -self.rc * np.cos(a)], -1)
            fv = np.stack([np.zeros(T), self.rc * wc * np.cos(a), self.rc * wc * np.sin(a)], -1)
            knee = 0.5 * (foot + np.array([0, -0.15, 0.95])) + np.array([0, 0.15, 0.0])
            kv = 0.5 * fv
            pos += [foot[None], knee[None]]
            vel += [fv[None], kv[None]]
            rcs += [self.foot_rcs, self.knee_rcs]
        return np.concatenate(pos), np.concatenate(vel), np.array(rcs), self.phase0
