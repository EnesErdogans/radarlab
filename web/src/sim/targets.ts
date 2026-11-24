// Target models. Each target expands into point scatterers whose position and
// velocity are evaluated analytically at any time t (relative to the target's
// reference state). Output stride per scatterer: [px,py,pz,vx,vy,vz,rcs,phase0].

export type TargetType = 'point' | 'car' | 'pedestrian' | 'drone' | 'bird';
export const TARGET_TYPES: TargetType[] = ['point', 'car', 'pedestrian', 'drone', 'bird'];

export interface Target {
  id: number;
  type: TargetType;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  rcs: number; // total RCS [m²]
  rpm?: number; // drone rotor speed
  bladeLength?: number; // drone blade length [m]
  nRotors?: number;
  flapHz?: number; // bird flapping frequency
}

export const STRIDE = 8;

export function defaultTarget(type: TargetType, id: number, x = 0, y = 20): Target {
  const base: Target = { id, type, x, y, z: 0, vx: 0, vy: 0, vz: 0, rcs: 1 };
  switch (type) {
    case 'point':
      return { ...base, z: 0, rcs: 1 };
    case 'car':
      return { ...base, z: 0.5, vy: -5, rcs: 10 };
    case 'pedestrian':
      return { ...base, z: 0, vx: 1.2, rcs: 0.5 };
    case 'drone':
      return { ...base, z: 2, vy: -2, rcs: 0.05, rpm: 6000, bladeLength: 0.12, nRotors: 4 };
    case 'bird':
      return { ...base, z: 3, vx: 6, rcs: 0.02, flapHz: 6 };
  }
}

export function scattererCount(t: Target): number {
  switch (t.type) {
    case 'point':
      return 1;
    case 'car':
      return 13;
    case 'pedestrian':
      return 6;
    case 'drone':
      return 1 + (t.nRotors ?? 4) * 4;
    case 'bird':
      return 3;
  }
}

// Car body points in local (forward, right) coordinates [m] + RCS weights.
const CAR_BODY: [number, number, number, number][] = [
  // fwd, right, up(rel. center), weight
  [2.25, 0, 0.0, 0.21],
  [2.05, -0.8, 0.0, 0.1],
  [2.05, 0.8, 0.0, 0.1],
  [0.0, -0.9, 0.0, 0.04], // side panels / mirrors
  [0.0, 0.9, 0.0, 0.04],
  [-2.2, -0.8, 0.1, 0.12],
  [-2.2, 0.8, 0.1, 0.12],
  [-2.3, 0, 0.2, 0.15],
  [0.2, 0, 0.9, 0.04], // roof
];
const CAR_WHEELS: [number, number][] = [
  [1.4, -0.85],
  [1.4, 0.85],
  [-1.4, -0.85],
  [-1.4, 0.85],
];
const WHEEL_R = 0.32;

/**
 * Write all scatterers of target `tg` at time t (seconds since the reference state)
 * into out[off..]. Returns number of scatterers written.
 */
export function evalTarget(tg: Target, t: number, out: Float64Array, off: number, t0 = 0): number {
  const tm = t0 + t; // absolute time for micro-motion phases
  const bx = tg.x + tg.vx * t,
    by = tg.y + tg.vy * t,
    bz = tg.z + tg.vz * t;
  const speedH = Math.hypot(tg.vx, tg.vy);
  // heading unit vectors (forward f, right r) in the x-y plane
  let fx = 0,
    fy = 1;
  if (speedH > 1e-6) {
    fx = tg.vx / speedH;
    fy = tg.vy / speedH;
  }
  const rx = fy,
    ry = -fx;
  let o = off;
  const put = (px: number, py: number, pz: number, vx: number, vy: number, vz: number, rcs: number, ph: number) => {
    out[o] = px;
    out[o + 1] = py;
    out[o + 2] = pz;
    out[o + 3] = vx;
    out[o + 4] = vy;
    out[o + 5] = vz;
    out[o + 6] = rcs;
    out[o + 7] = ph;
    o += STRIDE;
  };
  switch (tg.type) {
    case 'point':
      put(bx, by, bz, tg.vx, tg.vy, tg.vz, tg.rcs, 0);
      break;
    case 'car': {
      for (let i = 0; i < CAR_BODY.length; i++) {
        const [a, b, c, w] = CAR_BODY[i];
        put(bx + a * fx + b * rx, by + a * fy + b * ry, bz + c, tg.vx, tg.vy, tg.vz, tg.rcs * w, i * 1.7);
      }
      // rolling wheels: rim scatterer rotates in the (forward, up) plane
      const om = speedH / WHEEL_R;
      for (let i = 0; i < CAR_WHEELS.length; i++) {
        const [a, b] = CAR_WHEELS[i];
        const ang = -om * tm + i * 1.3;
        const ca = Math.cos(ang),
          sa = Math.sin(ang);
        const hx = bx + a * fx + b * rx,
          hy = by + a * fy + b * ry,
          hz = bz - 0.15;
        // offset = R(ca·f + sa·z); d/dt = R·(-om)(-sa·f + ca·z)
        const dv = -om * WHEEL_R;
        put(hx + WHEEL_R * ca * fx, hy + WHEEL_R * ca * fy, hz + WHEEL_R * sa, tg.vx - dv * sa * fx, tg.vy - dv * sa * fy, tg.vz + dv * ca, tg.rcs * 0.01, i * 2.1);
      }
      break;
    }
    case 'pedestrian': {
      const f = speedH > 0.1 ? Math.max(0.6, speedH / 1.4) : 0;
      const w = 2 * Math.PI * f;
      const s = Math.sin(w * tm),
        c = Math.cos(w * tm);
      // torso with slight vertical bob at 2f
      const bob = 0.03 * Math.sin(2 * w * tm),
        bobV = 0.03 * 2 * w * Math.cos(2 * w * tm);
      put(bx, by, bz + 1.1 + bob, tg.vx, tg.vy, tg.vz + bobV, tg.rcs * 0.55, 0);
      put(bx, by, bz + 1.65 + bob, tg.vx, tg.vy, tg.vz + bobV, tg.rcs * 0.1, 0.5);
      // arms (hands) swing ±0.3 m along heading, opposite phase
      const armA = f > 0 ? 0.3 : 0;
      for (const sg of [1, -1]) {
        const d = sg * armA * s,
          dd = sg * armA * w * c;
        put(bx + d * fx + sg * 0.25 * rx, by + d * fy + sg * 0.25 * ry, bz + 0.85, tg.vx + dd * fx, tg.vy + dd * fy, tg.vz, tg.rcs * 0.08, 1 + sg);
      }
      // legs (feet/shins) swing ±0.45 m, opposite to the same-side arm
      const legA = f > 0 ? 0.45 : 0;
      for (const sg of [1, -1]) {
        const d = -sg * legA * s,
          dd = -sg * legA * w * c;
        put(bx + d * fx + sg * 0.12 * rx, by + d * fy + sg * 0.12 * ry, bz + 0.3, tg.vx + dd * fx, tg.vy + dd * fy, tg.vz, tg.rcs * 0.095, 3 + sg);
      }
      break;
    }
    case 'drone': {
      const nR = tg.nRotors ?? 4;
      const L = tg.bladeLength ?? 0.12;
      const om = ((tg.rpm ?? 6000) / 60) * 2 * Math.PI;
      put(bx, by, bz, tg.vx, tg.vy, tg.vz, tg.rcs * 0.8, 0);
      const bladeRcs = (tg.rcs * 0.2) / (nR * 4);
      for (let i = 0; i < nR; i++) {
        const armAng = (2 * Math.PI * (i + 0.5)) / nR;
        const hx = bx + 0.22 * Math.cos(armAng),
          hy = by + 0.22 * Math.sin(armAng),
          hz = bz + 0.05;
        const dir = i % 2 === 0 ? 1 : -1; // counter-rotating pairs
        for (let b = 0; b < 2; b++) {
          const ang = dir * om * tm + b * Math.PI + i * 0.7;
          const ca = Math.cos(ang),
            sa = Math.sin(ang);
          for (const rho of [0.5 * L, L]) {
            put(hx + rho * ca, hy + rho * sa, hz, tg.vx - dir * om * rho * sa, tg.vy + dir * om * rho * ca, tg.vz, bladeRcs, i + b + rho);
          }
        }
      }
      break;
    }
    case 'bird': {
      const f = tg.flapHz ?? 6;
      const w = 2 * Math.PI * f;
      const amp = 0.12;
      put(bx, by, bz, tg.vx, tg.vy, tg.vz, tg.rcs * 0.8, 0);
      for (const sg of [1, -1]) {
        const zz = amp * Math.sin(w * tm),
          vzz = amp * w * Math.cos(w * tm);
        put(bx + sg * 0.3 * rx, by + sg * 0.3 * ry, bz + zz, tg.vx, tg.vy, tg.vz + vzz, tg.rcs * 0.1, sg);
      }
      break;
    }
  }
  return (o - off) / STRIDE;
}

/** Bulk-motion range, radial velocity and azimuth of a target's reference point. */
export function targetKinematics(tg: Target): { range: number; vr: number; azDeg: number } {
  const r = Math.hypot(tg.x, tg.y, tg.z);
  const vr = r > 0 ? (tg.x * tg.vx + tg.y * tg.vy + tg.z * tg.vz) / r : 0;
  return { range: r, vr, azDeg: (Math.atan2(tg.x, tg.y) * 180) / Math.PI };
}

/** Maximum |radial velocity| of any scatterer over one micro-motion period (sampled). */
export function maxMicroRadialSpeed(tg: Target): number {
  const n = scattererCount(tg);
  const buf = new Float64Array(n * STRIDE);
  let mx = 0;
  for (let i = 0; i < 64; i++) {
    evalTarget(tg, i * 1.37e-3, buf, 0);
    for (let s = 0; s < n; s++) {
      const o = s * STRIDE;
      const r = Math.hypot(buf[o], buf[o + 1], buf[o + 2]);
      const vr = (buf[o] * buf[o + 3] + buf[o + 1] * buf[o + 4] + buf[o + 2] * buf[o + 5]) / r;
      mx = Math.max(mx, Math.abs(vr));
    }
  }
  return mx;
}
