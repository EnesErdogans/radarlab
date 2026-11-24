// Constant-velocity EKF multi-target tracker.
// State [x, y, vx, vy]; measurement [R, θ, vr] (θ = atan2(x, y), vr > 0 receding).
// Global-nearest-neighbour assignment (Hungarian) with χ² gating on the position
// innovation (2 DOF, 99 %: 9.21). Confirm after 3 hits, delete after misses.

export interface Measurement {
  range: number;
  az: number; // rad
  vr: number;
  power: number;
}

export interface Track {
  id: number;
  s: Float64Array; // [x,y,vx,vy]
  P: Float64Array; // 4×4 row-major
  hits: number;
  misses: number;
  age: number;
  confirmed: boolean;
  hist: number[]; // flat x,y pairs (most recent last)
  lastD2: number;
}

export interface TrackerParams {
  sigmaAcc: number; // process noise accel std [m/s²]
  sigmaR: number;
  sigmaAz: number; // rad
  sigmaV: number;
  gate: number; // χ² gate (position, 2 DOF)
  confirmHits: number;
  maxMissTentative: number;
  maxMissConfirmed: number;
  vMax: number; // unambiguous velocity (for wrapped Doppler innovation); 0 = no wrap
  histLen: number;
  spawnSuppress: number; // no new track within this distance [m] of an existing one (extended targets)
  spawnGate: number; // … nor inside this χ² (position) gate of an existing track
}

export function defaultTrackerParams(): TrackerParams {
  return {
    sigmaAcc: 3,
    sigmaR: 0.3,
    sigmaAz: (3 * Math.PI) / 180,
    sigmaV: 0.3,
    gate: 9.21,
    confirmHits: 3,
    maxMissTentative: 2,
    maxMissConfirmed: 5,
    vMax: 0,
    histLen: 40,
    spawnSuppress: 3.5,
    spawnGate: 16,
  };
}

/** Hungarian algorithm (min-cost assignment). cost is n×m (n ≤ m), row-major. Returns col for each row. */
export function hungarian(cost: number[][], n: number, m: number): Int32Array {
  const INF = 1e18;
  const u = new Float64Array(n + 1),
    v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1),
    way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(INF);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF,
        j1 = 0;
      for (let j = 1; j <= m; j++)
        if (!used[j]) {
          const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const ans = new Int32Array(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j] > 0) ans[p[j] - 1] = j - 1;
  return ans;
}

function wrap(x: number, half: number): number {
  if (half <= 0) return x;
  const p = 2 * half;
  return x - p * Math.round(x / p);
}

export class Tracker {
  tracks: Track[] = [];
  private nextId = 1;
  p: TrackerParams;
  constructor(p: Partial<TrackerParams> = {}) {
    this.p = { ...defaultTrackerParams(), ...p };
  }

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }

  private predict(tr: Track, dt: number): void {
    const s = tr.s,
      P = tr.P;
    s[0] += dt * s[2];
    s[1] += dt * s[3];
    // P = F P Fᵀ + Q (F = [I dtI; 0 I])
    const q = this.p.sigmaAcc * this.p.sigmaAcc;
    const F = [1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1];
    const T = new Float64Array(16);
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) {
        let a = 0;
        for (let k = 0; k < 4; k++) a += F[i * 4 + k] * P[k * 4 + j];
        T[i * 4 + j] = a;
      }
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) {
        let a = 0;
        for (let k = 0; k < 4; k++) a += T[i * 4 + k] * F[j * 4 + k];
        P[i * 4 + j] = a;
      }
    const d4 = (dt ** 4) / 4,
      d3 = (dt ** 3) / 2,
      d2 = dt * dt;
    P[0] += q * d4;
    P[5] += q * d4;
    P[2] += q * d3;
    P[8] += q * d3;
    P[7] += q * d3;
    P[13] += q * d3;
    P[10] += q * d2;
    P[15] += q * d2;
  }

  /** Measurement prediction h(s) and Jacobian H (3×4). */
  private hJ(s: Float64Array, H: Float64Array): [number, number, number] {
    const [x, y, vx, vy] = s;
    const R = Math.max(1e-3, Math.hypot(x, y));
    const vr = (x * vx + y * vy) / R;
    H.fill(0);
    H[0] = x / R;
    H[1] = y / R;
    H[4] = y / (R * R);
    H[5] = -x / (R * R);
    H[8] = (vx - (vr * x) / R) / R;
    H[9] = (vy - (vr * y) / R) / R;
    H[10] = x / R;
    H[11] = y / R;
    return [R, Math.atan2(x, y), vr];
  }

  /** Innovation and innovation covariance S (3×3) for a measurement. */
  private innov(tr: Track, z: Measurement, H: Float64Array, nu: Float64Array, S: Float64Array): void {
    const h = this.hJ(tr.s, H);
    nu[0] = z.range - h[0];
    nu[1] = wrap(z.az - h[1], Math.PI);
    nu[2] = wrap(z.vr - h[2], this.p.vMax);
    const P = tr.P;
    const HP = new Float64Array(12);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 4; j++) {
        let a = 0;
        for (let k = 0; k < 4; k++) a += H[i * 4 + k] * P[k * 4 + j];
        HP[i * 4 + j] = a;
      }
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) {
        let a = 0;
        for (let k = 0; k < 4; k++) a += HP[i * 4 + k] * H[j * 4 + k];
        S[i * 3 + j] = a;
      }
    S[0] += this.p.sigmaR ** 2;
    S[4] += this.p.sigmaAz ** 2;
    S[8] += this.p.sigmaV ** 2;
  }

  private d2pos(nu: Float64Array, S: Float64Array): number {
    const a = S[0],
      b = S[1],
      c = S[3],
      d = S[4];
    const det = a * d - b * c;
    if (det <= 0) return Infinity;
    return (d * nu[0] * nu[0] - (b + c) * nu[0] * nu[1] + a * nu[1] * nu[1]) / det;
  }

  private update(tr: Track, z: Measurement): void {
    const H = new Float64Array(12),
      nu = new Float64Array(3),
      S = new Float64Array(9);
    this.innov(tr, z, H, nu, S);
    const Si = inv3(S);
    const P = tr.P;
    // K = P Hᵀ S⁻¹ (4×3)
    const PHt = new Float64Array(12);
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 3; j++) {
        let a = 0;
        for (let k = 0; k < 4; k++) a += P[i * 4 + k] * H[j * 4 + k];
        PHt[i * 3 + j] = a;
      }
    const K = new Float64Array(12);
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 3; j++) {
        let a = 0;
        for (let k = 0; k < 3; k++) a += PHt[i * 3 + k] * Si[k * 3 + j];
        K[i * 3 + j] = a;
      }
    for (let i = 0; i < 4; i++) tr.s[i] += K[i * 3] * nu[0] + K[i * 3 + 1] * nu[1] + K[i * 3 + 2] * nu[2];
    // P = (I − K H) P
    const IKH = new Float64Array(16);
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) {
        let a = i === j ? 1 : 0;
        for (let k = 0; k < 3; k++) a -= K[i * 3 + k] * H[k * 4 + j];
        IKH[i * 4 + j] = a;
      }
    const NP = new Float64Array(16);
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) {
        let a = 0;
        for (let k = 0; k < 4; k++) a += IKH[i * 4 + k] * P[k * 4 + j];
        NP[i * 4 + j] = a;
      }
    // symmetrize
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) P[i * 4 + j] = 0.5 * (NP[i * 4 + j] + NP[j * 4 + i]);
  }

  private spawn(z: Measurement): void {
    const sx = Math.sin(z.az),
      cy = Math.cos(z.az);
    const s = new Float64Array([z.range * sx, z.range * cy, z.vr * sx, z.vr * cy]);
    const P = new Float64Array(16);
    const sp = this.p.sigmaR ** 2 + (z.range * this.p.sigmaAz) ** 2;
    P[0] = sp;
    P[5] = sp;
    P[10] = 16;
    P[15] = 16;
    this.tracks.push({ id: this.nextId++, s, P, hits: 1, misses: 0, age: 1, confirmed: false, hist: [s[0], s[1]], lastD2: 0 });
  }

  step(meas: Measurement[], dt: number): void {
    const p = this.p;
    for (const tr of this.tracks) {
      this.predict(tr, dt);
      tr.age++;
    }
    const nT = this.tracks.length,
      nM = meas.length;
    const BIG = 1e6;
    const assigned = new Int32Array(nT).fill(-1);
    if (nT > 0 && nM > 0) {
      const H = new Float64Array(12),
        nu = new Float64Array(3),
        S = new Float64Array(9);
      const n = Math.max(nT, nM);
      const cost: number[][] = [];
      for (let i = 0; i < n; i++) {
        const row = new Array<number>(n).fill(BIG);
        if (i < nT)
          for (let j = 0; j < nM; j++) {
            this.innov(this.tracks[i], meas[j], H, nu, S);
            const d2 = this.d2pos(nu, S);
            // velocity consistency (soft): add normalized Doppler innovation
            const dv2 = (nu[2] * nu[2]) / S[8];
            if (d2 <= p.gate && dv2 < 25) row[j] = d2 + 0.25 * dv2;
          }
        cost.push(row);
      }
      const a = hungarian(cost, n, n);
      for (let i = 0; i < nT; i++) if (a[i] >= 0 && a[i] < nM && cost[i][a[i]] < BIG) assigned[i] = a[i];
    }
    const used = new Uint8Array(nM);
    for (let i = 0; i < nT; i++) {
      const tr = this.tracks[i];
      if (assigned[i] >= 0) {
        used[assigned[i]] = 1;
        this.update(tr, meas[assigned[i]]);
        tr.hits++;
        tr.misses = 0;
        if (tr.hits >= p.confirmHits) tr.confirmed = true;
      } else tr.misses++;
      tr.hist.push(tr.s[0], tr.s[1]);
      if (tr.hist.length > 2 * p.histLen) tr.hist.splice(0, tr.hist.length - 2 * p.histLen);
    }
    this.tracks = this.tracks.filter((t) => t.misses <= (t.confirmed ? p.maxMissConfirmed : p.maxMissTentative));
    const H = new Float64Array(12),
      nu = new Float64Array(3),
      S = new Float64Array(9);
    for (let j = 0; j < nM; j++) {
      if (used[j]) continue;
      const z = meas[j];
      const zx = z.range * Math.sin(z.az),
        zy = z.range * Math.cos(z.az);
      let near = false;
      for (const tr of this.tracks) {
        if (Math.hypot(tr.s[0] - zx, tr.s[1] - zy) < p.spawnSuppress) {
          near = true;
          break;
        }
        // inside a widened gate of an existing track (extended / poorly resolved targets)
        this.innov(tr, z, H, nu, S);
        if (this.d2pos(nu, S) < p.spawnGate) {
          near = true;
          break;
        }
      }
      if (!near) this.spawn(z);
    }
  }
}

function inv3(m: Float64Array): Float64Array {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h,
    B = -(d * i - f * g),
    C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const r = 1 / det;
  return new Float64Array([
    A * r,
    -(b * i - c * h) * r,
    (b * f - c * e) * r,
    B * r,
    (a * i - c * g) * r,
    -(a * f - c * d) * r,
    C * r,
    -(a * h - b * g) * r,
    (a * e - b * d) * r,
  ]);
}

/**
 * Merge raw detections into one measurement per object: single-linkage (DBSCAN,
 * minPts = 1) clustering in the x-y plane with radius `eps` [m]. Velocity is the
 * power-weighted mean, so micro-Doppler parts (wheels, limbs) follow the body.
 * Detections whose velocity differs by more than `dvMax` from the cluster seed
 * are kept separate (e.g. a pedestrian next to a car).
 */
export function mergeDetections(
  dets: { x: number; y: number; velocity: number; power: number; range?: number; angleDeg?: number }[],
  n: number,
  eps = 2.0,
  dvMax = 8,
  rEps = 0,
  azEpsDeg = 0,
): Measurement[] {
  // linked if close in Cartesian space, or (for coarse angle resolution) close in range and azimuth
  const linked = (a: (typeof dets)[number], b: (typeof dets)[number]) =>
    Math.hypot(a.x - b.x, a.y - b.y) < eps ||
    (rEps > 0 && a.range !== undefined && b.range !== undefined && Math.abs(a.range - b.range) < rEps && Math.abs((a.angleDeg ?? 0) - (b.angleDeg ?? 0)) < azEpsDeg);
  const label = new Int32Array(n).fill(-1);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => dets[b].power - dets[a].power);
  const queue = new Int32Array(n);
  const out: Measurement[] = [];
  let nc = 0;
  for (const i0 of order) {
    if (label[i0] >= 0) continue;
    let qh = 0,
      qt = 0;
    queue[qt++] = i0;
    label[i0] = nc;
    const v0 = dets[i0].velocity;
    let wx = 0,
      wy = 0,
      wv = 0,
      w = 0;
    while (qh < qt) {
      const i = queue[qh++];
      const s = dets[i];
      wx += s.x * s.power;
      wy += s.y * s.power;
      wv += s.velocity * s.power;
      w += s.power;
      for (let j = 0; j < n; j++) {
        if (label[j] >= 0) continue;
        const o = dets[j];
        if (linked(o, s) && Math.abs(o.velocity - v0) < dvMax) {
          label[j] = nc;
          queue[qt++] = j;
        }
      }
    }
    nc++;
    const x = wx / w,
      y = wy / w;
    out.push({ range: Math.hypot(x, y), az: Math.atan2(x, y), vr: wv / w, power: w });
  }
  return out;
}
