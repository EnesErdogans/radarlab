// Port of the Python reference tracker (radarlab.track.Tracker): EKF with a
// constant-velocity (white-noise acceleration) model, GNN association (Hungarian)
// on d² + ln det S with a χ²₃ gate on the full 3-D innovation, Joseph-form update,
// M-of-N confirmation and deletion after max_misses consecutive misses.
//
// It exists so that the web suite can check the Python tracker numerically
// (golden/tracker.json). The interactive UI uses `Tracker` (tracker.ts), a different
// design tuned for extended targets (2-D position gate, Doppler-wrapped innovation,
// spawn suppression). Measurement z = [R, θ, vr], θ = atan2(x, y), vr > 0 receding.

import { hungarian } from './tracker';
import { logGammaPQ } from '../dsp/gamma';

export interface RefTrackerParams {
  q: number; // acceleration PSD [m²/s³]
  sigmaR: number;
  sigmaTheta: number; // rad
  sigmaVr: number;
  gateProb: number; // χ²₃ gate probability
  confirmM: number;
  confirmN: number;
  maxMisses: number;
  initVtStd: number; // initial tangential-velocity std [m/s]
}

export function defaultRefTrackerParams(): RefTrackerParams {
  return { q: 1, sigmaR: 0.2, sigmaTheta: (1.5 * Math.PI) / 180, sigmaVr: 0.1, gateProb: 0.999, confirmM: 3, confirmN: 5, maxMisses: 3, initVtStd: 5 };
}

export interface RefTrack {
  id: number;
  x: number[]; // [px, py, vx, vy]
  P: number[]; // 4×4 row-major
  status: 'tentative' | 'confirmed' | 'deleted';
  hits: boolean[];
  misses: number;
  age: number;
}

/** χ² quantile with `dof` degrees of freedom (bisection on the regularised lower incomplete gamma). */
export function chi2Ppf(p: number, dof: number): number {
  let lo = 0,
    hi = 10 * dof + 100;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (Math.exp(logGammaPQ(dof / 2, mid / 2).lp) < p) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

// ---- small dense helpers (row-major) ----
function mul(A: number[], B: number[], n: number, k: number, m: number): number[] {
  const C = new Array<number>(n * m).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let a = 0;
      for (let l = 0; l < k; l++) a += A[i * k + l] * B[l * m + j];
      C[i * m + j] = a;
    }
  return C;
}
function tr(A: number[], n: number, m: number): number[] {
  const T = new Array<number>(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j * n + i] = A[i * m + j];
  return T;
}
function det3(S: number[]): number {
  return S[0] * (S[4] * S[8] - S[5] * S[7]) - S[1] * (S[3] * S[8] - S[5] * S[6]) + S[2] * (S[3] * S[7] - S[4] * S[6]);
}
function inv3(S: number[]): number[] {
  const d = det3(S);
  return [
    (S[4] * S[8] - S[5] * S[7]) / d,
    (S[2] * S[7] - S[1] * S[8]) / d,
    (S[1] * S[5] - S[2] * S[4]) / d,
    (S[5] * S[6] - S[3] * S[8]) / d,
    (S[0] * S[8] - S[2] * S[6]) / d,
    (S[2] * S[3] - S[0] * S[5]) / d,
    (S[3] * S[7] - S[4] * S[6]) / d,
    (S[1] * S[6] - S[0] * S[7]) / d,
    (S[0] * S[4] - S[1] * S[3]) / d,
  ];
}
function wrapPi(a: number): number {
  // Python: (a + π) % 2π − π with a floored modulo
  const t = 2 * Math.PI;
  return a + Math.PI - t * Math.floor((a + Math.PI) / t) - Math.PI;
}

export function measFn(x: number[]): number[] {
  const [px, py, vx, vy] = x;
  const R = Math.hypot(px, py);
  return [R, Math.atan2(px, py), (px * vx + py * vy) / R];
}
export function measJacobian(x: number[]): number[] {
  const [px, py, vx, vy] = x;
  const R2 = px * px + py * py;
  const R = Math.sqrt(R2);
  const vr = (px * vx + py * vy) / R;
  return [px / R, py / R, 0, 0, py / R2, -px / R2, 0, 0, vx / R - (vr * px) / R2, vy / R - (vr * py) / R2, px / R, py / R];
}

export class ReferenceTracker {
  readonly p: RefTrackerParams;
  tracks: RefTrack[] = [];
  readonly gate: number;
  private Rm: number[];
  private nextId = 1;

  constructor(p: Partial<RefTrackerParams> = {}) {
    this.p = { ...defaultRefTrackerParams(), ...p };
    this.gate = chi2Ppf(this.p.gateProb, 3);
    this.Rm = [this.p.sigmaR ** 2, 0, 0, 0, this.p.sigmaTheta ** 2, 0, 0, 0, this.p.sigmaVr ** 2];
  }

  private initTrack(z: number[]): RefTrack {
    const [R, th, vr] = z;
    const u = [Math.sin(th), Math.cos(th)];
    const t = [Math.cos(th), -Math.sin(th)];
    const x = [R * u[0], R * u[1], vr * u[0], vr * u[1]];
    // position covariance from polar errors; velocity: radial known, tangential unknown
    const J = [u[0], R * t[0], u[1], R * t[1]]; // 2×2, columns u and R·t
    const D = [this.p.sigmaR ** 2, 0, 0, this.p.sigmaTheta ** 2];
    const Pp = mul(mul(J, D, 2, 2, 2), tr(J, 2, 2), 2, 2, 2);
    const sv = this.p.sigmaVr ** 2,
      st = this.p.initVtStd ** 2;
    const Pv = [u[0] * u[0] * sv + t[0] * t[0] * st, u[0] * u[1] * sv + t[0] * t[1] * st, u[1] * u[0] * sv + t[1] * t[0] * st, u[1] * u[1] * sv + t[1] * t[1] * st];
    const P = new Array<number>(16).fill(0);
    P[0] = Pp[0];
    P[1] = Pp[1];
    P[4] = Pp[2];
    P[5] = Pp[3];
    P[10] = Pv[0];
    P[11] = Pv[1];
    P[14] = Pv[2];
    P[15] = Pv[3];
    return { id: this.nextId++, x, P, status: 'tentative', hits: [true], misses: 0, age: 0 };
  }

  predict(dt: number): void {
    const T = dt,
      q = this.p.q;
    const F = [1, 0, T, 0, 0, 1, 0, T, 0, 0, 1, 0, 0, 0, 0, 1];
    const a = (q * T ** 3) / 3,
      b = (q * T ** 2) / 2,
      c = q * T;
    const Q = [a, 0, b, 0, 0, a, 0, b, b, 0, c, 0, 0, b, 0, c];
    for (const t of this.tracks) {
      t.x = mul(F, t.x, 4, 4, 1);
      const FP = mul(F, t.P, 4, 4, 4);
      const FPF = mul(FP, tr(F, 4, 4), 4, 4, 4);
      t.P = FPF.map((v, i) => v + Q[i]);
    }
  }

  private innov(t: RefTrack, z: number[]): { nu: number[]; S: number[]; H: number[] } {
    const H = measJacobian(t.x);
    const h = measFn(t.x);
    const nu = [z[0] - h[0], wrapPi(z[1] - h[1]), z[2] - h[2]];
    const HPH = mul(mul(H, t.P, 3, 4, 4), tr(H, 3, 4), 3, 4, 3);
    const S = HPH.map((v, i) => v + this.Rm[i]);
    return { nu, S, H };
  }

  update(Z: number[][]): void {
    const nT = this.tracks.length,
      nZ = Z.length;
    const assigned = new Map<number, number>();
    if (nT && nZ) {
      const BIG = 1e9;
      const n = Math.max(nT, nZ);
      const cost: number[][] = [];
      for (let i = 0; i < n; i++) {
        const row = new Array<number>(n).fill(BIG);
        if (i < nT)
          for (let j = 0; j < nZ; j++) {
            const { nu, S } = this.innov(this.tracks[i], Z[j]);
            const Si = inv3(S);
            const d2 = mul(tr(nu, 3, 1), mul(Si, nu, 3, 3, 1), 1, 3, 1)[0];
            if (d2 <= this.gate) row[j] = d2 + Math.log(det3(S));
          }
        cost.push(row);
      }
      const a = hungarian(cost, n, n);
      for (let i = 0; i < nT; i++) if (a[i] >= 0 && a[i] < nZ && cost[i][a[i]] < BIG) assigned.set(i, a[i]);
    }
    const used = new Set(assigned.values());
    this.tracks.forEach((t, i) => {
      t.age++;
      const j = assigned.get(i);
      if (j !== undefined) {
        const { nu, S, H } = this.innov(t, Z[j]);
        const K = mul(mul(t.P, tr(H, 3, 4), 4, 4, 3), inv3(S), 4, 3, 3);
        const dx = mul(K, nu, 4, 3, 1);
        t.x = t.x.map((v, k) => v + dx[k]);
        const KH = mul(K, H, 4, 3, 4);
        const IKH = KH.map((v, k) => (k % 5 === 0 ? 1 : 0) - v);
        const A = mul(mul(IKH, t.P, 4, 4, 4), tr(IKH, 4, 4), 4, 4, 4);
        const B = mul(mul(K, this.Rm, 4, 3, 3), tr(K, 4, 3), 4, 3, 4);
        t.P = A.map((v, k) => v + B[k]); // Joseph form
        t.hits.push(true);
        t.misses = 0;
      } else {
        t.hits.push(false);
        t.misses++;
      }
      t.hits = t.hits.slice(-this.p.confirmN);
      const nh = t.hits.filter(Boolean).length;
      if (t.status === 'tentative') {
        if (nh >= this.p.confirmM) t.status = 'confirmed';
        else if (t.hits.length >= this.p.confirmN || t.misses >= this.p.maxMisses) t.status = 'deleted';
      } else if (t.misses >= this.p.maxMisses) t.status = 'deleted';
    });
    this.tracks = this.tracks.filter((t) => t.status !== 'deleted');
    for (let j = 0; j < nZ; j++) if (!used.has(j)) this.tracks.push(this.initTrack(Z[j]));
  }

  /** Predict by dt, then update with measurements [[R, θ, vr], …]. Returns confirmed tracks. */
  step(Z: number[][], dt: number): RefTrack[] {
    this.predict(dt);
    this.update(Z);
    return this.tracks.filter((t) => t.status === 'confirmed');
  }
}
