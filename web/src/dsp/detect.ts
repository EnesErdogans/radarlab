// Connected-component clustering of CFAR detections, parabolic peak interpolation,
// TDM-MIMO Doppler phase compensation and zero-padded FFT angle estimation,
// plus the range-angle map.

import type { Derived, RadarConfig } from '../config/radar';
import { getFFT } from './fft';

export interface Detection {
  rBin: number; // integer peak cell
  dBin: number;
  rFrac: number; // interpolated
  dFrac: number;
  range: number;
  velocity: number;
  power: number; // linear peak power
  snrDb: number; // peak power / CFAR noise statistic (threshold / α of the active CFAR type) in dB
  angleBin: number; // interpolated, fftshifted (A/2 = boresight)
  angleDeg: number; // asin(u_x): what the 1-D azimuth FFT measures (= θ when φ = 0)
  elevDeg: number; // φ from same-x / different-z virtual pairs (0 if the array has none)
  azimuthDeg: number; // θ = atan2(x, y) (brief convention), corrected for elevation
  x: number;
  y: number;
  z: number;
  cells: number;
}

export function newDetection(): Detection {
  return { rBin: 0, dBin: 0, rFrac: 0, dFrac: 0, range: 0, velocity: 0, power: 0, snrDb: 0, angleBin: 0, angleDeg: 0, elevDeg: 0, azimuthDeg: 0, x: 0, y: 0, z: 0, cells: 0 };
}

/** Parabolic interpolation offset in (−0.5, 0.5) from three samples (use log power). */
export function parabolic(a: number, b: number, c: number): number {
  const den = a - 2 * b + c;
  if (den >= 0 || !isFinite(den)) return 0;
  const o = (0.5 * (a - c)) / den;
  return Math.max(-0.5, Math.min(0.5, o));
}

export class Detector {
  readonly cfg: RadarConfig;
  readonly d: Derived;
  readonly nR: number;
  readonly M: number;
  readonly A: number;
  private visited: Uint8Array;
  /** cluster index of every mask cell after cluster() (−1 = none), for tests / tooling */
  readonly labels: Int32Array;
  private stack: Int32Array;
  readonly dets: Detection[];
  count = 0;
  private aRe: Float64Array;
  private aIm: Float64Array;
  private compRe: Float64Array; // [t·M + d]
  private compIm: Float64Array;
  private lagRe: Float64Array;
  private lagIm: Float64Array;
  private xr: Float64Array;
  private xi: Float64Array;
  private elPairs: Int32Array; // (low, high) virtual-channel index pairs with equal x, z_high > z_low
  private elDz = 0;
  private raAcc = new Float64Array(0);

  constructor(cfg: RadarConfig, d: Derived, nAngle = 64, maxDet = 256) {
    this.cfg = cfg;
    this.d = d;
    this.nR = d.nRangeBins;
    this.M = cfg.nChirps;
    this.A = nAngle;
    this.visited = new Uint8Array(this.nR * this.M);
    this.labels = new Int32Array(this.nR * this.M);
    this.stack = new Int32Array(this.nR * this.M);
    this.dets = Array.from({ length: maxDet }, newDetection);
    this.aRe = new Float64Array(nAngle);
    this.aIm = new Float64Array(nAngle);
    this.lagRe = new Float64Array(nAngle);
    this.lagIm = new Float64Array(nAngle);
    this.xr = new Float64Array(d.nVirt);
    this.xi = new Float64Array(d.nVirt);
    // elevation pairs (same rule as Python estimate_elevation)
    const pairs: number[] = [];
    for (let i = 0; i < d.nVirt; i++)
      for (let j = 0; j < d.nVirt; j++)
        if (Math.abs(d.virtPos[i] - d.virtPos[j]) < 1e-9 && d.virtPosZ[j] - d.virtPosZ[i] > 1e-9) {
          pairs.push(i, j);
          this.elDz = d.virtPosZ[j] - d.virtPosZ[i];
        }
    this.elPairs = new Int32Array(pairs);
    // TDM compensation phasors for each TX and Doppler bin (bin-centre velocity)
    this.compRe = new Float64Array(d.nTx * this.M);
    this.compIm = new Float64Array(d.nTx * this.M);
    for (let t = 0; t < d.nTx; t++)
      for (let k = 0; k < this.M; k++) {
        const v = (k - this.M / 2) * d.dv;
        const ph = -2 * Math.PI * ((2 * v) / d.lambda) * t * cfg.chirpPeriod;
        this.compRe[t * this.M + k] = Math.cos(ph);
        this.compIm[t * this.M + k] = Math.sin(ph);
      }
  }

  /** Cluster CFAR mask into detections (peak + interpolation). */
  cluster(P: Float64Array, mask: Uint8Array, noiseStat: Float64Array): number {
    const nR = this.nR,
      M = this.M;
    const vis = this.visited;
    vis.fill(0);
    const lab = this.labels;
    lab.fill(-1);
    const st = this.stack;
    let nd = 0;
    for (let i0 = 0; i0 < mask.length && nd < this.dets.length; i0++) {
      if (!mask[i0] || vis[i0]) continue;
      let sp = 0;
      st[sp++] = i0;
      vis[i0] = 1;
      let best = i0,
        bestP = P[i0],
        cells = 0;
      while (sp > 0) {
        const i = st[--sp];
        lab[i] = nd;
        cells++;
        if (P[i] > bestP) {
          bestP = P[i];
          best = i;
        }
        const r = (i / M) | 0,
          d = i - r * M;
        for (let dr = -1; dr <= 1; dr++) {
          const rr = r + dr;
          if (rr < 0 || rr >= nR) continue;
          for (let dd = -1; dd <= 1; dd++) {
            if (!dr && !dd) continue;
            let c = d + dd;
            if (c < 0) c += M;
            else if (c >= M) c -= M;
            const j = rr * M + c;
            if (mask[j] && !vis[j]) {
              vis[j] = 1;
              st[sp++] = j;
            }
          }
        }
      }
      const r = (best / M) | 0,
        dB = best - r * M;
      const det = this.dets[nd++];
      det.rBin = r;
      det.dBin = dB;
      det.cells = cells;
      det.power = bestP;
      const lp = Math.log(bestP + 1e-300);
      let ro = 0;
      if (r > 0 && r < nR - 1) ro = parabolic(Math.log(P[(r - 1) * M + dB] + 1e-300), lp, Math.log(P[(r + 1) * M + dB] + 1e-300));
      const dm = dB === 0 ? M - 1 : dB - 1,
        dp = dB === M - 1 ? 0 : dB + 1;
      const doff = parabolic(Math.log(P[r * M + dm] + 1e-300), lp, Math.log(P[r * M + dp] + 1e-300));
      det.rFrac = r + ro;
      det.dFrac = dB + doff;
      det.range = det.rFrac * this.d.rangeRes;
      det.velocity = (det.dFrac - M / 2) * this.d.dv;
      const noise = noiseStat[best];
      det.snrDb = 10 * Math.log10(bestP / (noise > 0 ? noise : 1e-30));
    }
    this.count = nd;
    return nd;
  }

  /** Interpolated range/velocity/angle of a single RD cell (used by golden tests and hover readouts). */
  measureCell(P: Float64Array, rdRe: Float64Array, rdIm: Float64Array, r: number, d: number): Detection {
    const M = this.M,
      nR = this.nR;
    const det = newDetection();
    det.rBin = r;
    det.dBin = d;
    det.power = P[r * M + d];
    const lp = Math.log(det.power + 1e-300);
    const ro = r > 0 && r < nR - 1 ? parabolic(Math.log(P[(r - 1) * M + d] + 1e-300), lp, Math.log(P[(r + 1) * M + d] + 1e-300)) : 0;
    const dm = d === 0 ? M - 1 : d - 1,
      dp = d === M - 1 ? 0 : d + 1;
    const doff = parabolic(Math.log(P[r * M + dm] + 1e-300), lp, Math.log(P[r * M + dp] + 1e-300));
    det.rFrac = r + ro;
    det.dFrac = d + doff;
    det.range = det.rFrac * this.d.rangeRes;
    det.velocity = (det.dFrac - M / 2) * this.d.dv;
    const save = this.dets[0],
      cnt = this.count;
    this.dets[0] = det;
    this.count = 1;
    this.estimateAngles(rdRe, rdIm);
    this.dets[0] = save;
    this.count = cnt;
    return det;
  }

  /** Zero-padded FFT DoA with TDM phase compensation (uses interpolated velocity). */
  estimateAngles(rdRe: Float64Array, rdIm: Float64Array): void {
    const d = this.d,
      cfg = this.cfg,
      A = this.A,
      M = this.M,
      nR = this.nR;
    const fft = getFFT(A);
    const re = this.aRe,
      im = this.aIm;
    for (let i = 0; i < this.count; i++) {
      const det = this.dets[i];
      re.fill(0);
      im.fill(0);
      const cr = this.xr,
        ci = this.xi;
      for (let k = 0; k < d.nVirt; k++) {
        const t = (k / d.nRx) | 0;
        const idx = ((k * nR + det.rBin) * M) + det.dBin;
        const ph = -2 * Math.PI * ((2 * det.velocity) / d.lambda) * t * cfg.chirpPeriod;
        const c = Math.cos(ph),
          s = Math.sin(ph);
        const xr = rdRe[idx],
          xi = rdIm[idx];
        // TDM-compensated snapshot
        cr[k] = xr * c - xi * s;
        ci[k] = xr * s + xi * c;
        const slot = d.angleIdx[k];
        if (slot < 0) continue;
        // conjugated snapshot + forward FFT ≡ +j kernel Σ g_k e^{+j2πka/A}, matched to the
        // −j steering vector of brief rev. 1 (same as the Python reference): θ > 0 → bin > A/2
        re[slot] = cr[k];
        im[slot] = -ci[k];
      }
      fft.forward(re, im);
      let bi = 0,
        bp = -1;
      for (let a = 0; a < A; a++) {
        const p = re[a] * re[a] + im[a] * im[a];
        if (p > bp) {
          bp = p;
          bi = a;
        }
      }
      const pa = re[(bi + A - 1) % A] ** 2 + im[(bi + A - 1) % A] ** 2;
      const pc = re[(bi + 1) % A] ** 2 + im[(bi + 1) % A] ** 2;
      const off = parabolic(Math.log(pa + 1e-300), Math.log(bp + 1e-300), Math.log(pc + 1e-300));
      // fftshift: unshifted bin b ↔ shifted (b + A/2) mod A
      const sb = ((bi + A / 2) % A) + off;
      det.angleBin = sb;
      let ux = (sb - A / 2) / (A * d.arraySpacing);
      // wrapped past the edge of visible space (d > ½ λ), as Python estimate_azimuth
      if (ux > 1) ux -= 1 / d.arraySpacing;
      else if (ux < -1) ux += 1 / d.arraySpacing;
      ux = Math.max(-1, Math.min(1, ux));
      det.angleDeg = (Math.asin(ux) * 180) / Math.PI;
      // elevation φ = asin(−arg(Σ conj(x_low)·x_high) / (2π·Δz))
      let uz = 0;
      if (this.elPairs.length) {
        let ar = 0,
          ai = 0;
        for (let q = 0; q < this.elPairs.length; q += 2) {
          const lo = this.elPairs[q],
            hi = this.elPairs[q + 1];
          ar += cr[lo] * cr[hi] + ci[lo] * ci[hi];
          ai += cr[lo] * ci[hi] - ci[lo] * cr[hi];
        }
        const v = -Math.atan2(ai, ar) / (2 * Math.PI * this.elDz);
        uz = Math.max(-1, Math.min(1, v));
      }
      det.elevDeg = (Math.asin(uz) * 180) / Math.PI;
      const uy = Math.sqrt(Math.max(1 - ux * ux - uz * uz, 0));
      det.x = det.range * ux;
      det.y = det.range * uy;
      det.z = det.range * uz;
      det.azimuthDeg = (Math.atan2(ux, uy) * 180) / Math.PI;
    }
  }

  /**
   * Range-angle power map (fftshifted angle bins, A per range bin):
   * out[r, a] = Σ_d |Σ_k w_k·x̃_k[r,d]·e^{+j2π·slot_k·a/A}|² / (Σw)², x̃ = TDM-compensated with the
   * bin-centre velocity of Doppler bin d, w = window over the L ULA slots (periodic; rect if
   * omitted) — the same definition and scaling as Python `range_angle_map`. Computed via the
   * lag-domain (spatial autocorrelation) identity when 2L−1 ≤ A, otherwise by a direct
   * zero-padded FFT per Doppler bin (L > A is rejected: the spectrum would be undersampled).
   */
  rangeAngle(rdRe: Float64Array, rdIm: Float64Array, out: Float64Array, window?: Float64Array): void {
    const d = this.d,
      A = this.A,
      M = this.M,
      nR = this.nR,
      K = d.nVirt;
    const fft = getFFT(A);
    const lr = this.lagRe,
      li = this.lagIm,
      xr = this.xr,
      xi = this.xi;
    const idx = d.angleIdx;
    const nRx = d.nRx;
    const L = d.arrayLen;
    if (L > A) throw new Error(`rangeAngle: ULA length ${L} > angle FFT size ${A}`);
    if (window && window.length !== L) throw new Error(`rangeAngle: window length ${window.length} != ULA length ${L}`);
    let sw = 0;
    for (let l = 0; l < L; l++) sw += window ? window[l] : 1;
    const norm = 1 / (sw * sw);
    const canLag = 2 * L - 1 <= A;
    const acc = canLag ? null : this.raAcc.length >= A ? this.raAcc : (this.raAcc = new Float64Array(A));
    for (let r = 0; r < nR; r++) {
      lr.fill(0);
      li.fill(0);
      if (acc) acc.fill(0);
      for (let dd = 0; dd < M; dd++) {
        for (let k = 0; k < K; k++) {
          const t = (k / nRx) | 0;
          const j = (k * nR + r) * M + dd;
          const cr = this.compRe[t * M + dd],
            ci = this.compIm[t * M + dd];
          const w = window && idx[k] >= 0 ? window[idx[k]] : 1;
          const a = rdRe[j] * w,
            b = rdIm[j] * w;
          xr[k] = a * cr - b * ci;
          xi[k] = a * ci + b * cr;
        }
        if (canLag) {
          for (let k = 0; k < K; k++) {
            if (idx[k] < 0) continue;
            const ar = xr[k],
              ai = xi[k];
            for (let l = 0; l <= k; l++) {
              if (idx[l] < 0) continue;
              // x_k · conj(x_l) at lag idx_k − idx_l
              const lag = idx[k] - idx[l];
              const p = lag >= 0 ? lag : lag + A;
              const pr = ar * xr[l] + ai * xi[l];
              const pi = ai * xr[l] - ar * xi[l];
              lr[p] += pr;
              li[p] += pi;
              if (l !== k) {
                const q = lag > 0 ? A - lag : -lag;
                lr[q] += pr;
                li[q] -= pi;
              }
            }
          }
        } else {
          // direct: +j kernel = conjugate input + forward FFT (|·|² unchanged), accumulate power
          lr.fill(0);
          li.fill(0);
          for (let k = 0; k < K; k++) {
            if (idx[k] < 0) continue;
            lr[idx[k]] = xr[k];
            li[idx[k]] = -xi[k];
          }
          fft.forward(lr, li);
          for (let a = 0; a < A; a++) acc![a] += lr[a] * lr[a] + li[a] * li[a];
        }
      }
      const base = r * A;
      if (acc) {
        // conj-input forward FFT bin a ≡ +j-kernel bin a; fftshift
        for (let a = 0; a < A; a++) out[base + ((a + A / 2) % A)] = acc[a] * norm;
        continue;
      }
      fft.forward(lr, li);
      // +j kernel (brief rev. 1): bin a of the +j transform = bin −a of the forward FFT; then fftshift
      for (let a = 0; a < A; a++) out[base + ((((A - a) % A) + A / 2) % A)] = Math.max(0, lr[a]) * norm;
    }
  }
}
