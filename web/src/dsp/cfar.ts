// 2D CFAR on the square-law range-Doppler power map P[r·M + d].
// Doppler axis is circular, range axis is clipped (α recomputed for the actual
// number of training cells). With K non-coherently summed channels the noise is
// Gamma(K): α is exact for every type (brief rev. 1, same definitions as the
// Python reference): CA closed form (caAlphaGamma), OS/GO/SO by quadrature
// (osAlphaGamma, gsAlphaGamma; see gamma.ts). K = 1 uses the closed forms.

import type { CfarParams, CfarType } from '../config/radar';
import { gsLogPfaGamma, osLogPfaGamma, solveAlpha } from './gamma';

/** CA-CFAR: threshold = α · mean(training). Pfa = (1 + α/N)^(−N). */
export function caAlpha(n: number, pfa: number): number {
  return n * (Math.pow(pfa, -1 / n) - 1);
}

function bisect(f: (a: number) => number, target: number, lo = 0, hi = 1e9): number {
  // f decreasing in α; find f(α) = target
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (f(mid) > target) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12 * Math.max(1, mid)) break;
  }
  return 0.5 * (lo + hi);
}

/** Pfa of OS-CFAR with N cells, rank k (1-based), threshold α·X_(k). */
export function osPfa(n: number, k: number, alpha: number): number {
  let lp = 0;
  for (let i = 0; i < k; i++) lp += Math.log(n - i) - Math.log(n - i + alpha);
  return Math.exp(lp);
}
export function osAlpha(n: number, k: number, pfa: number): number {
  const lt = Math.log(pfa);
  return bisect((a) => {
    let lp = 0;
    for (let i = 0; i < k; i++) lp += Math.log(n - i) - Math.log(n - i + a);
    return lp;
  }, lt);
}

/** Σ_{k=0}^{n-1} C(n-1+k,k)(2+T)^{-(n+k)}, T = α/n (threshold on the mean of n cells). */
function soSum(n: number, T: number): number {
  let lterm = -n * Math.log(2 + T);
  let s = 0;
  for (let k = 0; k < n; k++) {
    s += Math.exp(lterm);
    lterm += Math.log((n + k) / (k + 1)) - Math.log(2 + T);
  }
  return s;
}
export function soPfa(n: number, alpha: number): number {
  return 2 * soSum(n, alpha / n);
}
export function goPfa(n: number, alpha: number): number {
  return 2 * Math.pow(1 + alpha / n, -n) - 2 * soSum(n, alpha / n);
}
export function goAlpha(n: number, pfa: number): number {
  return bisect((a) => goPfa(n, a), pfa);
}
export function soAlpha(n: number, pfa: number): number {
  return bisect((a) => soPfa(n, a), pfa);
}

/** CA-CFAR α for Gamma(K)-distributed cells (non-coherent sum of K unit-exponential channels). */
export function caPfaGamma(n: number, K: number, alpha: number): number {
  const T = alpha / n;
  const NK = n * K;
  let lterm = -NK * Math.log1p(T);
  let s = 0;
  for (let i = 0; i < K; i++) {
    s += Math.exp(lterm);
    lterm += Math.log((NK + i) / (i + 1)) + Math.log(T) - Math.log1p(T);
  }
  return s;
}
export function caAlphaGamma(n: number, K: number, pfa: number): number {
  return bisect((a) => caPfaGamma(n, K, a), pfa);
}

/** OS-CFAR α (threshold α·X_(k)) for Gamma(K) cells; exact (quadrature for K > 1). */
export function osAlphaGamma(n: number, k: number, K: number, pfa: number): number {
  if (K <= 1) return osAlpha(n, k, pfa);
  return solveAlpha((a) => osLogPfaGamma(n, k, K, a), pfa);
}
/** Exact OS-CFAR Pfa for Gamma(K) cells. */
export function osPfaGamma(n: number, k: number, K: number, alpha: number): number {
  return K <= 1 ? osPfa(n, k, alpha) : Math.exp(osLogPfaGamma(n, k, K, alpha));
}

/**
 * GO/SO-CFAR α on the *mean* of a half with nHalf cells (α = nHalf·β) for Gamma(K) cells;
 * exact (closed form for K = 1, quadrature for K > 1).
 */
export function gsAlphaGamma(nHalf: number, K: number, pfa: number, kind: 'GO' | 'SO'): number {
  if (K <= 1) return kind === 'GO' ? goAlpha(nHalf, pfa) : soAlpha(nHalf, pfa);
  return nHalf * solveAlpha((b) => gsLogPfaGamma(nHalf, K, b, kind), pfa);
}
/** Exact GO/SO Pfa for threshold α·max/min(mean₁, mean₂) of two halves of nHalf Gamma(K) cells. */
export function gsPfaGamma(nHalf: number, K: number, alpha: number, kind: 'GO' | 'SO'): number {
  if (K <= 1) return kind === 'GO' ? goPfa(nHalf, alpha) : soPfa(nHalf, alpha);
  return Math.exp(gsLogPfaGamma(nHalf, K, alpha / nHalf, kind));
}

// process-wide α cache: the quadrature solve takes a few ms, engines are rebuilt on every
// parameter change, and edge cells need one α per distinct training-cell count
const ALPHA_CACHE = new Map<string, number>();

/** α for a CFAR type with n training cells (GO/SO: n = cells per half; OS: rank k), K channels. */
export function cfarAlpha(type: CfarType, n: number, pfa: number, K = 1, k = 0): number {
  const key = `${type}|${n}|${k}|${K}|${pfa}`;
  let a = ALPHA_CACHE.get(key);
  if (a === undefined) {
    if (n <= 0) a = Infinity;
    else if (type === 'CA') a = K > 1 ? caAlphaGamma(n, K, pfa) : caAlpha(n, pfa);
    else if (type === 'OS') a = osAlphaGamma(n, k, K, pfa);
    else a = gsAlphaGamma(n, K, pfa, type);
    if (ALPHA_CACHE.size > 20000) ALPHA_CACHE.clear();
    ALPHA_CACHE.set(key, a);
  }
  return a;
}

/** round(x) with ties to even (numpy.round), used for the OS rank k = round(rank·N). */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x),
    d = x - f;
  if (Math.abs(d - 0.5) < 1e-12) return f % 2 === 0 ? f : f + 1;
  return Math.round(x);
}
export function osRankIndex(n: number, rank: number): number {
  return Math.min(n, Math.max(1, roundHalfEven(rank * n)));
}

/** Quickselect: k-th smallest (0-based) of a[0..n), partially reorders a. */
export function quickselect(a: Float64Array, n: number, k: number): number {
  let lo = 0,
    hi = n - 1;
  while (hi > lo) {
    // median of three pivot
    const mid = (lo + hi) >> 1;
    const x = a[lo],
      y = a[mid],
      z = a[hi];
    const pivot = x < y ? (y < z ? y : x < z ? z : x) : x < z ? x : y < z ? z : y;
    let i = lo,
      j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}

/** α lookup bound to one parameter set and K (Gamma(K)-exact for all four types). */
export class AlphaCache {
  constructor(private p: CfarParams, readonly gammaK = 1) {}
  get(type: CfarType, n: number, k = 0): number {
    return cfarAlpha(type, n, this.p.pfa, this.gammaK, k);
  }
}

export interface CfarOutput {
  thr: Float64Array; // threshold per cell (NaN where skipped by prefilter)
  noise: Float64Array; // noise statistic per cell = thr/α of the active type (training mean, GO/SO half mean, OS k-th value)
  mask: Uint8Array; // 1 = detection
  count: number;
}

/**
 * 2D CFAR processor with preallocated buffers for a fixed (nR, M).
 */
export class Cfar2D {
  readonly nR: number;
  readonly M: number;
  private sat: Float64Array;
  private wExt = 0;
  private buf: Float64Array;
  private tmp: Float64Array;
  private osN: Int32Array;
  private osK: Int32Array;
  private osA: Float64Array;
  readonly out: CfarOutput;
  private alphas: AlphaCache;
  private params: CfarParams;

  constructor(nR: number, M: number, p: CfarParams, gammaK = 1) {
    this.nR = nR;
    this.M = M;
    this.params = p;
    this.alphas = new AlphaCache(p, gammaK);
    this.sat = new Float64Array(0);
    this.buf = new Float64Array(4096);
    const wr = p.guardR + p.trainR,
      wd = Math.min(p.guardD + p.trainD, (M - 1) >> 1);
    this.tmp = new Float64Array(Math.max(1, Math.min(nR, 8 + 2 * wr) * Math.min(M, 8 + 2 * wd)));
    this.osN = new Int32Array(nR);
    this.osK = new Int32Array(nR);
    this.osA = new Float64Array(nR);
    this.out = { thr: new Float64Array(nR * M), noise: new Float64Array(nR * M), mask: new Uint8Array(nR * M), count: 0 };
  }

  private buildSat(P: Float64Array, wd: number): void {
    const nR = this.nR,
      M = this.M;
    const W = M + 2 * wd;
    this.wExt = W;
    const need = (nR + 1) * (W + 1);
    if (this.sat.length < need) this.sat = new Float64Array(need);
    const sat = this.sat;
    for (let c = 0; c <= W; c++) sat[c] = 0;
    for (let r = 0; r < nR; r++) {
      const row = (r + 1) * (W + 1);
      const prev = r * (W + 1);
      sat[row] = 0;
      let acc = 0;
      for (let c = 0; c < W; c++) {
        let d = c - wd;
        d = ((d % M) + M) % M;
        acc += P[r * M + d];
        sat[row + c + 1] = sat[prev + c + 1] + acc;
      }
    }
  }

  /** Sum of rows [r0,r1], extended cols [c0,c1] (inclusive). */
  private rect(r0: number, r1: number, c0: number, c1: number): number {
    const W1 = this.wExt + 1;
    const s = this.sat;
    return s[(r1 + 1) * W1 + c1 + 1] - s[r0 * W1 + c1 + 1] - s[(r1 + 1) * W1 + c0] + s[r0 * W1 + c0];
  }

  run(P: Float64Array, type: CfarType = this.params.type, exact = true): CfarOutput {
    const p = this.params;
    const nR = this.nR,
      M = this.M;
    const { thr, mask, noise } = this.out;
    mask.fill(0);
    let count = 0;
    const wr = p.guardR + p.trainR,
      wd = Math.min(p.guardD + p.trainD, (M - 1) >> 1);
    const gd = Math.min(p.guardD, wd);
    if (type === 'OS') {
      count = this.runOS(P, exact);
      this.out.count = count;
      return this.out;
    }
    this.buildSat(P, wd);
    for (let r = 0; r < nR; r++) {
      const r0 = Math.max(0, r - wr),
        r1 = Math.min(nR - 1, r + wr);
      const g0 = Math.max(0, r - p.guardR),
        g1 = Math.min(nR - 1, r + p.guardR);
      for (let d = 0; d < M; d++) {
        const dc = d + wd;
        let t: number, z: number;
        if (type === 'CA') {
          const outer = this.rect(r0, r1, dc - wd, dc + wd);
          const inner = this.rect(g0, g1, dc - gd, dc + gd);
          const n = (r1 - r0 + 1) * (2 * wd + 1) - (g1 - g0 + 1) * (2 * gd + 1);
          z = n > 0 ? (outer - inner) / n : 0;
          t = n > 0 ? this.alphas.get('CA', n) * z : Infinity;
        } else {
          // halves: lead = Δr<0 or (Δr=0, Δd<0); lag = Δr>0 or (Δr=0, Δd>0) (training cells only)
          let sLead = 0,
            sLag = 0,
            nLead = 0,
            nLag = 0;
          const w = 2 * wd + 1;
          if (r0 <= r - 1) {
            const gt0 = Math.max(r0, r - p.guardR);
            sLead = this.rect(r0, r - 1, dc - wd, dc + wd);
            nLead = (r - r0) * w;
            if (gt0 <= r - 1) {
              sLead -= this.rect(gt0, r - 1, dc - gd, dc + gd);
              nLead -= (r - gt0) * (2 * gd + 1);
            }
          }
          if (r + 1 <= r1) {
            const gt1 = Math.min(r1, r + p.guardR);
            sLag = this.rect(r + 1, r1, dc - wd, dc + wd);
            nLag = (r1 - r) * w;
            if (r + 1 <= gt1) {
              sLag -= this.rect(r + 1, gt1, dc - gd, dc + gd);
              nLag -= (gt1 - r) * (2 * gd + 1);
            }
          }
          if (wd > gd) {
            sLead += this.rect(r, r, dc - wd, dc - gd - 1);
            sLag += this.rect(r, r, dc + gd + 1, dc + wd);
            nLead += wd - gd;
            nLag += wd - gd;
          }
          const nTr = nLead + nLag;
          const nMin = Math.min(nLead, nLag);
          // GO/SO → CA fallback when one half has < 1/4 of the cells (range edges); Gamma(K) CA α as in Python
          if (nMin < nTr / 4) {
            z = nTr > 0 ? (sLead + sLag) / nTr : 0;
            t = nTr > 0 ? this.alphas.get('CA', nTr) * z : Infinity;
          } else {
            const mLead = sLead / nLead,
              mLag = sLag / nLag;
            z = type === 'GO' ? Math.max(mLead, mLag) : Math.min(mLead, mLag);
            t = this.alphas.get(type, nMin) * z;
          }
        }
        const i = r * M + d;
        thr[i] = t;
        noise[i] = z;
        if (P[i] > t) {
          mask[i] = 1;
          count++;
        }
      }
    }
    this.out.count = count;
    return this.out;
  }

  private runOS(P: Float64Array, exact: boolean): number {
    const p = this.params;
    const nR = this.nR,
      M = this.M;
    const { thr, mask, noise } = this.out;
    const wr = p.guardR + p.trainR,
      wd = Math.min(p.guardD + p.trainD, (M - 1) >> 1);
    const gd = Math.min(p.guardD, wd);
    const maxN = (2 * wr + 1) * (2 * wd + 1);
    if (this.buf.length < maxN) this.buf = new Float64Array(maxN);
    const buf = this.buf;
    // per-row training-cell count, rank and α (only range is clipped; Doppler is circular)
    const nRow = this.osN,
      kRow = this.osK,
      aRow = this.osA;
    for (let r = 0; r < nR; r++) {
      const r0 = Math.max(0, r - wr),
        r1 = Math.min(nR - 1, r + wr);
      const g0 = Math.max(0, r - p.guardR),
        g1 = Math.min(nR - 1, r + p.guardR);
      const n = (r1 - r0 + 1) * (2 * wd + 1) - (g1 - g0 + 1) * (2 * gd + 1);
      nRow[r] = n;
      kRow[r] = osRankIndex(n, p.osRank);
      aRow[r] = this.alphas.get('OS', n, kRow[r]);
    }
    let count = 0;
    // Exact block prefilter (UI mode): for a block of CUTs let U be the union of their
    // windows. Every CUT's training set S ⊂ U, hence x_(k)(S) ≥ x_(k)(U) ≥ x_(k*)(U) with
    // k* = min k over the block, and its threshold ≥ α_min·x_(k*)(U). Cells with
    // P ≤ that bound cannot be detections and are skipped (thr = NaN) — identical
    // detections to the exact mode by construction (no noise-floor assumption).
    const BR = 8,
      BD = 8;
    const tmp = this.tmp;
    for (let rb = 0; rb < nR; rb += BR) {
      const re = Math.min(nR, rb + BR);
      let bound = -Infinity;
      for (let db = 0; db < M; db += BD) {
        const de = Math.min(M, db + BD);
        if (!exact) {
          let kMin = Infinity,
            aMin = Infinity;
          for (let r = rb; r < re; r++) {
            if (kRow[r] < kMin) kMin = kRow[r];
            if (aRow[r] < aMin) aMin = aRow[r];
          }
          const u0 = Math.max(0, rb - wr),
            u1 = Math.min(nR - 1, re - 1 + wr);
          const span = Math.min(M, de - db + 2 * wd);
          let m = 0;
          for (let rr = u0; rr <= u1; rr++) {
            const row = rr * M;
            for (let j = 0; j < span; j++) {
              let c = db - wd + j;
              if (c < 0) c += M;
              else if (c >= M) c -= M;
              tmp[m++] = P[row + c];
            }
          }
          bound = aMin * quickselect(tmp, m, kMin - 1);
        }
        for (let r = rb; r < re; r++) {
          const r0 = Math.max(0, r - wr),
            r1 = Math.min(nR - 1, r + wr);
          for (let d = db; d < de; d++) {
            const i = r * M + d;
            if (P[i] <= bound) {
              thr[i] = NaN;
              noise[i] = NaN;
              continue;
            }
            let n = 0;
            for (let rr = r0; rr <= r1; rr++) {
              const inG = Math.abs(rr - r) <= p.guardR;
              const row = rr * M;
              for (let dd = -wd; dd <= wd; dd++) {
                if (inG && Math.abs(dd) <= gd) continue;
                let c = d + dd;
                if (c < 0) c += M;
                else if (c >= M) c -= M;
                buf[n++] = P[row + c];
              }
            }
            const k = kRow[r];
            const xk = quickselect(buf, n, k - 1);
            const t = aRow[r] * xk;
            thr[i] = t;
            noise[i] = xk;
            if (P[i] > t) {
              mask[i] = 1;
              count++;
            }
          }
        }
      }
    }
    return count;
  }

  /** Thresholds of all four CFAR types along one Doppler column (for the CFAR-cut panel). */
  cut(P: Float64Array, d: number, out: Float64Array /* 4·nR: CA,GO,SO,OS */): void {
    const types: CfarType[] = ['CA', 'GO', 'SO', 'OS'];
    const p = this.params;
    const nR = this.nR,
      M = this.M;
    const wr = p.guardR + p.trainR,
      wd = Math.min(p.guardD + p.trainD, (M - 1) >> 1);
    const gd = Math.min(p.guardD, wd);
    const maxN = (2 * wr + 1) * (2 * wd + 1);
    if (this.buf.length < maxN) this.buf = new Float64Array(maxN);
    const buf = this.buf;
    for (let r = 0; r < nR; r++) {
      const r0 = Math.max(0, r - wr),
        r1 = Math.min(nR - 1, r + wr);
      let sum = 0,
        n = 0,
        sLead = 0,
        nLead = 0,
        sLag = 0,
        nLag = 0;
      for (let rr = r0; rr <= r1; rr++) {
        const inG = Math.abs(rr - r) <= p.guardR;
        for (let dd = -wd; dd <= wd; dd++) {
          let c = d + dd;
          if (c < 0) c += M;
          else if (c >= M) c -= M;
          const v = P[rr * M + c];
          if (inG && Math.abs(dd) <= gd) continue;
          if (rr < r || (rr === r && dd < 0)) {
            sLead += v;
            nLead++;
          } else {
            sLag += v;
            nLag++;
          }
          sum += v;
          buf[n++] = v;
        }
      }
      for (let ti = 0; ti < 4; ti++) {
        const type = types[ti];
        let t: number;
        if (type === 'CA') t = (this.alphas.get('CA', n) * sum) / n;
        else if (type === 'OS') {
          const k = osRankIndex(n, p.osRank);
          t = this.alphas.get('OS', n, k) * quickselect(buf, n, k - 1);
        } else if (Math.min(nLead, nLag) < n / 4) {
          t = (this.alphas.get('CA', n) * sum) / n;
        } else {
          const nn = Math.min(nLead, nLag);
          const a = sLead / nLead,
            b = sLag / nLag;
          t = this.alphas.get(type, nn) * (type === 'GO' ? Math.max(a, b) : Math.min(a, b));
        }
        out[ti * nR + r] = t;
      }
    }
  }
}
