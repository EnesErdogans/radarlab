// Hardware impairments (roadmap A4), matching the Python reference
// `radarlab.impairments` / `radarlab.signal.synthesize_cube`:
//
//  * phase noise: one realisation φ(t) per frame, shared by every TX, RX, scatterer and the
//    Tx→Rx leakage path; each path gets φ(t) − φ(t−τ) (range correlation, Budge & Burt 1993).
//    PSD (two-sided, rad²/Hz)  L(f) = 10^(L0/10)/(1 + (f/f_bw)²) + 10^(floor/10).
//  * Tx→Rx leakage: static path, τ_L = 2·distance/c, per-sample power `powerDb` above the unit
//    noise variance, no spatial term, a fixed phase per virtual channel.
//  * Swerling 1/3: one complex gain per scatterer per frame, shared by all channels and chirps;
//    |g|² ~ χ²₂/2 (SW1) or χ²₄/4 (SW3), mean 1; phase uniform.
//  * IQ imbalance: y = I + j·g·(Q·cosδ − I·sinδ).
//  * IF filter: digital Butterworth HPF (and optional LPF) along fast time, causal, zero initial
//    state per chirp (scipy.signal.butter + lfilter).
//  * ADC: mid-tread uniform quantiser with clipping per rail, LSB = FS/2^(bits−1), codes in
//    [−2^(bits−1), 2^(bits−1)−1], round half to even (numpy.round).
// The Python reference has no DC-offset or per-channel gain/phase-mismatch model yet (roadmap E1),
// so neither does the web engine.

const C0 = 299792458; // (local copy: avoids an import cycle with config/radar)
import { Gaussian } from './rng';

export interface PhaseNoiseParams {
  l0DbcHz: number; // plateau level L0 [dBc/Hz]
  loopBwHz: number; // PLL loop bandwidth f_bw [Hz]
  floorDbcHz: number; // white floor [dBc/Hz] (≤ −300 disables)
  fsGen: number; // generation grid rate [Hz] (Python default 50 MHz)
}

export interface ImpairmentSettings {
  phaseNoise: boolean;
  pn: PhaseNoiseParams;
  leakage: boolean;
  leakDistance: number; // [m]
  leakPowerDb: number; // per-sample power above unit noise [dB]
  swerling: 0 | 1 | 3;
  iq: boolean;
  iqGainDb: number;
  iqPhaseDeg: number;
  ifFilter: boolean;
  hpfHz: number; // 0 = off
  lpfHz: number; // 0 = off
  filterOrder: number;
  adc: boolean;
  adcBits: number;
  adcFullScaleDb: number; // full scale per rail, 20·log10(FS) relative to the unit noise σ (complex)
}

export function defaultImpairments(): ImpairmentSettings {
  return {
    phaseNoise: false,
    // Python PhaseNoise defaults (fidelity §1: L(1 MHz) ≈ −95 dBc/Hz, AWR1843)
    pn: { l0DbcHz: -87, loopBwHz: 450e3, floorDbcHz: -140, fsGen: 50e6 },
    leakage: false,
    leakDistance: 0.1,
    leakPowerDb: 40,
    swerling: 0,
    iq: false,
    iqGainDb: 0.5,
    iqPhaseDeg: 3,
    ifFilter: false,
    hpfHz: 150e3,
    lpfHz: 0,
    filterOrder: 1,
    adc: false,
    adcBits: 12,
    adcFullScaleDb: 50,
  };
}

export function anyImpairment(s: ImpairmentSettings | undefined): boolean {
  return !!s && (s.phaseNoise || s.leakage || s.swerling !== 0 || s.iq || s.ifFilter || s.adc);
}

// ------------------------------------------------------------------ phase noise

/** Two-sided phase-noise PSD L(f) [rad²/Hz]. */
export function pnPsd(p: PhaseNoiseParams, f: number): number {
  const q = f / p.loopBwHz;
  return Math.pow(10, p.l0DbcHz / 10) / (1 + q * q) + Math.pow(10, p.floorDbcHz / 10);
}

/** var(φ(t) − φ(t−τ)) = ∫ L(f)·4·sin²(πfτ) df over ±fsGen/2 (as Python `PhaseNoise.residual_variance`). */
export function pnResidualVariance(p: PhaseNoiseParams, tau: number, nPts = 400001): number {
  const a = -p.fsGen / 2,
    h = p.fsGen / (nPts - 1);
  let acc = 0;
  for (let i = 0; i < nPts; i++) {
    const f = a + i * h;
    const s = Math.sin(Math.PI * f * tau);
    const v = pnPsd(p, f) * 4 * s * s;
    acc += i === 0 || i === nPts - 1 ? 0.5 * v : v;
  }
  return acc * h;
}

/**
 * φ(t) sampled on a uniform grid t_i = t0 + i·dt; `eval` is linear interpolation clamped to
 * the end values (numpy.interp semantics).
 */
export class PhaseNoiseGrid {
  t0: number;
  dt: number;
  v: Float64Array;
  n: number;
  constructor(t0: number, dt: number, v: Float64Array, n = v.length) {
    this.t0 = t0;
    this.dt = dt;
    this.v = v;
    this.n = n;
  }
  eval(t: number): number {
    const x = (t - this.t0) / this.dt;
    if (!(x > 0)) return this.v[0];
    const i = Math.floor(x);
    if (i >= this.n - 1) return this.v[this.n - 1];
    const f = x - i;
    return this.v[i] + f * (this.v[i + 1] - this.v[i]);
  }
}

/**
 * Draw one frame of phase noise. The Lorentzian part L0/(1+(f/f_bw)²) is exactly the PSD of an
 * Ornstein–Uhlenbeck process (rate a = 2π·f_bw, stationary variance π·L0·f_bw), which can be
 * sampled exactly at arbitrary grid points; the white floor adds independent N(0, floor·fsGen)
 * samples (band-limited to ±fsGen/2 as in Python). Only the grid points needed by the ADC
 * windows [t_c + t_adc − τmax, t_c + t_adc + (N−1)/fs] of every chirp are generated; gaps
 * between windows are bridged with the exact OU transition, so the process is the same as a
 * full-frame realisation at those points. Python uses FFT shaping on the full grid instead;
 * both have PSD L(f) up to aliasing beyond ±fsGen/2 (< 1 % of the variance for the defaults).
 */
export class PhaseNoiseGenerator {
  grid: PhaseNoiseGrid;
  private buf: Float64Array;
  constructor() {
    this.buf = new Float64Array(16);
    this.grid = new PhaseNoiseGrid(0, 1, this.buf, 1);
  }
  /**
   * @param chirpStarts chirp start times (frame-relative) in any order
   * @param winStart window start relative to the chirp start (t_adc − τmax)
   * @param winEnd window end relative to the chirp start (t_adc + (N−1)/fs)
   */
  generate(p: PhaseNoiseParams, chirpStarts: number[], winStart: number, winEnd: number, g: Gaussian): PhaseNoiseGrid {
    const dt = 1 / p.fsGen;
    const starts = chirpStarts.slice().sort((a, b) => a - b);
    const t0 = starts[0] + winStart - 2 * dt;
    const n = Math.ceil((starts[starts.length - 1] + winEnd - t0) / dt) + 3;
    if (this.buf.length < n) this.buf = new Float64Array(Math.max(n, Math.ceil(this.buf.length * 1.5)));
    const v = this.buf;
    const a = 2 * Math.PI * p.loopBwHz;
    const varOu = Math.PI * Math.pow(10, p.l0DbcHz / 10) * p.loopBwHz;
    const sdOu = Math.sqrt(varOu);
    const sdFloor = p.floorDbcHz > -300 ? Math.sqrt(Math.pow(10, p.floorDbcHz / 10) * p.fsGen) : 0;
    const rho1 = Math.exp(-a * dt),
      s1 = sdOu * Math.sqrt(1 - rho1 * rho1);
    // Marsaglia polar method (no trig; ~2× faster than Box–Muller here)
    let spare = 0,
      hasSpare = false;
    const normal = (): number => {
      if (hasSpare) {
        hasSpare = false;
        return spare;
      }
      let u: number, w: number, q: number;
      do {
        u = 2 * g.uniform() - 1;
        w = 2 * g.uniform() - 1;
        q = u * u + w * w;
      } while (q >= 1 || q === 0);
      const f = Math.sqrt((-2 * Math.log(q)) / q);
      spare = w * f;
      hasSpare = true;
      return u * f;
    };
    let last = -1;
    let x = 0;
    for (const tc of starts) {
      let i0 = Math.max(0, Math.floor((tc + winStart - t0) / dt) - 1);
      const i1 = Math.min(n - 1, Math.ceil((tc + winEnd - t0) / dt) + 1);
      if (i0 <= last) i0 = last + 1;
      for (let i = i0; i <= i1; i++) {
        if (last < 0) x = sdOu * normal();
        else if (i === last + 1) x = rho1 * x + s1 * normal();
        else {
          const rho = Math.exp(-a * (i - last) * dt);
          x = rho * x + sdOu * Math.sqrt(1 - rho * rho) * normal();
        }
        v[i] = sdFloor > 0 ? x + sdFloor * normal() : x;
        last = i;
      }
    }
    this.grid = new PhaseNoiseGrid(t0, dt, v, n);
    return this.grid;
  }
}

// ------------------------------------------------------------------ Swerling

/**
 * Per-frame complex Swerling gains for `n` scatterers, interleaved [re0, im0, re1, im1, …].
 * SW1: |g|² ~ Exp(1) = χ²₂/2; SW3: |g|² ~ χ²₄/4; phase uniform; mean power 1.
 */
export function swerlingGains(model: 0 | 1 | 3, n: number, g: Gaussian, out?: Float64Array): Float64Array {
  const o = out && out.length >= 2 * n ? out : new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    let p = 1;
    if (model === 1) p = -Math.log(1 - g.uniform());
    else if (model === 3) p = 0.5 * (-Math.log(1 - g.uniform()) - Math.log(1 - g.uniform()));
    const ph = model === 0 ? 0 : 2 * Math.PI * g.uniform();
    const a = Math.sqrt(p);
    o[2 * i] = a * Math.cos(ph);
    o[2 * i + 1] = a * Math.sin(ph);
  }
  return o;
}

// ------------------------------------------------------------------ leakage

/** Leakage beat phase [rad] (without phase noise) at fast time t_f: 2π(fc·τ + S·τ·t_f − ½·S·τ²). */
export function leakagePhase(fc: number, slope: number, tau: number, tf: number): number {
  return 2 * Math.PI * (fc * tau + slope * tau * tf - 0.5 * slope * tau * tau);
}

export function leakageTau(distance: number): number {
  return (2 * distance) / C0;
}

// ------------------------------------------------------------------ IQ imbalance

/** In place: y = I + j·g·(Q·cosδ − I·sinδ), g = 10^(gainDb/20), δ = phaseDeg. */
export function iqImbalance(re: Float64Array, im: Float64Array, len: number, gainDb: number, phaseDeg: number): void {
  const g = Math.pow(10, gainDb / 20);
  const d = (phaseDeg * Math.PI) / 180;
  const gc = g * Math.cos(d),
    gs = g * Math.sin(d);
  for (let i = 0; i < len; i++) im[i] = gc * im[i] - gs * re[i];
}

/** Image-rejection ratio |μ|²/|ν|² [dB]. */
export function imageRejectionDb(gainDb: number, phaseDeg: number): number {
  const g = Math.pow(10, gainDb / 20),
    d = (phaseDeg * Math.PI) / 180;
  const mr = (1 + g * Math.cos(d)) / 2,
    mi = (-g * Math.sin(d)) / 2;
  const nr = (1 - g * Math.cos(d)) / 2,
    ni = (-g * Math.sin(d)) / 2;
  return 10 * Math.log10((mr * mr + mi * mi) / Math.max(nr * nr + ni * ni, 1e-300));
}

// ------------------------------------------------------------------ Butterworth IF filter

type Cx = [number, number];
const cmul = (a: Cx, b: Cx): Cx => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cdiv = (a: Cx, b: Cx): Cx => {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};

function poly(roots: Cx[]): number[] {
  let c: Cx[] = [[1, 0]];
  for (const r of roots) {
    const nc: Cx[] = c.map((x) => [x[0], x[1]] as Cx).concat([[0, 0]]);
    for (let i = 1; i < nc.length; i++) {
      const m = cmul(r, c[i - 1]);
      nc[i] = [nc[i][0] - m[0], nc[i][1] - m[1]];
    }
    c = nc;
  }
  return c.map((x) => x[0]);
}

/**
 * Digital Butterworth filter, identical to `scipy.signal.butter(order, wn, btype)` (wn = f/(fs/2)):
 * analog prototype poles, frequency pre-warping, lp2lp/lp2hp, bilinear transform (zpk), zpk→tf.
 */
export function butter(order: number, wn: number, type: 'lowpass' | 'highpass'): { b: number[]; a: number[] } {
  const N = order;
  const p0: Cx[] = [];
  for (let m = -N + 1; m < N; m += 2) {
    const th = (Math.PI * m) / (2 * N);
    p0.push([-Math.cos(th), -Math.sin(th)]); // −exp(jπm/(2N))
  }
  const fs = 2;
  const wo = 2 * fs * Math.tan((Math.PI * wn) / fs);
  let z: Cx[] = [];
  let p: Cx[];
  let k: number;
  if (type === 'lowpass') {
    p = p0.map((x) => [x[0] * wo, x[1] * wo] as Cx);
    k = Math.pow(wo, N);
  } else {
    p = p0.map((x) => cdiv([wo, 0], x));
    z = Array.from({ length: N }, () => [0, 0] as Cx);
    let prod: Cx = [1, 0];
    for (const x of p0) prod = cmul(prod, [-x[0], -x[1]]);
    k = 1 / prod[0];
  }
  // bilinear transform
  const fs2 = 2 * fs;
  const zz = z.map((x) => cdiv([fs2 + x[0], x[1]], [fs2 - x[0], -x[1]]));
  const pz = p.map((x) => cdiv([fs2 + x[0], x[1]], [fs2 - x[0], -x[1]]));
  while (zz.length < pz.length) zz.push([-1, 0]);
  let num: Cx = [1, 0],
    den: Cx = [1, 0];
  for (const x of z) num = cmul(num, [fs2 - x[0], -x[1]]);
  for (const x of p) den = cmul(den, [fs2 - x[0], -x[1]]);
  const kz = k * cdiv(num, den)[0];
  return { b: poly(zz).map((c) => c * kz), a: poly(pz) };
}

/**
 * Causal IIR filter (transposed direct form II, zero initial state) applied in place to every
 * row of length N (fast time) — `scipy.signal.lfilter(b, a, x, axis=-1)`. a[0] must be 1.
 */
export function lfilterRows(x: Float64Array, rows: number, N: number, b: number[], a: number[], zi?: Float64Array): void {
  const L = Math.max(a.length, b.length);
  const bb = new Float64Array(L),
    aa = new Float64Array(L);
  for (let i = 0; i < b.length; i++) bb[i] = b[i] / a[0];
  for (let i = 0; i < a.length; i++) aa[i] = a[i] / a[0];
  const z = zi && zi.length >= L ? zi : new Float64Array(L);
  if (L === 2) {
    // first-order fast path
    const b0 = bb[0],
      b1 = bb[1],
      a1 = aa[1];
    for (let r = 0; r < rows; r++) {
      const o = r * N;
      let s = 0;
      for (let n = 0; n < N; n++) {
        const xi = x[o + n];
        const y = b0 * xi + s;
        s = b1 * xi - a1 * y;
        x[o + n] = y;
      }
    }
    return;
  }
  for (let r = 0; r < rows; r++) {
    const o = r * N;
    z.fill(0);
    for (let n = 0; n < N; n++) {
      const xi = x[o + n];
      const y = bb[0] * xi + z[0];
      for (let j = 1; j < L; j++) z[j - 1] = bb[j] * xi + (j < L - 1 ? z[j] : 0) - aa[j] * y;
      x[o + n] = y;
    }
  }
}

// ------------------------------------------------------------------ ADC

/** numpy.round semantics (round half to even). */
export function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** 4 × rms of all rails (Python default full scale). */
export function autoFullScale(re: Float64Array, im: Float64Array | null, len: number): number {
  let acc = 0;
  for (let i = 0; i < len; i++) acc += re[i] * re[i];
  let cnt = len;
  if (im) {
    for (let i = 0; i < len; i++) acc += im[i] * im[i];
    cnt += len;
  }
  return 4 * (Math.sqrt(acc / cnt) || 1);
}

/**
 * In-place uniform quantisation with clipping of each rail to [−FS, FS). Returns the number of
 * clipped rail samples. `im` null = real IF.
 */
export function adcQuantize(re: Float64Array, im: Float64Array | null, len: number, bits: number, fullScale: number): number {
  const lsb = fullScale / Math.pow(2, bits - 1);
  const lo = -Math.pow(2, bits - 1),
    hi = Math.pow(2, bits - 1) - 1;
  let clipped = 0;
  const q = (v: number): number => {
    let c = roundHalfEven(v / lsb); // v/lsb as numpy (not v·(1/lsb))
    if (c < lo) {
      c = lo;
      clipped++;
    } else if (c > hi) {
      c = hi;
      clipped++;
    }
    return c * lsb;
  };
  for (let i = 0; i < len; i++) re[i] = q(re[i]);
  if (im) for (let i = 0; i < len; i++) im[i] = q(im[i]);
  return clipped;
}
