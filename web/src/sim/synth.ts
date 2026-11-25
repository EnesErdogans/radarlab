// FMCW TDM-MIMO beat-signal synthesis, exactly per PROJECT_BRIEF:
//   τ(t) = 2R(t)/c, R(t) = R_m + vr_m (t − t_{m,t}),   t_{m,t} = m·Te + t·Tc
//   x[k,m,n] = A·exp(j[2π(fc·τ + S·τ·t_f − ½S·τ²) − 2π(pos_k·sinθcosφ + posZ_k·sinφ) + φ0])  (brief rev. 1)
//   t_f = adcStartTime + n/fs.
// Because R(t) is linear inside a chirp the phase is a quadratic polynomial in n,
// so we evaluate it with a 2nd-order phasor recurrence (fast mode) or with direct
// cos/sin per sample (exact mode, used by the tests).

import { C0, type Derived, type RadarConfig, type SimSettings } from '../config/radar';
import { STRIDE, evalTarget, scattererCount, type Target } from './targets';
import { Gaussian, NoisePool, hashSeed, mulberry32 } from './rng';
import { leakagePhase, leakageTau, type PhaseNoiseGrid } from './impairments';

const TWO_PI = 2 * Math.PI;

export interface PointScatterer {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  rcs: number;
  phase0: number;
}

export function amplitudeFor(rcs: number, R: number, s: SimSettings, cfg: RadarConfig, lambda: number): number {
  if (s.amplitudeMode === 'snr') {
    const q = s.refRange / R;
    return Math.sqrt(rcs) * q * q * Math.pow(10, s.snrRefDb / 20);
  }
  const p = s.physical;
  const pt = Math.pow(10, (p.ptDbm - 30) / 10);
  const g = Math.pow(10, (p.gtDbi + p.grDbi) / 10);
  const L = Math.pow(10, p.lossDb / 10);
  const pr = (pt * g * lambda * lambda * rcs) / (Math.pow(4 * Math.PI, 3) * R * R * R * R * L);
  const pn = 1.380649e-23 * p.t0 * Math.pow(10, p.nfDb / 10) * cfg.fs;
  return Math.sqrt(pr / pn);
}

export class Synthesizer {
  readonly cfg: RadarConfig;
  readonly d: Derived;
  private zRe: Float64Array;
  private zIm: Float64Array;
  private scat: Float64Array;
  private cr: Float64Array;
  private ci: Float64Array;
  private bases: Int32Array;
  private pool: NoisePool | null = null;
  private poolSeed = -1;
  private clutterRe: Float64Array | null = null; // K×N static clutter (identical for all chirps)
  private clutterIm: Float64Array | null = null;
  private clutterKey = '';
  private clutterPts: Float64Array | null = null; // [x, y, z, rcs, phase0] per clutter scatterer
  /** Phase-noise realisation of the current frame (frame-relative time); null = off. */
  pn: PhaseNoiseGrid | null = null;
  private pnT: Float64Array; // φ(t_c + t_f[n]) of the current chirp
  private chirpT = 0;
  /** Per-frame Swerling gains (interleaved re/im per scatterer, in synthesis order); null = SW0. */
  private swPow: Float64Array | null = null;
  private swPh: Float64Array | null = null;

  constructor(cfg: RadarConfig, d: Derived) {
    this.cfg = cfg;
    this.d = d;
    this.zRe = new Float64Array(cfg.nSamples);
    this.zIm = new Float64Array(cfg.nSamples);
    this.scat = new Float64Array(64 * STRIDE);
    this.cr = new Float64Array(d.nRx);
    this.ci = new Float64Array(d.nRx);
    this.bases = new Int32Array(d.nRx);
    this.pnT = new Float64Array(cfg.nSamples);
  }

  /** Set the start time of the chirp being synthesised (needed for phase noise). */
  beginChirp(tc: number): void {
    this.chirpT = tc;
    const pn = this.pn;
    if (!pn) return;
    const N = this.cfg.nSamples,
      fs = this.cfg.fs,
      t0 = this.cfg.adcStartTime;
    for (let n = 0; n < N; n++) this.pnT[n] = pn.eval(tc + t0 + n / fs);
  }

  /**
   * Swerling 1/3: complex gains g_j (interleaved re/im) for the scatterers in synthesis order
   * (points: list order; targets: target order, then scatterer order), one per frame, shared by
   * all channels and chirps. null → Swerling 0.
   */
  setSwerlingGains(g: Float64Array | null, n = g ? g.length >> 1 : 0): void {
    if (!g) {
      this.swPow = this.swPh = null;
      return;
    }
    if (!this.swPow || this.swPow.length < n) {
      this.swPow = new Float64Array(n);
      this.swPh = new Float64Array(n);
    }
    for (let i = 0; i < n; i++) {
      const a = g[2 * i],
        b = g[2 * i + 1];
      this.swPow[i] = a * a + b * b;
      this.swPh![i] = Math.atan2(b, a);
    }
  }

  /**
   * Add the contribution of one point scatterer for chirp m of TX tx to the cube.
   * (px..vz) evaluated at t_{m,t}. Writes into channels k = tx·nRx + r.
   */
  addScatterer(
    re: Float64Array,
    im: Float64Array,
    tx: number,
    m: number,
    px: number,
    py: number,
    pz: number,
    vx: number,
    vy: number,
    vz: number,
    rcs: number,
    phase0: number,
    s: SimSettings,
    exact: boolean,
    segx = 0,
    segy = 0,
    segz = 0,
  ): void {
    const cfg = this.cfg,
      d = this.d;
    const N = cfg.nSamples,
      M = cfg.nChirps;
    const R = Math.sqrt(px * px + py * py + pz * pz);
    if (R < 1e-3) return;
    const vr = (px * vx + py * vy + pz * vz) / R;
    let A = amplitudeFor(rcs, R, s, cfg, d.lambda);
    // Line scatterer (e.g. rotor blade): weight by the uniform-blade sinc integral
    //   A ← A·sinc(Φ), Φ = (2π/λ)·(seg·û_LOS) = (2π/λ)·(seg·p)/R  (Martin & Mulgrew 1990).
    if (segx !== 0 || segy !== 0 || segz !== 0) {
      const Phi = ((TWO_PI / d.lambda) * (segx * px + segy * py + segz * pz)) / R;
      A *= Phi === 0 ? 1 : Math.sin(Phi) / Phi;
    }
    const S = cfg.slope,
      fc = cfg.fc,
      fs = cfg.fs,
      t0 = cfg.adcStartTime;
    const tau0 = (2 * R) / C0;
    const a = (2 * vr) / C0;
    // phase [cycles] as polynomial in t_f: q0 + q1 t_f + q2 t_f²
    const q0 = fc * tau0 - 0.5 * S * tau0 * tau0;
    const q1 = fc * a + S * tau0 - S * tau0 * a;
    const q2 = S * a - 0.5 * S * a * a;
    // in n: c0 + c1 n + c2 n²
    let c0 = q0 + q1 * t0 + q2 * t0 * t0;
    const c1 = q1 / fs + (2 * q2 * t0) / fs;
    const c2 = q2 / (fs * fs);
    const u = px / R; // sinθ·cosφ
    const uz = pz / R; // sinφ
    c0 = c0 - Math.floor(c0);
    const zRe = this.zRe,
      zIm = this.zIm;
    if (exact) {
      for (let n = 0; n < N; n++) {
        // evaluate the brief's formula directly
        const tf = t0 + n / fs;
        const tau = tau0 + a * tf;
        let ph = fc * tau + S * tau * tf - 0.5 * S * tau * tau;
        ph -= Math.floor(ph);
        zRe[n] = Math.cos(TWO_PI * ph);
        zIm[n] = Math.sin(TWO_PI * ph);
      }
    } else {
      // z_{n+1} = z_n·w_n,  w_n = exp(j2π(c1 + c2(2n+1))),  w_{n+1} = w_n·exp(j2π·2c2)
      let zr = Math.cos(TWO_PI * c0),
        zi = Math.sin(TWO_PI * c0);
      let p1 = c1 + c2;
      p1 -= Math.floor(p1);
      let wr = Math.cos(TWO_PI * p1),
        wi = Math.sin(TWO_PI * p1);
      let p2 = 2 * c2;
      p2 -= Math.floor(p2);
      const dr = Math.cos(TWO_PI * p2),
        di = Math.sin(TWO_PI * p2);
      for (let n = 0; n < N; n++) {
        zRe[n] = zr;
        zIm[n] = zi;
        const nr = zr * wr - zi * wi;
        zi = zr * wi + zi * wr;
        zr = nr;
        const nwr = wr * dr - wi * di;
        wi = wr * di + wi * dr;
        wr = nwr;
      }
    }
    const pn = this.pn;
    if (pn) {
      // shared phase-noise realisation: this path gets φ(t) − φ(t − τ(t)), t = t_c + t_f
      const pnT = this.pnT,
        tc = this.chirpT;
      // grid coordinate of t − τ = t_c + t_f(1 − a) − τ0 is linear in n: x_n = x0 + n·dx
      const v = pn.v,
        last = pn.n - 1,
        idt = 1 / pn.dt;
      const x0 = (tc + t0 * (1 - a) - tau0 - pn.t0) * idt,
        dx = ((1 - a) / fs) * idt;
      for (let n = 0; n < N; n++) {
        const x = x0 + n * dx;
        let ph: number;
        if (x > 0 && x < last) {
          const i = x | 0;
          ph = v[i] + (x - i) * (v[i + 1] - v[i]);
        } else ph = pn.eval(pn.t0 + x * pn.dt);
        const dphi = pnT[n] - ph;
        let c: number, sn: number;
        if (!exact && dphi < 0.05 && dphi > -0.05) {
          // UI mode: 6th/7th-order Taylor series, error < 3e-11 for |Δφ| < 0.05 rad
          const q = dphi * dphi;
          c = 1 - q * (0.5 - q * (1 / 24 - q / 720));
          sn = dphi * (1 - q * (1 / 6 - q * (1 / 120 - q / 5040)));
        } else {
          c = Math.cos(dphi);
          sn = Math.sin(dphi);
        }
        const zr = zRe[n];
        zRe[n] = zr * c - zIm[n] * sn;
        zIm[n] = zr * sn + zIm[n] * c;
      }
    }
    const nRx = d.nRx;
    const cr = this.cr,
      ci = this.ci,
      bases = this.bases;
    for (let r = 0; r < nRx; r++) {
      const k = tx * nRx + r;
      // brief rev. 1: spatial term is negative (element at +x is closer to a target at θ > 0)
      let ph = -(d.virtPos[k] * u + d.virtPosZ[k] * uz) + phase0 / TWO_PI;
      ph -= Math.floor(ph);
      cr[r] = A * Math.cos(TWO_PI * ph);
      ci[r] = A * Math.sin(TWO_PI * ph);
      bases[r] = (k * M + m) * N;
    }
    if (cfg.iqMode === 'complex') {
        for (let r = 0; r < nRx; r++) {
          const a = cr[r],
            b = ci[r],
            base = bases[r];
          for (let n = 0; n < N; n++) {
            re[base + n] += a * zRe[n] - b * zIm[n];
            im[base + n] += a * zIm[n] + b * zRe[n];
          }
        }
    } else {
      for (let r = 0; r < nRx; r++) {
        const a = cr[r],
          b = ci[r],
          base = bases[r];
        for (let n = 0; n < N; n++) re[base + n] += a * zRe[n] - b * zIm[n];
      }
    }
  }

  /** Synthesize a noiseless cube for a list of explicit point scatterers (state at t=0 = frame start). */
  synthPoints(re: Float64Array, im: Float64Array, pts: PointScatterer[], s: SimSettings, exact: boolean): void {
    const cfg = this.cfg,
      d = this.d;
    re.fill(0);
    im.fill(0);
    const swP = this.swPow,
      swF = this.swPh;
    for (let tx = 0; tx < d.nTx; tx++)
      for (let m = 0; m < cfg.nChirps; m++) {
        const t = m * d.te + tx * cfg.chirpPeriod;
        this.beginChirp(t);
        for (let j = 0; j < pts.length; j++) {
          const p = pts[j];
          const rcs = swP ? p.rcs * swP[j] : p.rcs,
            ph0 = swF ? p.phase0 + swF[j] : p.phase0;
          this.addScatterer(re, im, tx, m, p.x + p.vx * t, p.y + p.vy * t, p.z + p.vz * t, p.vx, p.vy, p.vz, rcs, ph0, s, exact);
        }
      }
  }

  /** Synthesize all targets (bulk state at frame start, micro-motion at absolute time simTime). */
  synthTargets(re: Float64Array, im: Float64Array, targets: Target[], simTime: number, s: SimSettings, exact: boolean): void {
    const cfg = this.cfg,
      d = this.d;
    re.fill(0);
    im.fill(0);
    let maxNs = 0;
    for (const tg of targets) maxNs = Math.max(maxNs, scattererCount(tg));
    if (this.scat.length < maxNs * STRIDE) this.scat = new Float64Array(maxNs * STRIDE);
    const sc = this.scat;
    const swP = this.swPow,
      swF = this.swPh;
    // chirp-major loop order: the K_rx×N block of one chirp stays in cache while all scatterers are added
    for (let tx = 0; tx < d.nTx; tx++)
      for (let m = 0; m < cfg.nChirps; m++) {
        const t = m * d.te + tx * cfg.chirpPeriod;
        this.beginChirp(t);
        let j = 0;
        for (const tg of targets) {
          const ns = evalTarget(tg, t, sc, 0, simTime);
          for (let i = 0; i < ns; i++, j++) {
            const o = i * STRIDE;
            const rcs = swP ? sc[o + 6] * swP[j] : sc[o + 6],
              ph0 = swF ? sc[o + 7] + swF[j] : sc[o + 7];
            this.addScatterer(re, im, tx, m, sc[o], sc[o + 1], sc[o + 2], sc[o + 3], sc[o + 4], sc[o + 5], rcs, ph0, s, exact, sc[o + 8], sc[o + 9], sc[o + 10]);
          }
        }
      }
  }

  /** Static clutter: random stationary scatterers; identical in every chirp, so cached as K×N. */
  addClutter(re: Float64Array, im: Float64Array, s: SimSettings): void {
    const cfg = this.cfg,
      d = this.d;
    const N = cfg.nSamples,
      M = cfg.nChirps,
      K = d.nVirt;
    const key = `${s.seed}|${s.nClutter}|${s.snrRefDb}|${s.refRange}|${s.amplitudeMode}`;
    if (!this.clutterRe || key !== this.clutterKey) {
      this.clutterKey = key;
      const cr = new Float64Array(K * N),
        ci = new Float64Array(K * N);
      // a tiny 1-chirp synthesizer writing into a K×1×N cube
      const one = new Synthesizer({ ...cfg, nChirps: 1 }, { ...d });
      const rng = mulberry32(hashSeed(s.seed, 7777));
      const rmax = d.rMax;
      const pts = new Float64Array(5 * s.nClutter);
      for (let i = 0; i < s.nClutter; i++) {
        const r = 0.08 * rmax + rng() * 0.85 * rmax;
        const az = (rng() * 2 - 1) * Math.min(70, d.fovDeg) * (Math.PI / 180);
        const rcs = Math.pow(10, (rng() * 25 - 5) / 10) * Math.pow(r / s.refRange, 1.2); // larger patches far away
        const ph = rng() * TWO_PI;
        pts.set([r * Math.sin(az), r * Math.cos(az), -0.5, rcs, ph], 5 * i);
        for (let tx = 0; tx < d.nTx; tx++) one.addScatterer(cr, ci, tx, 0, pts[5 * i], pts[5 * i + 1], -0.5, 0, 0, 0, rcs, ph, s, false);
      }
      this.clutterRe = cr;
      this.clutterIm = ci;
      this.clutterPts = pts;
    }
    if (this.pn) {
      // with phase noise every chirp differs (shared realisation): synthesise per chirp
      const pts = this.clutterPts!,
        nc = pts.length / 5;
      for (let tx = 0; tx < d.nTx; tx++)
        for (let m = 0; m < M; m++) {
          this.beginChirp(m * d.te + tx * cfg.chirpPeriod);
          for (let i = 0; i < nc; i++) {
            const o = 5 * i;
            this.addScatterer(re, im, tx, m, pts[o], pts[o + 1], pts[o + 2], 0, 0, 0, pts[o + 3], pts[o + 4], s, false);
          }
        }
      return;
    }
    const cr = this.clutterRe!,
      ci = this.clutterIm!;
    for (let k = 0; k < K; k++)
      for (let m = 0; m < M; m++) {
        const base = (k * M + m) * N,
          cb = k * N;
        for (let n = 0; n < N; n++) {
          re[base + n] += cr[cb + n];
          im[base + n] += ci[cb + n];
        }
      }
  }

  /**
   * Tx→Rx leakage (Python `signal.Leakage`): a static path with τ_L = 2·distance/c, amplitude
   * 10^(powerDb/20), no spatial term, fixed phase `phases[k]` per virtual channel, and the frame's
   * shared phase-noise residual φ(t) − φ(t − τ_L).
   */
  addLeakage(re: Float64Array, im: Float64Array, distance: number, powerDb: number, phases: ArrayLike<number>): void {
    const cfg = this.cfg,
      d = this.d;
    const N = cfg.nSamples,
      M = cfg.nChirps,
      nRx = d.nRx,
      fs = cfg.fs,
      t0 = cfg.adcStartTime;
    const tau = leakageTau(distance);
    const A = Math.pow(10, powerDb / 20);
    const real = cfg.iqMode === 'real';
    const zr = this.zRe,
      zi = this.zIm;
    let ready = false;
    for (let tx = 0; tx < d.nTx; tx++)
      for (let m = 0; m < M; m++) {
        const tc = m * d.te + tx * cfg.chirpPeriod;
        if (this.pn || !ready) {
          this.beginChirp(tc);
          for (let n = 0; n < N; n++) {
            const tf = t0 + n / fs;
            let psi = leakagePhase(cfg.fc, cfg.slope, tau, tf);
            if (this.pn) psi += this.pnT[n] - this.pn.eval(tc + tf - tau);
            zr[n] = Math.cos(psi);
            zi[n] = Math.sin(psi);
          }
          ready = true;
        }
        for (let r = 0; r < nRx; r++) {
          const k = tx * nRx + r;
          const a = A * Math.cos(phases[k]),
            b = A * Math.sin(phases[k]);
          const base = (k * M + m) * N;
          for (let n = 0; n < N; n++) {
            re[base + n] += a * zr[n] - b * zi[n];
            if (!real) im[base + n] += a * zi[n] + b * zr[n];
          }
        }
      }
  }

  /** Add complex white Gaussian noise of variance 1 (σ²/2 per I and Q). */
  addNoise(re: Float64Array, im: Float64Array, s: SimSettings, frameIdx: number): void {
    const len = re.length;
    const sc = Math.SQRT1_2;
    const real = this.cfg.iqMode === 'real';
    if (s.exactNoise) {
      const g = new Gaussian(hashSeed(s.seed, frameIdx));
      for (let i = 0; i < len; i++) {
        re[i] += sc * g.next();
        if (!real) im[i] += sc * g.next();
      }
      return;
    }
    // UI: block-permuted, randomly phase-rotated pool with rolling refill (see NoisePool)
    if (!this.pool || this.poolSeed !== s.seed || this.pool.size < Math.min(2 * len, 1 << 23)) {
      this.pool = new NoisePool(hashSeed(s.seed, 0x5eed), 2 * len);
      this.poolSeed = s.seed;
    }
    this.pool.addTo(re, im, len, sc, real);
  }
}
