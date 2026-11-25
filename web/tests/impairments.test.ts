// Impairments (roadmap A4): deterministic parts against golden/impairments.json (all random
// inputs stored there — no cross-language RNG), stochastic parts by statistical tests.

import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PRESETS, defaultSettings, derive, type RadarConfig, type SimSettings } from '../src/config/radar';
import { Synthesizer, type PointScatterer } from '../src/sim/synth';
import { Gaussian } from '../src/sim/rng';
import {
  PhaseNoiseGenerator,
  PhaseNoiseGrid,
  adcQuantize,
  autoFullScale,
  butter,
  defaultImpairments,
  imageRejectionDb,
  iqImbalance,
  lfilterRows,
  pnPsd,
  pnResidualVariance,
  roundHalfEven,
  swerlingGains,
  type PhaseNoiseParams,
} from '../src/sim/impairments';
import { getFFT } from '../src/dsp/fft';
import { Engine, allocOut } from '../src/engine/engine';
import { defaultScene } from '../src/sim/scenes';

const here = dirname(fileURLToPath(import.meta.url));
const file = join(here, '..', '..', 'golden', 'impairments.json');
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

function toConfig(c: Json): RadarConfig {
  return {
    fc: c.fc,
    slope: c.slope,
    fs: c.fs,
    nSamples: c.nSamples,
    adcStartTime: c.adcStartTime ?? 0,
    chirpPeriod: c.chirpPeriod,
    nChirps: c.nChirps,
    txPos: c.txPos,
    rxPos: c.rxPos,
    txPosZ: c.txPosZ ?? undefined,
    rxPosZ: c.rxPosZ ?? undefined,
    framePeriod: c.framePeriod ?? 0.05,
    iqMode: c.iqMode ?? 'complex',
    rampTime: c.rampTime ?? (c.idleTime !== undefined ? c.chirpPeriod - c.idleTime : undefined),
  };
}

function maxErr(aRe: ArrayLike<number>, aIm: ArrayLike<number>, bRe: ArrayLike<number>, bIm: ArrayLike<number>): { err: number; mx: number } {
  let err = 0,
    mx = 0;
  for (let i = 0; i < bRe.length; i++) {
    err = Math.max(err, Math.hypot(aRe[i] - bRe[i], aIm[i] - bIm[i]));
    mx = Math.max(mx, Math.hypot(bRe[i], bIm[i]));
  }
  return { err, mx };
}

describe.skipIf(!existsSync(file))('impairments golden (python reference)', () => {
  const g: Json = existsSync(file) ? JSON.parse(readFileSync(file, 'utf8')) : {};
  const T = g.tolerances ?? {};
  const cfg = g.config ? toConfig(g.config) : PRESETS.automotive.cfg;
  const d = derive(cfg);
  const s: SimSettings = { ...defaultSettings(), noise: false, snrRefDb: g.snrRefDb ?? 20, refRange: g.refRange ?? 10 };
  const pts: PointScatterer[] = (g.scatterers ?? []).map((p: Json) => ({ ...p }));
  const len = d.nVirt * cfg.nChirps * cfg.nSamples;

  function synth(withPn: boolean, exact: boolean): { re: Float64Array; im: Float64Array } {
    const sy = new Synthesizer(cfg, d);
    const pn = g.phaseNoise;
    sy.pn = withPn ? new PhaseNoiseGrid(pn.t0, 1 / pn.fsGen, Float64Array.from(pn.values)) : null;
    const gn = new Float64Array(2 * pts.length);
    g.swerling.gainsRe.forEach((v: number, i: number) => (gn[2 * i] = v));
    g.swerling.gainsIm.forEach((v: number, i: number) => (gn[2 * i + 1] = v));
    sy.setSwerlingGains(gn);
    const re = new Float64Array(len),
      im = new Float64Array(len);
    sy.synthPoints(re, im, pts, s, exact);
    sy.addLeakage(re, im, g.leakage.distance, g.leakage.powerDb, g.leakage.phases);
    return { re, im };
  }

  for (const exact of [true, false]) {
    it(`beat cube with shared phase noise + Tx→Rx leakage + Swerling-1 gains (${exact ? 'exact' : 'phasor recurrence'})`, () => {
      const rel = (T.cubeRelMax ?? 1e-9) * (exact ? 1 : 100);
      const a = synth(true, exact);
      const r = maxErr(a.re, a.im, g.cube.re, g.cube.im);
      expect(r.err, `max err ${r.err} vs max ${r.mx}`).toBeLessThanOrEqual(rel * r.mx);
      const b = synth(false, exact);
      const r2 = maxErr(b.re, b.im, g.cubeNoPhaseNoise.re, g.cubeNoPhaseNoise.im);
      expect(r2.err).toBeLessThanOrEqual(rel * r2.mx);
      // the phase noise really changes the cube (the test is not vacuous)
      expect(maxErr(a.re, a.im, b.re, b.im).err).toBeGreaterThan(0.05 * r.mx);
    });
  }

  it('IQ imbalance, Butterworth IF HPF+LPF, ADC codes (incl. clipping) on the stored cube', () => {
    const rel = T.chainRelMax ?? 1e-9;
    const re = Float64Array.from(g.cube.re),
      im = Float64Array.from(g.cube.im);
    const q = g.iqImbalance;
    iqImbalance(re, im, len, q.gainDb, q.phaseDeg);
    let r = maxErr(re, im, q.re, q.im);
    expect(r.err).toBeLessThanOrEqual(rel * r.mx);
    expect(imageRejectionDb(q.gainDb, q.phaseDeg)).toBeCloseTo(q.imageRejectionDb, 10);

    const f = g.ifFilter;
    const nyq = cfg.fs / 2;
    const hp = butter(f.order, f.hpfHz / nyq, 'highpass'),
      lp = butter(f.order, f.lpfHz / nyq, 'lowpass');
    const rows = len / cfg.nSamples;
    for (const x of [re, im]) {
      lfilterRows(x, rows, cfg.nSamples, hp.b, hp.a);
      lfilterRows(x, rows, cfg.nSamples, lp.b, lp.a);
    }
    r = maxErr(re, im, f.re, f.im);
    expect(r.err).toBeLessThanOrEqual(rel * r.mx);

    // ADC from the stored filter output (so that rounding sees the same input)
    const a = g.adc;
    const xr = Float64Array.from(f.re),
      xi = Float64Array.from(f.im);
    expect(autoFullScale(xr, xi, len)).toBeCloseTo(a.autoFullScale, 9);
    const clipped = adcQuantize(xr, xi, len, a.bits, a.fullScale);
    let bad = 0;
    for (let i = 0; i < len; i++) {
      if (Math.round(xr[i] / a.lsb) !== a.codesRe[i]) bad++;
      if (Math.round(xi[i] / a.lsb) !== a.codesIm[i]) bad++;
    }
    expect(bad).toBeLessThanOrEqual(T.codesMismatchMax ?? 0);
    expect(clipped).toBe(a.clippedRails);
    const yr = Float64Array.from(f.re),
      yi = Float64Array.from(f.im);
    const fsAuto = autoFullScale(yr, yi, len);
    adcQuantize(yr, yi, len, a.autoBits, fsAuto);
    const lsbA = fsAuto / Math.pow(2, a.autoBits - 1);
    for (let i = 0; i < a.autoCodesRe.length; i++) {
      expect(Math.round(yr[i] / lsbA)).toBe(a.autoCodesRe[i]);
      expect(Math.round(yi[i] / lsbA)).toBe(a.autoCodesIm[i]);
    }
  });

  it('Butterworth coefficients = scipy.signal.butter (orders 1–4, low/high pass)', () => {
    const rel = T.butterRel ?? 1e-10;
    for (const row of g.butter) {
      const { b, a } = butter(row.order, row.wn, row.type);
      expect(b.length).toBe(row.b.length);
      for (let i = 0; i < b.length; i++) expect(Math.abs(b[i] - row.b[i])).toBeLessThanOrEqual(rel * Math.max(1, Math.abs(row.b[i])));
      for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - row.a[i])).toBeLessThanOrEqual(rel * Math.max(1, Math.abs(row.a[i])));
    }
  });

  it('analytic phase-noise residual variance var(φ(t) − φ(t−τ)) = Python PhaseNoise.residual_variance', () => {
    const r = g.phaseNoiseResidualVariance;
    const p: PhaseNoiseParams = r.params;
    r.tau.forEach((tau: number, i: number) => {
      const v = pnResidualVariance(p, tau);
      expect(Math.abs(v - r.var[i]) / r.var[i]).toBeLessThanOrEqual(T.residualVarRel ?? 1e-6);
    });
  });
});

describe('impairment primitives', () => {
  it('round half to even (numpy.round) and quantiser clipping range', () => {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4, 2.6].map(roundHalfEven)).toEqual([0, 2, 2, -0, -2, -2, 2, 3]);
    const re = Float64Array.from([10, -10, 0.26, -0.24]),
      im = Float64Array.from([0, 0, 0, 0]);
    const c = adcQuantize(re, im, 4, 4, 1); // LSB 1/8, codes −8..7
    expect(c).toBe(2);
    expect(Array.from(re)).toEqual([7 / 8, -1, 0.25, -0.25]);
  });

  it('Swerling gains: one complex gain per scatterer, applied identically to every channel and chirp', () => {
    const cfg = { ...PRESETS.automotive.cfg, nChirps: 16, nSamples: 64 };
    const d = derive(cfg);
    const s: SimSettings = { ...defaultSettings(), noise: false };
    const p: PointScatterer[] = [{ x: 3, y: 20, z: 0, vx: 0, vy: 4, vz: 0, rcs: 1, phase0: 0.3 }];
    const sy = new Synthesizer(cfg, d);
    const L = d.nVirt * cfg.nChirps * cfg.nSamples;
    const r0 = new Float64Array(L),
      i0 = new Float64Array(L),
      r1 = new Float64Array(L),
      i1 = new Float64Array(L);
    sy.synthPoints(r0, i0, p, s, true);
    const gr = 0.6,
      gi = -1.1;
    sy.setSwerlingGains(Float64Array.from([gr, gi]));
    sy.synthPoints(r1, i1, p, s, true);
    let err = 0;
    for (let i = 0; i < L; i++) err = Math.max(err, Math.hypot(r1[i] - (gr * r0[i] - gi * i0[i]), i1[i] - (gr * i0[i] + gi * r0[i])));
    expect(err).toBeLessThan(1e-9);
  });
});

describe('impairments: statistics of the stochastic parts', () => {
  it('Swerling 1 / 3 power distribution (χ²₂/2, χ²₄/4) and uniform phase', () => {
    const n = 200000;
    for (const [model, varP, tail] of [
      [1, 1, Math.exp(-1)],
      [3, 0.5, 3 * Math.exp(-2)],
    ] as const) {
      const gg = swerlingGains(model, n, new Gaussian(42 + model));
      let m = 0,
        m2 = 0,
        above = 0,
        sr = 0,
        si = 0;
      for (let i = 0; i < n; i++) {
        const p = gg[2 * i] ** 2 + gg[2 * i + 1] ** 2;
        m += p;
        m2 += p * p;
        if (p > 1) above++;
        sr += gg[2 * i];
        si += gg[2 * i + 1];
      }
      m /= n;
      const v = m2 / n - m * m;
      expect(Math.abs(m - 1)).toBeLessThan(0.01);
      expect(Math.abs(v - varP) / varP).toBeLessThan(0.03);
      expect(Math.abs(above / n - tail)).toBeLessThan(0.005); // P(|g|² > 1): e^{-1} (SW1), 3e^{-2} (SW3)
      expect(Math.hypot(sr, si) / n).toBeLessThan(0.01); // E[g] = 0 (uniform phase)
    }
  });

  const pnp: PhaseNoiseParams = { ...defaultImpairments().pn }; // Python defaults

  it('phase-noise PSD matches L(f) = L0/(1+(f/f_bw)²) + floor (Welch estimate)', () => {
    const nseg = 1024,
      nAvg = 256;
    const total = nseg * nAvg;
    const gen = new PhaseNoiseGenerator();
    const grid = gen.generate(pnp, [0], 0, (total - 1) / pnp.fsGen, new Gaussian(9));
    const fft = getFFT(nseg);
    const w = new Float64Array(nseg);
    let u = 0;
    for (let i = 0; i < nseg; i++) {
      w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nseg);
      u += w[i] * w[i];
    }
    const psd = new Float64Array(nseg);
    const re = new Float64Array(nseg),
      im = new Float64Array(nseg);
    const i0 = Math.round((0 - grid.t0) / grid.dt);
    for (let k = 0; k < 2 * nAvg - 1; k++) {
      const o = i0 + (k * nseg) / 2; // 50 % overlap
      if (o + nseg > grid.n) break;
      for (let i = 0; i < nseg; i++) {
        re[i] = grid.v[o + i] * w[i];
        im[i] = 0;
      }
      fft.forward(re, im);
      for (let i = 0; i < nseg; i++) psd[i] += (re[i] ** 2 + im[i] ** 2) / (u * pnp.fsGen);
    }
    const nUsed = 2 * nAvg - 1;
    const df = pnp.fsGen / nseg;
    for (const f of [150e3, 450e3, 1e6, 2e6, 5e6]) {
      const b = Math.round(f / df);
      // average a few bins (both signs) to reduce the estimator variance
      let acc = 0,
        ref = 0;
      for (let j = b - 2; j <= b + 2; j++) {
        acc += psd[j] + psd[nseg - j];
        ref += 2 * pnPsd(pnp, j * df);
      }
      acc /= nUsed;
      const errDb = 10 * Math.log10(acc / ref);
      expect(Math.abs(errDb), `f = ${f / 1e6} MHz: ${errDb.toFixed(2)} dB`).toBeLessThan(0.6);
    }
  });

  it('phase-noise range correlation: var(φ(t) − φ(t−τ)) matches the analytic residual variance', () => {
    const gen = new PhaseNoiseGenerator();
    const p = { ...pnp, l0DbcHz: -80 };
    const taus = [20 / pnp.fsGen, 50 / pnp.fsGen, 33.3 / pnp.fsGen]; // 0.4 µs, 1 µs, off-grid 0.67 µs
    const acc = taus.map(() => 0);
    let cnt = 0;
    for (let rep = 0; rep < 40; rep++) {
      const grid = gen.generate(p, [0], 0, 20000 / pnp.fsGen, new Gaussian(100 + rep));
      for (let i = 0; i < 19000; i += 7) {
        const t = 1000 / pnp.fsGen + i / pnp.fsGen;
        const a = grid.eval(t);
        taus.forEach((tau, j) => (acc[j] += (a - grid.eval(t - tau)) ** 2));
        cnt++;
      }
    }
    taus.forEach((tau, j) => {
      const sim = acc[j] / cnt,
        th = pnResidualVariance(p, tau, 100001);
      // linear interpolation of the off-grid point smooths the floor slightly: 10 %
      expect(Math.abs(sim / th - 1), `τ = ${(tau * 1e6).toFixed(2)} µs sim ${sim} th ${th}`).toBeLessThan(0.1);
    });
  });

  it('gap bridging: φ is one stationary OU process across chirp windows (corr = e^{−2π f_bw Δ})', () => {
    const p: PhaseNoiseParams = { ...pnp, floorDbcHz: -400, loopBwHz: 200e3 };
    const gen = new PhaseNoiseGenerator();
    const gap = 3e-6;
    const winEnd = 1e-6;
    let sxy = 0,
      sxx = 0,
      syy = 0;
    const R = 4000;
    for (let rep = 0; rep < R; rep++) {
      const grid = gen.generate(p, [0, winEnd + gap], 0, winEnd, new Gaussian(5000 + rep));
      const x = grid.eval(winEnd),
        y = grid.eval(winEnd + gap);
      sxy += x * y;
      sxx += x * x;
      syy += y * y;
    }
    const rho = sxy / Math.sqrt(sxx * syy);
    const th = Math.exp(-2 * Math.PI * p.loopBwHz * gap);
    expect(Math.abs(rho - th)).toBeLessThan(0.05);
    const varOu = Math.PI * Math.pow(10, p.l0DbcHz / 10) * p.loopBwHz;
    expect(Math.abs(sxx / R / varOu - 1)).toBeLessThan(0.08);
  });
});

describe('impairments in the engine', () => {
  it('automotive preset runs with every impairment on; off = identical to the unimpaired chain', () => {
    const p = PRESETS.automotive;
    const base = { ...defaultSettings(), ...p.settings, exactNoise: true };
    const e0 = new Engine(p.cfg, base, defaultScene('automotive'));
    const imp = { ...defaultImpairments(), phaseNoise: true, leakage: true, swerling: 1 as const, iq: true, ifFilter: true, adc: true, adcFullScaleDb: 30 };
    const e1 = new Engine(p.cfg, { ...base, imp }, defaultScene('automotive'));
    const out = allocOut(p.cfg, e0.d, base.angleFft, e0.spec.cols, e0.spec.nfft);
    const m0 = e0.runFrame(out);
    const m1 = e1.runFrame(out);
    expect(m1.adcClipped).toBeGreaterThanOrEqual(0);
    expect(m0.adcClipped).toBe(0);
    for (const v of e1.P) expect(Number.isFinite(v)).toBe(true);
    expect(m1.detections.length).toBeGreaterThan(0);
    // all off → bit-identical to the default settings object without `imp` changes
    const e2 = new Engine(p.cfg, { ...base, imp: defaultImpairments() }, defaultScene('automotive'));
    e2.runFrame(out);
    expect(Array.from(e2.P.slice(0, 2000))).toEqual(Array.from(e0.P.slice(0, 2000)));
  });

  it('IF high-pass suppresses the near-DC Tx→Rx leakage in the range profile', () => {
    const p = PRESETS.automotive;
    const base = { ...defaultSettings(), ...p.settings, noise: false, mti: false };
    const leak = { ...defaultImpairments(), leakage: true, leakPowerDb: 40 };
    const eA = new Engine(p.cfg, { ...base, imp: leak }, []);
    const eB = new Engine(p.cfg, { ...base, imp: { ...leak, ifFilter: true, hpfHz: 500e3, filterOrder: 2 } }, []);
    eA.runFrame();
    eB.runFrame();
    // leakage beat 2·0.1 m·S/c ≈ 20 kHz → range bin 0
    expect(10 * Math.log10(eA.rangeP[0] / eB.rangeP[0])).toBeGreaterThan(20);
  });

  it('phase noise raises the skirt around a strong target (range correlation: larger at long range)', () => {
    const p = PRESETS.automotive;
    const base = { ...defaultSettings(), ...p.settings, noise: false, mti: false };
    const skirt = (R: number, on: boolean): number => {
      const imp = { ...defaultImpairments(), phaseNoise: on, pn: { ...defaultImpairments().pn, l0DbcHz: -75 } };
      const e = new Engine(p.cfg, { ...base, imp }, []);
      e.points = [{ x: 0, y: R, z: 0, vx: 0, vy: 0, vz: 0, rcs: 100, phase0: 0 }];
      e.runFrame(undefined, false, false);
      const M = p.cfg.nChirps,
        rb = Math.round(R / e.d.rangeRes);
      // off-zero-Doppler power at the target's range bin relative to the peak
      let off = 0;
      for (let dd = 0; dd < M; dd++) if (Math.abs(dd - M / 2) > 4) off += e.P[rb * M + dd];
      return off / e.P[rb * M + M / 2];
    };
    const near = skirt(5, true),
      far = skirt(40, true),
      clean = skirt(40, false);
    expect(far).toBeGreaterThan(10 * clean);
    expect(far).toBeGreaterThan(near);
  });
});
