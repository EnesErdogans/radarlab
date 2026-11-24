import { describe, it, expect } from 'vitest';
import { PRESETS, defaultSettings, type CfarParams, type CfarType } from '../src/config/radar';
import { Cfar2D, caAlpha, caAlphaGamma, caPfaGamma, goAlpha, goPfa, gsAlphaGamma, gsPfaGamma, osAlpha, osAlphaGamma, osPfa, osPfaGamma, quickselect, soAlpha, soPfa } from '../src/dsp/cfar';
import { Engine } from '../src/engine/engine';
import { Gaussian, mulberry32 } from '../src/sim/rng';
import { gsLogPfaGamma, osLogPfaGamma } from '../src/dsp/gamma';
import { defaultScene } from '../src/sim/scenes';

describe('CFAR scale factors', () => {
  it('CA α closed form gives the design Pfa', () => {
    const n = 248,
      a = caAlpha(n, 1e-5);
    expect(Math.pow(1 + a / n, -n)).toBeCloseTo(1e-5, 12);
    // Gamma(K=1) reduces to the exponential formula
    expect(caPfaGamma(n, 1, a)).toBeCloseTo(1e-5, 12);
  });
  it('OS α solves Π (N−i)/(N−i+α) = Pfa', () => {
    const a = osAlpha(100, 75, 1e-4);
    expect(osPfa(100, 75, a)).toBeCloseTo(1e-4, 9);
  });
  it('GO + SO = 2·CA(n) identity', () => {
    const n = 16,
      a = 5.3;
    expect(goPfa(n, a) + soPfa(n, a)).toBeCloseTo(2 * Math.pow(1 + a / n, -n), 12);
  });
  it('Gamma(K) quadrature reduces to the K=1 closed forms', () => {
    // the K>1 code path evaluated at K=1 (bypassing the closed-form shortcut)
    expect(Math.exp(osLogPfaGamma(32, 24, 1, 9.5)) / osPfa(32, 24, 9.5)).toBeCloseTo(1, 11);
    expect(Math.exp(gsLogPfaGamma(16, 1, 0.6, 'GO')) / goPfa(16, 9.6)).toBeCloseTo(1, 11);
    expect(Math.exp(gsLogPfaGamma(16, 1, 0.6, 'SO')) / soPfa(16, 9.6)).toBeCloseTo(1, 11);
  });
  it('Gamma(K): GO + SO = 2·CA(n/2) identity and solved α hits the design Pfa', () => {
    for (const K of [2, 4, 8]) {
      const n = 12,
        a = 4.2;
      // P(X > α·max) + P(X > α·min) = P(X > α·S₁/n) + P(X > α·S₂/n) = 2·Pfa_CA(n)
      expect((gsPfaGamma(n, K, a, 'GO') + gsPfaGamma(n, K, a, 'SO')) / (2 * caPfaGamma(n, K, a))).toBeCloseTo(1, 10);
      for (const pfa of [1e-3, 1e-6]) {
        expect(gsPfaGamma(n, K, gsAlphaGamma(n, K, pfa, 'GO'), 'GO') / pfa).toBeCloseTo(1, 9);
        expect(gsPfaGamma(n, K, gsAlphaGamma(n, K, pfa, 'SO'), 'SO') / pfa).toBeCloseTo(1, 9);
        expect(osPfaGamma(40, 30, K, osAlphaGamma(40, 30, K, pfa)) / pfa).toBeCloseTo(1, 9);
      }
      // K>1 α are below the misused K=1 formulas (noise of a K-sum is less spread)
      expect(osAlphaGamma(40, 30, K, 1e-5)).toBeLessThan(osAlpha(40, 30, 1e-5));
      expect(gsAlphaGamma(20, K, 1e-5, 'GO')).toBeLessThan(goAlpha(20, 1e-5));
      expect(gsAlphaGamma(20, K, 1e-5, 'SO')).toBeLessThan(soAlpha(20, 1e-5));
      expect(caAlphaGamma(40, K, 1e-5)).toBeLessThan(caAlpha(40, 1e-5));
    }
  });
  it('quickselect returns the k-th smallest', () => {
    const rng = mulberry32(5);
    const v = new Float64Array(301).map(() => rng());
    const sorted = Array.from(v).sort((a, b) => a - b);
    for (const k of [0, 1, 150, 225, 300]) expect(quickselect(v.slice(), 301, k)).toBe(sorted[k]);
  });
});

describe('CFAR empirical Pfa (K=1, exact Gaussian noise, full chain)', () => {
  // 1TX×1RX, rectangular windows → RD cells are i.i.d. exponential (DFT of white noise)
  const cfg = { ...PRESETS.automotive.cfg, txPos: [0], rxPos: [0], nSamples: 128, nChirps: 64 };
  const pfa = 2e-3;
  const cf: CfarParams = { type: 'CA', pfa, guardR: 2, guardD: 1, trainR: 6, trainD: 3, osRank: 0.75 };
  const s = { ...defaultSettings(), noise: true, exactNoise: true, mti: false, rangeWindow: 'rect' as const, dopplerWindow: 'rect' as const, cfar: cf };
  const frames = 60;
  const maps: Float64Array[] = [];
  const e = new Engine(cfg, s, []);
  for (let f = 0; f < frames; f++) {
    e.frame = f;
    e.synthesizeAndTransform(true);
    maps.push(e.P.slice());
  }
  for (const type of ['CA', 'GO', 'SO', 'OS'] as CfarType[]) {
    it(`${type}-CFAR Pfa within ±10 % of ${pfa}`, () => {
      const c = new Cfar2D(cfg.nSamples, cfg.nChirps, cf);
      let det = 0,
        cells = 0;
      for (const P of maps) {
        det += c.run(P, type, true).count;
        cells += P.length;
      }
      const emp = det / cells;
      expect(Math.abs(emp / pfa - 1)).toBeLessThan(0.1);
    });
  }
});

describe('CFAR empirical Pfa (K=4 non-coherent channels, exact Gamma(4) α, full chain)', () => {
  // 1TX×4RX: P = Σ of 4 independent exponential maps → i.i.d. Gamma(4) cells (rect windows)
  const cfg = { ...PRESETS.automotive.cfg, txPos: [0], rxPos: [0, 0.5, 1, 1.5], nSamples: 128, nChirps: 64 };
  const pfa = 2e-3;
  const cf: CfarParams = { type: 'CA', pfa, guardR: 2, guardD: 1, trainR: 6, trainD: 3, osRank: 0.75 };
  const s = { ...defaultSettings(), noise: true, exactNoise: true, mti: false, rangeWindow: 'rect' as const, dopplerWindow: 'rect' as const, cfar: cf };
  const frames = 80;
  const maps: Float64Array[] = [];
  const e = new Engine(cfg, s, []);
  for (let f = 0; f < frames; f++) {
    e.frame = f;
    e.synthesizeAndTransform(true);
    maps.push(e.P.slice());
  }
  // expected count ≈ 1300 → σ ≈ 2.8 %
  for (const type of ['CA', 'GO', 'SO', 'OS'] as CfarType[]) {
    it(`${type}-CFAR Pfa within ±10 % of ${pfa}`, () => {
      expect(e.d.nVirt).toBe(4);
      const c = new Cfar2D(cfg.nSamples, cfg.nChirps, cf, 4);
      let det = 0,
        cells = 0;
      for (const P of maps) {
        det += c.run(P, type, true).count;
        cells += P.length;
      }
      expect(Math.abs(det / cells / pfa - 1)).toBeLessThan(0.1);
    });
  }
});

describe('CFAR masking: OS vs CA', () => {
  // exponential noise + a strong target next to a weak one inside the training window
  const nR = 128,
    M = 64;
  const g = new Gaussian(42);
  const P = new Float64Array(nR * M);
  for (let i = 0; i < P.length; i++) {
    const a = g.next(),
      b = g.next();
    P[i] = 0.5 * (a * a + b * b);
  }
  P[60 * M + 20] = 1e4; // 40 dB
  P[65 * M + 20] = 200; // 23 dB, 5 range bins away (inside CA training)
  const p: CfarParams = { type: 'CA', pfa: 1e-5, guardR: 2, guardD: 2, trainR: 8, trainD: 4, osRank: 0.75 };
  it('CA misses the weak target, OS detects both', () => {
    const c = new Cfar2D(nR, M, p);
    const ca = c.run(P, 'CA').mask.slice();
    const os = c.run(P, 'OS', true).mask.slice();
    expect(ca[60 * M + 20]).toBe(1);
    expect(ca[65 * M + 20]).toBe(0);
    expect(os[60 * M + 20]).toBe(1);
    expect(os[65 * M + 20]).toBe(1);
  });
  it('SO resolves the pair, GO masks it', () => {
    const c = new Cfar2D(nR, M, p);
    expect(c.run(P, 'SO').mask[65 * M + 20]).toBe(1);
    expect(c.run(P, 'GO').mask[65 * M + 20]).toBe(0);
  });
});

describe('OS-CFAR UI block prefilter is exact', () => {
  const same = (c: Cfar2D, P: Float64Array) => {
    const ex = c.run(P, 'OS', true);
    const exMask = ex.mask.slice(),
      exThr = ex.thr.slice();
    const fast = c.run(P, 'OS', false);
    let diff = 0,
      n = 0,
      thrDiff = 0;
    for (let i = 0; i < exMask.length; i++) {
      diff += exMask[i] !== fast.mask[i] ? 1 : 0;
      n += exMask[i];
      if (!isNaN(fast.thr[i]) && fast.thr[i] !== exThr[i]) thrDiff++;
    }
    return { diff, n, thrDiff };
  };
  for (const id of ['automotive', 'drone', 'micro'] as const)
    for (const mti of [true, false])
      it(`${id} (MTI ${mti ? 'on' : 'off'}, clutter): same detections and thresholds as exact OS`, () => {
        const pr = PRESETS[id];
        const s = { ...defaultSettings(), ...pr.settings, clutter: true, mti };
        const e = new Engine(pr.cfg, s, defaultScene(id));
        e.synthesizeAndTransform();
        const c = new Cfar2D(e.d.nRangeBins, pr.cfg.nChirps, { ...s.cfar, type: 'OS' }, e.d.nVirt);
        const r = same(c, e.P);
        expect(r.n).toBeGreaterThan(0);
        expect(r.diff).toBe(0);
        expect(r.thrDiff).toBe(0);
      });
  it('weak target in a deep local noise notch (where a global-quantile prefilter fails)', () => {
    const nR = 128,
      M = 64;
    const g = new Gaussian(9);
    const P = new Float64Array(nR * M);
    for (let i = 0; i < P.length; i++) {
      const a = g.next(),
        b = g.next();
      // columns 28..36: 40 dB below the rest (like an MTI notch around zero Doppler)
      const d = i % M;
      P[i] = 0.5 * (a * a + b * b) * (d >= 28 && d <= 36 ? 1e-4 : 1);
    }
    P[70 * M + 32] = 0.05; // 27 dB above the notch floor, 13 dB below the global floor
    const c = new Cfar2D(nR, M, { type: 'OS', pfa: 1e-5, guardR: 2, guardD: 1, trainR: 8, trainD: 3, osRank: 0.75 });
    const r = same(c, P);
    expect(c.run(P, 'OS', false).mask[70 * M + 32]).toBe(1);
    expect(r.diff).toBe(0);
  });
});
