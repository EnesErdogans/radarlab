import { describe, it, expect } from 'vitest';
import { PRESETS, defaultSettings, type CfarParams, type CfarType } from '../src/config/radar';
import { Cfar2D, caAlpha, caPfaGamma, goPfa, osAlpha, osPfa, quickselect, soPfa } from '../src/dsp/cfar';
import { Engine } from '../src/engine/engine';
import { Gaussian, mulberry32 } from '../src/sim/rng';
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
    it(`${type}-CFAR Pfa within ±15 % of ${pfa}`, () => {
      const c = new Cfar2D(cfg.nSamples, cfg.nChirps, cf);
      let det = 0,
        cells = 0;
      for (const P of maps) {
        det += c.run(P, type, true).count;
        cells += P.length;
      }
      const emp = det / cells;
      expect(Math.abs(emp / pfa - 1)).toBeLessThan(0.15);
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

describe('OS-CFAR UI prefilter is safe', () => {
  it('gives the same detections as exact OS on the automotive scene', () => {
    const pr = PRESETS.automotive;
    const s = { ...defaultSettings(), ...pr.settings, clutter: true };
    const e = new Engine(pr.cfg, s, defaultScene('automotive'));
    e.synthesizeAndTransform();
    const c = new Cfar2D(e.d.nRangeBins, pr.cfg.nChirps, { ...s.cfar, type: 'OS' });
    const exact = c.run(e.P, 'OS', true).mask.slice();
    const fast = c.run(e.P, 'OS', false).mask;
    let diff = 0,
      n = 0;
    for (let i = 0; i < exact.length; i++) {
      diff += exact[i] !== fast[i] ? 1 : 0;
      n += exact[i];
    }
    expect(n).toBeGreaterThan(5);
    expect(diff).toBe(0);
  });
});
