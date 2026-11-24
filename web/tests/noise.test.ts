// UI noise generator (NoisePool): successive frames must be statistically independent
// (review round 1, m7: the old fixed pool made drone frames cyclic shifts of each other).

import { describe, it, expect } from 'vitest';
import { PRESETS, defaultSettings, derive } from '../src/config/radar';
import { Synthesizer } from '../src/sim/synth';
import { getFFT } from '../src/dsp/fft';

const cfg = PRESETS.drone.cfg;
const d = derive(cfg);
const L = d.nVirt * cfg.nChirps * cfg.nSamples; // drone: the frame that used to consume the whole old pool
const s = { ...defaultSettings(), noise: true, exactNoise: false };
const synth = new Synthesizer(cfg, d);
const frames: { re: Float64Array; im: Float64Array }[] = [];
for (let f = 0; f < 4; f++) {
  const re = new Float64Array(L),
    im = new Float64Array(L);
  synth.addNoise(re, im, s, f);
  frames.push({ re, im });
}

/** max over all cyclic lags of |Σ a[i]·conj(b[i−lag])| / sqrt(Σ|a|²·Σ|b|²) (via FFT). */
function maxXcorr(a: { re: Float64Array; im: Float64Array }, b: { re: Float64Array; im: Float64Array }): { max: number; at0: number } {
  let n = 1;
  while (n < L) n <<= 1;
  const ar = new Float64Array(n),
    ai = new Float64Array(n),
    br = new Float64Array(n),
    bi = new Float64Array(n);
  ar.set(a.re);
  ai.set(a.im);
  br.set(b.re);
  bi.set(b.im);
  const fft = getFFT(n);
  fft.forward(ar, ai);
  fft.forward(br, bi);
  for (let k = 0; k < n; k++) {
    const r = ar[k] * br[k] + ai[k] * bi[k],
      i = ai[k] * br[k] - ar[k] * bi[k];
    ar[k] = r;
    ai[k] = i;
  }
  fft.inverse(ar, ai);
  let ea = 0,
    eb = 0;
  for (let i = 0; i < L; i++) {
    ea += a.re[i] ** 2 + a.im[i] ** 2;
    eb += b.re[i] ** 2 + b.im[i] ** 2;
  }
  // cross-check the FFT scaling against the lag-0 direct sum
  let c0r = 0,
    c0i = 0;
  for (let i = 0; i < L; i++) {
    c0r += a.re[i] * b.re[i] + a.im[i] * b.im[i];
    c0i += a.im[i] * b.re[i] - a.re[i] * b.im[i];
  }
  const scale = Math.hypot(c0r, c0i) / Math.max(Math.hypot(ar[0], ai[0]), 1e-300);
  let mx = 0;
  for (let k = 0; k < n; k++) mx = Math.max(mx, Math.hypot(ar[k], ai[k]) * scale);
  const nrm = Math.sqrt(ea * eb);
  return { max: mx / nrm, at0: Math.hypot(c0r, c0i) / nrm };
}

describe('UI noise pool (NoisePool)', () => {
  it('has unit complex variance, zero mean and circular statistics', () => {
    for (const fr of frames) {
      let mr = 0,
        mi = 0,
        pr = 0,
        pi = 0,
        cross = 0;
      for (let i = 0; i < L; i++) {
        mr += fr.re[i];
        mi += fr.im[i];
        pr += fr.re[i] ** 2;
        pi += fr.im[i] ** 2;
        cross += fr.re[i] * fr.im[i];
      }
      const tol = 6 / Math.sqrt(L);
      expect(Math.abs(mr / L)).toBeLessThan(tol);
      expect(Math.abs(mi / L)).toBeLessThan(tol);
      expect(Math.abs(pr / L - 0.5)).toBeLessThan(tol);
      expect(Math.abs(pi / L - 0.5)).toBeLessThan(tol);
      expect(Math.abs(cross / L)).toBeLessThan(tol);
    }
  });

  it('successive frames are uncorrelated at every cyclic lag (|ρ| ≈ 1/√L)', () => {
    const lim0 = 5 / Math.sqrt(L),
      limMax = 6 / Math.sqrt(L);
    for (let f = 0; f + 1 < frames.length; f++) {
      const { max, at0 } = maxXcorr(frames[f], frames[f + 1]);
      expect(at0, `frames ${f},${f + 1} lag 0`).toBeLessThan(lim0);
      expect(max, `frames ${f},${f + 1} max over lags`).toBeLessThan(limMax);
    }
  });

  it('the cross-correlation check detects a shifted copy (test sanity)', () => {
    const a = frames[0];
    const sh = 12345;
    const b = { re: new Float64Array(L), im: new Float64Array(L) };
    for (let i = 0; i < L; i++) {
      b.re[i] = a.re[(i + sh) % L];
      b.im[i] = a.im[(i + sh) % L];
    }
    expect(maxXcorr(a, b).max).toBeGreaterThan(0.9);
  });

  it('noise powers |n|² of successive frames are uncorrelated', () => {
    for (let f = 0; f + 1 < frames.length; f++) {
      const p1 = frames[f],
        p2 = frames[f + 1];
      let s1 = 0,
        s2 = 0,
        s11 = 0,
        s22 = 0,
        s12 = 0;
      for (let i = 0; i < L; i++) {
        const a = p1.re[i] ** 2 + p1.im[i] ** 2,
          b = p2.re[i] ** 2 + p2.im[i] ** 2;
        s1 += a;
        s2 += b;
        s11 += a * a;
        s22 += b * b;
        s12 += a * b;
      }
      const cov = s12 / L - (s1 / L) * (s2 / L);
      const rho = cov / Math.sqrt((s11 / L - (s1 / L) ** 2) * (s22 / L - (s2 / L) ** 2));
      expect(Math.abs(rho)).toBeLessThan(5 / Math.sqrt(L));
    }
  });
});
