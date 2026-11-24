import { describe, it, expect } from 'vitest';
import { parabolic } from '../src/dsp/detect';
import { FFT, dft } from '../src/dsp/fft';
import { makeWindow } from '../src/dsp/window';
import { mulberry32 } from '../src/sim/rng';
import { PRESETS, derive } from '../src/config/radar';

describe('FFT', () => {
  for (const n of [1, 2, 4, 8, 16, 64, 128, 512, 1024]) {
    it(`matches naive DFT for n=${n}`, () => {
      const rng = mulberry32(n + 3);
      const re = new Float64Array(n).map(() => rng() * 2 - 1);
      const im = new Float64Array(n).map(() => rng() * 2 - 1);
      const ref = dft(re, im);
      const f = new FFT(n);
      const r2 = re.slice(),
        i2 = im.slice();
      f.forward(r2, i2);
      let err = 0;
      for (let k = 0; k < n; k++) err = Math.max(err, Math.abs(r2[k] - ref.re[k]), Math.abs(i2[k] - ref.im[k]));
      expect(err).toBeLessThan(1e-9 * Math.max(1, n));
      f.inverse(r2, i2);
      for (let k = 0; k < n; k++) {
        expect(r2[k]).toBeCloseTo(re[k], 10);
        expect(i2[k]).toBeCloseTo(im[k], 10);
      }
    });
  }
  it('forward sign: exp(+j2πk0 n/N) peaks at +k0', () => {
    const n = 64,
      k0 = 5;
    const re = new Float64Array(n),
      im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = Math.cos((2 * Math.PI * k0 * i) / n);
      im[i] = Math.sin((2 * Math.PI * k0 * i) / n);
    }
    new FFT(n).forward(re, im);
    expect(re[k0]).toBeCloseTo(n, 8);
    expect(Math.abs(re[n - k0])).toBeLessThan(1e-9);
  });
  it('works at an offset inside a larger buffer', () => {
    const re = new Float64Array(24),
      im = new Float64Array(24);
    re[8] = 1;
    new FFT(8).forward(re, im, 8);
    for (let k = 8; k < 16; k++) expect(re[k]).toBeCloseTo(1, 12);
    expect(re[16]).toBe(0);
  });
});

/** Peak sidelobe level [dB] of a window from a 32× zero-padded spectrum. */
function psl(w: Float64Array): number {
  const n = w.length,
    pad = 32 * n;
  const re = new Float64Array(pad),
    im = new Float64Array(pad);
  re.set(w);
  new FFT(pad).forward(re, im);
  const mag = (k: number) => Math.hypot(re[k], im[k]);
  const peak = mag(0);
  let k = 1;
  while (k < pad / 2 && mag(k + 1) < mag(k)) k++; // walk down the main lobe to the first null
  let side = 0;
  for (; k < pad / 2; k++) side = Math.max(side, mag(k));
  return 20 * Math.log10(side / peak);
}

describe('windows', () => {
  it('are periodic (DFT-even)', () => {
    const w = makeWindow('hann', 8);
    expect(w[0]).toBe(0);
    expect(w[4]).toBeCloseTo(1, 12);
    expect(w[1]).toBeCloseTo(w[7], 12);
  });
  const cases: [Parameters<typeof makeWindow>[0], number][] = [
    ['rect', -13.26],
    ['hann', -31.47],
    ['hamming', -42.7],
    ['blackman', -58.1],
  ];
  for (const [t, ref] of cases)
    it(`${t} peak sidelobe ≈ ${ref} dB`, () => {
      expect(Math.abs(psl(makeWindow(t, 256)) - ref)).toBeLessThan(1.0);
    });
});

describe('derived parameters', () => {
  it('automotive preset', () => {
    const d = derive(PRESETS.automotive.cfg);
    expect(d.nVirt).toBe(8);
    expect(d.rangeRes).toBeCloseTo(0.1952, 3);
    expect(d.rMax).toBeCloseTo(49.965, 2);
    expect(d.vMax).toBeCloseTo(12.17, 1);
    expect(d.dv).toBeCloseTo((2 * d.vMax) / 128, 9);
    expect(d.arraySpacing).toBeCloseTo(0.5, 12);
    expect(Array.from(d.angleIdx)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(d.te).toBeCloseTo(80e-6, 12);
  });
  it('drone preset', () => {
    const d = derive(PRESETS.drone.cfg);
    expect(d.bEff).toBeCloseTo(150e6, 0);
    expect(d.rangeRes).toBeCloseTo(0.9993, 3);
    expect(d.rMax).toBeCloseTo(511.6, 0);
    expect(d.nVirt).toBe(4);
  });
  it('micro-Doppler preset', () => {
    const d = derive(PRESETS.micro.cfg);
    expect(d.vMax).toBeCloseTo(156.1, 0);
    expect(d.adcTime).toBeCloseTo(16e-6, 12);
  });
  it('real IF halves Rmax and the range-bin count', () => {
    const c = { ...PRESETS.automotive.cfg, iqMode: 'real' as const };
    const d = derive(c);
    expect(d.rMax).toBeCloseTo(24.98, 1);
    expect(d.nRangeBins).toBe(128);
  });
});

describe('parabolic interpolation convention', () => {
  it('vertex offset at a maximum, clipped to ±0.5; 0 at a non-maximum (den ≥ 0)', () => {
    expect(parabolic(1, 2, 1)).toBeCloseTo(0, 15);
    expect(parabolic(1, 2, 1.5)).toBeCloseTo(0.5 * (1 - 1.5) / (1 - 4 + 1.5), 15);
    expect(parabolic(0, 1, 5)).toBe(0); // den = 3 > 0 (not a maximum); Python clips to ±0.5 here
    expect(parabolic(-10, 0, -0.01)).toBeLessThanOrEqual(0.5);
  });
});
