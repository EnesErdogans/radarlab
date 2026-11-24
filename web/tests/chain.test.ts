import { describe, it, expect } from 'vitest';
import { PRESETS, defaultSettings, derive, type RadarConfig, type SimSettings } from '../src/config/radar';
import { Engine } from '../src/engine/engine';
import { Synthesizer, type PointScatterer } from '../src/sim/synth';
import { Detector } from '../src/dsp/detect';
import { makeWindow } from '../src/dsp/window';

const auto = PRESETS.automotive.cfg;
const dA = derive(auto);

function settings(o: Partial<SimSettings> = {}): SimSettings {
  return { ...defaultSettings(), noise: false, mti: false, ...o };
}
function pt(R: number, azDeg: number, vr: number, rcs = 1): PointScatterer {
  const a = (azDeg * Math.PI) / 180;
  const ux = Math.sin(a),
    uy = Math.cos(a);
  return { x: R * ux, y: R * uy, z: 0, vx: vr * ux, vy: vr * uy, vz: 0, rcs, phase0: 0 };
}
function engineFor(cfg: RadarConfig, pts: PointScatterer[], o: Partial<SimSettings> = {}): Engine {
  const e = new Engine(cfg, settings(o), []);
  e.points = pts;
  return e;
}
function peak(e: Engine): { r: number; d: number } {
  const M = e.cfg.nChirps;
  let bi = 0;
  for (let i = 1; i < e.P.length; i++) if (e.P[i] > e.P[bi]) bi = i;
  return { r: Math.floor(bi / M), d: bi % M };
}

describe('synthesis', () => {
  it('phasor recurrence equals direct evaluation of the brief formula', () => {
    const d = derive(auto);
    const sy = new Synthesizer(auto, d);
    const n = d.nVirt * auto.nChirps * auto.nSamples;
    const a = [new Float64Array(n), new Float64Array(n)],
      b = [new Float64Array(n), new Float64Array(n)];
    const pts = [pt(37.3, 25, -9.1, 2), pt(3.1, -50, 4.4, 0.3)];
    pts[1].phase0 = 1.2;
    sy.synthPoints(a[0], a[1], pts, settings(), false);
    sy.synthPoints(b[0], b[1], pts, settings(), true);
    let err = 0,
      mx = 0;
    for (let i = 0; i < n; i++) {
      err = Math.max(err, Math.abs(a[0][i] - b[0][i]), Math.abs(a[1][i] - b[1][i]));
      mx = Math.max(mx, Math.abs(b[0][i]));
    }
    expect(err / mx).toBeLessThan(1e-9);
  });
  it('snr amplitude: unit RCS at refRange gives |x| = 10^(snrRefDb/20)', () => {
    const cfg = { ...auto, txPos: [0], rxPos: [0], nChirps: 2, nSamples: 8 };
    const sy = new Synthesizer(cfg, derive(cfg));
    const re = new Float64Array(16),
      im = new Float64Array(16);
    sy.synthPoints(re, im, [pt(10, 0, 0)], settings({ snrRefDb: 6, refRange: 10 }), true);
    expect(Math.hypot(re[3], im[3])).toBeCloseTo(Math.pow(10, 6 / 20), 9);
  });
  it('exact complex noise has unit variance', () => {
    const e = engineFor(auto, [], { noise: true, exactNoise: true });
    e.synthesizeAndTransform(true);
    // after range+Doppler FFT with Σw normalisation the noise power is K·Σw²/(Σw)² per FFT
    const sy = new Synthesizer(auto, dA);
    const n = 1 << 16;
    const re = new Float64Array(n),
      im = new Float64Array(n);
    sy.addNoise(re, im, settings({ noise: true, exactNoise: true }), 0);
    let p = 0,
      pr = 0;
    for (let i = 0; i < n; i++) {
      p += re[i] * re[i] + im[i] * im[i];
      pr += re[i] * re[i];
    }
    expect(p / n).toBeCloseTo(1, 1);
    expect(pr / n).toBeCloseTo(0.5, 1);
  });
});

describe('range / Doppler bins at analytic positions', () => {
  const M = auto.nChirps;
  it('static target at R = 40·ΔR → range bin 40, Doppler bin M/2', () => {
    const e = engineFor(auto, [pt(40 * dA.rangeRes, 0, 0)]);
    e.synthesizeAndTransform(true);
    expect(peak(e)).toEqual({ r: 40, d: M / 2 });
  });
  it('receding target (+5 Δv) → Doppler bin M/2+5', () => {
    const e = engineFor(auto, [pt(100 * dA.rangeRes, 10, 5 * dA.dv)]);
    e.synthesizeAndTransform(true);
    expect(peak(e)).toEqual({ r: 100, d: M / 2 + 5 });
  });
  it('approaching target (−7 Δv) → Doppler bin M/2−7', () => {
    const e = engineFor(auto, [pt(60 * dA.rangeRes, -10, -7 * dA.dv)]);
    e.synthesizeAndTransform(true);
    expect(peak(e)).toEqual({ r: 60, d: M / 2 - 7 });
  });
  it('velocity beyond vmax aliases: v = vmax + 3Δv → bin 3 (≙ −vmax + 3Δv)', () => {
    const v = dA.vMax + 3 * dA.dv;
    const R0 = 80 * dA.rangeRes;
    const e = engineFor(auto, [pt(R0, 0, v)]);
    e.synthesizeAndTransform(true);
    // analytic range bin: mid-frame range + Doppler contribution to the beat frequency
    const rMid = R0 + (v * dA.frameActive) / 2;
    const expR = Math.round(rMid / dA.rangeRes + ((2 * v) / dA.lambda) * (auto.nSamples / auto.fs));
    expect(expR).toBe(81);
    expect(peak(e)).toEqual({ r: expR, d: 3 });
  });
  it('interpolated range / velocity of an off-grid target', () => {
    const e = engineFor(auto, [pt(23.37, 5, -4.21)], { noise: true });
    const m = e.runFrame();
    const best = m.detections.reduce((a, b) => (b.power > a.power ? b : a));
    expect(Math.abs(best.range - 23.37)).toBeLessThan(dA.rangeRes * 0.25);
    expect(Math.abs(best.velocity + 4.21)).toBeLessThan(dA.dv * 0.25);
  });
  it('drone preset: target at 200 m', () => {
    const cfg = PRESETS.drone.cfg;
    const dd = derive(cfg);
    const e = engineFor(cfg, [pt(200 * dd.rangeRes, 0, -3 * dd.dv)], { snrRefDb: 20, refRange: 100 });
    e.synthesizeAndTransform(true);
    expect(peak(e)).toEqual({ r: 200, d: cfg.nChirps / 2 - 3 });
  });
});

describe('angle estimation (TDM-MIMO)', () => {
  for (const [az, v] of [
    [20, 7],
    [-35, -9],
    [0, 11],
    [48, 0],
  ] as const) {
    it(`θ = ${az}°, v = ${v} m/s`, () => {
      const e = engineFor(auto, [pt(20, az, v, 2)], { noise: true });
      const m = e.runFrame();
      const best = m.detections.reduce((a, b) => (b.power > a.power ? b : a));
      expect(Math.abs(best.angleDeg - az)).toBeLessThan(1.0);
      // rev. 1: −j steering + +j angle kernel → positive angle ↔ angle bin above A/2
      if (az !== 0) expect(Math.sign(best.angleBin - 32)).toBe(Math.sign(az));
      if (Math.abs(v) > 5) {
        // without TDM Doppler compensation the estimate is biased
        e.det.dets[0].velocity = 0;
        const save = e.det.count;
        // move the strongest detection to slot 0
        const idx = e.det.dets.slice(0, save).reduce((bi, x, i, arr) => (x.power > arr[bi].power ? i : bi), 0);
        const tmp = e.det.dets[0];
        e.det.dets[0] = e.det.dets[idx];
        e.det.dets[idx] = tmp;
        e.det.dets[0].velocity = 0;
        e.det.count = 1;
        e.det.estimateAngles(e.rdRe, e.rdIm);
        expect(Math.abs(e.det.dets[0].angleDeg - az)).toBeGreaterThan(3);
      }
    });
  }
});

describe('range–angle map orientation', () => {
  it('peak lies at the target azimuth (θ > 0 → bin > A/2)', () => {
    const e = engineFor(auto, [pt(20, 30, 5, 2)]);
    e.runFrame();
    const A = e.A;
    const r = Math.round(20 / dA.rangeRes);
    let best = 0;
    for (let a = 1; a < A; a++) if (e.raP[r * A + a] > e.raP[r * A + best]) best = a;
    const sinT = (best - A / 2) / (A * dA.arraySpacing);
    expect(Math.abs((Math.asin(sinT) * 180) / Math.PI - 30)).toBeLessThan(2.5);
  });
});

describe('range–angle map definition (lag-domain and direct paths)', () => {
  // brute force: out[r, shift(a)] = Σ_d |Σ_k w_slot·x̃_k e^{+j2π·slot·a/A}|² / (Σw)²
  function brute(e: Engine, A: number, w: Float64Array, r: number): Float64Array {
    const d = e.d,
      M = e.cfg.nChirps,
      nR = d.nRangeBins;
    const out = new Float64Array(A);
    let sw = 0;
    for (const x of w) sw += x;
    for (let dd = 0; dd < M; dd++) {
      const v = (dd - M / 2) * d.dv;
      for (let a = 0; a < A; a++) {
        let sr = 0,
          si = 0;
        for (let k = 0; k < d.nVirt; k++) {
          const sl = d.angleIdx[k];
          if (sl < 0) continue;
          const t = Math.floor(k / d.nRx);
          const j = (k * nR + r) * M + dd;
          const ph = -2 * Math.PI * ((2 * v) / d.lambda) * t * e.cfg.chirpPeriod + (2 * Math.PI * sl * a) / A;
          const c = Math.cos(ph),
            s = Math.sin(ph);
          sr += w[sl] * (e.rdRe[j] * c - e.rdIm[j] * s);
          si += w[sl] * (e.rdRe[j] * s + e.rdIm[j] * c);
        }
        out[(a + A / 2) % A] += (sr * sr + si * si) / (sw * sw);
      }
    }
    return out;
  }
  const e = engineFor(auto, [pt(20, 30, 5, 2), pt(20.3, -12, -3, 1)]);
  e.synthesizeAndTransform(true);
  const r = Math.round(20 / dA.rangeRes);
  for (const [A, win] of [
    [64, 'rect'],
    [64, 'hann'],
    [8, 'hann'], // 2L−1 > A → direct path
  ] as const) {
    it(`A=${A}, ${win} window`, () => {
      const L = dA.arrayLen;
      const w = makeWindow(win, L);
      const det = new Detector(auto, dA, A);
      const out = new Float64Array(dA.nRangeBins * A);
      det.rangeAngle(e.rdRe, e.rdIm, out, w);
      const ref = brute(e, A, w, r);
      let mx = 0;
      for (const x of ref) mx = Math.max(mx, x);
      for (let a = 0; a < A; a++) expect(Math.abs(out[r * A + a] - ref[a]) / mx, `bin ${a}`).toBeLessThan(1e-12);
    });
  }
  it('rejects a ULA longer than the angle FFT', () => {
    const det = new Detector(auto, dA, 4);
    expect(() => det.rangeAngle(e.rdRe, e.rdIm, new Float64Array(dA.nRangeBins * 4))).toThrow();
  });
});
