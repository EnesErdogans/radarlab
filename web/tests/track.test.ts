import { describe, it, expect } from 'vitest';
import { Tracker, hungarian, type Measurement } from '../src/track/tracker';
import { Gaussian } from '../src/sim/rng';
import { PRESETS, defaultSettings } from '../src/config/radar';
import { Engine } from '../src/engine/engine';
import { defaultTarget } from '../src/sim/targets';
import { Spectrogram } from '../src/dsp/spectrogram';

describe('Hungarian assignment', () => {
  it('finds the optimal assignment', () => {
    const c = [
      [4, 1, 3],
      [2, 0, 5],
      [3, 2, 2],
    ];
    const a = hungarian(c, 3, 3);
    const cost = a.reduce((s, j, i) => s + c[i][j], 0);
    expect(cost).toBe(5); // 1 + 2 + 2
  });
});

describe('EKF tracker', () => {
  it('converges on a constant-velocity target from noisy (R, θ, vr)', () => {
    const tr = new Tracker({ sigmaR: 0.2, sigmaAz: (1 * Math.PI) / 180, sigmaV: 0.1 });
    const g = new Gaussian(9);
    const dt = 0.05;
    let x = 5,
      y = 30;
    const vx = 1.5,
      vy = -4;
    for (let k = 0; k < 60; k++) {
      const R = Math.hypot(x, y);
      const z: Measurement = {
        range: R + 0.2 * g.next(),
        az: Math.atan2(x, y) + ((1 * Math.PI) / 180) * g.next(),
        vr: (x * vx + y * vy) / R + 0.1 * g.next(),
        power: 1,
      };
      tr.step([z], dt);
      x += vx * dt;
      y += vy * dt;
    }
    const conf = tr.tracks.filter((t) => t.confirmed);
    expect(conf.length).toBe(1);
    const t = conf[0];
    // the state is the posterior for the last measurement time (before the final x += v·dt)
    expect(Math.hypot(t.s[0] - (x - vx * dt), t.s[1] - (y - vy * dt))).toBeLessThan(0.4);
    expect(Math.hypot(t.s[2] - vx, t.s[3] - vy)).toBeLessThan(0.4);
  });
  it('confirms after 3 hits and deletes tentative tracks after misses', () => {
    const tr = new Tracker();
    const z = { range: 10, az: 0, vr: 0, power: 1 };
    tr.step([z], 0.05);
    tr.step([z], 0.05);
    expect(tr.tracks[0].confirmed).toBe(false);
    tr.step([z], 0.05);
    expect(tr.tracks[0].confirmed).toBe(true);
    const clutter = { range: 30, az: 0.5, vr: 3, power: 1 };
    tr.step([z, clutter], 0.05);
    expect(tr.tracks.length).toBe(2);
    for (let i = 0; i < 3; i++) tr.step([z], 0.05);
    expect(tr.tracks.length).toBe(1);
    expect(tr.tracks[0].id).toBe(1);
  });
  it('full chain: a moving car yields one confirmed track near the truth', () => {
    const p = PRESETS.automotive;
    const car = { ...defaultTarget('car', 1), x: 3, y: 25, vx: -0.5, vy: -6 };
    const e = new Engine(p.cfg, { ...defaultSettings(), ...p.settings }, [car]);
    let m = e.runFrame();
    for (let i = 0; i < 25; i++) m = e.runFrame();
    const conf = m.tracks.filter((t) => t.confirmed);
    expect(conf.length).toBe(1);
    const truth = m.truth[0];
    expect(Math.hypot(conf[0].x - truth.x, conf[0].y - truth.y)).toBeLessThan(2.5);
    expect(Math.hypot(conf[0].vx - truth.vx, conf[0].vy - truth.vy)).toBeLessThan(1.5);
  });
});

describe('micro-Doppler', () => {
  function spread(rpm: number): number {
    const p = PRESETS.micro;
    const drone = { ...defaultTarget('drone', 1), x: 0, y: 20, z: 2, vx: 0, vy: -3, rcs: 0.5, rpm };
    const e = new Engine(p.cfg, { ...defaultSettings(), ...p.settings, snrRefDb: 25 }, [drone]);
    e.selectedId = 1;
    for (let i = 0; i < 6; i++) e.runFrame(undefined, false, true);
    return e.spec.spreadBins(25);
  }
  it('rotor blades spread the Doppler spectrum (6000 rpm vs 0 rpm)', () => {
    const s0 = spread(0),
      s1 = spread(6000);
    expect(s1).toBeGreaterThan(4 * s0);
  });
  it('blade-tip speed is reflected in the Doppler extent', () => {
    // tip speed 2π·100 Hz·0.12 m ≈ 75 m/s; the spectrogram extent should reach ≳ 50 % of it
    const p = PRESETS.micro;
    const drone = { ...defaultTarget('drone', 1), x: 0, y: 20, z: 0, vx: 0, vy: -3, rcs: 2, rpm: 6000 };
    const e = new Engine(p.cfg, { ...defaultSettings(), ...p.settings, snrRefDb: 30 }, [drone]);
    e.selectedId = 1;
    for (let i = 0; i < 6; i++) e.runFrame(undefined, false, true);
    const sp: Spectrogram = e.spec;
    const vPerBin = e.d.lambda / (2 * sp.nfft * e.d.te);
    const extent = (sp.spreadBins(35) * vPerBin) / 2;
    expect(extent).toBeGreaterThan(0.5 * 75);
    expect(extent).toBeLessThan(e.d.vMax);
  });
});
