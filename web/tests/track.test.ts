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
  // The line-blade (sinc) rotor model (matching the Python reference) spreads each
  // blade's return across Doppler as flashes rather than as a single clean tip
  // tone, so these tests use a drone whose blades contribute visibly above the
  // body-return sidelobe floor (as a real multirotor's do).
  function spread(rpm: number, db: number): number {
    const p = PRESETS.micro;
    const drone = { ...defaultTarget('drone', 1), x: 0, y: 20, z: 2, vx: 0, vy: -3, rcs: 0.5, rpm, bodyRcs: 0.4, bladeRcs: 0.2 };
    const e = new Engine(p.cfg, { ...defaultSettings(), ...p.settings, snrRefDb: 25 }, [drone]);
    e.selectedId = 1;
    for (let i = 0; i < 6; i++) e.runFrame(undefined, false, true);
    return e.spec.spreadBins(db);
  }
  it('rotor blades spread the Doppler spectrum (6000 rpm vs 0 rpm)', () => {
    const s0 = spread(0, 30),
      s1 = spread(6000, 30);
    expect(s1).toBeGreaterThan(1.6 * s0);
  });
  it('blade-tip speed is reflected in the Doppler extent', () => {
    // tip speed 2π·f_rot·L (75 m/s at 6000 rpm, 0.12 m blades): the line-blade
    // micro-Doppler extent grows with rpm (tip speed) and stays below vmax.
    const p = PRESETS.micro;
    function extent(rpm: number): { ext: number; vMax: number } {
      const drone = { ...defaultTarget('drone', 1), x: 0, y: 20, z: 0, vx: 0, vy: -3, rcs: 2, rpm, bodyRcs: 0.5, bladeRcs: 0.15 };
      const e = new Engine(p.cfg, { ...defaultSettings(), ...p.settings, snrRefDb: 30 }, [drone]);
      e.selectedId = 1;
      for (let i = 0; i < 6; i++) e.runFrame(undefined, false, true);
      const sp: Spectrogram = e.spec;
      const vPerBin = e.d.lambda / (2 * sp.nfft * e.d.te);
      return { ext: (sp.spreadBins(40) * vPerBin) / 2, vMax: e.d.vMax };
    }
    const e3 = extent(3000),
      e6 = extent(6000),
      e9 = extent(9000);
    // a significant fraction of the 75 m/s tip speed, and faster tips spread wider
    expect(e6.ext).toBeGreaterThan(0.4 * 75);
    expect(e9.ext).toBeGreaterThan(e3.ext);
    expect(e6.ext).toBeLessThan(e6.vMax);
  });
});
