// Cross-language golden vectors written by python/scripts/export_golden.py into ../golden/*.json.
// Skips gracefully if the folder is empty or missing.

import { describe, it, expect } from 'vitest';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultSettings, derive, type Derived, type RadarConfig, type SimSettings, type WindowType } from '../src/config/radar';
import { Engine } from '../src/engine/engine';
import { Synthesizer, type PointScatterer } from '../src/sim/synth';
import { makeWindow } from '../src/dsp/window';
import { caAlpha, caAlphaGamma, goAlpha, osAlpha, osRankIndex, soAlpha } from '../src/dsp/cfar';

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, '..', '..', 'golden');
const files = existsSync(dir)
  ? readdirSync(dir)
      .filter((f: string) => f.endsWith('.json') && f !== 'index.json')
      .sort()
  : [];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

export function toConfig(c: Json): RadarConfig {
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

function toSettings(g: Json): SimSettings {
  const s = defaultSettings();
  s.noise = false;
  s.mti = false;
  s.amplitudeMode = g.amplitudeMode ?? 'snr';
  if (g.snrRefDb !== undefined) s.snrRefDb = g.snrRefDb;
  if (g.refRange !== undefined) s.refRange = g.refRange;
  const c = g.config ?? {};
  s.physical = {
    ptDbm: c.txPowerDbm ?? s.physical.ptDbm,
    gtDbi: c.txGainDbi ?? s.physical.gtDbi,
    grDbi: c.rxGainDbi ?? s.physical.grDbi,
    nfDb: c.noiseFigureDb ?? s.physical.nfDb,
    lossDb: c.systemLossDb ?? s.physical.lossDb,
    t0: c.t0 ?? 290,
  };
  const proc = g.processing ?? {};
  s.rangeWindow = (proc.rangeWindow ?? 'hann') as WindowType;
  s.dopplerWindow = (proc.dopplerWindow ?? 'hann') as WindowType;
  if (proc.mti !== undefined) s.mti = !!proc.mti;
  if (proc.angleFftSize) s.angleFft = proc.angleFftSize;
  return s;
}

function toPoints(g: Json): PointScatterer[] {
  return (g.scatterers ?? []).map((p: Json) => ({
    x: p.x,
    y: p.y,
    z: p.z ?? 0,
    vx: p.vx ?? 0,
    vy: p.vy ?? 0,
    vz: p.vz ?? 0,
    rcs: p.rcs ?? 1,
    phase0: p.phase0 ?? 0,
  }));
}

function relClose(a: number, b: number, rel: number, ctx: string): void {
  const err = Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);
  if (err > rel) throw new Error(`${ctx}: ts=${a} py=${b} (rel err ${err.toExponential(2)} > ${rel})`);
}

function checkDerived(d: Derived, cfg: RadarConfig, ref: Json, rel: number, ctx: string): void {
  const map: [string, number][] = [
    ['wavelength', d.lambda],
    ['te', d.te],
    ['bandwidthEff', d.bEff],
    ['rangeResolution', d.rangeRes],
    ['maxRange', d.rMax],
    ['velocityResolution', d.dv],
    ['maxVelocity', d.vMax],
    ['nVirtual', d.nVirt],
    ['nRangeBins', d.nRangeBins],
    ['frameActiveTime', d.frameActive],
  ];
  for (const [k, v] of map) if (ref[k] !== undefined) relClose(v, ref[k], rel, `${ctx} derived.${k}`);
  void cfg;
}

describe('golden vectors (python reference)', () => {
  if (files.length === 0) {
    it.skip('no golden vectors in ../golden yet', () => {});
    return;
  }
  for (const f of files) {
    const g: Json = JSON.parse(readFileSync(join(dir, f), 'utf8'));
    const T = g.tolerances ?? {};
    describe(`${f}`, () => {
      // ---------- CFAR α table ----------
      if (Array.isArray(g.table)) {
        it('CFAR scale factors (CA, GO, SO, OS, CA-Gamma K=4)', () => {
          const rel = T.rel ?? 1e-9;
          for (const row of g.table) {
            const ctx = `n=${row.n} pfa=${row.pfa}`;
            if (row.ca !== undefined) relClose(caAlpha(row.n, row.pfa), row.ca, rel, ctx + ' CA');
            if (row.osK !== undefined) expect(osRankIndex(row.n, g.osRank ?? 0.75), ctx + ' osK').toBe(row.osK);
            if (row.os !== undefined) relClose(osAlpha(row.n, row.osK, row.pfa), row.os, rel, ctx + ' OS');
            if (row.go !== undefined) relClose(goAlpha(Math.floor(row.n / 2), row.pfa), row.go, rel, ctx + ' GO');
            if (row.so !== undefined) relClose(soAlpha(Math.floor(row.n / 2), row.pfa), row.so, rel, ctx + ' SO');
            if (row.caGammaK4 !== undefined) relClose(caAlphaGamma(row.n, 4, row.pfa), row.caGammaK4, rel, ctx + ' CA-Gamma4');
          }
        });
        return;
      }
      // ---------- presets: derived quantities + windows ----------
      if (Array.isArray(g.presets)) {
        for (const p of g.presets) {
          it(`preset ${p.name}: derived, virtual array, windows`, () => {
            const rel = T.rel ?? 1e-12;
            const cfg = toConfig(p.config);
            const d = derive(cfg);
            checkDerived(d, cfg, p.derived ?? {}, rel, p.name);
            if (p.virtualPos) expect(Array.from(d.virtPos)).toEqual(p.virtualPos.map((x: number) => expect.closeTo(x, 12)));
            if (p.virtualPosZ) expect(Array.from(d.virtPosZ)).toEqual(p.virtualPosZ.map((x: number) => expect.closeTo(x, 12)));
            for (const [key, n] of [
              ['rangeWindow', cfg.nSamples],
              ['dopplerWindow', cfg.nChirps],
            ] as const) {
              if (!p[key]) continue;
              const w = makeWindow('hann', n);
              const ref: number[] = p[key];
              expect(ref.length).toBe(n);
              for (let i = 0; i < n; i++) expect(Math.abs(w[i] - ref[i]), `${key}[${i}]`).toBeLessThanOrEqual(1e-12);
            }
          });
        }
        return;
      }
      if (!g.config) return;
      // brief rev. 1 flipped the spatial-phase sign; scenario vectors older than v2 are stale
      if ((g.conventionVersion ?? 1) < 2) {
        it.skip(`stale golden file (conventionVersion ${g.conventionVersion ?? 1} < 2) — regenerate with python/scripts/export_golden.py`, () => {});
        return;
      }

      // ---------- scenario vectors ----------
      const cfg = toConfig(g.config);
      const d = derive(cfg);
      const s = toSettings(g);
      const pts = toPoints(g);
      const cubeRel = T.cubeRelMax ?? 1e-9;

      if (g.derived) it('derived parameters', () => checkDerived(d, cfg, g.derived, 1e-12, f));

      const synth = (exact: boolean) => {
        const n = d.nVirt * cfg.nChirps * cfg.nSamples;
        const re = new Float64Array(n),
          im = new Float64Array(n);
        new Synthesizer(cfg, d).synthPoints(re, im, pts, s, exact);
        return { re, im };
      };

      if (g.cube?.re) {
        it('raw beat cube (exact and phasor-recurrence synthesis)', () => {
          const [K, M, N] = g.cube.shape;
          expect([K, M, N]).toEqual([d.nVirt, cfg.nChirps, cfg.nSamples]);
          const gr: number[] = g.cube.re.flat(Infinity),
            gi: number[] = g.cube.im.flat(Infinity);
          let mx = 0;
          for (let i = 0; i < gr.length; i++) mx = Math.max(mx, Math.hypot(gr[i], gi[i]));
          for (const exact of [true, false]) {
            const { re, im } = synth(exact);
            let err = 0,
              at = 0;
            for (let i = 0; i < gr.length; i++) {
              const e = Math.hypot(re[i] - gr[i], im[i] - gi[i]);
              if (e > err) {
                err = e;
                at = i;
              }
            }
            const lim = cubeRel * mx * (exact ? 1 : 100);
            if (err > lim) {
              const k = Math.floor(at / (M * N)),
                m = Math.floor((at % (M * N)) / N);
              throw new Error(`cube (${exact ? 'exact' : 'fast'}) max|Δ|=${err.toExponential(3)} > ${lim.toExponential(3)} at k=${k} m=${m} n=${at % N}: ts=(${re[at]}, ${im[at]}) py=(${gr[at]}, ${gi[at]})`);
            }
          }
        });
      }

      if (Array.isArray(g.cubeSamples)) {
        it('cube spot samples', () => {
          const { re, im } = synth(true);
          const M = cfg.nChirps,
            N = cfg.nSamples;
          let mx = 0;
          for (let i = 0; i < re.length; i++) mx = Math.max(mx, Math.hypot(re[i], im[i]));
          for (const smp of g.cubeSamples) {
            const i = (smp.k * M + smp.m) * N + smp.n;
            const e = Math.hypot(re[i] - smp.re, im[i] - smp.im);
            expect(e, `sample k=${smp.k} m=${smp.m} n=${smp.n}: ts=(${re[i]}, ${im[i]}) py=(${smp.re}, ${smp.im})`).toBeLessThanOrEqual(cubeRel * mx);
          }
        });
      }

      const engine = () => {
        const e = new Engine(cfg, s, []);
        e.points = pts;
        e.synthesizeAndTransform(true);
        return e;
      };

      if (g.rdPowerDb) {
        it('range-Doppler power map (dB)', () => {
          const e = engine();
          const M = cfg.nChirps;
          const tdb = T.rdPowerDbAbs ?? 1e-6;
          const db = (i: number) => 10 * Math.log10(e.P[i]);
          let bi = 0;
          for (let i = 1; i < e.P.length; i++) if (e.P[i] > e.P[bi]) bi = i;
          const rd = g.rdPowerDb;
          if (rd.maxDb !== undefined) expect(Math.abs(db(bi) - rd.maxDb), `maxDb ts=${db(bi)} py=${rd.maxDb}`).toBeLessThanOrEqual(tdb);
          if (rd.maxCell) expect([Math.floor(bi / M), bi % M]).toEqual(rd.maxCell);
          if (rd.data) {
            const vals: number[] = rd.data.flat(Infinity);
            expect(vals.length).toBe(e.P.length);
            let worst = 0,
              at = 0;
            const top = db(bi);
            for (let i = 0; i < vals.length; i++) {
              // cells > 90 dB below the peak are dominated by float64 round-off (relative amplitude < 3e-5)
              if (vals[i] < top - 90) continue;
              const err = Math.abs(db(i) - vals[i]);
              if (err > worst) {
                worst = err;
                at = i;
              }
            }
            expect(worst, `worst cell (${Math.floor(at / M)}, ${at % M}) ts=${db(at)} py=${vals[at]}`).toBeLessThanOrEqual(tdb);
          }
          if (rd.meanDb !== undefined) {
            // python: to_db(P.mean()) — dB of the mean power
            let acc = 0;
            for (let i = 0; i < e.P.length; i++) acc += e.P[i];
            const mdb = 10 * Math.log10(acc / e.P.length);
            expect(Math.abs(mdb - rd.meanDb), `meanDb ts=${mdb} py=${rd.meanDb}`).toBeLessThanOrEqual(tdb);
          }
        });
      }

      const peaks: Json[] = g.expected?.peaks ?? [];
      if (peaks.length) {
        it('expected peaks: bins, interpolation, range/velocity/angle, TDM compensation, truth', () => {
          const e = engine();
          const M = cfg.nChirps;
          const fracT = T.binFracAbs ?? 1e-6;
          for (const pk of peaks) {
            const ctx = `${f} peak (${pk.rangeBin}, ${pk.dopplerBin})`;
            const P = e.P,
              c = pk.rangeBin * M + pk.dopplerBin;
            // integer cell must be a local maximum in our map
            for (let dr = -1; dr <= 1; dr++)
              for (let dd = -1; dd <= 1; dd++) {
                const r = pk.rangeBin + dr;
                if (r < 0 || r >= d.nRangeBins || (!dr && !dd)) continue;
                expect(P[r * M + ((pk.dopplerBin + dd + M) % M)], `${ctx} not a local max`).toBeLessThanOrEqual(P[c]);
              }
            const m = e.det.measureCell(P, e.rdRe, e.rdIm, pk.rangeBin, pk.dopplerBin);
            const chk = (name: string, ts: number, py: number | undefined, t: number) => {
              if (py === undefined) return;
              expect(Math.abs(ts - py), `${ctx} ${name}: ts=${ts} py=${py}`).toBeLessThanOrEqual(t);
            };
            chk('powerDb', 10 * Math.log10(m.power), pk.powerDb, T.rdPowerDbAbs ?? 1e-6);
            chk('rangeBinFrac', m.rFrac, pk.rangeBinFrac, fracT);
            chk('dopplerBinFrac', m.dFrac, pk.dopplerBinFrac, fracT);
            chk('range', m.range, pk.range, T.rangeAbs ?? 1e-6);
            chk('velocity', m.velocity, pk.velocity, T.velocityAbs ?? 1e-6);
            if (d.nVirt > 1) {
              chk('angleBin', Math.round(m.angleBin), pk.angleBin, T.integerBins ?? 0);
              chk('angleBinFrac', m.angleBin, pk.angleBinFrac, fracT);
              chk('angleDeg', m.angleDeg, pk.angleDeg, T.angleDegAbs ?? 1e-4);
              // without TDM compensation
              const v = m.velocity;
              const saved = { ...m };
              const det0 = e.det.dets[0];
              e.det.dets[0] = m;
              m.velocity = 0;
              e.det.count = 1;
              e.det.estimateAngles(e.rdRe, e.rdIm);
              e.det.dets[0] = det0;
              chk('angleDegNoTdmComp', m.angleDeg, pk.angleDegNoTdmComp, T.angleDegAbs ?? 1e-4);
              chk('angleBinNoTdmComp', Math.round(m.angleBin), pk.angleBinNoTdmComp, T.integerBins ?? 0);
              Object.assign(m, saved, { velocity: v });
            }
            if (pk.truth) {
              chk('truth range', m.range, pk.truth.range, T.truthRangeAbs ?? d.rangeRes);
              chk('truth velocity (aliased)', m.velocity, pk.truth.velocityAliased ?? pk.truth.velocity, T.truthVelocityAbs ?? d.dv);
              if (d.nVirt > 1 && Math.abs(pk.truth.velocity - (pk.truth.velocityAliased ?? pk.truth.velocity)) < 1e-9)
                chk('truth angle', m.angleDeg, pk.truth.angleDeg, T.truthAngleDegAbs ?? 2);
            }
          }
        });
      }
    });
  }
});
