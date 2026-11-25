// Cross-language golden vectors written by python/scripts/export_golden.py into ../golden/*.json.
// Skips gracefully if the folder is empty or missing.

import { describe, it, expect } from 'vitest';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultSettings, derive, type Derived, type RadarConfig, type SimSettings, type WindowType } from '../src/config/radar';
import { Engine } from '../src/engine/engine';
import { Synthesizer, type PointScatterer } from '../src/sim/synth';
import type { Target } from '../src/sim/targets';
import { makeWindow } from '../src/dsp/window';
import { Cfar2D, caAlpha, caAlphaGamma, goAlpha, gsAlphaGamma, osAlpha, osAlphaGamma, osRankIndex, soAlpha } from '../src/dsp/cfar';
import { Detector } from '../src/dsp/detect';
import { amplitudeFor } from '../src/sim/synth';
import { ReferenceTracker } from '../src/track/reference';
import type { CfarType } from '../src/config/radar';

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
      // impairments golden is consumed by tests/impairments.test.ts
      if (g.kind === 'impairments') {
        it('consumed by tests/impairments.test.ts', () => {});
        return;
      }
      // ---------- CFAR α table ----------
      if (Array.isArray(g.table)) {
        it('CFAR scale factors (CA, GO, SO, OS; K=1 closed forms and exact Gamma(K) for K>1)', () => {
          const rel = T.rel ?? 1e-9;
          let worst = 0;
          const cmp = (ts: number, py: number, ctx: string) => {
            relClose(ts, py, rel, ctx);
            worst = Math.max(worst, Math.abs(ts - py) / Math.abs(py));
          };
          for (const row of g.table) {
            const K = row.K ?? 1;
            const ctx = `K=${K} n=${row.n} pfa=${row.pfa}`;
            const half = Math.floor(row.n / 2);
            if (row.osK !== undefined) expect(osRankIndex(row.n, g.osRank ?? 0.75), ctx + ' osK').toBe(row.osK);
            if (K === 1) {
              if (row.ca !== undefined) cmp(caAlpha(row.n, row.pfa), row.ca, ctx + ' CA');
              if (row.os !== undefined) cmp(osAlpha(row.n, row.osK, row.pfa), row.os, ctx + ' OS');
              if (row.go !== undefined) cmp(goAlpha(half, row.pfa), row.go, ctx + ' GO');
              if (row.so !== undefined) cmp(soAlpha(half, row.pfa), row.so, ctx + ' SO');
            } else {
              if (row.ca !== undefined) cmp(caAlphaGamma(row.n, K, row.pfa), row.ca, ctx + ' CA-Gamma');
              if (row.os !== undefined) cmp(osAlphaGamma(row.n, row.osK, K, row.pfa), row.os, ctx + ' OS-Gamma');
              if (row.go !== undefined) cmp(gsAlphaGamma(half, K, row.pfa, 'GO'), row.go, ctx + ' GO-Gamma');
              if (row.so !== undefined) cmp(gsAlphaGamma(half, K, row.pfa, 'SO'), row.so, ctx + ' SO-Gamma');
            }
            if (row.caGammaK4 !== undefined) cmp(caAlphaGamma(row.n, 4, row.pfa), row.caGammaK4, ctx + ' CA-Gamma4');
          }
          console.info(`cfar_alpha: ${g.table.length} rows, worst rel diff ts vs py ${worst.toExponential(2)}`);
        });
        return;
      }
      // ---------- CFAR on stored power maps (goldenSet 2) ----------
      if (Array.isArray(g.maps) && g.shape) {
        const [nR, M] = g.shape;
        for (const mp of g.maps) {
          it(`K=${mp.K}: CA/GO/SO/OS thresholds, α, masks (incl. range edges, GO/SO fallback, Doppler wrap)`, () => {
            const P = Float64Array.from(mp.P);
            const relT = T.thresholdRel ?? 1e-9,
              relA = T.alphaRel ?? 1e-9;
            for (const type of ['CA', 'GO', 'SO', 'OS'] as CfarType[]) {
              const ref = mp[type.toLowerCase()];
              if (!ref) continue;
              const c = new Cfar2D(nR, M, { type, pfa: g.pfa, guardR: g.guard[0], guardD: g.guard[1], trainR: g.train[0], trainD: g.train[1], osRank: g.osRank ?? 0.75 }, mp.K);
              const out = c.run(P, type, true);
              let worstT = 0,
                worstA = 0;
              for (let i = 0; i < nR * M; i++) {
                const ctx = `K=${mp.K} ${type} cell (${Math.floor(i / M)}, ${i % M})`;
                relClose(out.thr[i], ref.threshold[i], relT, ctx + ' threshold');
                relClose(out.thr[i] / out.noise[i], ref.alpha[i], relA, ctx + ' alpha');
                worstT = Math.max(worstT, Math.abs(out.thr[i] / ref.threshold[i] - 1));
                worstA = Math.max(worstA, Math.abs(out.thr[i] / out.noise[i] / ref.alpha[i] - 1));
              }
              const det: [number, number][] = [];
              for (let i = 0; i < nR * M; i++) if (out.mask[i]) det.push([Math.floor(i / M), i % M]);
              const key = (a: number[][]) => a.map((x) => x.join(',')).sort();
              expect(key(det), `K=${mp.K} ${type} detections`).toEqual(key(ref.detections));
              if (ref.caFallbackCells) {
                // fallback cells use the CA statistic with the Gamma(K) CA α for nTrain cells
                const ca = new Cfar2D(nR, M, { type: 'CA', pfa: g.pfa, guardR: g.guard[0], guardD: g.guard[1], trainR: g.train[0], trainD: g.train[1], osRank: 0.75 }, mp.K).run(P, 'CA', true);
                for (const [r, d] of ref.caFallbackCells) relClose(out.thr[r * M + d], ca.thr[r * M + d], 1e-12, `${type} fallback (${r}, ${d})`);
              }
              console.info(`cfar_maps K=${mp.K} ${type}: worst rel diff threshold ${worstT.toExponential(1)}, α ${worstA.toExponential(1)}`);
            }
          });
        }
        return;
      }
      // ---------- clustering (goldenSet 2) ----------
      if (Array.isArray(g.clusters) && Array.isArray(g.mask)) {
        it('8-connected clustering with Doppler wrap: peaks, power, size, members', () => {
          const [nR, M] = g.shape;
          const cfg: RadarConfig = { fc: 77e9, slope: 30e12, fs: 5e6, nSamples: nR, adcStartTime: 0, chirpPeriod: 1e-5, nChirps: M, txPos: [0], rxPos: [0], framePeriod: 0.01, iqMode: 'complex' };
          const dd = derive(cfg);
          expect(dd.nRangeBins).toBe(nR);
          const det = new Detector(cfg, dd, 64, 64);
          const P = Float64Array.from(g.P);
          const mask = new Uint8Array(nR * M);
          for (const [r, d] of g.mask) mask[r * M + d] = 1;
          const n = det.cluster(P, mask, new Float64Array(nR * M).fill(1));
          expect(n).toBe(g.clusters.length);
          const ours = det.dets.slice(0, n).map((x, ci) => {
            const cells: number[][] = [];
            for (let i = 0; i < nR * M; i++) if (det.labels[i] === ci) cells.push([Math.floor(i / M), i % M]);
            return { rangeBin: x.rBin, dopplerBin: x.dBin, power: x.power, size: x.cells, cells };
          });
          // Python sorts by peak power, the web engine lists clusters in scan order → compare as sets
          ours.sort((a, b) => b.power - a.power);
          const rel = T.powerRel ?? 1e-12;
          for (let i = 0; i < n; i++) {
            const a = ours[i],
              b = g.clusters[i];
            expect([a.rangeBin, a.dopplerBin], `cluster ${i} peak`).toEqual([b.rangeBin, b.dopplerBin]);
            relClose(a.power, b.power, rel, `cluster ${i} power`);
            expect(a.size, `cluster ${i} size`).toBe(b.size);
            expect(a.cells, `cluster ${i} cells`).toEqual(b.cells);
          }
        });
        return;
      }
      // ---------- Python reference tracker (goldenSet 2) ----------
      if (Array.isArray(g.frames) && Array.isArray(g.measurements)) {
        it('EKF-CV + GNN + M/N tracker (TS port of the Python reference): state, covariance, status, hits', () => {
          const pp = g.params;
          const trk = new ReferenceTracker({
            q: pp.q,
            sigmaR: pp.sigmaR,
            sigmaTheta: pp.sigmaTheta,
            sigmaVr: pp.sigmaVr,
            gateProb: pp.gateProb,
            confirmM: pp.confirmM,
            confirmN: pp.confirmN,
            maxMisses: pp.maxMisses,
            initVtStd: pp.initVtStd,
          });
          if (pp.gateChi2 !== undefined) relClose(trk.gate, pp.gateChi2, 1e-10, 'χ²₃ gate');
          const sAbs = T.stateAbs ?? 1e-9,
            cRel = T.covRel ?? 1e-9;
          g.frames.forEach((fr: Json, f: number) => {
            trk.step(g.measurements[f], f === 0 ? 0 : g.dt);
            const ctx = `frame ${fr.frame ?? f}`;
            expect(trk.tracks.map((t) => t.id), `${ctx} track ids`).toEqual(fr.tracks.map((t: Json) => t.id));
            fr.tracks.forEach((rt: Json, i: number) => {
              const t = trk.tracks[i];
              const c = `${ctx} track ${rt.id}`;
              expect(t.status, `${c} status`).toBe(rt.status);
              expect(t.hits, `${c} hits`).toEqual(rt.hits);
              expect(t.misses, `${c} misses`).toBe(rt.misses);
              expect(t.age, `${c} age`).toBe(rt.age);
              for (let k = 0; k < 4; k++) expect(Math.abs(t.x[k] - rt.x[k]), `${c} x[${k}]: ts=${t.x[k]} py=${rt.x[k]}`).toBeLessThanOrEqual(sAbs);
              let mx = 0;
              for (const v of rt.P) mx = Math.max(mx, Math.abs(v));
              for (let k = 0; k < 16; k++) expect(Math.abs(t.P[k] - rt.P[k]), `${c} P[${k}]: ts=${t.P[k]} py=${rt.P[k]}`).toBeLessThanOrEqual(cRel * mx);
            });
          });
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
      // ---------- micro-Doppler drone: line-blade (sinc) rotor model (goldenSet 2) ----------
      if (g.target && g.target.type === 'drone' && g.cube?.re) {
        const cfg = toConfig(g.config);
        const d = derive(cfg);
        const s = toSettings(g);
        const tgt = g.target;
        // velocity (0,0,0) + heading 0 → the web forward/right basis equals the Python
        // rotation (heading 0), so the ported drone reproduces the reference exactly.
        const target: Target = {
          id: 1,
          type: 'drone',
          x: tgt.position[0],
          y: tgt.position[1],
          z: tgt.position[2],
          vx: tgt.velocity[0],
          vy: tgt.velocity[1],
          vz: tgt.velocity[2],
          rcs: 1,
          rpm: tgt.rpm,
          bladeLength: tgt.bladeLength,
          bladeRoot: tgt.bladeRoot,
          nRotors: tgt.nRotors,
          nBlades: tgt.nBlades,
          armLength: tgt.armLength,
          rotorZ: tgt.rotorZ,
          bodyRcs: tgt.bodyRcs,
          bladeRcs: tgt.bladeRcs,
        };
        const cubeRel = T.cubeRelMax ?? 1e-9;
        if (g.derived) it('derived parameters', () => checkDerived(d, cfg, g.derived, 1e-12, f));
        it('line-blade (sinc) rotor beat cube (exact and phasor-recurrence synthesis)', () => {
          const [K, M, N] = g.cube.shape;
          expect([K, M, N]).toEqual([d.nVirt, cfg.nChirps, cfg.nSamples]);
          const gr: number[] = g.cube.re.flat(Infinity),
            gi: number[] = g.cube.im.flat(Infinity);
          let mx = 0;
          for (let i = 0; i < gr.length; i++) mx = Math.max(mx, Math.hypot(gr[i], gi[i]));
          for (const exact of [true, false]) {
            const n = d.nVirt * cfg.nChirps * cfg.nSamples;
            const re = new Float64Array(n),
              im = new Float64Array(n);
            new Synthesizer(cfg, d).synthTargets(re, im, [target], 0, s, exact);
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
        if (g.slowTime?.re) {
          it('micro-Doppler slow-time y[m] at the target range bin (range FFT, Hann, Σw-normalised)', () => {
            const e = new Engine(cfg, s, [target]);
            e.synthesizeAndTransform(true); // range FFT in place → cubeRe/cubeIm (K×M×N)
            const M = cfg.nChirps,
              N = cfg.nSamples;
            const rb = g.slowTime.rangeBin,
              ch = g.slowTime.channel ?? 0;
            const gr: number[] = g.slowTime.re,
              gi: number[] = g.slowTime.im;
            expect(gr.length).toBe(M);
            let mx = 0;
            for (let i = 0; i < gr.length; i++) mx = Math.max(mx, Math.hypot(gr[i], gi[i]));
            const rel = T.slowTimeRelMax ?? 1e-9;
            let err = 0,
              at = 0;
            for (let m = 0; m < M; m++) {
              const j = (ch * M + m) * N + rb;
              const er = Math.hypot(e.cubeRe[j] - gr[m], e.cubeIm[j] - gi[m]);
              if (er > err) {
                err = er;
                at = m;
              }
            }
            const j = (ch * M + at) * N + rb;
            expect(err, `slow-time m=${at}: ts=(${e.cubeRe[j]}, ${e.cubeIm[j]}) py=(${gr[at]}, ${gi[at]})`).toBeLessThanOrEqual(rel * mx);
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

      if (Array.isArray(g.amplitudes)) {
        it('physical amplitude mode: per-scatterer amplitudes (radar equation, kT0F·fs noise)', () => {
          g.amplitudes.forEach((a: number, i: number) => {
            const p = pts[i];
            relClose(amplitudeFor(p.rcs, Math.hypot(p.x, p.y, p.z), s, cfg, d.lambda), a, 1e-12, `scatterer ${i}`);
          });
        });
      }

      if (g.rangeAngle) {
        const ra = g.rangeAngle;
        for (const win of ['rect', 'hann'] as const) {
          if (!ra[win]) continue;
          it(`range–angle map (${win} angle window, Σw-normalised)`, () => {
            const e = engine();
            const A = ra.angleFftSize ?? 64;
            expect(ra.excludeZeroDoppler ?? false).toBe(false);
            const [nR, nA] = ra[win].shape;
            expect([nR, nA]).toEqual([d.nRangeBins, A]);
            const det = new Detector(cfg, d, A);
            const out = new Float64Array(nR * A);
            det.rangeAngle(e.rdRe, e.rdIm, out, makeWindow(win, d.arrayLen));
            const ref: number[] = ra[win].data.flat(Infinity);
            let mx = 0;
            for (const v of ref) mx = Math.max(mx, v);
            const tol = (ra.tolerances?.relOfMax ?? 1e-9) * mx;
            let worst = 0,
              at = 0;
            for (let i = 0; i < ref.length; i++) {
              const err = Math.abs(out[i] - ref[i]);
              if (err > worst) {
                worst = err;
                at = i;
              }
            }
            expect(worst, `worst cell (${Math.floor(at / A)}, ${at % A}) ts=${out[at]} py=${ref[at]}`).toBeLessThanOrEqual(tol);
          });
        }
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
            if (d.nVirt > 1) {
              // point cloud (Python point_cloud): elevation from same-x/different-z pairs,
              // x = R·u_x, z = R·u_z, y = R·√(1−u_x²−u_z²), θ = atan2(x, y)
              chk('elevationDeg', m.elevDeg, pk.elevationDeg, T.angleDegAbs ?? 1e-4);
              chk('azimuthDeg', m.azimuthDeg, pk.azimuthDeg, T.angleDegAbs ?? 1e-4);
              if (pk.xyz) {
                const tx = T.xyzAbs ?? 1e-6;
                chk('x', m.x, pk.xyz[0], tx);
                chk('y', m.y, pk.xyz[1], tx);
                chk('z', m.z, pk.xyz[2], tx);
              }
            }
            if (pk.truth) {
              const tr = pk.truth;
              const notAliased = Math.abs(tr.velocity - (tr.velocityAliased ?? tr.velocity)) < 1e-9;
              // observable truth: interpolation bias only
              chk('truth rangeBeat', m.range, tr.rangeBeat, T.truthRangeBeatAbs ?? d.rangeRes);
              chk('truth velocityEffective', m.velocity, tr.velocityEffective, T.truthVelocityEffectiveAbs ?? d.dv);
              // (with an aliased velocity the TDM compensation is wrong, so the angle is not comparable)
              if (d.nVirt > 1 && notAliased) chk('truth angleDegFft', m.angleDeg, tr.angleDegFft, T.truthAngleFftDegAbs ?? 2);
              // geometric truth: + the known deterministic offsets of this file
              chk('truth range', m.range, tr.range, T.truthRangeAbs ?? d.rangeRes);
              chk('truth velocity (aliased)', m.velocity, tr.velocityAliased ?? tr.velocity, T.truthVelocityAbs ?? d.dv);
              if (d.nVirt > 1 && notAliased) {
                chk('truth angle', m.azimuthDeg, tr.angleDeg, T.truthAngleDegAbs ?? 2);
                chk('truth azimuth', m.azimuthDeg, tr.angleDeg, T.truthAzimuthDegAbs ?? 2);
                chk('truth elevation', m.elevDeg, tr.elevationDeg, T.truthElevationDegAbs ?? 2);
              }
            }
          }
        });
      }
    });
  }
});
