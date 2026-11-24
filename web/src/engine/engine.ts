// Full per-frame processing chain on preallocated typed arrays.

import { derive, type Derived, type RadarConfig, type SimSettings } from '../config/radar';
import { getFFT, type FFT } from '../dsp/fft';
import { makeNormWindow } from '../dsp/window';
import { AlphaCache, Cfar2D } from '../dsp/cfar';
import { Detector, type Detection } from '../dsp/detect';
import { Spectrogram } from '../dsp/spectrogram';
import { Synthesizer, type PointScatterer } from '../sim/synth';
import { targetKinematics, maxMicroRadialSpeed, type Target } from '../sim/targets';
import { Tracker, mergeDetections } from '../track/tracker';

export const STAGES = ['synth', 'noise', 'rangeFft', 'mti', 'dopplerFft', 'cfar', 'cluster', 'track', 'raMap', 'microDoppler', 'pack'] as const;
export type Stage = (typeof STAGES)[number];

/** Transferable output buffers (Float32) for the UI. */
export interface OutBuffers {
  rdDb: Float32Array; // nR × M
  rangeDb: Float32Array; // nR
  raDb: Float32Array; // nR × A
  cutDb: Float32Array; // nR (P along selected Doppler column)
  cutThrDb: Float32Array; // 4 × nR (CA, GO, SO, OS)
  ifRe: Float32Array; // N (channel 0, chirp 0)
  ifIm: Float32Array;
  specDb: Float32Array; // cols × nfft
}

export function allocOut(cfg: RadarConfig, d: Derived, A: number, specCols: number, specNfft: number): OutBuffers {
  const nR = d.nRangeBins,
    M = cfg.nChirps;
  return {
    rdDb: new Float32Array(nR * M),
    rangeDb: new Float32Array(nR),
    raDb: new Float32Array(nR * A),
    cutDb: new Float32Array(nR),
    cutThrDb: new Float32Array(4 * nR),
    ifRe: new Float32Array(cfg.nSamples),
    ifIm: new Float32Array(cfg.nSamples),
    specDb: new Float32Array(specCols * specNfft),
  };
}

export interface TruthInfo {
  id: number;
  type: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  range: number;
  vr: number;
  azDeg: number;
  rBin: number;
  dBin: number; // aliased Doppler bin (fftshifted)
  maxMicro: number;
}

export interface TrackInfo {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  confirmed: boolean;
  hist: number[];
}

export interface FrameMeta {
  frame: number;
  time: number;
  nR: number;
  M: number;
  A: number;
  N: number;
  specCols: number;
  specNfft: number;
  specWin: number;
  cutBin: number;
  selRangeBin: number;
  detections: Detection[];
  tracks: TrackInfo[];
  truth: TruthInfo[];
  timings: Record<Stage, number>;
  cfarCount: number;
  noiseFloorDb: number;
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export class Engine {
  cfg: RadarConfig;
  d: Derived;
  s: SimSettings;
  targets: Target[];
  initial: Target[];
  frame = 0;
  time = 0;
  selectedId = -1;
  cutOverride = -1; // user-chosen Doppler bin for the CFAR cut
  /** If set, synthesize these explicit point scatterers (state at frame start) instead of `targets`. */
  points: PointScatterer[] | null = null;
  readonly A: number;
  readonly synth: Synthesizer;
  readonly cubeRe: Float64Array; // K×M×N (range FFT in place)
  readonly cubeIm: Float64Array;
  readonly rdRe: Float64Array; // K×nR×M
  readonly rdIm: Float64Array;
  readonly P: Float64Array; // nR×M
  readonly rangeP: Float64Array;
  readonly raP: Float64Array;
  readonly cfar: Cfar2D;
  readonly det: Detector;
  readonly tracker: Tracker;
  readonly spec: Spectrogram;
  private wR: Float64Array;
  private wD: Float64Array;
  private fftN: FFT;
  private fftM: FFT;
  private seqRe: Float64Array;
  private seqIm: Float64Array;
  private cutThr: Float64Array;
  private noiseAlpha: number;
  timings: Record<Stage, number>;

  constructor(cfg: RadarConfig, s: SimSettings, targets: Target[]) {
    this.cfg = cfg;
    this.d = derive(cfg);
    this.s = s;
    this.targets = targets.map((t) => ({ ...t }));
    this.initial = targets.map((t) => ({ ...t }));
    const d = this.d,
      K = d.nVirt,
      M = cfg.nChirps,
      N = cfg.nSamples,
      nR = d.nRangeBins;
    this.A = s.angleFft;
    this.synth = new Synthesizer(cfg, d);
    this.cubeRe = new Float64Array(K * M * N);
    this.cubeIm = new Float64Array(K * M * N);
    this.rdRe = new Float64Array(K * nR * M);
    this.rdIm = new Float64Array(K * nR * M);
    this.P = new Float64Array(nR * M);
    this.rangeP = new Float64Array(nR);
    this.raP = new Float64Array(nR * this.A);
    // brief rev. 1: CA α is Gamma(K)-exact for K non-coherently summed channels
    this.cfar = new Cfar2D(nR, M, s.cfar, d.nVirt);
    this.det = new Detector(cfg, d, this.A, s.maxDetections);
    this.tracker = new Tracker({
      vMax: d.vMax,
      sigmaR: Math.max(0.15, d.rangeRes),
      sigmaV: Math.max(0.1, d.dv),
      sigmaAz: Math.max(1, d.angleResDeg / 3) * (Math.PI / 180),
    });
    this.spec = new Spectrogram(M);
    this.wR = makeNormWindow(s.rangeWindow, N);
    this.wD = makeNormWindow(s.dopplerWindow, M);
    this.fftN = getFFT(N);
    this.fftM = getFFT(M);
    this.seqRe = new Float64Array(M);
    this.seqIm = new Float64Array(M);
    this.cutThr = new Float64Array(4 * nR);
    const wr = s.cfar.guardR + s.cfar.trainR,
      wd = s.cfar.guardD + s.cfar.trainD;
    const nTrain = (2 * wr + 1) * (2 * wd + 1) - (2 * s.cfar.guardR + 1) * (2 * s.cfar.guardD + 1);
    this.noiseAlpha = new AlphaCache(s.cfar, d.nVirt).get('CA', nTrain);
    this.timings = Object.fromEntries(STAGES.map((k) => [k, 0])) as Record<Stage, number>;
  }

  setTargets(t: Target[], resetInitial = true): void {
    this.targets = t.map((x) => ({ ...x }));
    if (resetInitial) this.initial = t.map((x) => ({ ...x }));
  }

  truth(): TruthInfo[] {
    const d = this.d,
      M = this.cfg.nChirps;
    return this.targets.map((t) => {
      const k = targetKinematics(t);
      let db = Math.round(k.vr / d.dv) + M / 2;
      db = ((db % M) + M) % M;
      return {
        id: t.id,
        type: t.type,
        x: t.x,
        y: t.y,
        vx: t.vx,
        vy: t.vy,
        range: k.range,
        vr: k.vr,
        azDeg: k.azDeg,
        rBin: Math.round(k.range / d.rangeRes),
        dBin: db,
        maxMicro: maxMicroRadialSpeed(t),
      };
    });
  }

  /** Steps 1–6: synthesis → range FFT → MTI → Doppler FFT → power map. */
  synthesizeAndTransform(exactSynth = false, out?: OutBuffers): void {
    const cfg = this.cfg,
      d = this.d,
      s = this.s;
    const K = d.nVirt,
      M = cfg.nChirps,
      N = cfg.nSamples,
      nR = d.nRangeBins;
    const re = this.cubeRe,
      im = this.cubeIm;
    let t0 = now();
    if (this.points) this.synth.synthPoints(re, im, this.points, s, exactSynth);
    else this.synth.synthTargets(re, im, this.targets, this.time, s, exactSynth);
    if (s.clutter && s.nClutter > 0) this.synth.addClutter(re, im, s);
    let t1 = now();
    this.timings.synth = t1 - t0;
    if (s.noise) this.synth.addNoise(re, im, s, this.frame);
    t0 = now();
    this.timings.noise = t0 - t1;
    if (out) {
      // raw IF snapshot: virtual channel 0, chirp 0
      for (let n = 0; n < N; n++) {
        out.ifRe[n] = re[n];
        out.ifIm[n] = im[n];
      }
    }

    // range FFT (window normalised by Σw), in place, per (k, m)
    const w = this.wR;
    const rp = this.rangeP;
    rp.fill(0);
    for (let row = 0; row < K * M; row++) {
      const off = row * N;
      for (let n = 0; n < N; n++) {
        re[off + n] *= w[n];
        im[off + n] *= w[n];
      }
      this.fftN.forward(re, im, off);
      for (let r = 0; r < nR; r++) rp[r] += re[off + r] * re[off + r] + im[off + r] * im[off + r];
    }
    const inv = 1 / (K * M);
    for (let r = 0; r < nR; r++) rp[r] *= inv;
    t1 = now();
    this.timings.rangeFft = t1 - t0;

    // MTI: remove the slow-time mean (static clutter / zero-Doppler)
    if (s.mti) {
      for (let k = 0; k < K; k++)
        for (let r = 0; r < nR; r++) {
          let sr = 0,
            si = 0;
          for (let m = 0; m < M; m++) {
            const j = (k * M + m) * N + r;
            sr += re[j];
            si += im[j];
          }
          sr /= M;
          si /= M;
          for (let m = 0; m < M; m++) {
            const j = (k * M + m) * N + r;
            re[j] -= sr;
            im[j] -= si;
          }
        }
    }
    t0 = now();
    this.timings.mti = t0 - t1;

    // Doppler FFT + fftshift → rd[(k·nR + r)·M + d]; P = Σ_k |X|²
    const P = this.P;
    P.fill(0);
    const wd = this.wD,
      h = M >> 1;
    const rdRe = this.rdRe,
      rdIm = this.rdIm;
    // corner-turn: sequential read of the range cube, write rows of the RD layout
    for (let k = 0; k < K; k++)
      for (let m = 0; m < M; m++) {
        const src = (k * M + m) * N;
        const wm = wd[m];
        let dst = k * nR * M + m;
        for (let r = 0; r < nR; r++, dst += M) {
          rdRe[dst] = re[src + r] * wm;
          rdIm[dst] = im[src + r] * wm;
        }
      }
    for (let k = 0; k < K; k++)
      for (let r = 0; r < nR; r++) {
        const ob = (k * nR + r) * M;
        this.fftM.forward(rdRe, rdIm, ob);
        const pb = r * M;
        for (let dd = 0; dd < h; dd++) {
          const a0 = rdRe[ob + dd],
            b0 = rdIm[ob + dd];
          const a1 = rdRe[ob + dd + h],
            b1 = rdIm[ob + dd + h];
          rdRe[ob + dd] = a1;
          rdIm[ob + dd] = b1;
          rdRe[ob + dd + h] = a0;
          rdIm[ob + dd + h] = b0;
          P[pb + dd] += a1 * a1 + b1 * b1;
          P[pb + dd + h] += a0 * a0 + b0 * b0;
        }
      }
    t1 = now();
    this.timings.dopplerFft = t1 - t0;
  }

  /** Run one complete frame; fills `out` (if given) and returns metadata. Advances the scene. */
  runFrame(out?: OutBuffers, advance = true, updateState = advance): FrameMeta {
    const cfg = this.cfg,
      d = this.d,
      s = this.s;
    const M = cfg.nChirps,
      N = cfg.nSamples,
      nR = d.nRangeBins,
      A = this.A;
    const truth = this.truth();
    this.synthesizeAndTransform(false, out);

    let t0 = now();
    const cf = this.cfar.run(this.P, s.cfar.type, s.cfarExact);
    let t1 = now();
    this.timings.cfar = t1 - t0;

    this.det.cluster(this.P, cf.mask, cf.thr, this.noiseAlpha);
    this.det.estimateAngles(this.rdRe, this.rdIm);
    t0 = now();
    this.timings.cluster = t0 - t1;

    const meas = mergeDetections(this.det.dets, this.det.count, Math.max(2.5, 2.5 * d.rangeRes), Math.max(8, 4 * d.dv));
    if (updateState) this.tracker.step(meas, cfg.framePeriod);
    t1 = now();
    this.timings.track = t1 - t0;

    this.det.rangeAngle(this.rdRe, this.rdIm, this.raP);
    t0 = now();
    this.timings.raMap = t0 - t1;

    // micro-Doppler: slow-time sequence at the selected target's range bin,
    // TX0 channels beamformed toward the target's true azimuth.
    let sel = truth.find((t) => t.id === this.selectedId);
    if (!sel && truth.length) sel = truth[0];
    let selR = -1;
    if (sel) {
      selR = Math.min(nR - 1, Math.max(0, sel.rBin));
      // choose the strongest of rBin−1..rBin+1 (range-migration / extended targets)
      let best = selR,
        bp = -1;
      for (let r = Math.max(0, selR - 1); r <= Math.min(nR - 1, selR + 1); r++) if (this.rangeP[r] > bp) {
        bp = this.rangeP[r];
        best = r;
      }
      selR = best;
      const u = Math.sin((sel.azDeg * Math.PI) / 180);
      this.seqRe.fill(0);
      this.seqIm.fill(0);
      for (let rx = 0; rx < d.nRx; rx++) {
        const ph = 2 * Math.PI * d.virtPos[rx] * u; // undo the (negative) spatial phase
        const c = Math.cos(ph),
          sn = Math.sin(ph);
        for (let m = 0; m < M; m++) {
          const j = (rx * M + m) * N + selR;
          const a = this.cubeRe[j],
            b = this.cubeIm[j];
          this.seqRe[m] += a * c - b * sn;
          this.seqIm[m] += a * sn + b * c;
        }
      }
      if (updateState) this.spec.push(this.seqRe, this.seqIm, M);
    }
    t1 = now();
    this.timings.microDoppler = t1 - t0;

    // CFAR cut column
    let cutBin = this.cutOverride >= 0 && this.cutOverride < M ? this.cutOverride : sel ? sel.dBin : M / 2;
    this.cfar.cut(this.P, cutBin, this.cutThr);

    let floorDb = 0;
    if (out) {
      const L10 = 10 / Math.LN10;
      const P = this.P;
      let acc = 0;
      for (let i = 0; i < P.length; i++) {
        const v = L10 * Math.log(P[i] + 1e-30);
        out.rdDb[i] = v;
        acc += v;
      }
      floorDb = acc / P.length;
      for (let r = 0; r < nR; r++) {
        out.rangeDb[r] = L10 * Math.log(this.rangeP[r] + 1e-30);
        out.cutDb[r] = L10 * Math.log(P[r * M + cutBin] + 1e-30);
      }
      for (let i = 0; i < 4 * nR; i++) out.cutThrDb[i] = L10 * Math.log(this.cutThr[i] + 1e-30);
      const ra = this.raP;
      for (let i = 0; i < nR * A; i++) out.raDb[i] = L10 * Math.log(ra[i] + 1e-30);
      this.spec.toDb(out.specDb);
    }
    t0 = now();
    this.timings.pack = t0 - t1;

    const meta: FrameMeta = {
      frame: this.frame,
      time: this.time,
      nR,
      M,
      A,
      N,
      specCols: this.spec.cols,
      specNfft: this.spec.nfft,
      specWin: this.spec.win,
      cutBin,
      selRangeBin: selR,
      detections: this.det.dets.slice(0, this.det.count).map((x) => ({ ...x })),
      tracks: this.tracker.tracks.map((t) => ({ id: t.id, x: t.s[0], y: t.s[1], vx: t.s[2], vy: t.s[3], confirmed: t.confirmed, hist: t.hist.slice() })),
      truth,
      timings: { ...this.timings },
      cfarCount: cf.count,
      noiseFloorDb: floorDb,
    };
    if (advance) this.advance();
    return meta;
  }

  /** Advance bulk target motion by one frame period; reset targets that leave the coverage. */
  advance(): void {
    const dt = this.cfg.framePeriod;
    this.frame++;
    this.time += dt;
    const rMax = this.d.rMax;
    for (let i = 0; i < this.targets.length; i++) {
      const t = this.targets[i];
      t.x += t.vx * dt;
      t.y += t.vy * dt;
      t.z += t.vz * dt;
      const r = Math.hypot(t.x, t.y);
      const az = Math.abs(Math.atan2(t.x, t.y)) * (180 / Math.PI);
      if (r > rMax * 1.02 || r < 1 || t.y < 0.5 || az > 75) {
        const init = this.initial.find((q) => q.id === t.id);
        if (init) Object.assign(t, init);
      }
    }
  }
}
