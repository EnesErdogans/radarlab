// Radar configuration, derived parameters and presets.
// Field names follow docs/PROJECT_BRIEF.md (camelCase). SI units internally.

export const C0 = 299792458;

export type IqMode = 'complex' | 'real';

export interface RadarConfig {
  fc: number; // carrier start frequency [Hz]
  slope: number; // chirp slope [Hz/s]
  fs: number; // ADC sample rate [Hz]
  nSamples: number; // N, ADC samples per chirp (power of 2)
  adcStartTime: number; // [s]
  chirpPeriod: number; // Tc, ramp + idle, per TX [s]
  nChirps: number; // M, chirps per TX per frame (power of 2)
  txPos: number[]; // element x-positions [wavelengths]
  rxPos: number[];
  txPosZ?: number[];
  rxPosZ?: number[];
  framePeriod: number; // [s]
  iqMode: IqMode;
  /** TS-only extra: ramp duration [s] (for "ADC window > ramp" checks). Defaults to chirpPeriod. */
  rampTime?: number;
}

export interface Derived {
  lambda: number;
  nTx: number;
  nRx: number;
  nVirt: number; // K
  te: number; // effective chirp repetition per TX (nTx·Tc)
  adcTime: number; // N/fs
  bEff: number; // slope·N/fs
  bRamp: number; // slope·rampTime (swept bandwidth)
  rangeRes: number;
  rMax: number;
  nRangeBins: number; // N (complex) or N/2 (real)
  dv: number;
  vMax: number;
  frameActive: number; // M·Te
  virtPos: Float64Array; // K virtual positions (x, wavelengths)
  virtPosZ: Float64Array;
  arraySpacing: number; // minimum spacing of the virtual ULA [wavelengths]
  angleIdx: Int32Array; // ULA slot of each virtual channel (−1 = not used for azimuth)
  arrayLen: number; // max index + 1
  angleResDeg: number; // ≈ asin(1/(L·d)) at boresight (Rayleigh, uniform array)
  dataRate: number; // bytes/s for 16-bit I/Q samples
  fovDeg: number; // unambiguous FOV half-angle (|sinθ| ≤ 1/(2d))
}

export function derive(cfg: RadarConfig): Derived {
  const lambda = C0 / cfg.fc;
  const nTx = cfg.txPos.length;
  const nRx = cfg.rxPos.length;
  const nVirt = nTx * nRx;
  const te = nTx * cfg.chirpPeriod;
  const adcTime = cfg.nSamples / cfg.fs;
  const bEff = cfg.slope * adcTime;
  const rampTime = cfg.rampTime ?? cfg.chirpPeriod;
  const complex = cfg.iqMode === 'complex';
  const rMaxComplex = (cfg.fs * C0) / (2 * cfg.slope);
  const virtPos = new Float64Array(nVirt);
  const virtPosZ = new Float64Array(nVirt);
  for (let t = 0; t < nTx; t++)
    for (let r = 0; r < nRx; r++) {
      const k = t * nRx + r;
      virtPos[k] = cfg.txPos[t] + cfg.rxPos[r];
      virtPosZ[k] = (cfg.txPosZ?.[t] ?? 0) + (cfg.rxPosZ?.[r] ?? 0);
    }
  // Azimuth ULA (same rule as the Python reference): channels in the lowest z-plane,
  // unique x positions (first channel in x-sorted stable order wins), spacing = min Δx.
  let zmin = Infinity;
  for (let k = 0; k < nVirt; k++) zmin = Math.min(zmin, virtPosZ[k]);
  const cand: number[] = [];
  for (let k = 0; k < nVirt; k++) if (Math.abs(virtPosZ[k] - zmin) < 1e-9) cand.push(k);
  cand.sort((a, b) => virtPos[a] - virtPos[b] || a - b);
  const sel: number[] = [];
  for (const k of cand) if (!sel.some((j) => Math.abs(virtPos[j] - virtPos[k]) < 1e-9)) sel.push(k);
  let dmin = Infinity;
  for (let i = 1; i < sel.length; i++) {
    const dd = virtPos[sel[i]] - virtPos[sel[i - 1]];
    if (dd > 1e-9 && dd < dmin) dmin = dd;
  }
  if (!isFinite(dmin)) dmin = 0.5;
  const pmin = sel.length ? virtPos[sel[0]] : 0;
  const angleIdx = new Int32Array(nVirt).fill(-1);
  let arrayLen = 1;
  for (const k of sel) {
    angleIdx[k] = Math.round((virtPos[k] - pmin) / dmin);
    arrayLen = Math.max(arrayLen, angleIdx[k] + 1);
  }
  const aperture = arrayLen * dmin;
  const angleResDeg = (Math.asin(Math.min(1, 1 / aperture)) * 180) / Math.PI;
  return {
    lambda,
    nTx,
    nRx,
    nVirt,
    te,
    adcTime,
    bEff,
    bRamp: cfg.slope * rampTime,
    rangeRes: C0 / (2 * bEff),
    rMax: complex ? rMaxComplex : rMaxComplex / 2,
    nRangeBins: complex ? cfg.nSamples : cfg.nSamples / 2,
    dv: lambda / (2 * cfg.nChirps * te),
    vMax: lambda / (4 * te),
    frameActive: cfg.nChirps * te,
    virtPos,
    virtPosZ,
    arraySpacing: dmin,
    angleIdx,
    arrayLen,
    angleResDeg,
    dataRate: (nVirt * cfg.nSamples * cfg.nChirps * (complex ? 4 : 2)) / cfg.framePeriod,
    fovDeg: (Math.asin(Math.min(1, 1 / (2 * dmin))) * 180) / Math.PI,
  };
}

/** Standard filled ULA: RX at 0.5λ spacing, TX at nRx·0.5λ spacing → K-element 0.5λ virtual ULA. */
export function ulaLayout(nTx: number, nRx: number): { txPos: number[]; rxPos: number[] } {
  return {
    txPos: Array.from({ length: nTx }, (_, t) => t * nRx * 0.5),
    rxPos: Array.from({ length: nRx }, (_, r) => r * 0.5),
  };
}

export type WindowType = 'hann' | 'hamming' | 'blackman' | 'rect';
export type CfarType = 'CA' | 'GO' | 'SO' | 'OS';
export type AmplitudeMode = 'snr' | 'physical';

export interface CfarParams {
  type: CfarType;
  pfa: number;
  guardR: number;
  guardD: number;
  trainR: number;
  trainD: number;
  osRank: number; // fraction, k = round(rank·N)
}

export interface PhysicalParams {
  ptDbm: number;
  gtDbi: number;
  grDbi: number;
  nfDb: number;
  lossDb: number;
  t0: number;
}

export interface SimSettings {
  amplitudeMode: AmplitudeMode;
  snrRefDb: number;
  refRange: number;
  physical: PhysicalParams;
  noise: boolean;
  seed: number;
  exactNoise: boolean; // exact sequential Box–Muller (tests) vs Gaussian pool (UI)
  mti: boolean;
  clutter: boolean;
  nClutter: number;
  rangeWindow: WindowType;
  dopplerWindow: WindowType;
  cfar: CfarParams;
  cfarExact: boolean; // OS-CFAR without noise-floor prefilter
  angleFft: number;
  maxDetections: number;
}

export function defaultSettings(): SimSettings {
  return {
    amplitudeMode: 'snr',
    snrRefDb: 10,
    refRange: 10,
    physical: { ptDbm: 12, gtDbi: 10, grDbi: 10, nfDb: 12, lossDb: 3, t0: 290 },
    noise: true,
    seed: 1,
    exactNoise: false,
    mti: true,
    clutter: false,
    nClutter: 24,
    rangeWindow: 'hann',
    dopplerWindow: 'hann',
    cfar: { type: 'CA', pfa: 1e-5, guardR: 2, guardD: 2, trainR: 8, trainD: 4, osRank: 0.75 },
    cfarExact: false,
    angleFft: 64,
    maxDetections: 256,
  };
}

export type PresetId = 'automotive' | 'drone' | 'micro';

export interface Preset {
  id: PresetId;
  name: { en: string; tr: string };
  cfg: RadarConfig;
  settings: Partial<SimSettings>;
}

const MHZ_US = 1e12; // 1 MHz/µs in Hz/s

export const PRESETS: Record<PresetId, Preset> = {
  automotive: {
    id: 'automotive',
    name: { en: 'Automotive 77 GHz (2TX×4RX)', tr: 'Otomotiv 77 GHz (2TX×4RX)' },
    cfg: {
      fc: 77e9,
      slope: 30 * MHZ_US,
      fs: 10e6,
      nSamples: 256,
      adcStartTime: 0,
      chirpPeriod: 40e-6,
      nChirps: 128,
      ...ulaLayout(2, 4),
      framePeriod: 0.05,
      iqMode: 'complex',
      rampTime: 30e-6,
    },
    settings: { snrRefDb: 10, refRange: 10 },
  },
  drone: {
    id: 'drone',
    name: { en: 'Drone surveillance 24 GHz (1TX×4RX)', tr: 'Drone gözetleme 24 GHz (1TX×4RX)' },
    cfg: {
      fc: 24e9,
      slope: 150e6 / (512 / 5.12e6), // B_eff = 150 MHz over the 100 µs ADC window
      fs: 5.12e6,
      nSamples: 512,
      adcStartTime: 0,
      chirpPeriod: 110e-6,
      nChirps: 256,
      ...ulaLayout(1, 4),
      framePeriod: 0.1,
      iqMode: 'complex',
      rampTime: 105e-6,
    },
    settings: { snrRefDb: 0, refRange: 100 },
  },
  micro: {
    id: 'micro',
    name: { en: 'Micro-Doppler 24 GHz fast-chirp', tr: 'Mikro-Doppler 24 GHz hızlı chirp' },
    cfg: {
      fc: 24e9,
      slope: 250e6 / 16e-6, // 250 MHz over a 16 µs ramp
      fs: 8e6,
      nSamples: 128,
      adcStartTime: 0,
      chirpPeriod: 20e-6,
      nChirps: 256,
      ...ulaLayout(1, 4),
      framePeriod: 0.05,
      iqMode: 'complex',
      rampTime: 16e-6,
    },
    settings: { snrRefDb: 5, refRange: 10, cfar: { type: 'CA', pfa: 1e-5, guardR: 2, guardD: 3, trainR: 6, trainD: 6, osRank: 0.75 } },
  },
};

export function cloneConfig(c: RadarConfig): RadarConfig {
  return JSON.parse(JSON.stringify(c));
}

export function isPow2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

export interface Warning {
  key: string; // i18n key
  args?: Record<string, string>;
  level: 'warn' | 'error';
}

export interface TargetSummary {
  name: string;
  range: number;
  vr: number;
  maxMicroSpeed: number; // max |radial| speed of any scatterer (micro-Doppler)
  azDeg: number;
}

export function checkWarnings(cfg: RadarConfig, d: Derived, targets: TargetSummary[]): Warning[] {
  const w: Warning[] = [];
  const ramp = cfg.rampTime ?? cfg.chirpPeriod;
  if (cfg.adcStartTime + d.adcTime > ramp * (1 + 1e-9))
    w.push({ key: 'warnAdcRamp', level: 'error', args: { adc: ((cfg.adcStartTime + d.adcTime) * 1e6).toFixed(1), ramp: (ramp * 1e6).toFixed(1) } });
  if (ramp > cfg.chirpPeriod * (1 + 1e-9)) w.push({ key: 'warnRampTc', level: 'error' });
  if (d.frameActive > cfg.framePeriod) w.push({ key: 'warnFrame', level: 'error', args: { act: (d.frameActive * 1e3).toFixed(2) } });
  if (!isPow2(cfg.nSamples) || !isPow2(cfg.nChirps)) w.push({ key: 'warnPow2', level: 'error' });
  for (const t of targets) {
    if (t.range > d.rMax) w.push({ key: 'warnRmax', level: 'warn', args: { name: t.name, r: t.range.toFixed(1) } });
    if (Math.abs(t.vr) > d.vMax) w.push({ key: 'warnVel', level: 'warn', args: { name: t.name, v: t.vr.toFixed(1) } });
    else if (t.maxMicroSpeed > d.vMax) w.push({ key: 'warnMicro', level: 'warn', args: { name: t.name, v: t.maxMicroSpeed.toFixed(1) } });
    if (Math.abs(t.azDeg) > d.fovDeg) w.push({ key: 'warnFov', level: 'warn', args: { name: t.name } });
  }
  return w;
}
