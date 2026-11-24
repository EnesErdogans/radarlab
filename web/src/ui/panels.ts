// The eight visualisation panels.

import { C0, type Derived, type RadarConfig } from '../config/radar';
import type { FrameMeta, OutBuffers, TruthInfo } from '../engine/engine';
import { STAGES } from '../engine/engine';
import type { CmapName } from './colormap';
import { t } from './i18n';
import { AutoRange, Heatmap, MARGIN, Surface, THEME, drawAxes, drawColorbar, legend, makeAxes, plotRect, type Axes, type Rect } from './plot';

export interface Frame {
  meta: FrameMeta;
  bufs: OutBuffers;
  cfg: RadarConfig;
  d: Derived;
}

export interface AppApi {
  frame(): Frame | null;
  cmap(): CmapName;
  selectedId(): number;
  select(id: number): void;
  setCut(dBin: number): void;
  dragTarget(id: number, x: number, y: number, done: boolean): void;
  computeMs(): number;
  uiFps(): number;
  simFps(): number;
  warnings(): { key: string; args?: Record<string, string>; level: string }[];
}

export abstract class Panel {
  readonly id: string;
  readonly el: HTMLElement;
  readonly body: HTMLElement;
  readonly readout: HTMLElement;
  constructor(
    id: string,
    titleKey: string,
    protected app: AppApi,
  ) {
    this.id = id;
    this.el = document.createElement('section');
    this.el.className = `panel panel-${id}`;
    this.el.dataset.panel = id;
    this.el.innerHTML = `<header><h2 data-i18n="${titleKey}">${t(titleKey)}</h2><span class="readout"></span></header><div class="pbody"></div>`;
    this.body = this.el.querySelector('.pbody')!;
    this.readout = this.el.querySelector('.readout')!;
  }
  abstract render(): void;
  setReadout(s: string): void {
    this.readout.textContent = s;
  }
}

abstract class CanvasPanel extends Panel {
  readonly s: Surface;
  protected axes: Axes | null = null;
  constructor(id: string, titleKey: string, app: AppApi) {
    super(id, titleKey, app);
    this.s = new Surface(this.body);
    this.s.onResize = () => this.render();
    this.s.canvas.addEventListener('pointermove', (e) => this.hover(this.local(e)));
    this.s.canvas.addEventListener('pointerleave', () => this.setReadout(''));
  }
  protected local(e: PointerEvent | MouseEvent): { x: number; y: number } {
    const r = this.s.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  protected inside(p: { x: number; y: number }): boolean {
    const a = this.axes;
    if (!a) return false;
    const r = a.rect;
    return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h;
  }
  protected hover(_p: { x: number; y: number }): void {}
}

function selTruth(f: Frame, id: number): TruthInfo | undefined {
  return f.meta.truth.find((q) => q.id === id) ?? f.meta.truth[0];
}

function typeName(tr: { type: string; id: number }): string {
  return `${t(tr.type)} #${tr.id}`;
}

function wrapVel(v: number, vmax: number): number {
  return v - 2 * vmax * Math.round(v / (2 * vmax));
}

const VCOL = (v: number, vmax: number, a = 1) => {
  const x = Math.max(-1, Math.min(1, v / Math.max(1e-6, vmax * 0.6)));
  // approaching (negative) → cyan-blue, receding (positive) → red-orange
  const r = x > 0 ? 248 : Math.round(248 + x * 200),
    g = Math.round(180 - Math.abs(x) * 70),
    b = x < 0 ? 250 : Math.round(250 - x * 170);
  return `rgba(${r},${g},${b},${a})`;
};

// ------------------------------------------------------------------ 1 Chirp & IF
export class ChirpPanel extends CanvasPanel {
  constructor(app: AppApi) {
    super('chirp', 'pChirp', app);
  }
  render(): void {
    const f = this.app.frame();
    const c = this.s.begin();
    if (!f) return;
    const { cfg, d } = f;
    const tr = selTruth(f, this.app.selectedId());
    const H = this.s.h;
    const top: Rect = { x: MARGIN.l, y: 24, w: this.s.w - MARGIN.l - MARGIN.r, h: Math.max(30, H * 0.46 - 36) };
    const bot: Rect = { x: MARGIN.l, y: H * 0.46 + 20, w: top.w, h: Math.max(30, H * 0.54 - 16 - MARGIN.b) };
    // --- f–t sketch
    const tc = cfg.chirpPeriod * 1e6,
      ramp = (cfg.rampTime ?? cfg.chirpPeriod) * 1e6;
    const span = 2.35 * tc;
    const fTop = (cfg.slope * (cfg.rampTime ?? cfg.chirpPeriod)) / 1e6;
    const a = makeAxes(top, 0, span, -0.05 * fTop, fTop * 1.12);
    drawAxes(c, a, '', `f − f_c [MHz]`, { yTicks: 3 });
    c.font = THEME.fontSmall;
    // ADC windows
    const adc0 = cfg.adcStartTime * 1e6,
      adcT = d.adcTime * 1e6;
    c.fillStyle = 'rgba(56,189,248,0.10)';
    for (let k = 0; k < 3; k++) {
      const x0 = a.px(Math.min(span, k * tc + adc0)),
        x1 = a.px(Math.min(span, k * tc + adc0 + adcT));
      if (x1 > x0) c.fillRect(x0, top.y, x1 - x0, top.h);
    }
    const tau = tr ? (2 * tr.range) / C0 : 0;
    const tauUs = tau * 1e6;
    const vis = Math.max(tauUs, 0.03 * tc);
    const k = tauUs > 0 ? vis / tauUs : 1;
    const saw = (shift: number, col: string, dash: number[]) => {
      c.strokeStyle = col;
      c.setLineDash(dash);
      c.lineWidth = 1.6;
      c.beginPath();
      for (let m = 0; m < 3; m++) {
        const t0 = m * tc + shift;
        c.moveTo(a.px(Math.max(0, t0)), a.py(t0 < 0 ? (-t0 / ramp) * fTop : 0));
        c.lineTo(a.px(Math.min(span, t0 + ramp)), a.py(((Math.min(span, t0 + ramp) - t0) / ramp) * fTop));
        if (t0 + ramp < span) {
          c.lineTo(a.px(Math.min(span, t0 + ramp + 0.04 * tc)), a.py(0));
        }
      }
      c.stroke();
      c.setLineDash([]);
    };
    saw(0, THEME.accent, []);
    saw(vis, THEME.accent2, [5, 3]);
    legend(c, top.x + 6, 10, [
      { label: t('tx'), color: THEME.accent },
      { label: t('rx') + (k > 1.01 ? ` · ${t('delayExag', { k: k.toFixed(0) })}` : ''), color: THEME.accent2, dash: [5, 3] },
      { label: t('adcWin'), color: 'rgba(56,189,248,0.5)', width: 6 },
    ]);
    c.fillStyle = THEME.dim;
    c.textAlign = 'right';
    c.textBaseline = 'top';
    c.fillText(`${t('time')} [µs]`, top.x + top.w - 4, top.y + 3);
    // --- IF waveform
    const N = cfg.nSamples;
    const re = f.bufs.ifRe,
      im = f.bufs.ifIm;
    let mx = 1e-9;
    for (let n = 0; n < N; n++) mx = Math.max(mx, Math.abs(re[n]), Math.abs(im[n]));
    const ab = makeAxes(bot, adc0, adc0 + adcT, -mx * 1.05, mx * 1.05);
    this.axes = ab;
    drawAxes(c, ab, `${t('sampleTime')} [µs]`, t('amplitude'), { yTicks: 4 });
    const line = (arr: Float32Array, col: string) => {
      c.strokeStyle = col;
      c.lineWidth = 1.1;
      c.beginPath();
      for (let n = 0; n < N; n++) {
        const x = ab.px(adc0 + (n / cfg.fs) * 1e6),
          y = ab.py(arr[n]);
        if (n) c.lineTo(x, y);
        else c.moveTo(x, y);
      }
      c.stroke();
    };
    c.save();
    c.beginPath();
    c.rect(bot.x, bot.y, bot.w, bot.h);
    c.clip();
    line(im, 'rgba(167,139,250,0.55)');
    line(re, THEME.accent);
    c.restore();
    const fb = cfg.slope * tau + (tr ? (2 * tr.vr) / d.lambda : 0);
    const fbS = Math.abs(fb) >= 1e6 ? `${(fb / 1e6).toFixed(3)} MHz` : `${(fb / 1e3).toFixed(1)} kHz`;
    c.fillStyle = 'rgba(17,22,29,0.85)';
    c.fillRect(bot.x + bot.w - 86, bot.y + 2, 84, 16);
    legend(c, bot.x + bot.w - 84, bot.y + 10, [
      { label: 'Re', color: THEME.accent },
      { label: 'Im', color: THEME.violet },
    ]);
    c.font = THEME.fontSmall;
    c.fillStyle = THEME.text;
    c.textAlign = 'left';
    c.textBaseline = 'bottom';
    if (tr) c.fillText(`${typeName(tr)}: τ = ${(tau * 1e9).toFixed(1)} ns · ${t('beat')} ≈ ${fbS}`, bot.x, bot.y - 2);
  }
  protected hover(p: { x: number; y: number }): void {
    const a = this.axes;
    if (!a || !this.inside(p)) return this.setReadout('');
    this.setReadout(`t = ${a.ix(p.x).toFixed(2)} µs`);
  }
}

// ------------------------------------------------------------------ 2 Range profile
export class RangePanel extends CanvasPanel {
  private ar = new AutoRange();
  constructor(app: AppApi) {
    super('range', 'pRange', app);
  }
  render(): void {
    const f = this.app.frame();
    const c = this.s.begin();
    if (!f) return;
    const { d } = f;
    const nR = f.meta.nR;
    const data = f.bufs.rangeDb;
    const [lo, hi] = this.ar.update(data, 0.5, 70, 30);
    const a = makeAxes(plotRect(this.s), 0, nR * d.rangeRes, lo - 5, hi + 6);
    this.axes = a;
    drawAxes(c, a, `${t('range')} [m]`, `${t('power')} [dB]`);
    c.save();
    c.beginPath();
    c.rect(a.rect.x, a.rect.y, a.rect.w, a.rect.h);
    c.clip();
    // fill under curve
    c.beginPath();
    c.moveTo(a.px(0), a.py(a.y0));
    for (let r = 0; r < nR; r++) c.lineTo(a.px(r * d.rangeRes), a.py(data[r]));
    c.lineTo(a.px((nR - 1) * d.rangeRes), a.py(a.y0));
    c.closePath();
    const g = c.createLinearGradient(0, a.rect.y, 0, a.rect.y + a.rect.h);
    g.addColorStop(0, 'rgba(56,189,248,0.35)');
    g.addColorStop(1, 'rgba(56,189,248,0.02)');
    c.fillStyle = g;
    c.fill();
    c.strokeStyle = THEME.accent;
    c.lineWidth = 1.3;
    c.beginPath();
    for (let r = 0; r < nR; r++) {
      const x = a.px(r * d.rangeRes),
        y = a.py(data[r]);
      if (r) c.lineTo(x, y);
      else c.moveTo(x, y);
    }
    c.stroke();
    // truth markers
    c.font = THEME.fontSmall;
    c.textAlign = 'left';
    c.textBaseline = 'top';
    let lab = 0;
    for (const tr of f.meta.truth) {
      const x = a.px(tr.range);
      c.strokeStyle = tr.id === this.app.selectedId() ? THEME.accent2 : 'rgba(245,158,11,0.6)';
      c.setLineDash([4, 3]);
      c.beginPath();
      c.moveTo(x, a.rect.y);
      c.lineTo(x, a.rect.y + a.rect.h);
      c.stroke();
      c.setLineDash([]);
      c.fillStyle = THEME.accent2;
      c.fillText(typeName(tr), x + 3, a.rect.y + 3 + (lab++ % 3) * 12);
    }
    c.restore();
  }
  protected hover(p: { x: number; y: number }): void {
    const f = this.app.frame();
    const a = this.axes;
    if (!f || !a || !this.inside(p)) return this.setReadout('');
    const R = a.ix(p.x);
    const r = Math.max(0, Math.min(f.meta.nR - 1, Math.round(R / f.d.rangeRes)));
    this.setReadout(`R = ${(r * f.d.rangeRes).toFixed(2)} m · ${f.bufs.rangeDb[r].toFixed(1)} dB`);
  }
}

// ------------------------------------------------------------------ 3 Range–Doppler
export class RDPanel extends CanvasPanel {
  private hm = new Heatmap();
  private ar = new AutoRange();
  constructor(app: AppApi) {
    super('rd', 'pRD', app);
    this.s.canvas.addEventListener('click', (e) => {
      const f = this.app.frame();
      const p = this.local(e);
      if (!f || !this.axes || !this.inside(p)) return;
      const v = this.axes.ix(p.x);
      const dBin = Math.max(0, Math.min(f.meta.M - 1, Math.round(v / f.d.dv + f.meta.M / 2)));
      this.app.setCut(dBin);
    });
  }
  render(): void {
    const f = this.app.frame();
    const c = this.s.begin();
    if (!f) return;
    const { d } = f;
    const { nR, M } = f.meta;
    const rect = plotRect(this.s, { ...MARGIN, r: 44 });
    const a = makeAxes(rect, (-M / 2 - 0.5) * d.dv, (M / 2 - 0.5) * d.dv, -0.5 * d.rangeRes, (nR - 0.5) * d.rangeRes);
    this.axes = a;
    const [lo, hi] = this.ar.update(f.bufs.rdDb, 0.5, 55, 25);
    this.hm.ensure(M, nR);
    this.hm.fill(f.bufs.rdDb, (col, row) => (nR - 1 - row) * M + col, lo, hi, this.app.cmap());
    this.hm.draw(c, rect, false);
    drawAxes(c, a, `${t('velocity')} [m/s]`, `${t('range')} [m]`, { grid: false });
    drawColorbar(c, { x: rect.x + rect.w + 6, y: rect.y, w: 8, h: rect.h }, lo, hi, this.app.cmap());
    c.save();
    c.beginPath();
    c.rect(rect.x, rect.y, rect.w, rect.h);
    c.clip();
    // cut column
    const vc = (f.meta.cutBin - M / 2) * d.dv;
    c.strokeStyle = 'rgba(255,255,255,0.35)';
    c.setLineDash([3, 4]);
    c.beginPath();
    c.moveTo(a.px(vc), rect.y);
    c.lineTo(a.px(vc), rect.y + rect.h);
    c.stroke();
    c.setLineDash([]);
    // truth
    c.strokeStyle = 'rgba(255,255,255,0.9)';
    c.lineWidth = 1.2;
    for (const tr of f.meta.truth) {
      const x = a.px(wrapVel(tr.vr, d.vMax)),
        y = a.py(tr.range);
      c.beginPath();
      c.moveTo(x - 5, y);
      c.lineTo(x + 5, y);
      c.moveTo(x, y - 5);
      c.lineTo(x, y + 5);
      c.stroke();
    }
    // detections
    c.strokeStyle = THEME.bad;
    c.lineWidth = 1.4;
    for (const dt of f.meta.detections) {
      c.beginPath();
      c.arc(a.px(dt.velocity), a.py(dt.range), 5, 0, 7);
      c.stroke();
    }
    c.restore();
    legend(c, rect.x + 4, rect.y + 10, [
      { label: t('truth'), color: '#fff', marker: 'plus' },
      { label: t('detsN', { n: f.meta.detections.length }), color: THEME.bad, marker: 'circle' },
    ]);
  }
  protected hover(p: { x: number; y: number }): void {
    const f = this.app.frame();
    const a = this.axes;
    if (!f || !a || !this.inside(p)) return this.setReadout('');
    const M = f.meta.M;
    const dBin = Math.max(0, Math.min(M - 1, Math.round(a.ix(p.x) / f.d.dv + M / 2)));
    const r = Math.max(0, Math.min(f.meta.nR - 1, Math.round(a.iy(p.y) / f.d.rangeRes)));
    this.setReadout(`R = ${(r * f.d.rangeRes).toFixed(2)} m · v = ${((dBin - M / 2) * f.d.dv).toFixed(2)} m/s · ${f.bufs.rdDb[r * M + dBin].toFixed(1)} dB`);
  }
}

// ------------------------------------------------------------------ 4 CFAR cut
export class CutPanel extends CanvasPanel {
  private ar = new AutoRange();
  activeType = 'CA';
  constructor(app: AppApi) {
    super('cut', 'pCut', app);
  }
  render(): void {
    const f = this.app.frame();
    const c = this.s.begin();
    if (!f) return;
    const { d } = f;
    const nR = f.meta.nR;
    const P = f.bufs.cutDb,
      T = f.bufs.cutThrDb;
    const [lo, hi] = this.ar.update(P, 0.3, 60, 30);
    const rect = plotRect(this.s, { ...MARGIN, t: 24 });
    const a = makeAxes(rect, 0, nR * d.rangeRes, lo - 6, hi + 6);
    this.axes = a;
    drawAxes(c, a, `${t('range')} [m]`, `${t('power')} [dB]`);
    const types = ['CA', 'GO', 'SO', 'OS'];
    const cols = [THEME.accent, THEME.accent2, THEME.good, THEME.pink];
    c.save();
    c.beginPath();
    c.rect(rect.x, rect.y, rect.w, rect.h);
    c.clip();
    const line = (arr: Float32Array, off: number, col: string, w: number, dash: number[]) => {
      c.strokeStyle = col;
      c.lineWidth = w;
      c.setLineDash(dash);
      c.beginPath();
      for (let r = 0; r < nR; r++) {
        const x = a.px(r * d.rangeRes),
          y = a.py(arr[off + r]);
        if (r) c.lineTo(x, y);
        else c.moveTo(x, y);
      }
      c.stroke();
      c.setLineDash([]);
    };
    line(P, 0, 'rgba(226,232,240,0.85)', 1.1, []);
    const ai = Math.max(0, types.indexOf(this.activeType));
    for (let i = 0; i < 4; i++) if (i !== ai) line(T, i * nR, cols[i], 1, [4, 3]);
    line(T, ai * nR, cols[ai], 2, []);
    c.fillStyle = THEME.bad;
    for (let r = 0; r < nR; r++)
      if (P[r] > T[ai * nR + r]) {
        c.beginPath();
        c.arc(a.px(r * d.rangeRes), a.py(P[r]), 3, 0, 7);
        c.fill();
      }
    c.restore();
    legend(
      c,
      rect.x - 40,
      10,
      types.map((ty, i) => ({ label: ty + (i === ai ? ` (${t('active')})` : ''), color: cols[i], dash: i === ai ? [] : [4, 3], width: i === ai ? 2 : 1 })),
    );
    c.font = THEME.fontSmall;
    c.fillStyle = THEME.dim;
    c.textAlign = 'right';
    c.textBaseline = 'top';
    c.fillText(t('cutAt', { d: f.meta.cutBin, v: ((f.meta.cutBin - f.meta.M / 2) * d.dv).toFixed(2) }), rect.x + rect.w - 4, rect.y + rect.h - 14);
    c.fillText(t('cutGamma', { k: String(d.nVirt) }), rect.x + rect.w - 4, rect.y + 4);
  }
  protected hover(p: { x: number; y: number }): void {
    const f = this.app.frame();
    const a = this.axes;
    if (!f || !a || !this.inside(p)) return this.setReadout('');
    const r = Math.max(0, Math.min(f.meta.nR - 1, Math.round(a.ix(p.x) / f.d.rangeRes)));
    const nR = f.meta.nR;
    const T = f.bufs.cutThrDb;
    const th = ['CA', 'GO', 'SO', 'OS'].map((ty, i) => `${ty} ${T[i * nR + r].toFixed(1)}`).join(' · ');
    this.setReadout(`R = ${(r * f.d.rangeRes).toFixed(2)} m · P ${f.bufs.cutDb[r].toFixed(1)} · ${th} dB`);
  }
}

/** Equal-aspect Cartesian axes for the radar fan / bird's-eye views. */
function fanAxes(rect: Rect, rMax: number, xHalfMin: number, yPad = 0.03): Axes {
  // radar anchored at the bottom; spare room goes to the top / sides
  const yRange = rMax * (1 + 2 * yPad);
  const sc = Math.min(rect.w / (2 * xHalfMin), rect.h / yRange);
  const xh = rect.w / (2 * sc),
    yr = rect.h / sc;
  const y0 = -rMax * yPad;
  return makeAxes(rect, -xh, xh, y0, y0 + yr);
}

function drawFanGrid(c: CanvasRenderingContext2D, a: Axes, rMax: number, fovDeg: number): void {
  c.save();
  const ox = a.px(0),
    oy = a.py(0);
  const sc = a.rect.w / (a.x1 - a.x0);
  const fov = (Math.min(fovDeg, 80) * Math.PI) / 180;
  c.strokeStyle = THEME.grid;
  c.fillStyle = THEME.dim;
  c.font = THEME.fontSmall;
  c.lineWidth = 1;
  const step = niceStep(rMax / 4);
  for (let r = step; r <= rMax * 1.001; r += step) {
    c.beginPath();
    c.arc(ox, oy, r * sc, -Math.PI / 2 - fov, -Math.PI / 2 + fov);
    c.stroke();
    c.textAlign = 'left';
    c.fillText(`${+r.toFixed(1)} m`, ox + 3 + r * sc * Math.sin(fov) * 0, oy - r * sc - 2);
  }
  for (const deg of [-60, -30, 0, 30, 60]) {
    if (Math.abs(deg) > Math.min(fovDeg, 80)) continue;
    const th = (deg * Math.PI) / 180;
    c.beginPath();
    c.moveTo(ox, oy);
    c.lineTo(ox + rMax * sc * Math.sin(th), oy - rMax * sc * Math.cos(th));
    c.stroke();
    if (deg !== 0) {
      c.textAlign = 'center';
      c.fillText(`${deg}°`, ox + rMax * 1.02 * sc * Math.sin(th), oy - rMax * 1.02 * sc * Math.cos(th) - 4);
    }
  }
  c.restore();
}

function niceStep(x: number): number {
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  const m = x / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

// ------------------------------------------------------------------ 5 Range–angle
export class RAPanel extends CanvasPanel {
  private hm = new Heatmap();
  private ar = new AutoRange();
  private idx: Int32Array | null = null;
  private key = '';
  constructor(app: AppApi) {
    super('ra', 'pRA', app);
  }
  render(): void {
    const f = this.app.frame();
    const c = this.s.begin();
    if (!f) return;
    const { d } = f;
    const { nR, A } = f.meta;
    const rMax = nR * d.rangeRes;
    const rect = plotRect(this.s, { ...MARGIN, r: 44 });
    const fovDisp = Math.min(80, (Math.asin(Math.min(1, 1 / (2 * d.arraySpacing))) * 180) / Math.PI);
    const a = fanAxes(rect, rMax, rMax * Math.sin((fovDisp * Math.PI) / 180) * 1.04, 0.04);
    this.axes = a;
    // index table at ≤ 2 css px resolution
    const W = Math.max(16, Math.min(480, Math.round(rect.w / 1.5))),
      Hh = Math.max(16, Math.min(360, Math.round(rect.h / 1.5)));
    const key = `${W}x${Hh}|${nR}|${A}|${d.rangeRes}|${d.arraySpacing}|${rect.w}|${rect.h}`;
    if (key !== this.key) {
      this.key = key;
      this.hm.ensure(W, Hh);
      this.idx = new Int32Array(W * Hh);
      const sMax = Math.sin((fovDisp * Math.PI) / 180);
      for (let row = 0; row < Hh; row++)
        for (let col = 0; col < W; col++) {
          const x = a.ix(rect.x + ((col + 0.5) / W) * rect.w),
            y = a.iy(rect.y + ((row + 0.5) / Hh) * rect.h);
          const r = Math.hypot(x, y);
          let v = -1;
          if (y > 0 && r < rMax && r > 0) {
            const sn = x / r;
            if (Math.abs(sn) <= sMax) {
              const ab = Math.round(sn * A * d.arraySpacing + A / 2),
                rb = Math.round(r / d.rangeRes);
              if (ab >= 0 && ab < A && rb < nR) v = rb * A + ab;
            }
          }
          this.idx[row * W + col] = v;
        }
    }
    const [lo, hi] = this.ar.update(f.bufs.raDb, 0.6, 45, 25);
    this.hm.fillIndexed(f.bufs.raDb, this.idx!, lo, hi, this.app.cmap());
    c.fillStyle = '#080b10';
    c.fillRect(rect.x, rect.y, rect.w, rect.h);
    this.hm.draw(c, rect, true);
    drawFanGrid(c, a, rMax, fovDisp);
    drawAxes(c, a, `${t('xAxis')} [m]`, `${t('yAxis')} [m]`, { grid: false });
    drawColorbar(c, { x: rect.x + rect.w + 6, y: rect.y, w: 8, h: rect.h }, lo, hi, this.app.cmap());
    c.save();
    c.beginPath();
    c.rect(rect.x, rect.y, rect.w, rect.h);
    c.clip();
    c.strokeStyle = '#fff';
    for (const tr of f.meta.truth) {
      const x = a.px(tr.x),
        y = a.py(tr.y);
      c.beginPath();
      c.moveTo(x - 5, y);
      c.lineTo(x + 5, y);
      c.moveTo(x, y - 5);
      c.lineTo(x, y + 5);
      c.stroke();
    }
    c.strokeStyle = THEME.bad;
    for (const dt of f.meta.detections) {
      c.beginPath();
      c.arc(a.px(dt.x), a.py(dt.y), 4, 0, 7);
      c.stroke();
    }
    c.restore();
  }
  protected hover(p: { x: number; y: number }): void {
    const f = this.app.frame();
    const a = this.axes;
    if (!f || !a || !this.inside(p)) return this.setReadout('');
    const x = a.ix(p.x),
      y = a.iy(p.y);
    const r = Math.hypot(x, y),
      th = (Math.atan2(x, y) * 180) / Math.PI;
    const rb = Math.round(r / f.d.rangeRes),
      ab = Math.round(Math.sin((th * Math.PI) / 180) * f.meta.A * f.d.arraySpacing + f.meta.A / 2);
    const v = rb < f.meta.nR && ab >= 0 && ab < f.meta.A ? f.bufs.raDb[rb * f.meta.A + ab].toFixed(1) + ' dB' : '';
    this.setReadout(`R = ${r.toFixed(1)} m · θ = ${th.toFixed(1)}° ${v ? '· ' + v : ''}`);
  }
}

// ------------------------------------------------------------------ 6 Bird's-eye
export class BevPanel extends CanvasPanel {
  private drag: { id: number; dx: number; dy: number } | null = null;
  constructor(app: AppApi) {
    super('bev', 'pBev', app);
    const cv = this.s.canvas;
    cv.addEventListener('pointerdown', (e) => {
      const f = this.app.frame();
      const p = this.local(e);
      if (!f || !this.axes) return;
      const hit = this.pick(f, p);
      if (hit) {
        this.app.select(hit.id);
        this.drag = { id: hit.id, dx: this.axes.ix(p.x) - hit.x, dy: this.axes.iy(p.y) - hit.y };
        cv.setPointerCapture(e.pointerId);
        cv.style.cursor = 'grabbing';
      }
    });
    cv.addEventListener('pointermove', (e) => {
      const p = this.local(e);
      const f = this.app.frame();
      if (this.drag && this.axes) {
        const x = this.axes.ix(p.x) - this.drag.dx,
          y = Math.max(0.5, this.axes.iy(p.y) - this.drag.dy);
        this.app.dragTarget(this.drag.id, x, y, false);
      } else if (f) cv.style.cursor = this.pick(f, p) ? 'grab' : 'crosshair';
    });
    const end = (e: PointerEvent) => {
      if (this.drag && this.axes) {
        const p = this.local(e);
        this.app.dragTarget(this.drag.id, this.axes.ix(p.x) - this.drag.dx, Math.max(0.5, this.axes.iy(p.y) - this.drag.dy), true);
      }
      this.drag = null;
      cv.style.cursor = 'crosshair';
    };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
  }
  private pick(f: Frame, p: { x: number; y: number }): TruthInfo | null {
    const a = this.axes!;
    let best: TruthInfo | null = null,
      bd = 16;
    for (const tr of f.meta.truth) {
      const dd = Math.hypot(a.px(tr.x) - p.x, a.py(tr.y) - p.y);
      if (dd < bd) {
        bd = dd;
        best = tr;
      }
    }
    return best;
  }
  render(): void {
    const f = this.app.frame();
    const c = this.s.begin();
    if (!f) return;
    const { d } = f;
    const rMax = d.rMax;
    const rect = plotRect(this.s, { ...MARGIN, t: 26 });
    const a = fanAxes(rect, rMax, rMax * 0.42);
    this.axes = a;
    const sc = rect.w / (a.x1 - a.x0);
    c.fillStyle = '#0a0e13';
    c.fillRect(rect.x, rect.y, rect.w, rect.h);
    c.save();
    c.beginPath();
    c.rect(rect.x, rect.y, rect.w, rect.h);
    c.clip();
    // FOV wedge
    const fov = (Math.min(d.fovDeg, 80) * Math.PI) / 180;
    const ox = a.px(0),
      oy = a.py(0);
    const g = c.createRadialGradient(ox, oy, 0, ox, oy, rMax * sc);
    g.addColorStop(0, 'rgba(56,189,248,0.16)');
    g.addColorStop(1, 'rgba(56,189,248,0.03)');
    c.fillStyle = g;
    c.beginPath();
    c.moveTo(ox, oy);
    c.arc(ox, oy, rMax * sc, -Math.PI / 2 - fov, -Math.PI / 2 + fov);
    c.closePath();
    c.fill();
    c.strokeStyle = 'rgba(56,189,248,0.35)';
    c.stroke();
    drawFanGrid(c, a, rMax, Math.min(d.fovDeg, 80));
    // radar
    c.fillStyle = THEME.accent;
    c.fillRect(ox - 7, oy - 3, 14, 6);
    // tracks: trails
    for (const tk of f.meta.tracks) {
      if (!tk.confirmed) continue;
      c.strokeStyle = 'rgba(52,211,153,0.45)';
      c.lineWidth = 1.5;
      c.beginPath();
      for (let i = 0; i < tk.hist.length; i += 2) {
        const x = a.px(tk.hist[i]),
          y = a.py(tk.hist[i + 1]);
        if (i) c.lineTo(x, y);
        else c.moveTo(x, y);
      }
      c.stroke();
    }
    // ground truth
    const sel = this.app.selectedId();
    for (const tr of f.meta.truth) this.drawTruth(c, a, sc, tr, tr.id === sel);
    // detections
    for (const dt of f.meta.detections) {
      c.fillStyle = VCOL(dt.velocity, d.vMax, 0.95);
      c.beginPath();
      c.arc(a.px(dt.x), a.py(dt.y), 3, 0, 7);
      c.fill();
    }
    // tracks
    c.font = THEME.font;
    for (const tk of f.meta.tracks) {
      const x = a.px(tk.x),
        y = a.py(tk.y);
      if (!tk.confirmed) {
        c.strokeStyle = 'rgba(148,163,184,0.5)';
        c.strokeRect(x - 4, y - 4, 8, 8);
        continue;
      }
      c.strokeStyle = THEME.good;
      c.lineWidth = 2;
      c.strokeRect(x - 6, y - 6, 12, 12);
      c.beginPath();
      c.moveTo(x, y);
      c.lineTo(x + tk.vx * sc * 0.6, y - tk.vy * sc * 0.6);
      c.stroke();
      c.fillStyle = THEME.good;
      c.textAlign = 'left';
      c.textBaseline = 'bottom';
      c.fillText(`T${tk.id}`, x + 8, y - 6);
    }
    c.restore();
    drawAxes(c, a, `${t('xAxis')} [m]`, `${t('yAxis')} [m]`, { grid: false });
    const conf = f.meta.tracks.filter((q) => q.confirmed).length;
    legend(c, rect.x, 12, [
      { label: t('truth'), color: '#e2e8f0', marker: 'plus' },
      { label: t('dets'), color: VCOL(-5, 10), marker: 'dot' },
      { label: `${t('tracks')} (${conf})`, color: THEME.good, marker: 'square' },
    ]);
  }
  private drawTruth(c: CanvasRenderingContext2D, a: Axes, sc: number, tr: TruthInfo, sel: boolean): void {
    const x = a.px(tr.x),
      y = a.py(tr.y);
    c.save();
    c.translate(x, y);
    c.strokeStyle = sel ? THEME.accent2 : 'rgba(226,232,240,0.85)';
    c.fillStyle = sel ? 'rgba(245,158,11,0.18)' : 'rgba(226,232,240,0.08)';
    c.lineWidth = sel ? 2 : 1.3;
    const sp = Math.hypot(tr.vx, tr.vy);
    const hd = sp > 1e-6 ? Math.atan2(tr.vx, tr.vy) : 0;
    switch (tr.type) {
      case 'car': {
        const L = Math.max(14, 4.6 * sc),
          W = Math.max(7, 1.8 * sc);
        c.rotate(hd);
        c.beginPath();
        c.rect(-W / 2, -L / 2, W, L);
        c.fill();
        c.stroke();
        c.beginPath();
        c.moveTo(-W / 2, -L / 2 + 3);
        c.lineTo(W / 2, -L / 2 + 3);
        c.stroke();
        c.rotate(-hd);
        break;
      }
      case 'pedestrian':
        c.beginPath();
        c.arc(0, 0, 5, 0, 7);
        c.fill();
        c.stroke();
        c.beginPath();
        c.arc(0, 0, 1.5, 0, 7);
        c.stroke();
        break;
      case 'drone':
        c.beginPath();
        for (const q of [-1, 1]) {
          c.moveTo(-6, -6 * q);
          c.lineTo(6, 6 * q);
        }
        c.stroke();
        for (const [px, py] of [
          [-6, -6],
          [6, 6],
          [-6, 6],
          [6, -6],
        ]) {
          c.beginPath();
          c.arc(px, py, 3.2, 0, 7);
          c.stroke();
        }
        break;
      case 'bird':
        c.beginPath();
        c.moveTo(-8, -3);
        c.quadraticCurveTo(-4, -6, 0, 0);
        c.quadraticCurveTo(4, -6, 8, -3);
        c.stroke();
        break;
      default:
        c.beginPath();
        c.moveTo(0, -6);
        c.lineTo(6, 0);
        c.lineTo(0, 6);
        c.lineTo(-6, 0);
        c.closePath();
        c.fill();
        c.stroke();
    }
    // velocity arrow
    if (sp > 0.05) {
      c.strokeStyle = sel ? THEME.accent2 : 'rgba(226,232,240,0.6)';
      c.lineWidth = 1;
      const ex = tr.vx * sc * 0.6,
        ey = -tr.vy * sc * 0.6;
      c.beginPath();
      c.moveTo(0, 0);
      c.lineTo(ex, ey);
      const an = Math.atan2(ey, ex);
      c.lineTo(ex - 5 * Math.cos(an - 0.4), ey - 5 * Math.sin(an - 0.4));
      c.moveTo(ex, ey);
      c.lineTo(ex - 5 * Math.cos(an + 0.4), ey - 5 * Math.sin(an + 0.4));
      c.stroke();
    }
    c.font = THEME.fontSmall;
    c.fillStyle = sel ? THEME.accent2 : THEME.dim;
    c.textAlign = 'left';
    c.textBaseline = 'top';
    c.fillText(typeName(tr), 9, 6);
    c.restore();
  }
  protected hover(p: { x: number; y: number }): void {
    const a = this.axes;
    if (!a || !this.inside(p)) return this.setReadout('');
    const x = a.ix(p.x),
      y = a.iy(p.y);
    this.setReadout(`x = ${x.toFixed(1)} m · y = ${y.toFixed(1)} m · R = ${Math.hypot(x, y).toFixed(1)} m · θ = ${((Math.atan2(x, y) * 180) / Math.PI).toFixed(1)}°`);
  }
}

// ------------------------------------------------------------------ 7 Micro-Doppler
export class MicroPanel extends CanvasPanel {
  private hm = new Heatmap();
  private ar = new AutoRange();
  constructor(app: AppApi) {
    super('micro', 'pMicro', app);
  }
  render(): void {
    const f = this.app.frame();
    const c = this.s.begin();
    if (!f) return;
    const { d } = f;
    const { specCols: cols, specNfft: nf, M, specWin } = f.meta;
    const cpf = Math.floor((M - specWin) / (specWin >> 2)) + 1;
    const vSpan = d.lambda / (4 * d.te);
    const rect = plotRect(this.s, { ...MARGIN, r: 44, t: 22 });
    const a = makeAxes(rect, -cols / cpf, 0, -vSpan, vSpan);
    this.axes = a;
    const [lo, hi] = this.ar.update(f.bufs.specDb, 0.6, 45, 25);
    this.hm.ensure(cols, nf);
    this.hm.fill(f.bufs.specDb, (col, row) => col * nf + (nf - 1 - row), lo, hi, 'inferno');
    this.hm.draw(c, rect, true);
    drawAxes(c, a, `${t('time')} [${t('frames')}]`, `${t('velocity')} [m/s]`, { grid: false });
    drawColorbar(c, { x: rect.x + rect.w + 6, y: rect.y, w: 8, h: rect.h }, lo, hi, 'inferno');
    const tr = selTruth(f, this.app.selectedId());
    c.font = THEME.fontSmall;
    c.fillStyle = THEME.text;
    c.textAlign = 'left';
    c.textBaseline = 'middle';
    if (tr) {
      const extra = tr.maxMicro > 0.1 ? ` · ${t('microParts', { v: tr.maxMicro.toFixed(1) })}` : '';
      c.fillText(`${t('selected')}: ${typeName(tr)} · R = ${tr.range.toFixed(1)} m${extra}`, rect.x, 11);
    } else c.fillText(t('noSel'), rect.x, 11);
  }
  protected hover(p: { x: number; y: number }): void {
    const a = this.axes;
    if (!a || !this.inside(p)) return this.setReadout('');
    this.setReadout(`v = ${a.iy(p.y).toFixed(2)} m/s · t = ${a.ix(p.x).toFixed(1)} ${t('frames')}`);
  }
}

// ------------------------------------------------------------------ 8 Budget
export class BudgetPanel extends Panel {
  constructor(app: AppApi) {
    super('budget', 'pBudget', app);
    this.body.classList.add('budget');
  }
  render(): void {
    const f = this.app.frame();
    if (!f) return;
    const { d, cfg, meta } = f;
    const row = (k: string, v: string) => `<tr><th>${t(k)}</th><td>${v}</td></tr>`;
    const fmtRate = (b: number) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB/s` : `${(b / 1e6).toFixed(1)} MB/s`);
    const tbl =
      '<table class="kv">' +
      row('rangeRes', `${d.rangeRes.toFixed(3)} m`) +
      row('rMax', `${d.rMax.toFixed(1)} m`) +
      row('vRes', `${d.dv.toFixed(3)} m/s`) +
      row('vMax', `${d.vMax.toFixed(2)} m/s`) +
      row('angRes', `${d.angleResDeg.toFixed(1)}° (K=${d.nVirt})`) +
      row('fovL', `±${d.fovDeg.toFixed(0)}°`) +
      row('frameTime', `${(d.frameActive * 1e3).toFixed(2)} / ${(cfg.framePeriod * 1e3).toFixed(0)} ms`) +
      row('dataRate', fmtRate(d.dataRate)) +
      row('bEff', `${(d.bEff / 1e6).toFixed(0)} MHz`) +
      row('lambda', `${(d.lambda * 1e3).toFixed(2)} mm`) +
      row('cube', `${d.nVirt}×${cfg.nChirps}×${cfg.nSamples}`) +
      '</table>';
    const ws = this.app.warnings();
    const warn =
      `<h3>${t('warnings')}</h3>` +
      (ws.length ? '<ul class="warn">' + ws.map((w) => `<li class="${w.level}">${t(w.key, w.args)}</li>`).join('') + '</ul>' : `<p class="ok">${t('noWarnings')}</p>`);
    const tm = meta.timings;
    const tot = STAGES.reduce((s, k) => s + tm[k], 0);
    const mx = Math.max(1, ...STAGES.map((k) => tm[k]));
    const bars =
      `<h3>${t('timings')}</h3><div class="tsum">${t('total')} <b>${tot.toFixed(1)} ms</b> · ${t('fpsSim')} <b>${this.app.simFps().toFixed(0)}</b> · ${t('fpsUi')} <b>${this.app.uiFps().toFixed(0)}</b></div><div class="bars">` +
      STAGES.map((k) => `<div class="bar"><span>${k}</span><i style="width:${((tm[k] / mx) * 100).toFixed(1)}%"></i><b>${tm[k].toFixed(1)}</b></div>`).join('') +
      '</div>';
    this.body.innerHTML = `<div class="bcol">${tbl}</div><div class="bcol">${bars}</div><div class="bwarn">${warn}</div>`;
  }
}
