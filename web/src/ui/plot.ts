// Minimal Canvas2D plotting toolkit: HiDPI canvas, axes with nice ticks, heatmaps.

import { lut, type CmapName } from './colormap';

export const THEME = {
  bg: '#0b0f14',
  panel: '#11161d',
  grid: 'rgba(148,163,184,0.10)',
  axis: 'rgba(148,163,184,0.55)',
  text: '#c9d4e0',
  dim: '#7d8a99',
  accent: '#38bdf8',
  accent2: '#f59e0b',
  good: '#34d399',
  bad: '#f87171',
  violet: '#a78bfa',
  pink: '#f472b6',
  font: '11px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
  fontSmall: '10px "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
};

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export class Surface {
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  w = 0; // CSS px
  h = 0;
  dpr = 1;
  onResize: (() => void) | null = null;
  constructor(parent: HTMLElement) {
    this.canvas = document.createElement('canvas');
    parent.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
    const ro = new ResizeObserver(() => this.resize());
    ro.observe(parent);
    this.resize();
  }
  resize(): void {
    const p = this.canvas.parentElement!;
    const w = Math.max(10, p.clientWidth),
      h = Math.max(10, p.clientHeight);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (w === this.w && h === this.h && dpr === this.dpr) return;
    this.w = w;
    this.h = h;
    this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.onResize?.();
  }
  begin(): CanvasRenderingContext2D {
    const c = this.ctx;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.w, this.h);
    return c;
  }
}

export function niceTicks(lo: number, hi: number, target = 6): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return (+v.toFixed(1)).toString();
  if (a >= 1) return (+v.toFixed(2)).toString();
  return (+v.toPrecision(2)).toString();
}

export interface Axes {
  rect: Rect;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  px(x: number): number;
  py(y: number): number;
  ix(px: number): number;
  iy(py: number): number;
}

export function makeAxes(rect: Rect, x0: number, x1: number, y0: number, y1: number): Axes {
  return {
    rect,
    x0,
    x1,
    y0,
    y1,
    px: (x) => rect.x + ((x - x0) / (x1 - x0)) * rect.w,
    py: (y) => rect.y + rect.h - ((y - y0) / (y1 - y0)) * rect.h,
    ix: (px) => x0 + ((px - rect.x) / rect.w) * (x1 - x0),
    iy: (py) => y0 + ((rect.y + rect.h - py) / rect.h) * (y1 - y0),
  };
}

export const MARGIN = { l: 50, r: 12, t: 10, b: 34 };

export function plotRect(s: Surface, m = MARGIN): Rect {
  return { x: m.l, y: m.t, w: Math.max(10, s.w - m.l - m.r), h: Math.max(10, s.h - m.t - m.b) };
}

export function drawAxes(c: CanvasRenderingContext2D, a: Axes, xLabel: string, yLabel: string, opts: { grid?: boolean; xTicks?: number; yTicks?: number } = {}): void {
  const r = a.rect;
  c.save();
  c.font = THEME.font;
  c.lineWidth = 1;
  const xt = niceTicks(Math.min(a.x0, a.x1), Math.max(a.x0, a.x1), opts.xTicks ?? Math.max(3, Math.floor(r.w / 70)));
  const yt = niceTicks(Math.min(a.y0, a.y1), Math.max(a.y0, a.y1), opts.yTicks ?? Math.max(3, Math.floor(r.h / 40)));
  if (opts.grid !== false) {
    c.strokeStyle = THEME.grid;
    c.beginPath();
    for (const v of xt) {
      const x = Math.round(a.px(v)) + 0.5;
      c.moveTo(x, r.y);
      c.lineTo(x, r.y + r.h);
    }
    for (const v of yt) {
      const y = Math.round(a.py(v)) + 0.5;
      c.moveTo(r.x, y);
      c.lineTo(r.x + r.w, y);
    }
    c.stroke();
  }
  c.strokeStyle = THEME.axis;
  c.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  c.fillStyle = THEME.dim;
  c.textAlign = 'center';
  c.textBaseline = 'top';
  for (const v of xt) {
    const x = a.px(v);
    c.fillText(fmt(v), x, r.y + r.h + 4);
  }
  c.textAlign = 'right';
  c.textBaseline = 'middle';
  for (const v of yt) c.fillText(fmt(v), r.x - 5, a.py(v));
  c.fillStyle = THEME.text;
  c.textAlign = 'center';
  c.textBaseline = 'bottom';
  c.fillText(xLabel, r.x + r.w / 2, r.y + r.h + 32);
  c.save();
  c.translate(12, r.y + r.h / 2);
  c.rotate(-Math.PI / 2);
  c.textBaseline = 'middle';
  c.fillText(yLabel, 0, 0);
  c.restore();
  c.restore();
}

/** Off-screen heatmap image; data[row·cols + col] mapped by (lo..hi) through a LUT. */
export class Heatmap {
  private cv: HTMLCanvasElement | OffscreenCanvas | null = null;
  private img: ImageData | null = null;
  private u32: Uint32Array | null = null;
  cols = 0;
  rows = 0;
  ensure(cols: number, rows: number): void {
    if (cols === this.cols && rows === this.rows && this.img) return;
    this.cols = cols;
    this.rows = rows;
    this.cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(cols, rows) : Object.assign(document.createElement('canvas'), { width: cols, height: rows });
    const ctx = this.cv.getContext('2d') as CanvasRenderingContext2D;
    this.img = ctx.createImageData(cols, rows);
    this.u32 = new Uint32Array(this.img.data.buffer);
  }
  /**
   * Fill from a source with an index mapper: pixel (col,row) top-left origin → src index (or −1 = transparent).
   */
  fill(src: Float32Array, map: (col: number, row: number) => number, lo: number, hi: number, cmap: CmapName): void {
    const L = lut(cmap),
      u = this.u32!;
    const k = 255 / Math.max(1e-9, hi - lo);
    let p = 0;
    for (let row = 0; row < this.rows; row++)
      for (let col = 0; col < this.cols; col++, p++) {
        const i = map(col, row);
        if (i < 0) {
          u[p] = 0;
          continue;
        }
        let v = (src[i] - lo) * k;
        v = v < 0 ? 0 : v > 255 ? 255 : v;
        u[p] = L[v | 0];
      }
    this.commit();
  }
  /** Fast path for a precomputed index table. */
  fillIndexed(src: Float32Array, idx: Int32Array, lo: number, hi: number, cmap: CmapName): void {
    const L = lut(cmap),
      u = this.u32!;
    const k = 255 / Math.max(1e-9, hi - lo);
    for (let p = 0; p < idx.length; p++) {
      const i = idx[p];
      if (i < 0) {
        u[p] = 0;
        continue;
      }
      let v = (src[i] - lo) * k;
      v = v < 0 ? 0 : v > 255 ? 255 : v;
      u[p] = L[v | 0];
    }
    this.commit();
  }
  private commit(): void {
    (this.cv!.getContext('2d') as CanvasRenderingContext2D).putImageData(this.img!, 0, 0);
  }
  draw(c: CanvasRenderingContext2D, r: Rect, smooth = false): void {
    if (!this.cv) return;
    c.save();
    c.imageSmoothingEnabled = smooth;
    c.drawImage(this.cv as CanvasImageSource, r.x, r.y, r.w, r.h);
    c.restore();
  }
}

/** Smoothed auto color range (EMA) from a sampled percentile and the max. */
export class AutoRange {
  lo = NaN;
  hi = NaN;
  update(data: Float32Array, pct = 0.5, dyn = 50, minSpan = 25): [number, number] {
    const n = data.length;
    const step = Math.max(1, Math.floor(n / 2048));
    const s: number[] = [];
    let mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = data[i];
      if (v > mx) mx = v;
      if (i % step === 0 && v > -250) s.push(v);
    }
    s.sort((a, b) => a - b);
    const med = s.length ? s[Math.floor(pct * (s.length - 1))] : mx - dyn;
    const lo = Math.max(med - 3, mx - dyn);
    const hi = Math.max(mx, lo + minSpan);
    const a = isNaN(this.lo) ? 1 : 0.25;
    this.lo = isNaN(this.lo) ? lo : this.lo + a * (lo - this.lo);
    this.hi = isNaN(this.hi) ? hi : this.hi + a * (hi - this.hi);
    return [this.lo, this.hi];
  }
}

export function drawColorbar(c: CanvasRenderingContext2D, r: Rect, lo: number, hi: number, cmap: CmapName, unit = 'dB'): void {
  const L = lut(cmap);
  for (let i = 0; i < r.h; i++) {
    const v = L[Math.round((1 - i / r.h) * 255)];
    c.fillStyle = `rgb(${v & 255},${(v >> 8) & 255},${(v >> 16) & 255})`;
    c.fillRect(r.x, r.y + i, r.w, 1.5);
  }
  c.strokeStyle = THEME.axis;
  c.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
  c.font = THEME.fontSmall;
  c.fillStyle = THEME.dim;
  c.textAlign = 'left';
  c.textBaseline = 'top';
  c.fillText(hi.toFixed(0), r.x + r.w + 3, r.y);
  c.textBaseline = 'bottom';
  c.fillText(lo.toFixed(0), r.x + r.w + 3, r.y + r.h);
  c.textBaseline = 'middle';
  c.fillText(unit, r.x + r.w + 3, r.y + r.h / 2);
}

export function legend(c: CanvasRenderingContext2D, x: number, y: number, items: { label: string; color: string; dash?: number[]; width?: number; marker?: 'dot' | 'circle' | 'plus' | 'square' }[]): void {
  c.save();
  c.font = THEME.fontSmall;
  c.textBaseline = 'middle';
  let cx = x;
  for (const it of items) {
    c.strokeStyle = it.color;
    c.fillStyle = it.color;
    c.lineWidth = it.width ?? 1.5;
    c.setLineDash(it.dash ?? []);
    if (it.marker === 'dot') {
      c.beginPath();
      c.arc(cx + 7, y, 3, 0, 7);
      c.fill();
    } else if (it.marker === 'circle') {
      c.beginPath();
      c.arc(cx + 7, y, 4, 0, 7);
      c.stroke();
    } else if (it.marker === 'plus') {
      c.beginPath();
      c.moveTo(cx + 3, y);
      c.lineTo(cx + 11, y);
      c.moveTo(cx + 7, y - 4);
      c.lineTo(cx + 7, y + 4);
      c.stroke();
    } else if (it.marker === 'square') {
      c.strokeRect(cx + 3, y - 4, 8, 8);
    } else {
      c.beginPath();
      c.moveTo(cx, y);
      c.lineTo(cx + 14, y);
      c.stroke();
    }
    c.setLineDash([]);
    c.fillStyle = THEME.text;
    c.fillText(it.label, cx + 18, y);
    cx += 24 + c.measureText(it.label).width + 8;
  }
  c.restore();
}
