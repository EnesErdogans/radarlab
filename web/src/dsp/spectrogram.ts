// Rolling micro-Doppler STFT spectrogram of a slow-time sequence.
// Each frame contributes columns computed within that frame's M chirps
// (frames are not contiguous in time, so windows never straddle frames).

import { getFFT } from './fft';
import { makeWindow } from './window';

export class Spectrogram {
  readonly win: number;
  readonly hop: number;
  readonly nfft: number;
  readonly cols: number;
  readonly data: Float64Array; // ring: cols × nfft power (fftshifted: bin nfft/2 = 0 Hz)
  head = 0; // next column to write
  filled = 0;
  private w: Float64Array;
  private re: Float64Array;
  private im: Float64Array;

  constructor(M: number, cols = 192, nfft = 128) {
    this.win = Math.min(64, M >> 1);
    this.hop = this.win >> 2;
    this.nfft = Math.max(nfft, this.win);
    this.cols = cols;
    this.data = new Float64Array(cols * this.nfft);
    this.w = makeWindow('hann', this.win);
    this.re = new Float64Array(this.nfft);
    this.im = new Float64Array(this.nfft);
  }

  columnsPerFrame(M: number): number {
    return Math.floor((M - this.win) / this.hop) + 1;
  }

  reset(): void {
    this.data.fill(0);
    this.head = 0;
    this.filled = 0;
  }

  /** Append STFT columns of sequence s (length M). */
  push(sRe: Float64Array, sIm: Float64Array, M: number): void {
    const fft = getFFT(this.nfft);
    const nf = this.nfft,
      W = this.win;
    for (let st = 0; st + W <= M; st += this.hop) {
      this.re.fill(0);
      this.im.fill(0);
      for (let i = 0; i < W; i++) {
        this.re[i] = sRe[st + i] * this.w[i];
        this.im[i] = sIm[st + i] * this.w[i];
      }
      fft.forward(this.re, this.im);
      const base = this.head * nf;
      for (let b = 0; b < nf; b++) {
        const sb = (b + nf / 2) % nf;
        this.data[base + sb] = this.re[b] * this.re[b] + this.im[b] * this.im[b];
      }
      this.head = (this.head + 1) % this.cols;
      this.filled = Math.min(this.cols, this.filled + 1);
    }
  }

  /** Copy ring into out (oldest column first) as dB. */
  toDb(out: Float32Array): void {
    const nf = this.nfft;
    for (let c = 0; c < this.cols; c++) {
      const src = ((this.head + c) % this.cols) * nf;
      for (let b = 0; b < nf; b++) {
        const v = this.data[src + b];
        out[c * nf + b] = v > 0 ? 10 * Math.log10(v) : -300;
      }
    }
  }

  /** Doppler spread metric: number of bins within `db` dB of the peak, averaged over filled columns. */
  spreadBins(db = 20): number {
    const nf = this.nfft;
    let tot = 0,
      n = 0;
    for (let c = 0; c < this.filled; c++) {
      const col = ((this.head - 1 - c + this.cols * 2) % this.cols) * nf;
      let mx = 0;
      for (let b = 0; b < nf; b++) mx = Math.max(mx, this.data[col + b]);
      if (mx <= 0) continue;
      const lim = mx * Math.pow(10, -db / 10);
      let lo = nf,
        hi = -1;
      for (let b = 0; b < nf; b++)
        if (this.data[col + b] >= lim) {
          lo = Math.min(lo, b);
          hi = Math.max(hi, b);
        }
      tot += hi - lo + 1;
      n++;
    }
    return n ? tot / n : 0;
  }
}
