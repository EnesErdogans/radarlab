// Radix-2 iterative complex FFT with cached twiddles and bit-reversal table.
// Forward transform uses exp(-j2πkn/N) (brief convention). Split re/im storage.

export class FFT {
  readonly n: number;
  readonly log2n: number;
  private cos: Float64Array;
  private sin: Float64Array;
  private rev: Uint32Array;

  constructor(n: number) {
    if (n < 1 || (n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of 2, got ${n}`);
    this.n = n;
    this.log2n = Math.round(Math.log2(n));
    const half = n >> 1;
    this.cos = new Float64Array(Math.max(1, half));
    this.sin = new Float64Array(Math.max(1, half));
    for (let i = 0; i < half; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = -Math.sin((2 * Math.PI * i) / n);
    }
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < this.log2n; b++) r |= ((i >> b) & 1) << (this.log2n - 1 - b);
      this.rev[i] = r;
    }
  }

  /** In-place forward FFT of re/im[off .. off+n). */
  forward(re: Float64Array | Float32Array, im: Float64Array | Float32Array, off = 0): void {
    this.transform(re, im, off, false);
  }

  /** In-place inverse FFT (scaled by 1/n). */
  inverse(re: Float64Array | Float32Array, im: Float64Array | Float32Array, off = 0): void {
    this.transform(re, im, off, true);
    const s = 1 / this.n;
    for (let i = 0; i < this.n; i++) {
      re[off + i] *= s;
      im[off + i] *= s;
    }
  }

  private transform(re: Float64Array | Float32Array, im: Float64Array | Float32Array, off: number, inv: boolean): void {
    const n = this.n;
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const a = off + i,
          b = off + j;
        let t = re[a];
        re[a] = re[b];
        re[b] = t;
        t = im[a];
        im[a] = im[b];
        im[b] = t;
      }
    }
    const cs = this.cos;
    const sn = this.sin;
    const sgn = inv ? -1 : 1;
    let first = 2;
    if (n >= 4) {
      // fused size-2 and size-4 stages (twiddles 1 and ∓j: no multiplications)
      for (let s0 = off; s0 < off + n; s0 += 4) {
        const ar = re[s0] + re[s0 + 1],
          ai = im[s0] + im[s0 + 1];
        const br = re[s0] - re[s0 + 1],
          bi = im[s0] - im[s0 + 1];
        const cr = re[s0 + 2] + re[s0 + 3],
          ci = im[s0 + 2] + im[s0 + 3];
        const dr = re[s0 + 2] - re[s0 + 3],
          di = im[s0 + 2] - im[s0 + 3];
        // d·w, w = exp(∓jπ/2): forward → (di, −dr), inverse → (−di, dr)
        const er = sgn * di,
          ei = -sgn * dr;
        re[s0] = ar + cr;
        im[s0] = ai + ci;
        re[s0 + 2] = ar - cr;
        im[s0 + 2] = ai - ci;
        re[s0 + 1] = br + er;
        im[s0 + 1] = bi + ei;
        re[s0 + 3] = br - er;
        im[s0 + 3] = bi - ei;
      }
      first = 8;
    }
    for (let size = first; size <= n; size <<= 1) {
      const halfSize = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        let k = 0;
        for (let j = 0; j < halfSize; j++) {
          const wr = cs[k];
          const wi = sgn * sn[k];
          k += step;
          const a = off + start + j;
          const b = a + halfSize;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }
}

const cache = new Map<number, FFT>();
export function getFFT(n: number): FFT {
  let f = cache.get(n);
  if (!f) {
    f = new FFT(n);
    cache.set(n, f);
  }
  return f;
}

/** Naive DFT (for tests). */
export function dft(re: ArrayLike<number>, im: ArrayLike<number>): { re: Float64Array; im: Float64Array } {
  const n = re.length;
  const or = new Float64Array(n);
  const oi = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let sr = 0,
      si = 0;
    for (let t = 0; t < n; t++) {
      const a = (-2 * Math.PI * ((k * t) % n)) / n;
      const c = Math.cos(a),
        s = Math.sin(a);
      sr += re[t] * c - im[t] * s;
      si += re[t] * s + im[t] * c;
    }
    or[k] = sr;
    oi[k] = si;
  }
  return { re: or, im: oi };
}

/** In-place fftshift of a length-n segment (n even). */
export function fftshiftInPlace(a: Float64Array | Float32Array, off: number, n: number): void {
  const h = n >> 1;
  for (let i = 0; i < h; i++) {
    const t = a[off + i];
    a[off + i] = a[off + i + h];
    a[off + i + h] = t;
  }
}
