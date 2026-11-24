// Seeded PRNG: mulberry32 + Box–Muller.

/** mulberry32 by Tommy Ettinger (public domain, https://gist.github.com/tommyettinger/46a874533244883189143505d203312c). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal generator (Box–Muller, both outputs used). */
export class Gaussian {
  private rng: () => number;
  private spare = 0;
  private hasSpare = false;
  constructor(seed: number) {
    this.rng = mulberry32(seed);
  }
  uniform(): number {
    return this.rng();
  }
  next(): number {
    if (this.hasSpare) {
      this.hasSpare = false;
      return this.spare;
    }
    let u1 = this.rng();
    if (u1 < 1e-300) u1 = 1e-300;
    const u2 = this.rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const a = 2 * Math.PI * u2;
    this.spare = r * Math.sin(a);
    this.hasSpare = true;
    return r * Math.cos(a);
  }
}

/**
 * Fast complex white Gaussian noise for the interactive UI (tests use exact sequential
 * Box–Muller instead, `SimSettings.exactNoise`).
 *
 * A pool of N(0,1) values (≥ 2^20, sized to hold at least one frame) is read in blocks
 * of B = 256 complex pairs. For every frame and every output block j the source block
 * is π(j) = (a·j + b) mod n_blocks with a random odd a and random b (a permutation, so a
 * frame that fits in the pool never repeats a value), and the pair is rotated by an
 * independent uniform phase θ_j: n = e^{jθ_j}·(x + j·y). Because θ_j is uniform and
 * independent between frames, E[n_f · conj(n_{f'})] = 0 and E[n_f · n_{f'}] = 0 for any
 * two samples of different frames, i.e. successive frames are uncorrelated (and circular
 * complex Gaussian marginally). In addition 1/16 of the pool is regenerated after every
 * frame, so no pool value lives longer than 16 frames. The old scheme (one fixed pool
 * read at a random offset) made successive drone frames cyclic shifts of each other.
 * Residual higher-order dependence (the same |x+jy| may reappear within 16 frames at a
 * different, randomly permuted cube position) is invisible after the range/Doppler FFTs.
 */
export class NoisePool {
  static readonly BLOCK = 512; // floats per block (256 complex pairs)
  pool: Float32Array;
  private mask: number;
  private g: Gaussian;
  private refillPos = 0;
  constructor(seed: number, minFloats = 0) {
    let n = 1 << 20;
    while (n < minFloats && n < 1 << 23) n <<= 1;
    this.pool = new Float32Array(n);
    this.mask = n - 1;
    this.g = new Gaussian(seed);
    for (let i = 0; i < n; i++) this.pool[i] = this.g.next();
  }
  get size(): number {
    return this.pool.length;
  }
  /** Regenerate `count` pool values (round-robin). */
  refill(count: number): void {
    const p = this.pool,
      m = this.mask,
      g = this.g;
    let o = this.refillPos;
    for (let i = 0; i < count; i++) p[(o + i) & m] = g.next();
    this.refillPos = (o + count) & m;
  }
  /**
   * Add scale·(complex white Gaussian, E|n|² = 2) to re/im (im untouched if `realOnly`;
   * the real part alone is N(0, 1)·scale). Then refresh 1/16 of the pool.
   */
  addTo(re: Float64Array, im: Float64Array, len: number, scale: number, realOnly: boolean): void {
    const B = NoisePool.BLOCK,
      half = B >> 1;
    const nBlocks = this.pool.length / B;
    const bm = nBlocks - 1;
    const g = this.g;
    const a = (Math.floor(g.uniform() * nBlocks) | 1) >>> 0;
    const b = Math.floor(g.uniform() * nBlocks);
    const p = this.pool;
    for (let j = 0, i = 0; i < len; j++) {
      const src = ((Math.imul(a, j) + b) & bm) * B;
      const th = 2 * Math.PI * g.uniform();
      const c = scale * Math.cos(th),
        s = scale * Math.sin(th);
      const end = Math.min(len, i + half);
      let q = src;
      if (realOnly)
        for (; i < end; i++, q += 2) re[i] += c * p[q] - s * p[q + 1];
      else
        for (; i < end; i++, q += 2) {
          const x = p[q],
            y = p[q + 1];
          re[i] += c * x - s * y;
          im[i] += s * x + c * y;
        }
    }
    this.refill(this.pool.length >> 4);
  }
}

/** Hash two integers into a 32-bit seed. */
export function hashSeed(a: number, b: number): number {
  let h = (a ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (b + 0x7f4a7c15), 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}
