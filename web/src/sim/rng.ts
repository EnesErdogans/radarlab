// Seeded PRNG: mulberry32 + Box–Muller.

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

/** Precomputed pool of N(0,1) samples; read with a random offset per frame (UI speed). */
export class GaussianPool {
  readonly pool: Float32Array;
  readonly mask: number;
  private g: Gaussian;
  constructor(seed: number, log2Size = 20) {
    const n = 1 << log2Size;
    this.pool = new Float32Array(n);
    this.mask = n - 1;
    this.g = new Gaussian(seed);
    for (let i = 0; i < n; i++) this.pool[i] = this.g.next();
  }
  randomOffset(): number {
    return Math.floor(this.g.uniform() * this.pool.length);
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
