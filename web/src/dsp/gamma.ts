// Special functions and quadrature for the Gamma(K)-exact CFAR thresholds.
//
// With P = Σ_k |X_k|² over K non-coherently summed unit-power channels the noise
// in each cell is Gamma(K, 1). The exact OS / GO / SO false-alarm probabilities
// are one-dimensional integrals (same definitions as the Python reference
// `radarlab.dsp.cfar`, which integrates over u = F(x) with scipy.quad); here
// they are evaluated in x after the substitution u = F(x), on a log-x grid
// (s = ln x) with composite Gauss–Legendre quadrature refined until the
// relative change is < 1e-13:
//   OS:    Pfa = ∫ Q(K, α·x) · f_(k)(x) dx,   f_(k) = pdf of the k-th smallest of N Gamma(K) cells
//   GO/SO: Pfa = ∫ Q(K, β·y) · w(G(y)) · g(y) dy,  g/G = Gamma(nK) pdf/cdf of a half sum,
//          w(u) = 2u (GO, max of two halves) or 2(1−u) (SO, min), β = α/n.
// Q(a, x) is the regularised upper incomplete gamma function.

const LN_SQRT_2PI = 0.9189385332046728;

/** ln Γ(x) for x > 0 (Stirling series after upward recursion; |err| < 1e-14 relative). */
export function lgamma(x: number): number {
  let shift = 0;
  while (x < 15) {
    shift += Math.log(x);
    x += 1;
  }
  const x2 = 1 / (x * x);
  const series = (1 / 12 - x2 * (1 / 360 - x2 * (1 / 1260 - x2 * (1 / 1680 - x2 / 1188)))) / x;
  return (x - 0.5) * Math.log(x) - x + LN_SQRT_2PI + series - shift;
}

/**
 * ln P(a, x) and ln Q(a, x) (regularised lower / upper incomplete gamma), accurate in
 * both tails: the smaller one is computed directly (series for x < a+1, Lentz continued
 * fraction otherwise), the other as log1p(−small).
 */
export function logGammaPQ(a: number, x: number, lga = lgamma(a)): { lp: number; lq: number } {
  if (x <= 0) return { lp: -Infinity, lq: 0 };
  const lpre = a * Math.log(x) - x - lga;
  if (x < a + 1) {
    let ap = a,
      del = 1 / a,
      sum = del;
    for (let i = 0; i < 100000; i++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (del < sum * 1e-17) break;
    }
    const lp = lpre + Math.log(sum);
    return { lp, lq: Math.log1p(-Math.exp(lp)) };
  }
  // continued fraction for Q (modified Lentz)
  const tiny = 1e-300;
  let b = x + 1 - a,
    c = 1 / tiny,
    d = 1 / b,
    h = d;
  for (let i = 1; i < 100000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  const lq = lpre + Math.log(h);
  return { lp: Math.log1p(-Math.exp(lq)), lq };
}

/** ln Q(a, x) only. */
export function logGammaQ(a: number, x: number, lga = lgamma(a)): number {
  return logGammaPQ(a, x, lga).lq;
}

// Gauss–Legendre 20-point nodes/weights on [−1, 1]
const GL_N = 20;
const GL_X = new Float64Array(GL_N);
const GL_W = new Float64Array(GL_N);
(() => {
  for (let i = 0; i < GL_N; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (GL_N + 0.5));
    let pp = 0;
    for (let it = 0; it < 100; it++) {
      let p1 = 1,
        p2 = 0;
      for (let j = 1; j <= GL_N; j++) {
        const p3 = p2;
        p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      pp = (GL_N * (z * p1 - p2)) / (z * z - 1);
      const z1 = z;
      z = z1 - p1 / pp;
      if (Math.abs(z - z1) < 1e-16) break;
    }
    GL_X[i] = z;
    GL_W[i] = 2 / ((1 - z * z) * pp * pp);
  }
})();

/**
 * ∫₀^∞ exp(logf(x)) dx for a smooth, unimodal-ish positive integrand, evaluated in s = ln x.
 * `center` is a typical scale (e.g. the mean) of x. Returns the natural log of the integral.
 */
export function logIntegratePositive(logf: (x: number) => number, center: number): number {
  const g = (s: number) => {
    const x = Math.exp(s);
    return logf(x) + s; // dx = x ds
  };
  // locate the support on a coarse grid
  const s0 = Math.log(center);
  const lo = s0 - 60,
    hi = s0 + 8,
    step = 0.125;
  const nG = Math.round((hi - lo) / step) + 1;
  let mx = -Infinity,
    imx = 0;
  const vals = new Float64Array(nG);
  for (let i = 0; i < nG; i++) {
    const v = g(lo + i * step);
    vals[i] = isNaN(v) ? -Infinity : v;
    if (vals[i] > mx) {
      mx = vals[i];
      imx = i;
    }
  }
  if (!isFinite(mx)) return -Infinity;
  const cut = mx - 50; // e^-50 ≈ 2e-22 relative
  let ia = imx,
    ib = imx;
  while (ia > 0 && vals[ia] > cut) ia--;
  while (ib < nG - 1 && vals[ib] > cut) ib++;
  const a = lo + ia * step,
    b = lo + ib * step;
  // composite Gauss–Legendre, integrand scaled by e^{-mx}; double panels until converged
  const comp = (panels: number): number => {
    const h = (b - a) / panels;
    let acc = 0;
    for (let p = 0; p < panels; p++) {
      const c = a + (p + 0.5) * h;
      for (let i = 0; i < GL_N; i++) acc += GL_W[i] * Math.exp(g(c + 0.5 * h * GL_X[i]) - mx);
    }
    return acc * 0.5 * h;
  };
  let panels = 4;
  let prev = comp(panels);
  for (let it = 0; it < 8; it++) {
    panels *= 2;
    const cur = comp(panels);
    if (Math.abs(cur - prev) <= 1e-13 * cur) return mx + Math.log(cur);
    prev = cur;
  }
  return mx + Math.log(prev);
}

/** ln B(a, b). */
function lbeta(a: number, b: number): number {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

/**
 * Exact OS-CFAR Pfa for threshold α·X_(k), k-th smallest of n i.i.d. Gamma(K, 1) cells.
 * Returns ln Pfa.
 */
export function osLogPfaGamma(n: number, k: number, K: number, alpha: number): number {
  const lgK = lgamma(K);
  const lb = lbeta(k, n - k + 1);
  const logf = (x: number): number => {
    const { lp, lq } = logGammaPQ(K, x, lgK);
    const lpdf = (K - 1) * Math.log(x) - x - lgK;
    const lthr = logGammaQ(K, alpha * x, lgK);
    return lthr + lpdf + (k - 1) * lp + (n - k) * lq - lb;
  };
  return logIntegratePositive(logf, K);
}

/**
 * Exact GO/SO-CFAR Pfa for threshold β·max/min(S₁, S₂), S_i sums of nHalf i.i.d.
 * Gamma(K, 1) cells (S_i ~ Gamma(nHalf·K)). Returns ln Pfa.
 */
export function gsLogPfaGamma(nHalf: number, K: number, beta: number, kind: 'GO' | 'SO'): number {
  const a = nHalf * K;
  const lga = lgamma(a),
    lgK = lgamma(K);
  const logf = (y: number): number => {
    const { lp, lq } = logGammaPQ(a, y, lga);
    const lpdf = (a - 1) * Math.log(y) - y - lga;
    const lw = Math.LN2 + (kind === 'GO' ? lp : lq);
    return logGammaQ(K, beta * y, lgK) + lpdf + lw;
  };
  return logIntegratePositive(logf, a);
}

/**
 * Solve logPfa(α) = ln(pfa) for α > 0 (logPfa decreasing): bracket by doubling, then
 * Illinois regula falsi on (ln α, ln Pfa), which is nearly linear. Relative tol 1e-14.
 */
export function solveAlpha(logPfa: (a: number) => number, pfa: number): number {
  const target = Math.log(pfa);
  const f = (la: number) => logPfa(Math.exp(la)) - target;
  let hi = 0; // ln α
  let fhi = f(hi);
  let lo = hi,
    flo = fhi;
  if (fhi > 0) {
    while (fhi > 0) {
      lo = hi;
      flo = fhi;
      hi += Math.LN2;
      fhi = f(hi);
    }
  } else {
    while (flo <= 0) {
      hi = lo;
      fhi = flo;
      lo -= Math.LN2;
      flo = f(lo);
      if (lo < -60) return Math.exp(lo);
    }
  }
  // flo > 0 ≥ fhi
  let side = 0,
    last = 0.5 * (lo + hi);
  for (let it = 0; it < 200; it++) {
    let m = (lo * fhi - hi * flo) / (fhi - flo);
    if (!(m > lo && m < hi)) m = 0.5 * (lo + hi);
    const fm = f(m);
    last = m;
    if (Math.abs(fm) < 1e-15) break;
    if (fm > 0) {
      lo = m;
      flo = fm;
      if (side === -1) fhi *= 0.5;
      side = -1;
    } else {
      hi = m;
      fhi = fm;
      if (side === 1) flo *= 0.5;
      side = 1;
    }
    if (hi - lo < 1e-15) break;
  }
  return Math.exp(last);
}
