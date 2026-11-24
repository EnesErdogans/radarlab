// 256-entry RGB lookup tables from 6th-order polynomial fits of matplotlib's
// viridis / inferno (fits by Matt Zucker, public domain).

export type CmapName = 'viridis' | 'inferno';

const COEF: Record<CmapName, number[][]> = {
  viridis: [
    [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
    [0.1050930431085774, 1.404613529898575, 1.384590162594685],
    [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
    [-4.634230498983486, -5.799100973351585, -19.33244095627987],
    [6.228269936347081, 14.17993336680509, 56.69055260068105],
    [4.776384997670288, -13.74514537774601, -65.35303263337234],
    [-5.435455855934631, 4.645852612178535, 26.3124352495832],
  ],
  inferno: [
    [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
    [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
    [11.60249308247187, -3.972853965665698, -15.9423941062914],
    [-41.70399613139459, 17.43639888205313, 44.35414519872813],
    [77.162935699427, -33.40235894210092, -81.80730925738993],
    [-71.31942824499214, 32.62606426397723, 73.20951985803202],
    [25.13112622477341, -12.24266895238567, -23.07032500287172],
  ],
};

const cache = new Map<CmapName, Uint32Array>();

/** LUT of packed little-endian RGBA (for Uint32 views on ImageData). */
export function lut(name: CmapName): Uint32Array {
  let l = cache.get(name);
  if (l) return l;
  l = new Uint32Array(256);
  const c = COEF[name];
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    const rgb = [0, 1, 2].map((ch) => {
      let v = 0;
      for (let k = 6; k >= 0; k--) v = v * x + c[k][ch];
      return Math.max(0, Math.min(255, Math.round(v * 255)));
    });
    l[i] = (255 << 24) | (rgb[2] << 16) | (rgb[1] << 8) | rgb[0];
  }
  cache.set(name, l);
  return l;
}

export function cssColor(name: CmapName, x: number): string {
  const v = lut(name)[Math.max(0, Math.min(255, Math.round(x * 255)))];
  return `rgb(${v & 255},${(v >> 8) & 255},${(v >> 16) & 255})`;
}
