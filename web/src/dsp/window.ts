import type { WindowType } from '../config/radar';

/** Periodic (DFT-even) windows: w[n] uses 2πn/N. */
export function makeWindow(type: WindowType, n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = (2 * Math.PI * i) / n;
    switch (type) {
      case 'hann':
        w[i] = 0.5 - 0.5 * Math.cos(x);
        break;
      case 'hamming':
        w[i] = 0.54 - 0.46 * Math.cos(x);
        break;
      case 'blackman':
        w[i] = 0.42 - 0.5 * Math.cos(x) + 0.08 * Math.cos(2 * x);
        break;
      default:
        w[i] = 1;
    }
  }
  return w;
}

/** Window normalised for coherent gain: w / Σw (brief: divide by Σw). */
export function makeNormWindow(type: WindowType, n: number): Float64Array {
  const w = makeWindow(type, n);
  let s = 0;
  for (let i = 0; i < n; i++) s += w[i];
  for (let i = 0; i < n; i++) w[i] /= s;
  return w;
}
