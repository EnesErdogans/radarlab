### Noise floor, noise distribution, CFAR false alarms

| scene | data | σ [LSB] | Gamma shape (K=8) | KS D / p | shape / KS p incl. 0-Doppler | Pfa ratio @1e-3 (n) | @1e-4 (n) | @1e-5 (n) | @1e-4 incl. 0-Doppler | noise-floor shape rms err [dB] (|bin|>2) |
|---|---|---|---|---|---|---|---|---|---|---|
| pedestrian | real | 7.62 | 7.95 | 0.0043 / 0.20 | 7.42 / 2e-09 | 1.14 (272) | 1.55 (37) | 1.26 (3) | 21.85 | 0 |
|  | sim ideal | 1 (def.) | 7.98 | 0.0016 / 1.00 | 7.98 / 1 | 1.12 (266) | 1.01 (24) | 1.26 (3) | 0.98 | 2.4 |
|  | sim all | 1 (def.) | 7.91 | 0.0050 / 0.10 | 7.90 / 0.1 | 1.06 (253) | 1.43 (34) | 1.68 (4) | 1.39 | 1.0 |
| cyclist | real | 7.57 | 7.88 | 0.0044 / 0.17 | 7.58 / 3e-05 | 1.12 (267) | 1.09 (26) | 1.26 (3) | 10.54 | 0 |
|  | sim ideal | 1 (def.) | 8.04 | 0.0025 / 0.84 | 8.03 / 0.8 | 1.14 (272) | 1.13 (27) | 1.68 (4) | 1.10 | 2.4 |
|  | sim all | 1 (def.) | 7.86 | 0.0047 / 0.12 | 7.86 / 0.1 | 1.12 (267) | 1.01 (24) | 0.84 (2) | 0.98 | 1.0 |
| car | real | 7.60 | 7.88 | 0.0041 / 0.28 | 7.41 / 3e-05 | 1.18 (281) | 1.55 (37) | 2.52 (6) | 15.85 | 0 |
|  | sim ideal | 1 (def.) | 7.99 | 0.0029 / 0.71 | 0.98 / 0 | 1.09 (259) | 1.47 (35) | 0.42 (1) | 14.38 | 2.4 |
|  | sim all | 1 (def.) | 7.84 | 0.0056 / 0.05 | 0.90 / 0 | 1.22 (291) | 1.22 (29) | 1.26 (3) | 13.56 | 1.0 |


Pfa ratio = empirical/nominal in the target-free region (CA-CFAR, Hann, exact Gamma(8) α; 238080 cells over 30 frames; n = false alarms). Hann-correlated training cells give ≈1.5× in theory (validation V1c).

### Dominant target (strongest CFAR detection per frame)

| scene | data | frames found | R median [m] | dR/dt Theil–Sen [m/s] | v median [m/s] | v IQR | az median [°] | az IQR [°] | peak [dB re noise] (fitted) | peak IQR [dB] | array coherence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pedestrian | real | 22 | 6.95 | -0.64 | -0.38 | 0.36 | -1.9 | 1.2 | 35.4 | 16.5 | 0.916 |
|  | sim ideal | 29 | 7.14 | -0.32 | -0.30 | 0.11 | -1.9 | 0.1 | 35.2 | 3.4 | 1.000 |
|  | sim all | 29 | 7.14 | -0.32 | -0.30 | 0.11 | -1.9 | 0.1 | 35.1 | 3.2 | 0.930 |
| cyclist | real | 30 | 10.43 | -0.41 | -0.52 | 0.27 | -7.5 | 1.9 | 38.6 | 5.7 | 0.927 |
|  | sim ideal | 26 | 10.50 | -0.51 | -0.44 | 0.22 | -7.6 | 0.1 | 40.1 | 4.2 | 1.000 |
|  | sim all | 26 | 10.50 | -0.51 | -0.44 | 0.22 | -7.6 | 0.1 | 40.1 | 4.2 | 0.930 |
| car | real | 30 | 10.71 | -0.00 | 0.00 | 0.00 | -25.2 | 0.1 | 43.4 | 0.2 | 0.926 |
|  | sim ideal | 30 | 10.74 | 0.00 | 0.00 | 0.00 | -26.2 | 0.0 | 43.4 | 0.0 | 0.852 |
|  | sim all | 30 | 10.74 | 0.00 | 0.00 | 0.00 | -26.3 | 0.0 | 44.5 | 0.0 | 0.745 |

### RD-map statistics and impairment ablation (mean over the three scenes)

| data | detections / frame | at 0-Doppler (±2 bins) | near DC (range bin ≤2 or ≥126) | power fraction at 0-Doppler | noise KS p (excl. 0-Doppler) | noise-floor shape rms err [dB] (|bin|>2) | array coherence (ped., cyc.) |
|---|---|---|---|---|---|---|---|
| real | 19.8 | 9.6 | 7.3 | 0.78 | 0.22 | 0 | 0.922 |
| ideal | 3.3 | 1.6 | 0.0 | 0.27 | 0.85 | 2.4 | 1.000 |
| +hpf | 3.7 | 1.6 | 0.0 | 0.28 | 0.25 | 1.0 | 1.000 |
| +dc | 5.3 | 3.6 | 2.0 | 0.78 | 0.85 | 2.4 | 1.000 |
| +iq | 3.5 | 1.7 | 0.0 | 0.27 | 0.80 | 2.4 | 1.000 |
| +mismatch | 3.3 | 1.6 | 0.0 | 0.28 | 0.85 | 2.4 | 0.929 |
| +phase_noise | 3.3 | 1.6 | 0.0 | 0.27 | 0.78 | 2.4 | 1.000 |
| +clutter | 7.6 | 5.8 | 0.0 | 0.51 | 0.85 | 2.4 | 1.000 |
| all | 10.7 | 8.4 | 2.0 | 0.82 | 0.09 | 1.0 | 0.930 |

### Micro-Doppler signature features

| scene | data | torso v [m/s] | torso accel [m/s²] | extent median [m/s] | extent p95 [m/s] | v min p5 [m/s] | v max p95 [m/s] | rms bandwidth [m/s] | limb-envelope period [s] | autocorr. peak |
|---|---|---|---|---|---|---|---|---|---|---|
| pedestrian | real | -0.25 | -0.20 | 0.63 | 1.14 | -0.89 | 0.38 | 0.126 | 0.75 | 0.08 |
|  | sim ideal | -0.25 | -0.17 | 1.08 | 1.77 | -1.27 | 0.51 | 0.237 | 0.83 | 0.21 |
|  | sim all | -0.38 | -0.18 | 1.01 | 1.77 | -1.27 | 0.51 | 0.228 | 0.83 | 0.20 |
| cyclist | real | -0.38 | -0.58 | 1.01 | 1.39 | -1.65 | 0.38 | 0.181 | 0.69 | 0.14 |
|  | sim ideal | -0.38 | -0.49 | 0.63 | 1.01 | -1.27 | -0.25 | 0.096 | 0.45 | 0.12 |
|  | sim all | -0.38 | -0.49 | 0.51 | 0.76 | -1.27 | -0.25 | 0.095 | 0.26 | 0.16 |

### Fitted constants (all fitted to the real data)

| scene | σ² [LSB²] | R(t=0) [m] | v0 [m/s] | a [m/s²] | az [°] | snr_ref_db (1 m² @10 m) | implied RCS [dBsm] | clutter scale ×1e4 |
|---|---|---|---|---|---|---|---|---|
| pedestrian | 58.0 | 7.23 | -0.21 | -0.20 | -1.9 | -7.5 | -8.3 | 3.43 |
| cyclist | 57.4 | 10.57 | -0.11 | -0.58 | -7.5 | 1.6 | 0.8 | 4.59 |
| car | 57.8 | 10.71 | 0.00 | 0.00 | -25.2 | 5.6 | 4.9 | 11.56 |


Global: IF HPF 300 kHz order 1 (noise-floor rms error 2.42 → 1.01 dB); image rejection 39.1 dB → IQ phase 1.27°; channel gain [-4.3, -3.3, -1.0, -0.8, +0.9, +1.0, +2.3, +1.3] dB, phase [+0, +13, -7, -11, -9, +12, -2, -1]°.

### Loader / ordering checks (parked car, θ ≈ −25°)

| hypothesis | array coherence (median of 30 frames) | θ̂ [°] |
|---|---|---|
| order(0,1) d=2 | 0.926 | -25.2 |
| order(1,0) d=2 | 0.754 | -32.0 |
| order(0,1) d=1 | 0.279 | -37.8 |
| order(0,1) d=3 | 0.769 | -17.5 |


Doppler sign: range-rate fit -0.38 m/s vs median Doppler velocity -0.54 m/s (24 frames) → consistent: True. Positive-beat fraction of moving power: pedestrian 84 %, cyclist 90 %, car 51 %.

adcStartTime 0 / 6 / 12 µs (simulated target, 7 m, −0.7 m/s): range 6.99017 / 6.99016 / 6.99015 m, velocity -0.7033 / -0.7045 / -0.7058 m/s, azimuth -2.0012 / -2.0013 / -2.0013°.
