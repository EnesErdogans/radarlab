# RadarLab web — STATUS

## Done
- **Engine** (`src/`, pure TS on preallocated typed arrays, runs in a module Web Worker):
  - config + derived params + 3 presets (automotive 77 GHz 2TX×4RX, drone 24 GHz 1TX×4RX, micro-Doppler 24 GHz fast-chirp); warnings (ADC window > ramp, ramp > Tc, frame overrun, R > Rmax, Doppler aliasing, micro-Doppler aliasing, outside FOV).
  - radix-2 FFT (cached twiddles/bit reversal, fused first two stages), periodic Hann/Hamming/Blackman/rect windows (Σw-normalised).
  - beat-signal synthesis per brief (TDM timing, intra-chirp motion, RVP term, **rev. 1 negative spatial phase**, z term, **sinc line-blade weighting** for rotor/blade line scatterers), 2nd-order phasor recurrence (fast) or direct evaluation (exact); mulberry32 + Box–Muller; UI noise from `NoisePool` (block-permuted reads, independent uniform phase per 256-sample block per frame, 1/16 of the pool regenerated per frame → successive frames uncorrelated at every lag, tested) or exact sequential noise (tests); real-IQ mode.
  - targets: point, car (body + radar-side rolling wheels), pedestrian (arms/legs/torso bob), **drone — line-blade (sinc) rotor model** (N rotors × N_b blades, one uniform line scatterer per blade, Martin & Mulgrew 1990, matching Python `scene.Drone(blade_model='line')`; segment vector carried in the scatterer stride and weighted by `sinc(Φ)` in synthesis), bird (flapping wings); cached static clutter; MTI (slow-time mean removal).
  - range FFT, MTI, Doppler FFT + fftshift, RD power map, 2D CFAR CA/GO/SO/OS (SAT for CA/GO/SO; OS with quickselect + an exact block prefilter in UI mode — bound from the k-th smallest of a superset window, identical results). α is Gamma(K)-exact for all four types (CA closed form; OS/GO/SO by log-x Gauss–Legendre quadrature + root solve, `src/dsp/gamma.ts`, process-wide cache), GO/SO edge fallback uses the Gamma(K) CA α (as Python). Detection SNR = P / noise statistic of the active type.
  - connected-component clustering + parabolic (log-power) interpolation; TDM compensation + zero-padded 64-pt angle FFT over the azimuth ULA (z=min channels, unique x, same rule as Python); elevation from same-x/different-z pairs and Python point-cloud geometry (x = R·u_x, z = R·u_z, y = R√(1−u_x²−u_z²), θ = atan2(x, y)); range–angle map (lag-domain identity, direct FFT if 2L−1 > A; angle window selectable, default Hann as Python; Σw-normalised); point cloud; DBSCAN-style measurement merging (Cartesian + range/azimuth resolution); EKF CV tracker (R, θ, v_r with wrapped Doppler innovation), Hungarian GNN, χ² gate 9.21, confirm after 3 hits, deletion after misses, gated spawn suppression; `ReferenceTracker` = TS port of the Python EKF-CV/GNN/M-of-N tracker (checked against golden, not used by the UI); rolling STFT micro-Doppler spectrogram.
- **Worker protocol** with transferable Float32Array buffers (ping-pong pool), per-stage timings.
- **UI** (vanilla TS + Canvas2D, viridis/inferno LUTs, dark theme, responsive desktop/tablet/mobile, EN/TR): 8 panels, target editor (add/remove/edit, drag in bird's-eye view), radar/processing editors with live parameter budget, hover readouts, click-to-select CFAR cut, 8-step Learn mode (EN/TR), keyboard: space = play/pause, ←/→ in Learn mode. URL params: `?preset=micro|drone`, `?lang=tr`, `?learn=0..7`, `?paused`.
- **Tests** (Vitest): FFT vs DFT, windows PSL, derived params, analytic range/Doppler bins (incl. negative velocity, aliasing, range–Doppler coupling), TDM-compensated angle, synthesis recurrence = exact formula, CFAR Pfa ±10 % at K=1 and K=4 (all 4 types), Gamma(K) α identities, OS block prefilter = exact (all presets, MTI on/off, notch case), noise-pool independence, RA map vs brute force, OS vs CA / SO vs GO masking, OS prefilter safety, Hungarian, EKF convergence, track lifecycle, full-chain tracking, micro-Doppler (line-blade rotor: rotation spreads the Doppler spectrum, tip speed reflected in the extent), golden vectors.
- **CI**: `.github/workflows/ci.yml` (pytest + vitest + build), `pages.yml` (vitest, then build and deploy `web/dist`).

## Timings (this 2-core sandbox, Node, ms/frame)
automotive ≈ 50 (≈ 20 fps; synth 22, range FFT 5, Doppler 6, RA map 7, CFAR 4), micro ≈ 30, drone ≈ 110–135 (realtime needs 10 fps). First frame after a parameter change adds ≈ 0.1–0.25 s for the Gamma(K) α solves (cached afterwards).

## Golden vectors
`tests/golden.test.ts` reads every `../golden/*.json` (skips scenario files with `conventionVersion < 2`). Against **goldenSet 2** everything passes:
- presets/derived/windows (1e-12); raw cubes (exact 1e-9, phasor recurrence 1e-7); RD dB maps (1e-6 dB); peaks (bins, fractional bins, range, velocity, angle with/without TDM comp., 1e-6 / 1e-4°); elevation, azimuth θ = atan2(x, y) and xyz (`awr1843_elevated`); observable truth (rangeBeat, velocityEffective, angleDegFft) and geometric truth with the file's tolerances;
- `tiny_mti`, `tiny_real_iq`, `tiny_physical` (incl. per-scatterer amplitudes 1e-12); range–angle map rect + Hann (1e-9 of max);
- `cfar_alpha`: all rows K ∈ {1, 4, 8}, CA/GO/SO/OS, worst rel. diff 4e-10;
- `cfar_maps`: thresholds, α (thr/statistic) and masks for all four types at K = 1 and 8, incl. edges and fallback cells — worst rel. diff 7e-13, 0 mask differences;
- `clustering`: peak, power, size and member cells (compared as sets: web lists clusters in scan order);
- `tracker`: `ReferenceTracker` state (1e-9), covariance (1e-9 rel), status, hits, misses, age for all 12 frames.
- `microdoppler_drone`: **line-blade (sinc) rotor** beat cube (exact 1e-9, phasor recurrence 1e-7) and the range-FFT slow-time `y[m]` at the target range bin (1e-9 of max); the ported drone (`sim/targets.ts` + the sinc weighting in `sim/synth.ts`) reproduces the Python `scene.Drone(blade_model='line')` reference.

## Known issues / next
- The UI tracker (`Tracker`) is a different design from the Python reference (2-D position gate, Doppler-wrapped innovation, spawn suppression for extended targets); only `ReferenceTracker` is golden-checked.
- Parabolic interpolation at a non-maximum (den ≥ 0): web returns 0, Python a clipped ±0.5 offset (only reachable at non-peaks; documented in `tests/dsp.test.ts`).
- UI noise: frames are uncorrelated, but a pool value can reappear (randomly rotated and moved) within 16 frames — fine for visuals; tests use exact sequential noise.
- Synthesis dominates frame time (compute bound, ~3 GFLOP/s scalar JS); WASM-SIMD would help.
- Spectrogram concatenates frames (time axis in frames, gaps not shown).
- Extended targets can still create an extra track briefly; tracker has no IMM.
