# RadarLab web — STATUS

## Done
- **Engine** (`src/`, pure TS on preallocated typed arrays, runs in a module Web Worker):
  - config + derived params + 3 presets (automotive 77 GHz 2TX×4RX, drone 24 GHz 1TX×4RX, micro-Doppler 24 GHz fast-chirp); warnings (ADC window > ramp, ramp > Tc, frame overrun, R > Rmax, Doppler aliasing, micro-Doppler aliasing, outside FOV).
  - radix-2 FFT (cached twiddles/bit reversal, fused first two stages), periodic Hann/Hamming/Blackman/rect windows (Σw-normalised).
  - beat-signal synthesis per brief (TDM timing, intra-chirp motion, RVP term, **rev. 1 negative spatial phase**, z term), 2nd-order phasor recurrence (fast) or direct evaluation (exact); mulberry32 + Box–Muller, Gaussian pool with a random offset (UI) or exact sequential noise (tests); real-IQ mode.
  - targets: point, car (body + radar-side rolling wheels), pedestrian (arms/legs/torso bob), drone (N rotors × 2 blades × 2 points), bird (flapping wings); cached static clutter; MTI (slow-time mean removal).
  - range FFT, MTI, Doppler FFT + fftshift, RD power map, 2D CFAR CA/GO/SO/OS (SAT for CA/GO/SO; OS with quickselect + noise-floor prefilter in UI mode, exact mode for tests). CA α is Gamma(K)-exact for K summed channels (brief rev. 1); GO/SO/OS use K=1 formulas.
  - connected-component clustering + parabolic (log-power) interpolation; TDM compensation + zero-padded 64-pt angle FFT over the azimuth ULA (z=min channels, unique x, same rule as Python); range–angle map (lag-domain identity); point cloud; DBSCAN-style measurement merging (Cartesian + range/azimuth resolution); EKF CV tracker (R, θ, v_r with wrapped Doppler innovation), Hungarian GNN, χ² gate 9.21, confirm after 3 hits, deletion after misses, gated spawn suppression; rolling STFT micro-Doppler spectrogram.
- **Worker protocol** with transferable Float32Array buffers (ping-pong pool), per-stage timings.
- **UI** (vanilla TS + Canvas2D, viridis/inferno LUTs, dark theme, responsive desktop/tablet/mobile, EN/TR): 8 panels, target editor (add/remove/edit, drag in bird's-eye view), radar/processing editors with live parameter budget, hover readouts, click-to-select CFAR cut, 8-step Learn mode (EN/TR), keyboard: space = play/pause, ←/→ in Learn mode. URL params: `?preset=micro|drone`, `?lang=tr`, `?learn=0..7`, `?paused`.
- **Tests** (Vitest): FFT vs DFT, windows PSL, derived params, analytic range/Doppler bins (incl. negative velocity, aliasing, range–Doppler coupling), TDM-compensated angle, synthesis recurrence = exact formula, CFAR Pfa ±15 % (all 4 types), OS vs CA / SO vs GO masking, OS prefilter safety, Hungarian, EKF convergence, track lifecycle, full-chain tracking, micro-Doppler spread, golden vectors.
- **CI**: `.github/workflows/ci.yml` (pytest + vitest + build), `pages.yml` (deploy `web/dist`).

## Timings (this 2-core sandbox, Node, ms/frame)
automotive ≈ 46 (≈ 22 fps; synth 22, range FFT 5, Doppler 6, RA map 6.5, CFAR 2), micro ≈ 26, drone ≈ 106 (realtime needs 10 fps).

## Golden vectors
`tests/golden.test.ts` reads every `../golden/*.json` (skips if none; skips scenario files with `conventionVersion < 2`). Against the **conventionVersion 2** files all checks pass: derived params, presets/windows (1e-12), raw cube (exact 1e-9, phasor recurrence 1e-7), cube samples, RD dB map (1e-6 dB, cells within 90 dB of peak), peaks (integer bins, fractional bins / range / velocity 1e-6, angle 1e-4°, angle without TDM compensation), truth checks, CFAR α (K=1 all types; CA Gamma(K) for K=4, 8). Not checked: OS/GO/SO Gamma(K>1) rows (TS uses K=1 formulas there).
Angle kernel: +j (conjugated snapshot + forward FFT), θ > 0 → bin > A/2, sinθ = (a − A/2)/(A·d), as in the reference.

## Known issues / next
- Synthesis dominates frame time (compute bound, ~3 GFLOP/s scalar JS); WASM-SIMD would help.
- Spectrogram concatenates frames (time axis in frames, gaps not shown).
- Extended targets can still create an extra track briefly; tracker has no IMM.
- GitHub link is a placeholder (`GITHUB_URL` in `src/main.ts`).
