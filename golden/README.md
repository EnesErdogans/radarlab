# Golden vectors (Python reference → TypeScript engine)

Written only by `python/scripts/export_golden.py` (code: `python/src/radarlab/golden.py`).
Schema and conventions: `docs/PROJECT_BRIEF.md`. Current files: `"goldenSet": 2`,
`"conventionVersion": 2` (also in `index.json`).

All inputs a consumer needs are stored in the files. No file needs a random
number generator to be read or checked. Scenario cases are noiseless and use φ0 = 0.
Some files were *created* with a seeded generator (the CFAR noise maps), but
the generated map itself is stored.

## Files

| File | Input stored | What is checked | Tolerance |
|---|---|---|---|
| `presets.json` | preset configs | derived quantities, virtual array, periodic Hann windows | rel 1e-12 |
| `cfar_alpha.json` | (K, N, Pfa) grid | CA/GO/SO/OS α, K ∈ {1, 4, 8} (Gamma(K) exact) | rel 1e-7 |
| `tiny_cube.json` | scatterers + full cube | beat cube, full RD power map, peaks, **range–angle map** (rect and Hann angle window, Na = 64) | cube 1e-9 of max, 1e-6 dB |
| `automotive_*.json`, `awr1843_elevated.json` | scatterers + cube spot samples | RD max/mean, peaks: integer/fractional bins, range, velocity, angle with and without TDM compensation | 1e-6 |
| `awr1843_elevated.json` | (as above) | also **elevation** (`elevationDeg`), point-cloud azimuth θ = atan2(x, y) (`azimuthDeg`) and `xyz` | 0.05° vs truth |
| `tiny_mti.json` | scatterers + full cube | **MTI** (slow-time mean removed after the range FFT); the static scatterer vanishes; full RD map | as `tiny_cube` |
| `tiny_real_iq.json` | scatterers + full real cube (`im` = 0) | **`iqMode: 'real'`**: Re(beat), N/2 range bins, halved Rmax | as `tiny_cube` |
| `tiny_physical.json` | scatterers + full cube | **`amplitudeMode: 'physical'`** (radar equation, kT0F·fs noise); per-scatterer `amplitudes` | as `tiny_cube` |
| `cfar_maps.json` | two 20×16 power maps (K = 1 and K = 8) | **CA/GO/SO/OS thresholds, per-cell α, detection masks**, training-cell counts incl. range edges (r = 0, 1, Nr−1), GO/SO lead/lag counts and CA-fallback cells, Doppler wrap | rel 1e-9, 0 mask differences |
| `clustering.json` | 12×10 mask + power map | 8-connected clustering with Doppler wrap: peak cell, size, member cells | exact |
| `tracker.json` | 12 frames of [R, θ, vr] | Python EKF-CV tracker: state, covariance, status, hit history per frame | 1e-9 |

### Truth tolerances (scenario files)

Each peak carries `truth` values. Two kinds of truth are stored:

* **Observable truth.** `rangeBeat` is the range including range–Doppler coupling.
  `velocityEffective` is the velocity scaled with the carrier at the centre of the ADC window,
  fc + S·(t_adc + N/(2fs)), because `fc` is the chirp *start* frequency.
  `angleDegFft` is asin(sinθ·cosφ), which is what a 1-D azimuth FFT measures.
  The tolerances `truthRangeBeatAbs`, `truthVelocityEffectiveAbs` and `truthAngleFftDegAbs` allow
  only the interpolation bias: 0.03 bin and 0.05°. The largest observed errors are 0.018 bin and 0.026°.
* **Geometric truth.** `range`, `velocityAliased`, `angleDeg` = atan2(x, y) and `elevationDeg`.
  The tolerances `truthRangeAbs`, `truthVelocityAbs` and `truthAngleDegAbs` add the largest
  known deterministic offset in that file. These offsets are range–Doppler coupling (up to 0.21 ΔR),
  carrier scaling (up to 0.19 Δv) and the cosφ projection (0.29° in `awr1843_elevated`).
  So the tolerance is never a blanket half-bin or 2°.

The engine-to-engine tolerances (`*Abs` without `truth`) are 1e-6 or tighter.

## Not covered (known gaps)

* Noise, Swerling fluctuation, phase noise, impairments (IQ imbalance, leakage, quantisation,
  interference, multipath) and micro-Doppler targets. These are random or exist only in Python.
* The interactive **web tracker** is a different design from the Python tracker. It uses a 2-D
  position gate, a different Q, and spawn suppression with `mergeDetections`. `tracker.json`
  specifies the Python reference. The web suite checks it with `ReferenceTracker`, a TS port of
  the Python tracker that the UI does not use.
* Clustering order. The web engine lists clusters in scan order and Python lists them by power,
  so compare them as sets.
* **Elevation without vertical diversity.** For arrays that have no same-x/different-z pairs, φ is
  taken as 0 (in both engines). Targets with z ≠ 0 are then reported at asin(sinθ·cosφ), which is
  0.29° off at φ = 10°. `angleDeg` is always compared with `truth.angleDegFft`.
* MVDR/MUSIC, DBSCAN, the micro-Doppler STFT, the OS-CFAR UI prefilter (web only) and DCA1000 I/O.
* Zero-padded range/Doppler FFTs and non-Hann range/Doppler windows.

## History

* Set 1: synthesis → Hann FFTs → P map → peak interpolation → FFT angle; α table; presets.
* Set 2 (2026-11-24, review round 1, M1/M2/m5/m16): adds CFAR maps, MTI, real IQ, physical amplitude,
  range–angle map, elevation / point-cloud azimuth, clustering and tracker. It also tightens the truth
  tolerances and adds the observable-truth fields. Parabolic peak interpolation now follows the web
  rule: the offset is 0 when the 3-point fit is not concave. Conventions are unchanged
  (`conventionVersion` 2).
