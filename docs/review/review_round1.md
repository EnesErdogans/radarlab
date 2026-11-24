# RadarLab — independent review, round 1

Reviewer role: radar signal-processing journal reviewer and senior software engineer. I did not write any of the reviewed code.
Scope: `docs/PROJECT_BRIEF.md`, `README.md`, `docs/literature/fidelity_requirements.md`, `python/`, `web/`, `golden/`, `python/examples/out/validation_report.md`, `.github/workflows/`.
Date: 2026-11-24. Reviewed commit: `97280ab`.
The throwaway scripts were in `/tmp/review/` (not committed). The commands below show how to reproduce each result.

## Summary verdict

**Core physics and statistics: sound. Claims and cross-implementation coverage: need revision before a paper submission (major revision).**

What I verified independently and found correct:

- **Beat model and signs.** I wrote an independent first-principles synthesiser. It uses exact bistatic path lengths |p−tx|+|p−rx| per element, real TDM chirp start times, a moving target, and dechirp = s_TX·conj(s_RX). It agrees with `synthesize_cube` to 0.037 rad, and the residual comes from the near-field and τ(t) approximations. Processed by the repo chain, it gives θ = 20.05° for a +20° target, and the same at v = 12 m/s (0.8·vmax, TDM compensation active). So the negative spatial term, the +j angle kernel, the TDM compensation sign, and "receding → Doppler bin > M/2" are physically right.
- **θ > 0 = right in both engines.** Python: automotive and awr1843 at ±20° give ±19.99°. Web: all three presets at ±20° give ±20.000° in the detections, and the range–angle map peak is at bins 43/21 (±20.1°). The UI maps x/r → bin with the same sign.
- **Derived quantities.** I recomputed every row of README Table 2 by hand. All values match except the Δθ column note (m1).
- **Real IF.** Rmax is halved and N/2 bins are kept. A target beyond the real-IF Rmax folds to the mirrored range with Doppler and angle sign flips, which is physically correct.
- **CFAR α (all four types, Gamma(K)).** My own Monte-Carlo: 2·10⁶ trials per case, K ∈ {1, 4, 8}, N ∈ {16, 32}, CA/OS/GO/SO, design Pfa 1e-3. All 24 cases have |z| < 2.
- **CRLB.** `6/(η·L·(N²−1))` matches Rife & Boorstyn and Kay for complex noise with η = |A|²/σ². An independent 1-D ML Monte-Carlo gives MSE/CRLB = 1.02 (0 dB) and 1.01 (10 dB). The SNR definition (per complex sample, σ² = 1) is the same in `signal.py`, `metrics.py` and V3.
- **The "4.1–6.1 dB" claim.** Reproduced exactly: 4.51/4.21/4.07 dB for K = 4 and 6.10/5.74/5.57 dB for K = 8. The resulting SNR loss is 3.9–7.4 dB (Pd = 0.5–0.9), so "a comparable amount of detection SNR" is fair. The actual Pfa becomes about 5·10⁻¹³ instead of 10⁻⁴.
- **Test suites.** Python: 114 passed (+7 slow). Web: 79 passed. `tsc` and `vite build` succeed. The Pages build with `base: './'` emits `./assets/...` and a relative worker URL, so it is correct for a project page.
- **Secrets and licences.** No secrets found (only `id-token: write` in pages.yml, which is required). I found no third-party code copied without attribution. The only borrowed algorithm is mulberry32, which is public domain.

No critical findings. Six major and eighteen minor findings follow.

---

## Major findings

### M1 — Python point cloud: azimuth and x/y are wrong for targets with elevation ≠ 0, and a test locks in the bug

`dsp/pointcloud.py` sets `azimuth = asin(u)`, where the angle FFT measures u = sinθ·cosφ. It then computes `x = R·sin(az)·cos(el)` and `y = R·cos(az)·cos(el)`. So the reported azimuth is asin(sinθ cosφ) and not the brief's θ = atan2(x, y), and x picks up an extra cosφ factor.

Evidence (`awr1843_like`, R = 10 m, θ = 15°):
```
el=12: true xyz [2.532 9.448 2.079]  est (2.477, 9.466, 2.080)  az_est 14.67
el=30: true xyz [2.241 8.365 5.000]  est (1.942, 8.442, 5.001)  az_est 12.95   (x error 13 %)
```
The golden file `awr1843_elevated.json` shows the same thing: angle − truth = **+0.288°**, while every other golden peak is within ≤ 0.03°. The truth tolerance is 2°, so this goes unnoticed. In `tests/test_doa.py::test_elevation_awr1843`, the assertion `sin(az) == sinθ·cosφ` states the buggy behaviour as if it were correct.

**Fix.** Use u_x = sin(az_meas) and u_z = sinφ, then x = R·u_x, z = R·u_z, y = R·√(1−u_x²−u_z²), and θ = atan2(x, y). Change the test to `azimuth ≈ 15°`. Tighten `truthAngleDegAbs` to about 0.05° (see m5). Regenerate golden (conventionVersion 3). The web engine has no elevation estimate, so for targets with z ≠ 0 it has the same small bias in BEV x/y. Document this.

### M2 — "Golden vectors keep the two implementations numerically equivalent" is overclaimed; important conventions are not covered

The abstract, contribution 4 and CITATION.cff all make this claim. The golden files cover only noiseless synthesis → Hann FFTs → P map → peak interpolation → FFT angle, plus the α table and the presets. They do **not** cover:

1. CFAR masks and thresholds, including range-edge clipping, the GO/SO half split, the GO/SO→CA fallback and OS rank at edges.
2. MTI.
3. `iqMode: 'real'`.
4. `amplitudeMode: 'physical'`.
5. Clustering, the tracker and the range–angle map. Python uses a Hann angle window by default and TS uses rect, so the RA maps differ.
6. Elevation.

I cross-checked CFAR myself (`/tmp/review/mkP.py` and `cfar_x.test.ts`) by feeding the same P maps (64×32, with targets at the range edges) to both engines:
```
K=1 CA/GO/SO/OS: mask diffs 0, max threshold rel diff ≤ 4e-13 (interior and edges)   <- good
K=8 CA:          mask diffs 0, rel diff 3e-13
K=8 GO: thr rel diff 1.8–1.9 (×2.8), SO: 2.0–2.3 (×3.3), OS: 1.5–1.6 (×2.6); mask diffs 0/1/1
```
There is also an **undocumented divergence**. Web `Cfar2D.run` uses `alphas.get('CA1', …)`, the single-channel α, in the GO/SO edge fallback. Python uses `ca_alpha(n, pfa, K)`.

**Fix.** Add golden cases: a fixed P map (K = 1 and K = 8) with masks and thresholds for all four types, including edges; one MTI case; one real-IQ case; one physical-amplitude case; and the RA map with the window stated. Use the Gamma(K) CA α in the TS fallback. Scope the abstract to "numerically equivalent for synthesis, FFT processing, peak and angle estimation and CA-CFAR".

### M3 — Web GO/SO/OS-CFAR thresholds are 4–5 dB too high at K = 8, which misleads the educational CFAR comparison

README §10 documents this. However, the web app's main teaching device is the CFAR-cut panel, which shows CA, GO, SO and OS thresholds on one line, and Learn step 4. Both compare an exact CA threshold with GO/SO/OS thresholds that are ×2.6–3.3 too high (M2 evidence), because the defaults sum K = 8 or K = 4 channels. Users will therefore conclude that OS/GO/SO are "less sensitive" than CA, but that difference comes from the implementation. In addition, `Detector.cluster` computes `snrDb = P / (thr/α_CA)` for every CFAR type, so the SNR readout is also wrong for GO/SO/OS.

**Fix.** Ship Gamma(K) α tables for OS/GO/SO from Python as JSON. They are already in `golden/cfar_alpha.json` for K ∈ {1, 4, 8} and a range of N; extend this to all K ≤ 32 and all N reachable in the UI, and interpolate in log Pfa if needed. Alternatively, port the quadrature. Divide by the α of the active type when computing the SNR readout.

### M4 — The validation report hides V7 and V8 results, so README numbers cannot be traced

`examples/validation_report.py::_table` takes its columns from `rows[0].keys()`. When later rows have different keys, they print as empty cells. In `validation_report.md`, the V7 "equal slopes (ghost)" row and the V8 "flash rate N_b = 2/3" and "bird wing-beat" rows are blank. The values exist (my run of `V.ALL["V7"]()` and `V.ALL["V8"](quick=False)`): ghost at 9.076 m vs theory 8.994 m; flash rate error −1e-16 and −4e-6; wing-beat error 6.7e-4. README §6 says "All numbers in this section come from validation_report.md", but the V7 ghost and V8 flash and wing-beat claims are not visible there.

**Fix.** Build each table from the union of the row keys, or write one table per row schema. Regenerate the report.

### M5 — The Gamma(K) CFAR "contribution" overstates novelty

The exact CA threshold under M-pulse or K-channel non-coherent integration follows from the Beta/F-distribution of X/(X+S). It is standard in the classical CFAR literature: for example, Ritcey's censored mean-level detector analysis and Shor & Levanon's OS-CFAR analysis both treat noncoherent integration of M pulses. The reviewer recommends that the authors check these primary sources and cite them. The "4.1–6.1 dB" figure measures the cost of *misusing* the K = 1 formula. That is a useful warning for practitioners, but it is not a scientific contribution.

The numbers themselves are correct (see summary). One more point: README says "verified by Monte-Carlo simulation in V1", but V1 only covers K = 8 with N = 54 (interior cells). The K = 4 and N = 16–32 rows in the table are not MC-verified in the repo. My MC above confirms them.

**Fix.** Reword contribution 3 as "a verified implementation of the known exact thresholds (CA closed form; OS/GO/SO by quadrature) and a quantification of the error of a common misuse". Add the classical references. Add K = 4 to V1.

### M6 — Validation matrix: consistency checks are presented as "validation against closed-form theory"

The abstract and Table 3 present V1–V11 as experiments that "compare the simulator with closed-form theory", and conclude "All validations pass". Several entries are self-consistency checks:

- **V6** (IQ image): the model is defined as y = μx + νx*, so image/signal = |ν|²/|μ|² by algebra. The measured error, 1.2e-11 dB, shows it is an identity.
- **V9** (TDM): the synthesiser and the compensator use the same t·Tc formula. My independent bistatic model confirms the physics, but V9 cannot detect a shared error.
- **V10**: the stored golden files are compared with a fresh export by the same code.
- **V11**: the DCA1000 writer and reader are written from the same assumption, and no real capture is used.
- **V8 flash rate** (rel. error 1e-16): this holds by construction of a strictly periodic rotor.

**Fix.** Split Table 3 into "T1 theory validations" (V1–V5, V7, V8 extent and wing-beat) and "implementation consistency checks" (V6, V9–V11, V8 flash rate). Change "All validations pass" to "All 7 theory validations and 5 consistency checks pass". In V11, check the decoder against TI's published `readDCA1000.m` on a byte-level fixture, or against one public capture (`docs/real_data/candidates.md` already lists some).

---

## Minor findings

- **m1 — Δθ note is factually wrong.** README §3.4 says the Δθ column is "the same value that the web parameter budget shows". Web `derive()` uses `asin(1/(L·d))`, so it shows 14.5° and 30.0°, not 14.3° and 28.6°. *Fix:* either make the budget show `1/(L·d)` rad or change the README.
- **m2 — V5 theory reference uses λ at the start frequency.** The simulated nulls are systematically 0.19–0.25 % farther than predicted (0.03→0.19 m). This matches the ADC-window centre frequency, which is 0.29 % above `fc` for `automotive_77ghz`. The ±1-bin criterion is too loose to show this. README §4.2 says "null positions are exact (V5)". The fidelity spec also asks for "peak power follows 4sin²", but only the nulls are tested (rms deviation 1.61 dB, not part of the criterion). *Fix:* use λ_c = c/(fc + S·t_mid) in the theory, tighten the tolerance to about 0.1 %, and add an amplitude criterion or state the 1.6 dB as a known approximation.
- **m3 — V2 criterion and detector.** The spec says "within the 95 % CI", but the verdict uses the 99.9 % CI. That choice is defensible for multiple comparisons, but it should be stated up front, or replaced by a pre-registered Bonferroni or χ² goodness-of-fit test. V2 also uses a *fixed* threshold, not a CFAR. `metrics.pd_ca_cfar_swerling1` exists but is never used or tested. *Fix:* add a CA-CFAR Pd validation.
- **m4 — V3 does not test the pipeline estimator.** V3 validates an unwindowed 3-D ML estimator. The processing chain actually uses Hann windows with parabolic log-power interpolation, which is biased and not efficient. README is transparent about this, but readers will assume the chain is CRLB-efficient. *Fix:* add an RMSE curve for `process_frame`. Also, `crlb_angle` claims to hold for an "arbitrary linear array". For TDM-MIMO with unknown v, the TX-slot Doppler phase couples v and θ (non-separable), so the bound is optimistic. Note this or derive the coupled FIM.
- **m5 — Golden truth tolerances are too loose.** `truthAngleDegAbs` is 2°, while the noiseless error is ≤ 0.03° except for the M1 bug. Range and velocity tolerances are half a bin. *Fix:* use about 0.05° for angle and ΔR/10 for range.
- **m6 — Web CFAR Pfa test is loose and K = 1 only.** The tolerance of ±15 % is about ±4.7σ (60 frames, Pfa 2e-3). There is no empirical K > 1 Pfa test in the web suite. *Fix:* use ±10 % and add a K = 4 CA test.
- **m7 — Web UI noise reuses one pool.** The noise pool (`GaussianPool`, 2²⁰ float32 values) is read from a random offset each frame. The drone preset consumes exactly 2²⁰ normals per frame, so every frame's noise is a cyclic shift of the same pool. The automotive preset overlaps about 50 % between frames. Repeated noise patterns and correlated false alarms are possible. *Fix:* document that this is for visuals only, or refill part of the pool every frame.
- **m8 — `Detector.rangeAngle` can return an all-zero map.** If `2L−1 > A`, the `canLag` path does nothing. The UI cannot reach this (at most 32 elements with A = 64), but API users can. *Fix:* add a direct-FFT fallback or throw an error.
- **m9 — The OS-CFAR UI prefilter is an unproven heuristic.** It skips cells with P ≤ α·q10(global) and is tested on one scene only. It can miss detections where the local noise is below the global 10 % quantile, for example in the MTI notch. *Fix:* document it, or limit it to regions without MTI.
- **m10 — Python tracker splits extended targets.** The README quick-start gives three confirmed tracks for one `Car`. README describes extended-target merging only for the web version. *Fix:* document this or port `mergeDetections`.
- **m11 — CI gaps.** Python is tested only on 3.12, while the README claims "3.10+" (badge and `requires-python`). The "Check for python package" step is obsolete. The Pages workflow deploys without running the tests, so a build that fails its tests can still be deployed. *Fix:* add a {3.10, 3.12} matrix, delete the step, and make Pages depend on CI (`workflow_run` or a test job).
- **m12 — Repository hygiene.** `python/.gitignore` ignores `examples/out/`, yet README links tracked files there. The files were force-added, so new figures would stay silently untracked. The web `package.json` version is 1.0.0, while Python and CITATION use 0.1.0. The brief lists a `docs/paper/` folder that does not exist.
- **m13 — Docstring error.** `drone_24ghz` says "vmax ≈ 25.9 m/s", but the value is 26.02 m/s.
- **m14 — Real IF in Python `range_fft` is chosen by `np.isrealobj`, not `cfg.iq_mode`.** The README hardware example calls `cubes[0].astype(complex)`. For a real-format capture (`adcOutputFmt 0`) this keeps all N bins, including the mirror image. *Fix:* pass `real_input=(cfg.iq_mode == 'real')` in `process_frame`.
- **m15 — Dead code.** `metrics.pd_ca_cfar_swerling1` is never called. Its formula is correct, but it is untested.
- **m16 — Web/Python behaviour difference in parabolic interpolation.** For `den ≥ 0`, web returns 0 while Python returns a clipped offset. This only matters at non-maxima, so it is cosmetic, but it is an untested convention.
- **m17 — References.** I spot-checked eight entries against publisher or index pages:
  - [3] Schüßler et al.: J. Microwaves **1**(4) 962–974, 2021 is correct. The DOI is 10.1109/JMW.2021.3104722, so "(DOI unverified)" can be replaced.
  - [4] SHENRON: authors K. Bansal, G. Reddy, D. Bharadia; *IEEE RA-L* **9**(2) 1644–1651. Complete the entry.
  - [5] Magosi et al.: correct. It has an article number, not pages (fixed in README).
  - [14] Rahman & Robertson: *Sci. Rep.* **8**, art. no. **17396**, 2018. Add this and remove "unverified".
  - [17] Siddiq et al.: TAES **55**(1) 70–81, DOI correct.
  - [24] Gandhi & Kassam: AES-24, July 1988, pp. 427–445 is correct. Some citing works print the title as "Analysis of *some* CFAR processors in nonhomogeneous background"; check against IEEE Xplore.
  - [25] Rohling: AES-19, July 1983 (no. 4), pp. 608–621 is correct.
  - [26] Bechter, Roos, Waldschmidt: *IEEE MWCL* **27**(12) 1164–1166, Dec. 2017 is correct. "(DOI unverified)" remains, but the details are confirmed.
- **m18 — Licensing courtesy.** Add a one-line attribution for mulberry32 (public domain) in `web/src/sim/rng.ts`.

## README edits made by the reviewer (typos only)

- Table 3, V3: "0.957–1.041" → "0.956–1.041". The report's minimum is 0.9557. The same fix was made in README.tr.md.
- Ref. [5]: "pp. 5693" → "art. no. 5693".

## Reproduction (reviewer scripts, `/tmp/review/`)

| Script | Purpose | Key output |
|---|---|---|
| `indep_geom.py`, `indep_geom_fast.py` | bistatic first-principles synthesis vs repo; process with repo chain | phase diff 0.037 rad; θ = 20.05° at v = 3 and 12 m/s |
| `angle_py.py`, `web_angle.test.ts` | ±20° in both engines | py ±19.99°, web ±20.000°, RA map ±20.1° |
| `cfar_mc.py` | MC of CA/OS/GO/SO α, K ∈ {1, 4, 8} | 24/24 cases with \|z\| < 2 |
| `ca_err.py` | 4.1–6.1 dB claim, actual Pfa, SNR loss | 4.07–6.10 dB; Pfa ≈ 5e-13; loss 3.9–7.4 dB |
| `crlb1d.py` | 1-D CRLB vs independent ML | MSE/CRLB 1.02, 1.01 |
| `mkP.py` + `cfar_x.test.ts` | py vs TS CFAR on identical P maps | K = 1 identical; K = 8 GO/SO/OS ×2.6–3.3 |
| `realif.py`, `quick.py` | real-IF behaviour; README quick-start | correct folding; quick-start runs (3 tracks for one car) |

Web scripts are run from `web/` with `npx vitest run --config /tmp/review/vitest.config.mjs` (include `/tmp/review/*.test.ts`).
