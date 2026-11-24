# Python reference: STATUS

## Done
* Phase 1–3 as in README. Convention version 2 (brief REVISION 1): negative spatial phase, +j angle kernel. All golden files are regenerated with `conventionVersion: 2`.
* Exact Gamma(K) CFAR α for CA, OS, GO and SO. It is the pipeline default; the table for K = 1, 4, 8 is in `golden/cfar_alpha.json`.
* Leakage is synthesised inside the frame and shares the phase-noise realisation. Swerling 1/3 gains are shared across channels.
* Line-blade (sinc) rotor model, with the flash-rate law tested for N_b = 2, 3, 4, 5. 2-segment bird (fidelity §12.2).
* Closed-form Pd for SW1 (correlated over K), SW2 and SW3 (K = 1). CA-CFAR Pd for SW1.
* Validation matrix V1–V11 in `radarlab.validation`, tested in pytest (quick sizes plus slow paper-grade sizes). The report script writes `examples/out/validation_report.md`.
* Closed loop .cfg → simulate → int16 DCA1000 .bin → read → process gives identical codes and identical detections.
* **Golden set 2 is written** (`"goldenSet": 2` in every file and in `golden/index.json`; conventions unchanged, `conventionVersion` 2). New files: `cfar_maps` (CA/GO/SO/OS thresholds, α, masks on stored 20×16 maps, K = 1 and 8, incl. range edges and GO/SO fallback), `clustering`, `tracker`, `tiny_mti`, `tiny_real_iq`, `tiny_physical`. `tiny_cube` adds range–angle maps (rect and Hann), `awr1843_elevated` adds elevation, point-cloud azimuth and xyz. Truth tolerances are tightened to the maths (0.03 bin, 0.05°, plus the documented deterministic offsets). All inputs are stored in the files. Coverage and gaps: `golden/README.md`. The web suite passes against set 2.
* Review round 1 (Python side): M1 point-cloud geometry with elevation (x = R·u_x, z = R·sinφ, y = R·√(1−u_x²−sin²φ), θ = atan2(x, y)). M4 report tables use the union of row keys. M5/M6: V1 adds K = 4; each result has `kind` = theory | consistency; V8 flash rate is split out as consistency check V8f. m2: V5 uses λ at the ADC-window centre, a 0.1 % null criterion and an amplitude criterion. m3/m15: CA-CFAR SW1 Pd Monte-Carlo test. m4: crlb_angle documents the TDM v–θ coupling. m13: docstring. m14: `process_frame` chooses real IF from `cfg.iq_mode`. m16: parabolic offset is 0 for a non-concave fit, as in the web engine. V10 compares with a relative tolerance of 1e-9.

## Next
* TI IF-filter model (two first-order HPF sections at TI corner indices, 4th-order LPF) and the tone-mode vs time-domain comparison (fidelity §0.2, §4).
* Chirp nonlinearity (§3), element patterns and channel calibration (§7), distributed clutter (§14), Fresnel Γ (§10).
* Full Boulic pedestrian model. Validation on public DCA1000 captures (V11, second half).

## Known issues / approximations
* Two-ray: the mixed path is an equivalent scatterer on the direct line of sight, so its elevation is approximate. Null positions agree within 0.1 % with λ at the ADC-window centre (V5).
* Without same-x/different-z virtual pairs φ = 0 is assumed, so targets with z ≠ 0 are reported at asin(sinθ·cosφ).
* The Python tracker has no extended-target merging, so one car can give several tracks.
* V11 is a consistency check only: it is not yet checked against TI's readDCA1000.m or a public capture.
* FMCW interference uses an ideal-LPF mask on the instantaneous difference frequency.
* GO/SO α at range edges uses n = min(halves).
* For |v| > vmax, TDM compensation uses the aliased velocity, so the angle is biased.
* SW2/SW4 are per chirp. The fidelity doc recommends per-frame-per-channel decorrelation instead; that is not implemented.
