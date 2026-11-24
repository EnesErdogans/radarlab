# Python reference: STATUS

## Done
* Phase 1–3 as in README. Convention version 2 (brief REVISION 1): negative spatial phase, +j angle kernel. All golden files are regenerated with `conventionVersion: 2`.
* Exact Gamma(K) CFAR α for CA, OS, GO and SO. It is the pipeline default; the table for K = 1, 4, 8 is in `golden/cfar_alpha.json`.
* Leakage is synthesised inside the frame and shares the phase-noise realisation. Swerling 1/3 gains are shared across channels.
* Line-blade (sinc) rotor model, with the flash-rate law tested for N_b = 2, 3, 4, 5. 2-segment bird (fidelity §12.2).
* Closed-form Pd for SW1 (correlated over K), SW2 and SW3 (K = 1). CA-CFAR Pd for SW1.
* Validation matrix V1–V11 in `radarlab.validation`, tested in pytest (quick sizes plus slow paper-grade sizes). The report script writes `examples/out/validation_report.md`.
* Closed loop .cfg → simulate → int16 DCA1000 .bin → read → process gives identical codes and identical detections.

## Next
* TI IF-filter model (two first-order HPF sections at TI corner indices, 4th-order LPF) and the tone-mode vs time-domain comparison (fidelity §0.2, §4).
* Chirp nonlinearity (§3), element patterns and channel calibration (§7), distributed clutter (§14), Fresnel Γ (§10).
* Full Boulic pedestrian model. Validation on public DCA1000 captures (V11, second half).

## Known issues / approximations
* Two-ray: the mixed path is an equivalent scatterer on the direct line of sight, so its elevation is approximate. Null positions are exact (V5).
* FMCW interference uses an ideal-LPF mask on the instantaneous difference frequency.
* GO/SO α at range edges uses n = min(halves).
* For |v| > vmax, TDM compensation uses the aliased velocity, so the angle is biased.
* SW2/SW4 are per chirp. The fidelity doc recommends per-frame-per-channel decorrelation instead; that is not implemented.
