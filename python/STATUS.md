# Python reference: STATUS

## Done
* Phase 1: config and presets, scene targets (point, car+wheels, pedestrian, drone, bird, clutter), exact beat synthesis,
  full DSP chain (FFT, MTI, CFAR CA/GO/SO/OS, clustering, DoA FFT/MVDR/MUSIC, point cloud, micro-Doppler), EKF/GNN tracker,
  pipeline/Simulator, metrics (CRLB, Pd/Pfa), golden vectors (7 files), 4 example figures.
* Phase 2: Swerling 1–4, ADC quantization, IF filters, phase noise with range correlation, leakage, IQ imbalance, two-ray, FMCW interference.
* Phase 3: TI `.cfg` parser and DCA1000 reader/writer (complex and real, round-trip tested).
* Tests: 84 passing in about 10 s (`pytest python/tests`).

## Next
* Antenna element patterns, chirp nonlinearity, distributed (range-spread) clutter.
* Monte-Carlo ROC scripts (Pd vs SNR, empirical vs analytic) for the paper.
* Validation against a real DCA1000 capture. The `conj` flag is in place for devices that need it.
* Pedestrian model: full Boulic joint-angle trajectories (the current model is a simplified pendulum).

## Known issues / approximations
* Two-ray "mixed" path is an equivalent scatterer on the direct line of sight (angle of arrival approximated).
* FMCW interferer uses an ideal-LPF mask on the instantaneous difference frequency.
* GO/SO α at range edges is approximate (see README).
* TDM compensation uses the aliased velocity estimate. For |v| > vmax the angle is biased.
  This is shown by `golden/automotive_velocity_alias.json`.
