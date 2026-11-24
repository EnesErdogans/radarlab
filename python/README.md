# radarlab (Python reference implementation)

Research-grade FMCW TDM-MIMO radar simulation and processing in numpy/scipy.
It follows the binding conventions in [`../docs/PROJECT_BRIEF.md`](../docs/PROJECT_BRIEF.md).
It is also the generator of the cross-language golden vectors in [`../golden/`](../golden/).

## Install

```bash
pip install -e "python[dev]" --break-system-packages   # from the repo root
pytest python/tests                                   # about 10 s; -m "not slow" to skip Monte-Carlo
python python/scripts/export_golden.py                # regenerate ../golden/*.json
```

## Quick start

```python
import radarlab as rl

cfg = rl.automotive_77ghz()                     # 2TX x 4RX, 8-element virtual ULA
scene = rl.Scene([
    rl.Car((-3.5, 40, 0), (0, -8, 0), wheels=True),
    rl.Pedestrian((6, 15, 0), (-1.3, 0, 0)),
    rl.PointTarget.from_polar(25.0, 12.0, radial_velocity=-5.0, rcs=5.0),
])
cube = rl.simulate_frame(cfg, scene, rng=1)      # (K, M, N) complex, noise variance 1
res = rl.process_frame(cube, cfg, rl.ProcessingParams(cfar_kind="os"))
res.power_db                                     # RD map (Nr, M), Doppler fftshifted
res.points[["range", "velocity", "azimuth", "snr_db"]]

# frame stepping + tracking
from radarlab.track import Tracker
sim = rl.Simulator(cfg, scene, tracker=Tracker(), seed=0)
for fr in sim.run(20):
    print([ (t.id, t.position.round(1)) for t in fr.tracks ])
```

Real data: `radarlab.io.parse_ti_cfg("profile.cfg")` gives a `TiCfg` (`.config` is a `RadarConfig`).
`radarlab.io.read_dca1000("adc_data.bin", cfg)` gives cubes of shape (F, K, M, N).

## Modules

| module | contents |
|---|---|
| `config` | `RadarConfig` (+ derived quantities, validation warnings, camelCase JSON) and presets `automotive_77ghz`, `awr1843_like`, `drone_24ghz`, `micro_doppler_24ghz` |
| `scene` | `PointTarget`, `Car`, `Pedestrian` (Boulic/Chen), `Drone` (Martin & Mulgrew blades), `Bird`, `ClutterField`/`static_clutter`, `Scene` |
| `signal` | exact beat synthesis (TDM timing, intra-chirp motion, RVP term), 'snr'/'physical' amplitude, complex/real IQ |
| `impairments` | Swerling 1–4, ADC quantization/clipping, IF HPF/LPF, range-correlated phase noise, TX→RX leakage, IQ imbalance, two-ray multipath, FMCW interferer |
| `dsp` | windows, range/Doppler FFT, MTI, RD power, 2-D CFAR (CA/GO/SO/OS, cached α), clustering (connected/DBSCAN), parabolic refinement, TDM compensation, FFT/MVDR/MUSIC DoA, elevation, range-angle map, point cloud, micro-Doppler STFT |
| `track` | EKF constant-velocity tracker, measurement (R, θ, vr), GNN with χ² gate, M/N confirm, delete after misses |
| `pipeline` | `simulate_frame`, `process_frame` → `FrameResult`, `Simulator` |
| `metrics` | CRLB (range/velocity/angle), ML tone estimator, RMSE, analytic Pd for Swerling 0–3 with non-coherent integration, empirical Pd/Pfa |
| `io` | TI SDK `.cfg` parser, DCA1000 `.bin` reader/writer (xWR16/18/68xx, SWRA581 layout) |
| `golden` | golden-vector case builders (used by `scripts/export_golden.py` and tests) |
| `validation` | validation matrix V1–V11 (Pfa, Pd, CRLB, phase noise, multipath, IQ, interference, micro-Doppler, TDM, golden, DCA1000 closed loop) |

## Figures

Run `python examples/<name>.py` from `python/examples` (matplotlib is required). Output goes to `examples/out/`.

* `rd_map_automotive.py`: RD map with OS-CFAR clusters and a bird's-eye point cloud
* `micro_doppler.py`: quadrotor vs bird spectrograms
* `cfar_pfa.py`: empirical vs design Pfa (CA, OS)
* `rmse_crlb.py`: ML RMSE vs CRLB for angle, velocity and range
* `micro_doppler_regimes.py`: unaliased, K-band 4.25 kHz CRF (aliased) and W-band 20 µs regimes (after Rahman & Robertson)
* `validation_report.py`: the full validation matrix as a markdown report

## Convention notes (convention version 2, brief REVISION 1)

* **Spatial phase is negative.** The beat term is `−2π·(x_k·sinθ·cosφ + z_k·sinφ)`, and the steering vectors are `a_k = exp(−j2π(x_k sinθ cosφ + z_k sinφ))`.
  MVDR and MUSIC use the same steering vectors.
* **Angle spectrum uses the +j kernel.** `A[a] = Σ_k x_k·exp(+j2πka/Na)`, computed as `Na·ifft`, then fftshifted.
  As before, bin a maps to sinθ = (a − Na/2)/(Na·d), and θ > 0 lands in a bin > Na/2.
  The window is rectangular by default, normalised by Σw, and Na = 64 in the golden files.
  The ULA is built from the z = 0 virtual channels with unique x, on a grid of the smallest spacing d. Gaps are zero-filled.
* Elevation comes from same-x pairs as φ = asin(−arg(Σ conj(x_low)·x_high)/(2πΔz)).
* `fc` is the **chirp start frequency** (TI `startFreq`). λ = c/fc is used everywhere.
* Optional `txPosZ`/`rxPosZ` arrays hold element z-positions in wavelengths.
* Real IQ is the real part of the complex cube, noise included (real noise variance ½). The range FFT keeps bins 0…N/2−1.
* Peak refinement is a 3-point parabola on natural-log power.
* **CFAR with K summed channels uses exact Gamma(K) α. This is the default in `process_frame`, which uses K = number of channels.**
  * CA: α = N·u/(1−u) with u = betaincinv(K, NK, 1−Pfa).
  * OS: exact quadrature ∫Q(K, α F⁻¹(u))·Beta(u; k, N−k+1) du.
  * GO/SO: exact quadrature over the half-sum quantiles.
  * K = 1 uses the closed forms.
  * All of these assume i.i.d. training cells, which holds for rect windows. Hann inflates Pfa about 1.5× (V1c).
  * OS uses k = round(0.75·N) with round-half-even.
  * GO/SO split cells into halves by the sign of the range offset; cells with zero range offset are split by the sign of the Doppler offset.
    At range edges, n = min(halves). If a half has fewer than a quarter of the training cells, CA is used instead.
* **Phase noise.** One realisation per frame is shared by all TX, RX, scatterers and `Leakage` paths. Each path gets φ(t) − φ(t − τ).
* **Swerling 1/3.** One gain per scatterer per frame, shared by all channels and chirps.
* **Rotor blades.** The default is the line-blade model: one line scatterer per blade, weighted by sinc((2πℓ/λ)·û_blade·û_LOS) (Martin & Mulgrew).
  `blade_model='points'` is the discretised reference.
  The flash rate is N_b·f_rot for even N_b and 2·N_b·f_rot for odd N_b.
* **Micro-Doppler slow time.** Use a single range bin. With Hann, a coherent sum over adjacent bins cancels.

## Validation

`python examples/validation_report.py [--quick]` runs V1–V11 (fidelity §15) and writes
`examples/out/validation_report.md` with figures. `pytest -m slow` runs the paper-grade sizes.
