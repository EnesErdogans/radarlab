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

## Figures

Run `python examples/<name>.py` from `python/examples` (matplotlib is required). Output goes to `examples/out/`.

* `rd_map_automotive.py`: RD map with OS-CFAR clusters and a bird's-eye point cloud
* `micro_doppler.py`: quadrotor vs bird spectrograms
* `cfar_pfa.py`: empirical vs design Pfa (CA, OS)
* `rmse_crlb.py`: ML RMSE vs CRLB for angle, velocity and range

## Convention notes (beyond the brief)

* `fc` is the **chirp start frequency** (TI `startFreq`). The brief's beat model needs this because `t_f` is measured from the chirp start.
  λ = c/fc is used for all Doppler scaling.
* Element positions: `txPos`/`rxPos` are x in wavelengths. Optional `txPosZ`/`rxPosZ` add the phase term `2π·z_k·sinφ`.
* Real IQ (`iq_mode='real'`) means the real part of the complex cube, noise included (real noise variance ½).
  The range FFT keeps the first N/2 bins.
* The angle FFT runs over the z = 0 virtual channels with unique x, placed on a grid of the smallest spacing d.
  Gaps are zero-filled. The window is rectangular by default, the FFT is fftshifted, and bin a maps to sinθ = (a − Na/2)/(Na·d).
* Peak refinement is a 3-point parabola on natural-log power (Gaussian interpolation).
* CFAR GO/SO halves are split by range-offset sign; cells with zero range offset are split by Doppler-offset sign.
  At range edges α uses n = min(halves), and CA is used if a half holds fewer than a quarter of the cells.
  OS and GO/SO always use the K = 1 (exponential) formulas.
  For CA, `n_channels=K` gives the Gamma-exact α: Pfa = 1 − I_{c/(1+c)}(K, NK), with c = α/N.
