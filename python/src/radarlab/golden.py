"""Cross-language golden test vectors (schema: ``docs/PROJECT_BRIEF.md``).

Generated only via ``python/scripts/export_golden.py``. Each case is a dict
that is JSON-serialised verbatim. All cases are noiseless and use φ0 = 0;
scatterers move with constant velocity from their t = 0 positions and the
frame starts at t = 0.

Processing used for ``expected``/``rdPowerDb`` (TS must match):
  * periodic Hann range and Doppler windows, Σw-normalised, no zero padding;
  * Doppler fftshift (bin M/2 = 0 m/s);
  * P[r, d] = Σ_k |X[k, r, d]|², rdPowerDb = 10·log10(P) (floor 1e-30);
  * peaks: local maximum of P nearest to the analytic bin; fractional bins by
    parabolic interpolation on natural-log power;
  * angle (conventionVersion 2): synthesis spatial phase is −2π(x_k sinθ cosφ
    + z_k sinφ); TDM compensation with the fractional-bin velocity, rectangular
    window, ``angleFftSize``-point zero-padded spectrum with the **+j kernel**
    A[a] = Σ_k x_k·exp(+j2πka/Na) (= Na·IFFT) over the azimuth ULA, fftshift;
    angle bin a ↔ sinθ = (a − Na/2)/(Na·d), θ > 0 → a > Na/2.
  * the azimuth FFT measures the direction cosine u_x = sinθ·cosφ, so
    ``angleDeg`` = asin(u_x); ``truth.angleDegFft`` is the noiseless value of
    that observable and ``truth.angleDeg`` the true azimuth atan2(x, y).
    ``azimuthDeg``/``elevationDeg``/``xyz`` are the Python point-cloud outputs
    (elevation from same-x/different-z virtual pairs, θ = atan2(x, y)).

Golden set 2 (``"goldenSet": 2``) adds non-scenario files whose **inputs are
stored in the file** (no RNG dependency in the consumer): ``cfar_maps``
(CA/GO/SO/OS thresholds and masks on fixed power maps, K = 1 and K = 8),
``clustering``, ``tracker``, and the scenario cases ``tiny_mti``,
``tiny_real_iq``, ``tiny_physical``; ``tiny_cube`` gains range–angle maps.
See ``golden/README.md`` for what is and is not covered.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import dsp
from .config import C0, PRESETS, RadarConfig
from .scene import Drone, PointTarget, Scene
from .signal import amplitude, synthesize_cube

CONVENTION_VERSION = 2       # brief REVISION 1: negative spatial phase, +j angle kernel
GOLDEN_SET = 2               # set 2: CFAR maps, MTI, real IQ, physical amplitude, RA map, elevation, clustering, tracker
SNR_REF_DB = 20.0
REF_RANGE = 10.0
ANGLE_FFT = 64


def _scat(x: float, y: float, z: float, vx: float, vy: float, vz: float, rcs: float = 1.0) -> dict[str, float]:
    return {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz, "rcs": rcs, "phase0": 0.0}


def polar_scatterer(R: float, az_deg: float, vr: float, el_deg: float = 0.0, rcs: float = 1.0) -> dict[str, float]:
    th, ph = np.deg2rad(az_deg), np.deg2rad(el_deg)
    u = np.array([np.sin(th) * np.cos(ph), np.cos(th) * np.cos(ph), np.sin(ph)])
    p, v = R * u, vr * u
    return _scat(*map(float, p), *map(float, v), rcs)


def _scene(scats: list[dict[str, float]]) -> Scene:
    return Scene([PointTarget((s["x"], s["y"], s["z"]), (s["vx"], s["vy"], s["vz"]), s["rcs"], s["phase0"])
                  for s in scats])


def _process(cube: np.ndarray, cfg: RadarConfig, mti: str | None = None):
    rf, rd = dsp.range_doppler(cube, mti_kind=mti)
    P = dsp.rd_power(rd)
    return rf, rd, P


def _expected_peaks(cfg: RadarConfig, rd: np.ndarray, P: np.ndarray, scats: list[dict[str, float]],
                    n_angle: int = ANGLE_FFT) -> list[dict[str, Any]]:
    out = []
    peaks = dsp.local_maxima(P)
    ula = dsp.azimuth_ula(cfg)
    Md = P.shape[1]
    t_mid = 0.5 * cfg.frame_active_time
    for s in scats:
        p = np.array([s["x"], s["y"], s["z"]]) + t_mid * np.array([s["vx"], s["vy"], s["vz"]])
        v = np.array([s["vx"], s["vy"], s["vz"]])
        R = float(np.linalg.norm(p))
        vr = float(p @ v / R)
        az = float(np.degrees(np.arctan2(p[0], p[1])))
        az_fft = float(np.degrees(np.arcsin(p[0] / R)))       # asin(sinθ·cosφ)
        el = float(np.degrees(np.arcsin(p[2] / R)))
        rb = R / cfg.range_resolution
        # effective carrier at the ADC-window centre (fc is the chirp *start* frequency)
        f_eff = cfg.fc + cfg.slope * (cfg.adc_start_time + 0.5 * cfg.n_samples / cfg.fs)
        fd = 2 * vr * f_eff / C0
        r_beat = R + fd * C0 / (2 * cfg.slope)                 # range-Doppler coupling
        v_eff = vr * f_eff / cfg.fc                            # Doppler scaled with λ = c/fc
        v_eff_alias = (v_eff + cfg.max_velocity) % (2 * cfg.max_velocity) - cfg.max_velocity
        vr_alias = (vr + cfg.max_velocity) % (2 * cfg.max_velocity) - cfg.max_velocity
        db = vr_alias / cfg.velocity_resolution + Md // 2
        dist = np.hypot(peaks[:, 0] - rb, np.minimum(np.abs(peaks[:, 1] - db), Md - np.abs(peaks[:, 1] - db)))
        r, d = map(int, peaks[np.argmin(dist)])
        rf, df = dsp.refine_peak(P, r, d)
        vel = float(dsp.bin_to_velocity(df[0], cfg))
        x_raw = rd[:, r, d]
        x = dsp.tdm_compensate(x_raw, cfg, vel)
        af, a, th = dsp.estimate_azimuth(x, cfg, n_angle)
        af0, a0, th0 = dsp.estimate_azimuth(x_raw, cfg, n_angle)
        pc = dsp.point_cloud(rd, P, np.array([[r, d]]), cfg, n_angle_fft=n_angle)[0]
        out.append({
            "rangeBin": r, "dopplerBin": d, "angleBin": a,
            "rangeBinFrac": float(rf[0]), "dopplerBinFrac": float(df[0]), "angleBinFrac": af,
            "range": float(dsp.bin_to_range(rf[0], cfg)), "velocity": vel,
            "angleDeg": float(np.degrees(th)),
            "angleDegNoTdmComp": float(np.degrees(th0)), "angleBinNoTdmComp": a0,
            "powerDb": float(10 * np.log10(P[r, d])),
            "azimuthDeg": float(np.degrees(pc["azimuth"])), "elevationDeg": float(np.degrees(pc["elevation"])),
            "xyz": [float(pc["x"]), float(pc["y"]), float(pc["z"])],
            "truth": {"range": R, "velocity": vr, "velocityAliased": float(vr_alias), "angleDeg": az,
                      "rangeBeat": float(r_beat), "velocityEffective": float(v_eff_alias),
                      "angleDegFft": az_fft, "elevationDeg": el, "xyz": [float(q) for q in p],
                      "rangeBinAnalytic": rb, "dopplerBinAnalytic": db,
                      "angleBinAnalytic": float(dsp.sin_to_angle_bin(np.sin(np.radians(az)), n_angle, ula.d))},
        })
    return out


INTERP_BINS = 0.03     # parabolic-log interpolation bias on a Hann main lobe (≤ 0.018 bin observed)
ANGLE_DEG = 0.05       # noiseless FFT-DoA error (≤ 0.026° observed, 4-element tiny array)


def _tol(cfg: RadarConfig, peaks: list[dict[str, Any]]) -> dict[str, Any]:
    """Engine-vs-engine tolerances (``*Abs`` without ``truth``) and estimator-vs-truth
    tolerances. The truth tolerances are what the noiseless math supports:

    * ``truthRangeBeatAbs``/``truthVelocityEffectiveAbs``/``truthAngleFftDegAbs``
      compare with the truth of the *observable* (range incl. range–Doppler
      coupling, velocity with the ADC-centre carrier, asin(sinθ·cosφ)) and only
      allow interpolation bias;
    * ``truthRangeAbs``/``truthVelocityAbs``/``truthAngleDegAbs`` compare with the
      geometric truth and add, per case, the largest known deterministic offset
      (coupling, carrier scaling, cosφ projection) of its peaks."""
    ok = [p for p in peaks if abs(p["truth"]["velocity"] - p["truth"]["velocityAliased"]) < 1e-9]
    t = [p["truth"] for p in peaks]
    coup = max([abs(q["rangeBeat"] - q["range"]) for q in t], default=0.0)
    vsc = max([abs(q["velocityEffective"] - q["velocityAliased"]) for q in t], default=0.0)
    proj = max([abs(p["truth"]["angleDegFft"] - p["truth"]["angleDeg"]) for p in ok], default=0.0)
    dR, dv = cfg.range_resolution, cfg.velocity_resolution
    return {"cubeRelMax": 1e-9, "rdPowerDbAbs": 1e-6, "binFracAbs": 1e-6, "integerBins": 0,
            "rangeAbs": 1e-6, "velocityAbs": 1e-6, "angleDegAbs": 1e-4,
            "truthRangeBeatAbs": INTERP_BINS * dR, "truthVelocityEffectiveAbs": INTERP_BINS * dv,
            "truthAngleFftDegAbs": ANGLE_DEG, "truthAzimuthDegAbs": ANGLE_DEG, "truthElevationDegAbs": ANGLE_DEG,
            "truthRangeAbs": INTERP_BINS * dR + coup, "truthVelocityAbs": INTERP_BINS * dv + vsc,
            "truthAngleDegAbs": ANGLE_DEG + proj,
            "toleranceNotes": "truth*Abs = interpolation bias (0.03 bin, 0.05 deg) + largest deterministic "
                              "offset in this case (range-Doppler coupling, carrier scaling, cos(phi) "
                              "projection of the 1-D azimuth FFT). angleDeg is compared with truth.angleDeg "
                              "only when the velocity is not aliased."}


def _rounded(a: np.ndarray, sig: int = 13) -> list[float]:
    """Flatten and round to ``sig`` significant digits (keeps large arrays small;
    rel. error 5e-14 ≪ every tolerance used)."""
    return [float(f"{v:.{sig - 1}e}") for v in np.asarray(a, float).ravel()]


def _case(name: str, desc: str, cfg: RadarConfig, scats: list[dict[str, float]], full_cube: bool,
          extra: dict[str, Any] | None = None, amplitude_mode: str = "snr", mti: str | None = None,
          peak_scats: list[dict[str, float]] | None = None, range_angle: bool = False) -> dict[str, Any]:
    cube = synthesize_cube(cfg, _scene(scats), 0.0, amplitude_mode, SNR_REF_DB, REF_RANGE, noise=False)
    rf, rd, P = _process(cube, cfg, mti)
    proc: dict[str, Any] = {"rangeWindow": "hann", "dopplerWindow": "hann", "angleWindow": "rect",
                            "angleFftSize": ANGLE_FFT, "angleKernel": "+j (ifft*Na), fftshift",
                            "peakInterpolation": "parabolic-log-power",
                            "tdmCompensation": True}
    if mti:
        proc["mti"] = mti
        proc["mtiNote"] = "after the range FFT, before the Doppler window: subtract the slow-time mean per (k, range bin)"
    case: dict[str, Any] = {
        "name": name, "goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION, "description": desc,
        "config": cfg.to_dict(), "derived": cfg.derived(),
        "scatterers": scats, "amplitudeMode": amplitude_mode, "snrRefDb": SNR_REF_DB, "refRange": REF_RANGE,
        "noise": False, "tFrame": 0.0,
        "processing": proc,
    }
    if amplitude_mode == "physical":
        case["physicalNoisePowerW"] = cfg.noise_power_w()
        case["amplitudes"] = [float(amplitude(np.array(sc["rcs"]), np.array(np.hypot(np.hypot(sc["x"], sc["y"]), sc["z"])),
                                              cfg, "physical")) for sc in scats]
    if full_cube:
        case["cube"] = {"shape": list(cube.shape), "re": _rounded(cube.real),
                        "im": _rounded(np.imag(cube))}
        case["rdPowerDb"] = {"shape": list(P.shape), "data": _rounded(dsp.to_db(P))}
    else:
        # spot samples of the cube for cheap cross-checks
        idx = [(0, 0, 0), (cfg.n_virtual - 1, cfg.n_chirps - 1, cfg.n_samples - 1),
               (cfg.n_virtual // 2, cfg.n_chirps // 3, cfg.n_samples // 5)]
        case["cubeSamples"] = [{"k": k, "m": m, "n": n, "re": float(cube[k, m, n].real),
                                "im": float(cube[k, m, n].imag)} for k, m, n in idx]
        r, d = np.unravel_index(int(np.argmax(P)), P.shape)
        case["rdPowerDb"] = {"maxDb": float(dsp.to_db(P).max()), "maxCell": [int(r), int(d)],
                             "meanDb": float(dsp.to_db(P.mean()))}
    if range_angle:
        case["rangeAngle"] = {
            "description": "Sum over all Doppler bins d of |AngleFFT(TDMcomp(X[:, r, d], v_d))|^2, v_d = "
                           "(d - M/2)*dv; +j kernel, fftshift, Sigma(w)-normalised; shape (Nr, Na), row-major",
            "angleFftSize": ANGLE_FFT, "excludeZeroDoppler": False,
            "rect": {"shape": [P.shape[0], ANGLE_FFT],
                     "data": _rounded(dsp.range_angle_map(rd, cfg, ANGLE_FFT, None))},
            "hann": {"shape": [P.shape[0], ANGLE_FFT], "note": "Python default angle window (periodic Hann over ULA slots)",
                     "data": _rounded(dsp.range_angle_map(rd, cfg, ANGLE_FFT, "hann"))},
            "tolerances": {"relOfMax": 1e-9}}
    case["expected"] = {"peaks": _expected_peaks(cfg, rd, P, peak_scats if peak_scats is not None else scats)}
    case["tolerances"] = _tol(cfg, case["expected"]["peaks"])
    if extra:
        case.update(extra)
    return case


def tiny_config() -> RadarConfig:
    """2TX×2RX (K=4), M=16, N=32 configuration for the full-cube golden case."""
    return RadarConfig(fc=77e9, slope=30e12, fs=5e6, n_samples=32, chirp_period=10e-6, n_chirps=16,
                       tx_pos=[0.0, 1.0], rx_pos=[0.0, 0.5], adc_start_time=1e-6, idle_time=2e-6,
                       frame_period=1e-3, name="tiny")


def case_tiny() -> dict[str, Any]:
    cfg = tiny_config()
    scats = [polar_scatterer(6.2, 15.0, 12.0, rcs=1.0), polar_scatterer(14.0, -25.0, -20.0, rcs=0.5)]
    return _case("tiny_cube", "Tiny noiseless 4ch x 16 chirps x 32 samples, 2 scatterers (TDM 2TX x 2RX); "
                 "includes range-angle maps (rect and Hann angle window)",
                 cfg, scats, True, range_angle=True)


def case_tiny_mti() -> dict[str, Any]:
    cfg = tiny_config()
    static = polar_scatterer(9.0, -10.0, 0.0, rcs=20.0)
    moving = polar_scatterer(6.2, 15.0, 12.0, rcs=1.0)
    return _case("tiny_mti", "Tiny config with a strong static scatterer and a mover; MTI (slow-time mean "
                 "removal) suppresses the static one exactly (expected peaks: mover only)",
                 cfg, [static, moving], True, mti="mean", peak_scats=[moving])


def case_tiny_real_iq() -> dict[str, Any]:
    cfg = tiny_config().replace(iq_mode="real", name="tiny_real")
    scats = [polar_scatterer(4.1, 15.0, 12.0, rcs=1.0), polar_scatterer(7.3, -25.0, -20.0, rcs=0.5)]
    return _case("tiny_real_iq", "Tiny config, iqMode 'real': cube = Re(complex beat), im stored as zeros; "
                 "range FFT keeps the first N/2 bins; Rmax halved",
                 cfg, scats, True)


def case_tiny_physical() -> dict[str, Any]:
    cfg = tiny_config()
    scats = [polar_scatterer(6.2, 15.0, 12.0, rcs=1.0), polar_scatterer(14.0, -25.0, -20.0, rcs=10.0)]
    return _case("tiny_physical", "Tiny config, amplitudeMode 'physical' (radar equation; noise k*T0*F*fs, "
                 "variance 1); per-scatterer amplitudes in 'amplitudes'",
                 cfg, scats, True, amplitude_mode="physical")


def case_automotive_single() -> dict[str, Any]:
    cfg = PRESETS["automotive_77ghz"]()
    scats = [polar_scatterer(37.3, 12.0, -6.4)]
    return _case("automotive_single", "automotive_77ghz preset, single point target", cfg, scats, False)


def case_multi() -> dict[str, Any]:
    cfg = PRESETS["automotive_77ghz"]()
    scats = [polar_scatterer(20.0, 0.0, 0.0, rcs=1.0),
             polar_scatterer(45.5, -30.0, -8.3, rcs=10.0),
             polar_scatterer(62.2, 25.0, 11.7, rcs=10.0),
             polar_scatterer(30.4, 40.0, 13.9, rcs=5.0)]
    return _case("automotive_multi",
                 "4 targets: static boresight, negative velocity at -30 deg, +25 deg receding, "
                 "large |v| at +40 deg where TDM compensation matters (compare angleDegNoTdmComp)",
                 cfg, scats, False)


def case_aliased() -> dict[str, Any]:
    cfg = PRESETS["automotive_77ghz"]()
    v = cfg.max_velocity + 3.3 * cfg.velocity_resolution
    scats = [polar_scatterer(25.0, -10.0, float(v))]
    return _case("automotive_velocity_alias",
                 f"target at vr = vmax + 3.3 dv = {v:.4f} m/s appears aliased at negative velocity "
                 "(TDM compensation with aliased v gives a biased angle; see angleDeg)",
                 cfg, scats, False)


def case_awr1843() -> dict[str, Any]:
    cfg = PRESETS["awr1843_like"]()
    scats = [polar_scatterer(12.5, -18.0, 3.1, el_deg=10.0)]
    return _case("awr1843_elevated", "awr1843_like 3TX (TX1 elevated by 0.5 lambda), target at 10 deg elevation",
                 cfg, scats, False)


def case_cfar_alpha() -> dict[str, Any]:
    ns = [8, 16, 24, 32, 48, 64, 128, 248]
    pfas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    rows = []
    for K in (1, 4, 8):
        for n in ns:
            for pfa in pfas:
                k = dsp.os_rank_index(n, 0.75)
                rows.append({"K": K, "n": n, "pfa": pfa, "osK": k,
                             "ca": dsp.ca_alpha(n, pfa, K), "os": dsp.os_alpha(n, pfa, k, K),
                             "go": dsp.gs_alpha(n // 2, pfa, "go", K), "so": dsp.gs_alpha(n // 2, pfa, "so", K)})
    return {"name": "cfar_alpha", "goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION, "description":
            "CFAR threshold multipliers for K non-coherently summed channels (noise Gamma(K), exact). "
            "CA and GO/SO multiply the mean of the training cells (GO/SO: of a half with n/2 cells); OS "
            "multiplies the k-th smallest, k = round(0.75 n) (round-half-even). K=1: closed forms; K>1: "
            "CA via betaincinv, OS/GO/SO via adaptive quadrature of the exact Gamma order-statistic / half-sum "
            "integrals (rel. accuracy ~1e-9).",
            "osRank": 0.75, "table": rows, "tolerances": {"rel": 1e-7}}


def case_presets() -> dict[str, Any]:
    out = []
    for name, fn in PRESETS.items():
        cfg = fn()
        out.append({"name": name, "config": cfg.to_dict(), "derived": cfg.derived(),
                    "virtualPos": cfg.virtual_pos.tolist(), "virtualPosZ": cfg.virtual_pos_z.tolist(),
                    "rangeWindow": dsp.get_window("hann", cfg.n_samples).tolist(),
                    "dopplerWindow": dsp.get_window("hann", cfg.n_chirps).tolist()})
    return {"name": "presets", "goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION, "description": "Preset configs, derived quantities, periodic Hann windows",
            "presets": out, "tolerances": {"rel": 1e-12}}


# ----------------------------------------------------------- golden set 2
CFAR_SHAPE = (20, 16)
CFAR_PARAMS = {"guard": [1, 1], "train": [3, 2], "pfa": 1e-3, "osRank": 0.75}
# (range, Doppler, target power p): range edges r = 0, 1, Nr−1, a Doppler-seam
# pair, a 2×2 blob, interior points and marginal targets (p = 6…12) on which
# CA/GO/SO/OS disagree. Added power: p·s_K with s_1 = 1, s_8 = 2.
CFAR_TARGETS = [(0, 3, 40.0), (1, 8, 12.0), (19, 12, 50.0), (10, 0, 25.0), (10, 15, 20.0),
                (5, 5, 30.0), (5, 6, 22.0), (6, 5, 18.0), (6, 6, 12.0), (14, 10, 15.0),
                (3, 12, 8.0), (8, 9, 6.0), (16, 3, 10.0), (18, 6, 7.0), (12, 4, 9.0), (7, 2, 6.5)]
CFAR_TARGET_SCALE = {1: 1.0, 8: 2.0}


def cfar_power_map(K: int, seed: int = 20261124) -> np.ndarray:
    """Fixed test map: Gamma(K) noise (mean K, i.e. Σ_k |n_k|² with σ² = 1) plus
    point powers p·s_K at :data:`CFAR_TARGETS`. Only used *to create* the stored
    map; consumers read ``P`` from ``cfar_maps.json``."""
    g = np.random.default_rng(seed + K)
    P = g.gamma(K, 1.0, CFAR_SHAPE)
    for r, d, pw in CFAR_TARGETS:
        P[r, d] += CFAR_TARGET_SCALE[K] * pw
    return P


def _cfar_halves(guard, train, shape):
    from .dsp.cfar import _box, _kernels, _pad
    hr, hd, trainm, lead, lag = _kernels(tuple(guard), tuple(train))
    Op = _pad(np.ones(shape), hr, hd, 0.0)
    return (np.rint(_box(Op, lead, shape)).astype(int), np.rint(_box(Op, lag, shape)).astype(int))


def case_cfar_maps() -> dict[str, Any]:
    cp = CFAR_PARAMS
    n_lead, n_lag = _cfar_halves(cp["guard"], cp["train"], CFAR_SHAPE)
    maps = []
    for K in (1, 8):
        P = cfar_power_map(K)
        kinds = {}
        n_tr = None
        for kind in ("ca", "go", "so", "os"):
            r = dsp.cfar_2d(P, tuple(cp["guard"]), tuple(cp["train"]), cp["pfa"], kind, cp["osRank"], n_channels=K)
            n_tr = r.n_train
            ent = {"threshold": _rounded(r.threshold), "alpha": _rounded(r.alpha),
                   "detections": np.argwhere(r.mask).tolist()}
            if kind in ("go", "so"):
                fb = np.minimum(n_lead, n_lag) < r.n_train / 4
                ent["caFallbackCells"] = np.argwhere(fb).tolist()
            kinds[kind] = ent
        maps.append({"K": K, "P": [float(v) for v in P.ravel()], "nTrain": n_tr.ravel().tolist(), **kinds})
    return {"name": "cfar_maps", "goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION,
            "description": "2-D CFAR on fixed stored power maps P[r, d] (row-major, Doppler circular, range "
                           "clipped). Inputs are stored (no RNG needed). For each K and kind: threshold T, "
                           "per-cell alpha (T = alpha * statistic), detections (P > T) as [r, d]. Statistic: CA "
                           "mean of training cells; GO/SO max/min of the means of the leading (dr < 0, or dr = 0 "
                           "and dd < 0) and lagging halves, alpha for n = min(nLead, nLag) cells; if min(nLead, "
                           "nLag) < nTrain/4 the cell falls back to CA with the Gamma(K) CA alpha for nTrain "
                           "cells (caFallbackCells); OS: k-th smallest training cell, k = round_half_even(0.75 "
                           "nTrain) clipped to [1, nTrain]. alpha is the exact Gamma(K) value for the actual "
                           "cell count (K = 1: closed forms).",
            "shape": list(CFAR_SHAPE), **cp, "nLead": n_lead.ravel().tolist(), "nLag": n_lag.ravel().tolist(),
            "maps": maps, "tolerances": {"thresholdRel": 1e-9, "alphaRel": 1e-9, "maskDiffs": 0}}


def clustering_input() -> tuple[np.ndarray, np.ndarray]:
    """Hand-made (mask, P) on a 12×10 grid: a Doppler-seam component, an L shape,
    a diagonal pair (8-connectivity), edge singletons and a seam-diagonal pair."""
    Nr, Md = 12, 10
    r, d = np.meshgrid(np.arange(Nr), np.arange(Md), indexing="ij")
    P = 1.0 + ((7 * r + 3 * d) % 11) / 4.0
    cells = [(3, 0), (3, 9), (4, 9), (4, 0),             # seam (merges across d = 0 / Md−1)
             (7, 3), (8, 3), (9, 3), (9, 4), (9, 5),     # L shape
             (1, 6), (2, 7),                             # diagonal pair
             (0, 2), (11, 8),                            # range-edge singletons
             (6, 0), (7, 9)]                             # diagonal across the seam
    mask = np.zeros((Nr, Md), bool)
    for i, (a, b) in enumerate(cells):
        mask[a, b] = True
        P[a, b] = 20.0 + 3.0 * ((5 * i) % 7) + 0.25 * i
    return mask, P


def case_clustering() -> dict[str, Any]:
    mask, P = clustering_input()
    cl = dsp.cluster_connected(mask, P)
    return {"name": "clustering", "goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION,
            "description": "8-connected components of a stored CFAR mask with circular Doppler axis (cells at d = 0 "
                           "and d = Md-1 in rows r-1..r+1 are neighbours), range not wrapped. Each cluster: its "
                           "strongest cell (peak), power there, size, member cells sorted. Clusters sorted by "
                           "peak power (descending).",
            "shape": list(mask.shape), "P": [float(v) for v in P.ravel()],
            "mask": np.argwhere(mask).tolist(),
            "clusters": [{"rangeBin": c.range_bin, "dopplerBin": c.doppler_bin, "power": c.power, "size": c.size,
                          "cells": sorted(map(list, c.cells.tolist()))} for c in cl],
            "tolerances": {"powerRel": 1e-12}}


def tracker_input() -> tuple[float, list[list[list[float]]]]:
    """Deterministic detection sequence (no RNG): two targets on CV paths with
    small deterministic perturbations, target B missed in frame 5, one clutter
    detection in frame 3. Returns (dt, frames of [R, θ, vr])."""
    dt = 0.1
    paths = [(np.array([-5.0, 30.0]), np.array([2.0, -3.0])), (np.array([8.0, 45.0]), np.array([-1.0, 1.0]))]
    frames = []
    for k in range(12):
        z = []
        for i, (p0, v) in enumerate(paths):
            if i == 1 and k == 5:
                continue
            p = p0 + v * k * dt
            R = float(np.hypot(*p))
            th = float(np.arctan2(p[0], p[1]))
            vr = float(p @ v / R)
            z.append([R + 0.15 * np.sin(1.7 * k + i), th + 0.01 * np.cos(2.3 * k + i), vr + 0.05 * np.sin(0.9 * k + 2 * i)])
        if k == 3:
            z.append([20.0, -0.6, 0.0])
        frames.append([[float(q) for q in m] for m in z])
    return dt, frames


def case_tracker() -> dict[str, Any]:
    from .track import Tracker, TrackerParams
    dt, frames = tracker_input()
    tp = TrackerParams()
    trk = Tracker(tp)
    out = []
    for k, z in enumerate(frames):
        trk.step(np.array(z, float).reshape(-1, 3), dt if k > 0 else 0.0)
        out.append({"frame": k, "tracks": [{"id": t.id, "status": t.status, "x": [float(v) for v in t.x],
                                            "P": [float(v) for v in t.P.ravel()], "hits": [bool(h) for h in t.hits],
                                            "misses": t.misses, "age": t.age} for t in trk.tracks]})
    return {"name": "tracker", "goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION,
            "description": "Python EKF-CV tracker (radarlab.track.Tracker, default TrackerParams) on a stored "
                           "detection sequence z = [R, theta, vr] (theta = atan2(x, y), vr > 0 receding). State "
                           "[px, py, vx, vy]; Q = q*[[T^3/3 I, T^2/2 I], [T^2/2 I, T I]]; GNN (Hungarian) on cost "
                           "d2 + ln det S with a chi2(3) gate (prob gateProb) on the full 3-D innovation; Joseph-form "
                           "update; init x = [R u, vr u] with u = (sin th, cos th), P from polar errors (radial "
                           "velocity std sigmaVr, tangential initVtStd); M/N confirmation, deletion after maxMisses "
                           "consecutive misses. Frame 0 uses dt = 0. Tracks listed in internal order after each "
                           "step (deleted tracks removed). NOTE: the web tracker is a different design (2-D "
                           "position gate, CWNA Q with sigmaAcc, spawn suppression), so this file specifies the "
                           "Python reference only.",
            "dt": dt, "params": {"q": tp.q, "sigmaR": tp.sigma_r, "sigmaTheta": tp.sigma_theta, "sigmaVr": tp.sigma_vr,
                                 "gateProb": tp.gate_prob, "gateChi2": trk.gate, "confirmM": tp.confirm_m,
                                 "confirmN": tp.confirm_n, "maxMisses": tp.max_misses, "initVtStd": tp.init_vt_std},
            "measurements": frames, "frames": out, "tolerances": {"stateAbs": 1e-9, "covRel": 1e-9}}


# ----------------------------------------------------------- micro-Doppler (line-blade rotor)
def md_drone_config() -> RadarConfig:
    """Small deterministic 1TX×1RX 24 GHz config for a drone micro-Doppler slow-time
    sequence (K=1, M=128, N=16; ΔR≈0.75 m, Rmax≈12 m, frame = M·Te gap-free)."""
    n, fs = 16, 4e6
    slope = 200e6 / (n / fs)
    return RadarConfig(fc=24e9, slope=slope, fs=fs, n_samples=n, chirp_period=80e-6,
                       n_chirps=128, tx_pos=np.array([0.0]), rx_pos=np.array([0.0]),
                       adc_start_time=2e-6, idle_time=5e-6, frame_period=128 * 80e-6,
                       iq_mode="complex", name="md_drone")


# Fully specified, RNG-free drone (random_blade_phase=False, rpm_jitter=0, heading=0),
# so both engines reproduce every blade trajectory and the sinc flash weighting.
MD_DRONE = dict(position=(0.0, 6.0, 0.0), velocity=(0.0, 0.0, 0.0), n_rotors=2, n_blades=2,
                rpm=9000.0, blade_length=0.10, blade_root=0.0, arm_length=0.2,
                body_rcs=0.05, blade_rcs=0.02, blade_model="line", rpm_jitter=0.0,
                random_blade_phase=False, heading=0.0)


def _md_target_block() -> dict[str, Any]:
    p = MD_DRONE
    return {"type": "drone", "position": list(p["position"]), "velocity": list(p["velocity"]),
            "heading": p["heading"], "nRotors": p["n_rotors"], "nBlades": p["n_blades"],
            "rpm": p["rpm"], "bladeLength": p["blade_length"], "bladeRoot": p["blade_root"],
            "armLength": p["arm_length"], "rotorZ": 0.05, "bodyRcs": p["body_rcs"],
            "bladeRcs": p["blade_rcs"], "bladeModel": p["blade_model"],
            "note": "rotorZ=0.05 m and the rotor-hub angles pi/4 + 2*pi*i/nRotors are the "
                    "scene.Drone defaults; rotor i spins at (-1)^i * 2*pi*rpm/60; blade k at "
                    "angle 2*pi*k/nBlades + spin*t; one line scatterer per blade centred at "
                    "r0+(L-r0)/2 with segment (L-r0)*(cos,sin,0) in the rotor plane."}


def case_microdoppler_drone() -> dict[str, Any]:
    from .dsp.fft import range_fft
    cfg = md_drone_config()
    drone = Drone(**MD_DRONE)
    cube = synthesize_cube(cfg, Scene([drone]), 0.0, "snr", SNR_REF_DB, REF_RANGE, noise=False)
    rf = range_fft(cube, "hann")                                  # (K, M, Nr), Σw-normalised
    r0 = int(np.argmax((np.abs(rf) ** 2).sum(axis=(0, 1))))       # target range bin
    y = rf[0, :, r0]                                              # slow-time at channel 0
    return {
        "name": "microdoppler_drone", "goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION,
        "description": "Drone micro-Doppler slow-time sequence: line-blade (sinc) rotor model "
                       "(Martin & Mulgrew 1990) in scene.Drone(blade_model='line'). 2 rotors x 2 "
                       "blades, 9000 rpm, hovering at 6 m boresight. Each blade is one uniform line "
                       "scatterer weighted by sinc(Phi), Phi = (2*pi/lambda)*(seg . p_hat); the "
                       "rotor flashes modulate the slow-time signal. Stored: the full noiseless beat "
                       "cube and the range-FFT slow-time y[m] at the target range bin (Hann range "
                       "window, Sigma(w)-normalised). The web engine reproduces it via the ported "
                       "line-blade model (sim/targets.ts drone + the sinc weighting in sim/synth.ts).",
        "config": cfg.to_dict(), "derived": cfg.derived(),
        "amplitudeMode": "snr", "snrRefDb": SNR_REF_DB, "refRange": REF_RANGE, "noise": False, "tFrame": 0.0,
        "target": _md_target_block(),
        "microDoppler": {
            "omegaRadPerS": [float(v) for v in drone.omega],
            "tipSpeed": float(drone.tip_speed[0]),
            "flashRateHz": [float(v) for v in drone.flash_rate()],
            "maxMicroDopplerHz": float(2 * drone.tip_speed[0] / cfg.wavelength),
            "rotationsPerFrame": float(drone.omega[0] / (2 * np.pi) * cfg.frame_active_time),
            "note": "flash rate = N_b*f_rot (even N_b); max micro-Doppler 2*v_tip/lambda (may alias: "
                    "|v_tip| > vmax here, as for real counter-UAS geometries)."},
        "cube": {"shape": list(cube.shape), "re": _rounded(cube.real), "im": _rounded(np.imag(cube))},
        "slowTime": {"rangeBin": r0, "channel": 0,
                     "note": "y[m] = range_fft(cube, 'hann')[0, m, rangeBin]; the micro-Doppler "
                             "slow-time signal (before the Doppler FFT).",
                     "re": _rounded(y.real), "im": _rounded(y.imag)},
        "tolerances": {"cubeRelMax": 1e-9, "slowTimeRelMax": 1e-9},
    }


ALL_CASES = {
    "tiny_cube": case_tiny,
    "automotive_single": case_automotive_single,
    "automotive_multi": case_multi,
    "automotive_velocity_alias": case_aliased,
    "awr1843_elevated": case_awr1843,
    "cfar_alpha": case_cfar_alpha,
    "presets": case_presets,
    "tiny_mti": case_tiny_mti,
    "tiny_real_iq": case_tiny_real_iq,
    "tiny_physical": case_tiny_physical,
    "cfar_maps": case_cfar_maps,
    "clustering": case_clustering,
    "tracker": case_tracker,
    "microdoppler_drone": case_microdoppler_drone,
}


def build_all() -> dict[str, dict[str, Any]]:
    return {k: f() for k, f in ALL_CASES.items()}
