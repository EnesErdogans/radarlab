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
  * angle: TDM compensation with the fractional-bin velocity, rectangular
    window, ``angleFftSize``-point zero-padded FFT over the azimuth ULA,
    fftshift; angle bin a ↔ sinθ = (a − Na/2)/(Na·d).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import dsp
from .config import PRESETS, RadarConfig
from .scene import PointTarget, Scene
from .signal import synthesize_cube

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


def _process(cube: np.ndarray, cfg: RadarConfig, n_angle: int = ANGLE_FFT):
    rf, rd = dsp.range_doppler(cube)
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
        rb = R / cfg.range_resolution
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
        out.append({
            "rangeBin": r, "dopplerBin": d, "angleBin": a,
            "rangeBinFrac": float(rf[0]), "dopplerBinFrac": float(df[0]), "angleBinFrac": af,
            "range": float(dsp.bin_to_range(rf[0], cfg)), "velocity": vel,
            "angleDeg": float(np.degrees(th)),
            "angleDegNoTdmComp": float(np.degrees(th0)), "angleBinNoTdmComp": a0,
            "powerDb": float(10 * np.log10(P[r, d])),
            "truth": {"range": R, "velocity": vr, "velocityAliased": float(vr_alias), "angleDeg": az,
                      "rangeBinAnalytic": rb, "dopplerBinAnalytic": db,
                      "angleBinAnalytic": float(dsp.sin_to_angle_bin(np.sin(np.radians(az)), n_angle, ula.d))},
        })
    return out


def _tol(cfg: RadarConfig) -> dict[str, float]:
    return {"cubeRelMax": 1e-9, "rdPowerDbAbs": 1e-6, "binFracAbs": 1e-6, "integerBins": 0,
            "rangeAbs": 1e-6, "velocityAbs": 1e-6, "angleDegAbs": 1e-4,
            "truthRangeAbs": cfg.range_resolution / 2, "truthVelocityAbs": cfg.velocity_resolution / 2,
            "truthAngleDegAbs": 2.0}


def _case(name: str, desc: str, cfg: RadarConfig, scats: list[dict[str, float]], full_cube: bool,
          extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cube = synthesize_cube(cfg, _scene(scats), 0.0, "snr", SNR_REF_DB, REF_RANGE, noise=False)
    rf, rd, P = _process(cube, cfg)
    case: dict[str, Any] = {
        "name": name, "description": desc, "config": cfg.to_dict(), "derived": cfg.derived(),
        "scatterers": scats, "amplitudeMode": "snr", "snrRefDb": SNR_REF_DB, "refRange": REF_RANGE,
        "noise": False, "tFrame": 0.0,
        "processing": {"rangeWindow": "hann", "dopplerWindow": "hann", "angleWindow": "rect",
                       "angleFftSize": ANGLE_FFT, "peakInterpolation": "parabolic-log-power",
                       "tdmCompensation": True},
    }
    if full_cube:
        case["cube"] = {"shape": list(cube.shape), "re": cube.real.ravel().tolist(),
                        "im": cube.imag.ravel().tolist()}
        case["rdPowerDb"] = {"shape": list(P.shape), "data": dsp.to_db(P).ravel().tolist()}
    else:
        # spot samples of the cube for cheap cross-checks
        idx = [(0, 0, 0), (cfg.n_virtual - 1, cfg.n_chirps - 1, cfg.n_samples - 1),
               (cfg.n_virtual // 2, cfg.n_chirps // 3, cfg.n_samples // 5)]
        case["cubeSamples"] = [{"k": k, "m": m, "n": n, "re": float(cube[k, m, n].real),
                                "im": float(cube[k, m, n].imag)} for k, m, n in idx]
        r, d = np.unravel_index(int(np.argmax(P)), P.shape)
        case["rdPowerDb"] = {"maxDb": float(dsp.to_db(P).max()), "maxCell": [int(r), int(d)],
                             "meanDb": float(dsp.to_db(P.mean()))}
    case["expected"] = {"peaks": _expected_peaks(cfg, rd, P, scats)}
    case["tolerances"] = _tol(cfg)
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
    return _case("tiny_cube", "Tiny noiseless 4ch x 16 chirps x 32 samples, 2 scatterers (TDM 2TX x 2RX)",
                 cfg, scats, True)


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
    for n in ns:
        for pfa in pfas:
            k = dsp.os_rank_index(n, 0.75)
            rows.append({"n": n, "pfa": pfa, "ca": dsp.ca_alpha(n, pfa), "osK": k,
                         "os": dsp.os_alpha(n, pfa, k), "go": dsp.gs_alpha(n // 2, pfa, "go"),
                         "so": dsp.gs_alpha(n // 2, pfa, "so"),
                         "caGammaK4": dsp.ca_alpha(n, pfa, 4)})
    return {"name": "cfar_alpha", "description":
            "CFAR threshold multipliers on the mean (CA, GO/SO: halves of n/2 cells) or on the k-th "
            "order statistic (OS, k=round(0.75 n)); caGammaK4 is CA for 4 non-coherently summed channels.",
            "osRank": 0.75, "table": rows, "tolerances": {"rel": 1e-9}}


def case_presets() -> dict[str, Any]:
    out = []
    for name, fn in PRESETS.items():
        cfg = fn()
        out.append({"name": name, "config": cfg.to_dict(), "derived": cfg.derived(),
                    "virtualPos": cfg.virtual_pos.tolist(), "virtualPosZ": cfg.virtual_pos_z.tolist(),
                    "rangeWindow": dsp.get_window("hann", cfg.n_samples).tolist(),
                    "dopplerWindow": dsp.get_window("hann", cfg.n_chirps).tolist()})
    return {"name": "presets", "description": "Preset configs, derived quantities, periodic Hann windows",
            "presets": out, "tolerances": {"rel": 1e-12}}


ALL_CASES = {
    "tiny_cube": case_tiny,
    "automotive_single": case_automotive_single,
    "automotive_multi": case_multi,
    "automotive_velocity_alias": case_aliased,
    "awr1843_elevated": case_awr1843,
    "cfar_alpha": case_cfar_alpha,
    "presets": case_presets,
}


def build_all() -> dict[str, dict[str, Any]]:
    return {k: f() for k, f in ALL_CASES.items()}
