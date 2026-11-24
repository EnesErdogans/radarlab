"""Simulated counterparts of the UW captures and the impairment variants.

Everything that is *fitted to the real data* is computed in :func:`fit_all`
and stored in ``results/03_fit.json`` (so the report can list it):

* σ² (noise variance per complex sample, ADC LSB²) — real data are divided by σ
  so that real and simulated cubes are both in units of the noise std;
* trajectory: torso radial velocity v(t) = v0 + a·t (micro-Doppler torso line),
  mean range and azimuth (CFAR detections);
* amplitude: ``snr_ref_db`` (the RCS × system-gain constant of the 'snr'
  amplitude mode, at ref_range = 10 m) so that the median dominant-cell
  power matches the real one; reported also as an implied RCS via the
  'physical' radar equation with the default AWR1843-like link budget;
* impairments: IF high-pass corner/order (noise-floor shape), IQ phase
  imbalance (image level), per-channel complex gain (array mismatch, from the
  parked car), static clutter level (median zero-Doppler power), measured DC.

Not fitted (assumed): radar height 1.0 m, pedestrian height 1.75 m,
cyclist/car geometry, car heading (end-on), phase-noise PSD (radarlab default).
"""
from __future__ import annotations

import numpy as np

import radarlab as rl
from radarlab import dsp, impairments as imp
from radarlab.signal import complex_noise, synthesize_cube

from common import (PROC, TARGET_WINDOW, array_coherence, load, load_json, noise_floor_profile,
                    noise_region_mask, window_noise_gain)
from mdfeat import md_features, md_spectrogram
from targets import Cyclist

H_RADAR = 1.0               # assumed radar height above ground [m]
REF_RANGE = 10.0
MD_BAND = {"pedestrian": (27, 35), "cyclist": (43, 50)}
IMPAIRMENTS = ("hpf", "dc", "iq", "mismatch", "phase_noise", "clutter")
VARIANTS = ("ideal",) + tuple("+" + i for i in IMPAIRMENTS) + ("all",)


def unit(az_deg: float) -> np.ndarray:
    a = np.deg2rad(az_deg)
    return np.array([np.sin(a), np.cos(a), 0.0])


def make_target(scene: str, fit: dict, rcs: float = 1.0):
    f = fit[scene]
    u = unit(f["azimuth_deg"])
    p0 = f["range0"] * u + np.array([0, 0, -H_RADAR])
    v0, a = f["v0"] * u, f["accel"] * u
    if scene == "pedestrian":
        return rl.Pedestrian(p0, v0, height=1.75, rcs_total=rcs, accel=a, seed=3)
    if scene == "cyclist":
        return Cyclist(p0, v0, rcs_total=rcs, accel=a, seed=3)
    # parked car, assumed end-on (heading along the line of sight, rear bumper nearest);
    # a side-on car spreads 4.5 m across the beam and gives a non-point-like return
    # (array coherence 0.65 vs 0.93 measured), so end-on is the better match.
    L = 4.5
    p0 = (f["range0"] + L / 2) * u + np.array([0, 0, -H_RADAR])
    return rl.Car(p0, (0, 0, 0), length=L, rcs_total=rcs, heading=np.deg2rad(f["azimuth_deg"]), seed=3)


def simulate(scene: str, cfg, fit: dict, variant: str = "ideal", seed: int = 0,
             snr_ref_db: float | None = None, noise: bool = True, n_frames: int = 30) -> np.ndarray:
    """Simulated cubes (F, K, M, N) in units of the noise std (σ = 1)."""
    on = set(IMPAIRMENTS) if variant == "all" else {variant.lstrip("+")} & set(IMPAIRMENTS)
    f = fit[scene]
    snr = f["snr_ref_db"] if snr_ref_db is None else snr_ref_db
    targets = [make_target(scene, fit)]
    if "clutter" in on:     # one synthesis call, so target and clutter share the phase-noise realisation
        cl = _clutter()
        cl.rcs = cl.rcs * fit["clutter_scale"][scene] / 10 ** (snr / 10)
        targets.append(cl)
    rng = np.random.default_rng(seed)
    pn = imp.PhaseNoise() if "phase_noise" in on else None
    g = np.asarray(fit["mismatch_re"]) + 1j * np.asarray(fit["mismatch_im"])
    dc = np.asarray(f["dc_re"]) + 1j * np.asarray(f["dc_im"])
    out = []
    for fr in range(n_frames):
        t = fr * cfg.frame_period
        sig = synthesize_cube(cfg, rl.Scene(targets), t, "snr", snr, REF_RANGE, noise=False,
                              rng=rng, phase_noise=pn)
        if "mismatch" in on:
            sig = sig * g[:, None, None]
        x = sig + (complex_noise(sig.shape, rng) if noise else 0)
        if "hpf" in on:
            x = imp.if_filter(x, cfg, fit["hpf_hz"], None, fit["hpf_order"])
        if "iq" in on:
            x = imp.iq_imbalance(x, 0.0, fit["iq_phase_deg"])
        if "dc" in on:
            x = x + dc[:, None, None]
        out.append(x)
    return np.stack(out)


# ------------------------------------------------------------------ fitting
def _peak_power(P: np.ndarray, cfg, scene: str) -> float:
    """Max RD power in the target window (moving: |v| > 0.2 m/s; car: |v| ≤ 2Δv)."""
    lo, hi, moving = TARGET_WINDOW[scene]
    R, v = cfg.range_axis(), cfg.velocity_axis()
    rs = (R >= lo) & (R <= hi)
    vs = np.abs(v) > 0.2 if moving else np.abs(v) <= 2 * cfg.velocity_resolution
    return float(P[np.ix_(rs, vs)].max())


def rd_power_stack(cubes: np.ndarray) -> np.ndarray:
    return np.stack([dsp.rd_power(dsp.doppler_fft(dsp.range_fft(c, PROC.range_window), PROC.doppler_window))
                     for c in cubes])


def fit_all(real: dict, seed: int = 100) -> dict:
    """Fit all real-data constants (see module docstring). ``real`` = results/02_real.json."""
    fit: dict = {"assumed": {"radar_height_m": H_RADAR, "pedestrian_height_m": 1.75, "ref_range_m": REF_RANGE,
                             "car_heading": "end-on (rear bumper nearest)", "phase_noise": "radarlab PhaseNoise() default",
                             "link_budget": "RadarConfig defaults: 12 dBm, 10+10 dBi, NF 15 dB, L 3 dB"}}
    cubes_n, cfg = {}, None
    for scene in ("pedestrian", "cyclist", "car"):
        cubes, cfg = load(scene)
        s2 = real[scene]["noise"]["sigma2"]
        cubes_n[scene] = cubes / np.sqrt(s2)
        ts = real[scene]["track_summary"]
        dc = np.asarray(real[scene]["dc_per_channel"]) / np.sqrt(s2)
        f = {"sigma2_lsb2": s2, "azimuth_deg": ts["azimuth_median"], "dc_re": dc[:, 0], "dc_im": dc[:, 1]}
        if scene in MD_BAND:
            spec, tr, ok = md_spectrogram(cubes_n[scene], cfg, MD_BAND[scene])
            md = md_features(spec, tr, ok)
            f["v0"], f["accel"] = md["torso_v0_fit"], md["torso_accel_fit"]
        else:
            f["v0"], f["accel"] = 0.0, 0.0
        t = np.asarray(real[scene]["track"]["t"])
        f["range0"] = ts["range_mean"] - float(np.mean(f["v0"] * t + 0.5 * f["accel"] * t ** 2))
        fit[scene] = f
    # channel mismatch from the parked car (strong static point target, θ ≈ −25°)
    rf = dsp.range_fft(cubes_n["car"], PROC.range_window)
    rb = int(real["car"]["target_bin"])
    snap = rf[..., rb].mean(axis=(0, 2))
    _, th = array_coherence(snap, cfg.virtual_pos)
    e = snap / np.exp(-2j * np.pi * cfg.virtual_pos * np.sin(np.deg2rad(th)))
    e = e / (np.sqrt(np.mean(np.abs(e) ** 2)) * np.exp(1j * np.angle(e[0])))
    fit["mismatch_re"], fit["mismatch_im"] = e.real, e.imag
    fit["mismatch_gain_db"] = 20 * np.log10(np.abs(e))
    fit["mismatch_phase_deg"] = np.rad2deg(np.angle(e))
    # IQ image: static car return vs its mirror bin (coherent slow-time mean, summed over K)
    st = (np.abs(rf.mean(axis=2)) ** 2).sum(axis=1).mean(axis=0)       # (Nr,)
    N = cfg.n_range_bins
    irr = 10 * np.log10(st[rb] / st[(N - rb) % N])
    fit["irr_db_measured"] = float(irr)
    ph = np.linspace(0.01, 30, 3000)
    fit["iq_phase_deg"] = float(ph[np.argmin(np.abs([imp.image_rejection_db(0.0, p) - irr for p in ph]))])
    # IF high-pass: corner and order that best reproduce the real noise-floor shape
    prof_real = np.mean([real[s]["noise_floor_profile_db"] for s in ("pedestrian", "cyclist", "car")], axis=0)
    rng = np.random.default_rng(seed)
    noise = complex_noise((3, cfg.n_virtual, cfg.n_chirps, cfg.n_samples), rng)
    m = noise_region_mask(cfg, exclude_zero_doppler=3)
    nb = cfg.n_range_bins
    off = np.minimum(np.arange(nb), nb - np.arange(nb))
    use = off > 2          # DC bins 0, ±1, ±2 are filled by the unmodelled DC Doppler skirt
    best = (np.inf, None, None)
    for order in (1, 2, 3, 4):
        for fc in np.arange(50e3, 1501e3, 25e3):
            P = rd_power_stack(imp.if_filter(noise, cfg, fc, None, order))
            s2 = P[:, m].mean() / (cfg.n_virtual * window_noise_gain(cfg))
            err = np.sqrt(np.mean((noise_floor_profile(P, cfg, s2) - prof_real)[use] ** 2))
            if err < best[0]:
                best = (float(err), float(fc), order)
    P = rd_power_stack(noise)
    s2 = P[:, m].mean() / (cfg.n_virtual * window_noise_gain(cfg))
    fit["hpf_rms_err_db"], fit["hpf_hz"], fit["hpf_order"] = best
    fit["hpf_fit_bins"] = "all except |bin| <= 2"
    fit["no_hpf_rms_err_db"] = float(np.sqrt(np.mean((noise_floor_profile(P, cfg, s2) - prof_real)[use] ** 2)))
    # static clutter level: median zero-Doppler power over 2–13 m (positive half)
    R = cfg.range_axis()
    rsel = (R > 2) & (R < 13)
    fit["clutter_scale"] = {}
    for scene in ("pedestrian", "cyclist", "car"):
        Pr = rd_power_stack(cubes_n[scene][:3])[:, rsel, cfg.n_chirps // 2]
        clut = simulate_clutter_only(cfg, seed)
        Ps = rd_power_stack(clut)[:, rsel, cfg.n_chirps // 2]
        fit["clutter_scale"][scene] = float(np.median(Pr) / np.median(Ps))
    # amplitude constant (snr_ref_db) from the median dominant-cell power
    for scene in ("pedestrian", "cyclist", "car"):
        P_real = rd_power_stack(cubes_n[scene])
        P0 = rd_power_stack(simulate(scene, cfg, fit, "ideal", seed, snr_ref_db=0.0, noise=False))
        pr = np.median([_peak_power(p, cfg, scene) for p in P_real])
        p0 = np.median([_peak_power(p, cfg, scene) for p in P0])
        fit[scene]["snr_ref_db"] = float(10 * np.log10(pr / p0))
        phys = rl.signal.snr_per_sample_db(1.0, REF_RANGE, cfg, "physical")
        fit[scene]["snr_ref_physical_1m2_db"] = phys
        fit[scene]["implied_rcs_dbsm"] = fit[scene]["snr_ref_db"] - phys
    return fit


def _clutter():
    """150 static scatterers, 1–14 m, ±60°, 0.5 m above ground, exponential RCS (seeded)."""
    return rl.static_clutter(150, 1.0, 14.0, -60, 60, z=-H_RADAR + 0.5, rcs_mean=1.0, rng=11)


def simulate_clutter_only(cfg, seed: int, n_frames: int = 3) -> np.ndarray:
    cl = _clutter()
    return np.stack([synthesize_cube(cfg, cl, f * cfg.frame_period, "snr", 0.0, REF_RANGE, noise=False,
                                     rng=seed) for f in range(n_frames)])


def load_fit() -> dict:
    return load_json("03_fit")
