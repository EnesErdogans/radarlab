#!/usr/bin/env python3
"""Write cross-language golden vectors to ``<repo>/golden/*.json``.

Usage: python python/scripts/export_golden.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import scipy.signal

from radarlab import impairments as imp
from radarlab.golden import CONVENTION_VERSION, GOLDEN_SET, _rounded, _scene, build_all, polar_scatterer, tiny_config
from radarlab.signal import Leakage, synthesize_cube


_NUM_LIST = re.compile(r"\[\s*(?:-?[\d.eE+-]+|true|false)(?:,\s*(?:-?[\d.eE+-]+|true|false))*\s*\]")


def dumps(obj) -> str:
    """Indented JSON with numeric/boolean arrays on one line (smaller files, readable structure)."""
    txt = json.dumps(obj, indent=1)
    return _NUM_LIST.sub(lambda m: "[" + ",".join(x.strip() for x in m.group(0)[1:-1].split(",")) + "]", txt)


def case_impairments() -> dict:
    """Deterministic impairments golden (roadmap A4). Every random input is stored, so the
    consumer needs no random number generator:

    * ``phaseNoise``: one stored φ(t) grid (t_i = t0 + i/fsGen, linear interpolation clamped at
      the ends = numpy.interp), passed to ``synthesize_cube`` as a callable; it is shared by both
      scatterers and the Tx→Rx leakage path (each path gets φ(t) − φ(t − τ)).
    * ``swerling``: stored per-frame Swerling-1 complex gains g_s (one per scatterer, shared by
      all channels and chirps). The reference cube is synthesised with rcs·|g|² and phase0 + arg g
      (identical to ``synthesize_cube(swerling=1)`` with those draws; checked below).
    * ``leakage``: ``Leakage(distance, powerDb, phase=<stored per-channel phases>)``.
    * receiver chain on that cube: ``iq_imbalance`` → ``if_filter`` (Butterworth HPF then LPF,
      causal, zero state per chirp) → ``adc_quantize`` (codes; clipping active).
    * ``butter``: scipy.signal.butter (b, a) for orders 1–4, low/high pass.
    """
    cfg = tiny_config()
    scats = [polar_scatterer(6.2, 15.0, 12.0, rcs=1.0), polar_scatterer(14.0, -25.0, -20.0, rcs=0.5)]
    rng = np.random.default_rng(20261126)
    # phase noise: realise once with the Python PhaseNoise FFT shaping, store the grid
    pn = imp.PhaseNoise(l0_dbc_hz=-70.0, loop_bw_hz=450e3, floor_dbc_hz=-130.0, fs_gen=10e6)
    t0, t1 = -2e-6, cfg.n_chirps * cfg.te + cfg.chirp_period
    n = int(np.ceil((t1 - t0) * pn.fs_gen)) + 2
    nfft = 1 << int(np.ceil(np.log2(n)))
    w = rng.standard_normal(nfft)
    f = np.fft.rfftfreq(nfft, 1 / pn.fs_gen)
    phi = np.fft.irfft(np.fft.rfft(w) * np.sqrt(pn.psd(f) * pn.fs_gen), nfft)[:n]
    phi = np.array(_rounded(phi))                                   # stored values are the input
    tg = t0 + np.arange(n) / pn.fs_gen
    pn_fn = lambda t: np.interp(t, tg, phi)                         # noqa: E731
    # Swerling 1 gains (one per scatterer per frame)
    g_rng = np.random.default_rng(7)
    gains = imp.swerling_gains(1, len(scats), cfg.n_tx, cfg.n_chirps, g_rng)[:, 0, 0]
    gains = np.array(_rounded(gains.real)) + 1j * np.array(_rounded(gains.imag))
    folded = [dict(sc, rcs=sc["rcs"] * abs(gk) ** 2, phase0=sc["phase0"] + float(np.angle(gk)))
              for sc, gk in zip(scats, gains)]
    leak_ph = np.array(_rounded(np.random.default_rng(3).uniform(0, 2 * np.pi, cfg.n_virtual)))
    lk = Leakage(distance=0.3, power_db=30.0, phase=leak_ph)
    cube = synthesize_cube(cfg, _scene(folded), 0.0, "snr", 20.0, 10.0, noise=False,
                           phase_noise=pn_fn, leakage=lk)
    # self-check: the folded gains equal synthesize_cube's own Swerling-1 path for the same draws
    raw = imp.swerling_gains(1, len(scats), cfg.n_tx, cfg.n_chirps, np.random.default_rng(7))[:, 0, 0]
    chk_f = [dict(sc, rcs=sc["rcs"] * abs(gk) ** 2, phase0=sc["phase0"] + float(np.angle(gk))) for sc, gk in zip(scats, raw)]
    c_fold = synthesize_cube(cfg, _scene(chk_f), 0.0, "snr", 20.0, 10.0, noise=False, phase_noise=pn_fn, leakage=lk)
    c_sw = synthesize_cube(cfg, _scene(scats), 0.0, "snr", 20.0, 10.0, noise=False, rng=np.random.default_rng(7),
                           swerling=1, phase_noise=pn_fn, leakage=lk)
    assert np.max(np.abs(c_fold - c_sw)) <= 1e-9 * np.max(np.abs(c_sw)), "Swerling folding mismatch"
    cube_no_pn = synthesize_cube(cfg, _scene(folded), 0.0, "snr", 20.0, 10.0, noise=False, leakage=lk)
    # receiver chain
    iq = {"gainDb": 0.8, "phaseDeg": 4.0}
    y1 = imp.iq_imbalance(cube, iq["gainDb"], iq["phaseDeg"])
    flt = {"hpfHz": 300e3, "lpfHz": 1.5e6, "order": 2}
    y2 = imp.if_filter(y1, cfg, flt["hpfHz"], flt["lpfHz"], flt["order"])
    rails = np.concatenate([y2.real.ravel(), y2.imag.ravel()])
    fs_adc = float(_rounded(np.array([0.5 * np.max(np.abs(rails))]))[0])
    bits = 8
    y3, codes = imp.adc_quantize(y2, bits, fs_adc, return_codes=True)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    lsb = fs_adc / 2 ** (bits - 1)
    n_clip = int(sum(np.sum((np.round(r / lsb) < lo) | (np.round(r / lsb) > hi)) for r in (y2.real, y2.imag)))
    auto_fs = 4 * float(np.sqrt(np.mean(rails ** 2)))
    _, codes_auto = imp.adc_quantize(y2, 12, None, return_codes=True)
    bt = []
    for order in (1, 2, 3, 4):
        for wn in (0.06, 0.3):
            for btype in ("lowpass", "highpass"):
                b, a = scipy.signal.butter(order, wn, btype=btype)
                bt.append({"order": order, "wn": wn, "type": btype, "b": [float(v) for v in b], "a": [float(v) for v in a]})
    taus = [2 * 0.3 / 299792458.0, 2 * 6.2 / 299792458.0, 2 * 14.0 / 299792458.0, 2 * 50.0 / 299792458.0, 2 * 150 / 299792458.0]
    pn_ref = imp.PhaseNoise()
    return {
        "name": "impairments", "goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION,
        "kind": "impairments",
        "description": case_impairments.__doc__.strip(),
        "config": cfg.to_dict(), "amplitudeMode": "snr", "snrRefDb": 20.0, "refRange": 10.0, "noise": False,
        "tFrame": 0.0, "scatterers": scats,
        "phaseNoise": {"l0DbcHz": pn.l0_dbc_hz, "loopBwHz": pn.loop_bw_hz, "floorDbcHz": pn.floor_dbc_hz,
                       "fsGen": pn.fs_gen, "t0": t0, "values": [float(v) for v in phi],
                       "note": "phi(t) = numpy.interp(t, t0 + i/fsGen, values), frame-relative time"},
        "swerling": {"model": 1, "gainsRe": [float(v) for v in gains.real], "gainsIm": [float(v) for v in gains.imag],
                     "note": "one complex gain per scatterer per frame, shared by all channels and chirps"},
        "leakage": {"distance": lk.distance, "powerDb": lk.power_db, "phases": [float(v) for v in leak_ph]},
        "cube": {"shape": list(cube.shape), "re": _rounded(cube.real), "im": _rounded(cube.imag)},
        "cubeNoPhaseNoise": {"re": _rounded(cube_no_pn.real), "im": _rounded(cube_no_pn.imag)},
        "iqImbalance": dict(iq, imageRejectionDb=imp.image_rejection_db(iq["gainDb"], iq["phaseDeg"]),
                            re=_rounded(y1.real), im=_rounded(y1.imag)),
        "ifFilter": dict(flt, re=_rounded(y2.real), im=_rounded(y2.imag)),
        "adc": {"bits": bits, "fullScale": fs_adc, "lsb": fs_adc / 2 ** (bits - 1), "clippedRails": n_clip,  # rails whose unclipped code lies outside [lo, hi]
               
                "codesRe": [int(v) for v in codes.real.ravel()], "codesIm": [int(v) for v in codes.imag.ravel()],
                "autoFullScale": auto_fs, "autoBits": 12,
                "autoCodesRe": [int(v) for v in codes_auto.real.ravel()[:256]],
                "autoCodesIm": [int(v) for v in codes_auto.imag.ravel()[:256]],
                "note": "input = ifFilter output; autoCodes* are the first 256 codes with full_scale=None (4x rail rms)"},
        "butter": bt,
        "phaseNoiseResidualVariance": {"params": {"l0DbcHz": pn_ref.l0_dbc_hz, "loopBwHz": pn_ref.loop_bw_hz,
                                                  "floorDbcHz": pn_ref.floor_dbc_hz, "fsGen": pn_ref.fs_gen},
                                       "tau": taus, "var": [pn_ref.residual_variance(t) for t in taus]},
        "tolerances": {"cubeRelMax": 1e-9, "chainRelMax": 1e-9, "butterRel": 1e-10, "codesMismatchMax": 0,
                       "residualVarRel": 1e-6},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "golden")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = build_all()
    cases["impairments"] = case_impairments()
    for name, case in cases.items():
        p = args.out / f"{name}.json"
        p.write_text(dumps(case) + "\n")
        print(f"wrote {p} ({p.stat().st_size / 1024:.1f} KiB)")
    (args.out / "index.json").write_text(json.dumps({"goldenSet": GOLDEN_SET, "conventionVersion": CONVENTION_VERSION,
                                                     "files": [f"{n}.json" for n in cases]}, indent=1) + "\n")


if __name__ == "__main__":
    main()
