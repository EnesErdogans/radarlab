"""Micro-Doppler spectrogram over frames and signature features (identical
for real and simulated data).

Slow time: per frame, range FFT (Hann) + slow-time mean removal (``dsp.mti``),
then ``dsp.slow_time`` concatenates the 255 chirps of TX0 over all 30 frames
(as the UW MATLAB script does: the 2.7 ms inter-frame gap is *not* filled).
STFT columns whose window straddles a frame boundary see a phase jump
(broadband stripe) and are marked invalid: features use only the ~16 of 31
columns per frame that lie inside one frame; figures leave the others blank. STFT
(``dsp.spectrogram``): Hann, 128 chirps (15.4 ms, Δv = 0.127 m/s), hop 8.
Power is summed incoherently over the 4 RX of TX0 and over the range band
that contains the target in all frames.
"""
from __future__ import annotations

import numpy as np

from radarlab import dsp

NPERSEG, NOVERLAP = 128, 120
THR_DB = 10.0          # envelope threshold above the noise floor
NOTCH = 0.15           # |v| below this is ignored (MTI notch) [m/s]


def md_spectrogram(cubes: np.ndarray, cfg, band: tuple[int, int]
                   ) -> tuple[dsp.Spectrogram, np.ndarray, np.ndarray]:
    """Spectrogram (power summed over RX of TX0 and range bins band[0]..band[1]),
    the real (wall-clock) time of each STFT column and a validity mask
    (window inside one frame)."""
    rf = [dsp.mti(dsp.range_fft(c, "hann"), axis=-2, kind="mean") for c in cubes]
    prf = 1.0 / cfg.te
    spec = None
    for b in range(band[0], band[1] + 1):
        for r in range(cfg.n_rx):
            y = dsp.slow_time(rf, b, channels=r, cfg=cfg, tx=0)
            s = dsp.spectrogram(y, prf, cfg.wavelength, NPERSEG, NOVERLAP)
            spec = s if spec is None else dsp.Spectrogram(s.times, s.velocity, s.freq, spec.power + s.power)
    # chirp-index time → wall-clock time (frame gaps)
    idx = spec.times * prf
    M = cfg.n_chirps
    t_real = (idx // M) * cfg.frame_period + (idx % M) * cfg.te
    start = np.rint(idx - NPERSEG / 2).astype(int)
    valid = (start // M) == ((start + NPERSEG - 1) // M)
    return spec, t_real, valid


def md_features(spec: dsp.Spectrogram, t_real: np.ndarray, valid: np.ndarray,
                motion_sign: float = -1.0) -> dict:
    """Signature features from the valid columns. ``motion_sign`` = sign of the
    bulk radial velocity (−1 approaching); the limb excursion is measured on
    that side. The cadence is the dominant period (0.25–0.9 s) of the limb
    envelope; with ~1 s of data it is resolved to no better than ±0.1 s."""
    v = spec.velocity
    S = spec.power[:, valid]
    t_real = t_real[valid]
    far = np.abs(v) > 6.0
    nf = np.median(S[far])
    Sdb = 10 * np.log10(S / nf)
    use = np.abs(v) >= NOTCH
    torso, lo, hi, bw = [], [], [], []
    for j in range(S.shape[1]):
        col = np.where(use, Sdb[:, j], -np.inf)
        torso.append(v[np.argmax(col)])
        above = np.where(col > THR_DB)[0]
        if len(above):
            lo.append(v[above.min()]), hi.append(v[above.max()])
            w = 10 ** (col[above] / 10)
            mu = np.sum(w * v[above]) / w.sum()
            bw.append(np.sqrt(np.sum(w * (v[above] - mu) ** 2) / w.sum()))
        else:
            lo.append(np.nan), hi.append(np.nan), bw.append(np.nan)
    torso, lo, hi, bw = map(np.asarray, (torso, lo, hi, bw))
    ext = hi - lo
    limb = lo if motion_sign < 0 else hi                 # envelope on the motion side
    # cadence: dominant period of the limb envelope (uniform wall-clock grid)
    tg = np.arange(t_real[0], t_real[-1], 0.005)
    ok = ~np.isnan(limb)
    vt = np.polyfit(t_real, torso, 1)            # torso velocity trend v(t) = a·t + v0
    s = np.interp(tg, t_real[ok], limb[ok])
    s = s - np.polyval(np.polyfit(tg, s, 1), tg)
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac /= ac[0]
    lags = np.arange(len(ac)) * 0.005
    win = (lags >= 0.25) & (lags <= 0.9)
    period = float(lags[win][np.argmax(ac[win])]) if win.any() else float("nan")
    nfft = 4096
    Sp = np.abs(np.fft.rfft(s * np.hanning(len(s)), nfft)) ** 2
    fr = np.fft.rfftfreq(nfft, 0.005)
    band = (fr > 0.8) & (fr < 5)
    return {
        "noise_floor": float(nf),
        "torso_velocity_median": float(np.median(torso)),
        "torso_velocity_iqr": float(np.subtract(*np.percentile(torso, [75, 25]))),
        "torso_accel_fit": float(vt[0]), "torso_v0_fit": float(vt[1]),
        "extent_median": float(np.nanmedian(ext)), "extent_p95": float(np.nanpercentile(ext, 95)),
        "v_min_p5": float(np.nanpercentile(lo, 5)), "v_max_p95": float(np.nanpercentile(hi, 95)),
        "bandwidth_rms_median": float(np.nanmedian(bw)),
        "limb_envelope_period_s": period,
        "limb_envelope_rate_hz": float(1 / period) if period == period else float("nan"),
        "limb_envelope_peak_hz": float(fr[band][np.argmax(Sp[band])]),
        "ac_peak": float(ac[win].max()) if win.any() else float("nan"),
        "duration_s": float(t_real[-1] - t_real[0]),
        "series": {"t": t_real, "torso": torso, "lo": lo, "hi": hi},
    }
