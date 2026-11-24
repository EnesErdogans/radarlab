r"""TI mmWave interfaces: SDK/mmWave-Studio CLI ``.cfg`` parser and DCA1000 raw
``.bin`` reader/writer for xWR16xx/18xx/68xx devices.

``.cfg`` → :class:`RadarConfig` mapping (mmWave SDK user guide, CLI commands)::

    profileCfg  id startFreq[GHz] idle[us] adcStart[us] rampEnd[us] txPwr txPhase
                slope[MHz/us] txStart[us] numAdcSamples fs[ksps] hpf1 hpf2 rxGain
    chirpCfg    start end profileId fVar slopeVar idleVar adcVar txEnableMask
    frameCfg    chirpStart chirpEnd numLoops numFrames period[ms] trigSel trigDelay
    channelCfg  rxEnMask txEnMask cascading
    adcCfg      numADCBits(0:12,1:14,2:16) adcOutputFmt(0:real,1:complex1x,2:complex2x)

    fc = startFreq (chirp start frequency, see RadarConfig), Tc = idle + rampEnd,
    nTx = chirps per loop (TDM, one TX bit per chirp), M = numLoops,
    txPos/rxPos from the device antenna geometry for the enabled elements.

DCA1000 LVDS format for xWR16xx/18xx/68xx (2 lanes), following TI's
``readDCA1000.m`` in "Mmwave Radar Device ADC Raw Data Capture" (SWRA581B):

* int16 little-endian stream;
* complex: groups of four words [I0, I1, Q0, Q1] → samples I0+jQ0, I1+jQ1
  (same layout for complex1x and complex2x);
* real: one word per sample;
* per chirp: all N samples of RX0, then RX1, … (non-interleaved);
* chirps in transmission order c = m·nTx + t; frames concatenated.

The reader returns cubes (F, K, M, N) with k = t·nRx + r (brief layout).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RadarConfig

# antenna geometry (wavelengths) per device: tx (x, z), rx x
DEVICE_GEOMETRY: dict[str, dict[str, Any]] = {
    "awr1843": {"tx": [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)], "rx": [0.0, 0.5, 1.0, 1.5]},
    "iwr1843": {"tx": [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)], "rx": [0.0, 0.5, 1.0, 1.5]},
    "iwr6843isk": {"tx": [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)], "rx": [0.0, 0.5, 1.0, 1.5]},
    "awr1642": {"tx": [(0.0, 0.0), (2.0, 0.0)], "rx": [0.0, 0.5, 1.0, 1.5]},
}


@dataclass
class TiCfg:
    """Parsed TI configuration: the RadarConfig plus capture metadata."""

    config: RadarConfig
    adc_bits: int = 16
    adc_format: str = "complex1x"          # 'real' | 'complex1x' | 'complex2x'
    n_frames: int = 0
    tx_order: list[int] = field(default_factory=list)
    rx_enabled: list[int] = field(default_factory=list)
    commands: dict[str, list[list[str]]] = field(default_factory=dict)


def _bits(mask: int) -> list[int]:
    return [i for i in range(16) if mask >> i & 1]


def parse_ti_cfg(text_or_path: str | Path, device: str = "awr1843") -> TiCfg:
    """Parse a TI mmWave SDK CLI ``.cfg`` (text or file path) into a :class:`TiCfg`.

    Only single-profile TDM configurations are supported (one TX per chirp)."""
    p = Path(str(text_or_path))
    text = p.read_text() if ("\n" not in str(text_or_path) and p.exists()) else str(text_or_path)
    cmds: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        line = line.split("%", 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        cmds.setdefault(tok[0], []).append(tok[1:])
    if "profileCfg" not in cmds:
        raise ValueError("profileCfg missing")
    if len(cmds["profileCfg"]) > 1:
        raise NotImplementedError("multiple profiles not supported")
    pr = [float(v) for v in cmds["profileCfg"][0]]
    prof_id = int(pr[0])
    fc = pr[1] * 1e9
    idle, adc_start, ramp_end = pr[2] * 1e-6, pr[3] * 1e-6, pr[4] * 1e-6
    slope = pr[7] * 1e12
    n_samples = int(pr[9])
    fs = pr[10] * 1e3
    geom = DEVICE_GEOMETRY[device.lower()]
    rx_en = list(range(len(geom["rx"])))
    if "channelCfg" in cmds:
        rx_en = _bits(int(cmds["channelCfg"][0][0]))
    chirp_tx: dict[int, int] = {}
    for c in cmds.get("chirpCfg", []):
        s, e, pid = int(c[0]), int(c[1]), int(c[2])
        if pid != prof_id:
            continue
        txb = _bits(int(c[7]))
        if len(txb) != 1:
            raise NotImplementedError("chirps with simultaneous TX (BPM) are not supported")
        for i in range(s, e + 1):
            chirp_tx[i] = txb[0]
    n_loops, n_frames, frame_period = 1, 0, 0.1
    c0, c1 = (min(chirp_tx), max(chirp_tx)) if chirp_tx else (0, 0)
    if "frameCfg" in cmds:
        fcmd = cmds["frameCfg"][0]
        c0, c1 = int(fcmd[0]), int(fcmd[1])
        n_loops, n_frames = int(fcmd[2]), int(fcmd[3])
        frame_period = float(fcmd[4]) * 1e-3
    tx_order = [chirp_tx.get(i, 0) for i in range(c0, c1 + 1)]
    bits, fmt = 16, "complex1x"
    if "adcCfg" in cmds:
        bits = {0: 12, 1: 14, 2: 16}[int(cmds["adcCfg"][0][0])]
        fmt = {0: "real", 1: "complex1x", 2: "complex2x"}[int(cmds["adcCfg"][0][1])]
    tx = np.array([geom["tx"][t] for t in tx_order], float)
    rx = np.array([geom["rx"][r] for r in rx_en], float)
    cfg = RadarConfig(
        fc=fc, slope=slope, fs=fs, n_samples=n_samples, chirp_period=idle + ramp_end,
        n_chirps=n_loops, tx_pos=tx[:, 0], tx_pos_z=tx[:, 1] if np.any(tx[:, 1]) else None,
        rx_pos=rx, adc_start_time=adc_start, idle_time=idle, frame_period=frame_period,
        iq_mode="real" if fmt == "real" else "complex", name=f"ti_{device}")
    return TiCfg(cfg, bits, fmt, n_frames, tx_order, rx_en, cmds)


# ------------------------------------------------------------------ DCA1000
def _frame_words(cfg: RadarConfig) -> int:
    per = cfg.n_virtual * cfg.n_chirps * cfg.n_samples
    return per * (2 if cfg.iq_mode == "complex" else 1)


def dca1000_decode(raw: np.ndarray, cfg: RadarConfig, adc_bits: int = 16,
                   sign_extend: bool = False, conj: bool = False) -> np.ndarray:
    """Decode an int16 word stream into cubes (F, K, M, N).

    ``sign_extend`` re-applies two's complement for 12/14-bit data stored
    unsigned (as in TI's script); ``conj`` flips the Q sign."""
    w = np.asarray(raw).astype(np.int32)
    if sign_extend and adc_bits != 16:
        lmax = 2 ** (adc_bits - 1) - 1
        w = np.where(w > lmax, w - 2 ** adc_bits, w)
    fw = _frame_words(cfg)
    nf = w.size // fw
    if nf == 0:
        raise ValueError("file shorter than one frame")
    w = w[: nf * fw].astype(float)
    if cfg.iq_mode == "complex":
        g = w.reshape(-1, 4)
        s = np.empty(g.shape[0] * 2, complex)
        s[0::2] = g[:, 0] + 1j * g[:, 2]
        s[1::2] = g[:, 1] + 1j * g[:, 3]
        if conj:
            s = s.conj()
    else:
        s = w
    nT, nR, M, N = cfg.n_tx, cfg.n_rx, cfg.n_chirps, cfg.n_samples
    # stream order: frame, loop m, tx t, rx r, sample n
    s = s.reshape(nf, M, nT, nR, N).transpose(0, 2, 3, 1, 4)
    return s.reshape(nf, nT * nR, M, N)


def dca1000_encode(cubes: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """Inverse of :func:`dca1000_decode`: cubes (F, K, M, N) or (K, M, N)
    of integer-valued samples → int16 word stream."""
    c = np.asarray(cubes)
    if c.ndim == 3:
        c = c[None]
    nf = c.shape[0]
    nT, nR, M, N = cfg.n_tx, cfg.n_rx, cfg.n_chirps, cfg.n_samples
    s = c.reshape(nf, nT, nR, M, N).transpose(0, 3, 1, 2, 4).reshape(-1)
    if cfg.iq_mode == "complex":
        s = s.astype(complex)
        g = np.empty((s.size // 2, 4))
        g[:, 0], g[:, 1] = s[0::2].real, s[1::2].real
        g[:, 2], g[:, 3] = s[0::2].imag, s[1::2].imag
        w = g.reshape(-1)
    else:
        w = np.real(s)
    return np.clip(np.round(w), -32768, 32767).astype("<i2")


def read_dca1000(path: str | Path, cfg: RadarConfig, adc_bits: int = 16, **kw: Any) -> np.ndarray:
    """Read a DCA1000 ``.bin`` capture → (F, K, M, N)."""
    return dca1000_decode(np.fromfile(path, dtype="<i2"), cfg, adc_bits, **kw)


def write_dca1000(path: str | Path, cubes: np.ndarray, cfg: RadarConfig) -> None:
    """Write cubes in DCA1000 ``.bin`` format (for tests / synthetic captures)."""
    dca1000_encode(cubes, cfg).tofile(path)
