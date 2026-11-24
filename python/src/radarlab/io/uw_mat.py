r"""Loader for the University of Washington AWR1843 template captures.

Source: X. Gao, G. Xing, S. Roy, H. Liu, "Experiments with mmWave Automotive
Radar Test-bed", Asilomar 2019, DOI 10.1109/IEEECONF44664.2019.9048939
(arXiv:1912.12566). Data and MATLAB code (MIT licence):
github.com/Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification,
files ``template data/{pms,bms,cms}1000_30fs.mat``.

File format (``generate_ra_3dfft.m``, ``read_uwcr.m`` in that repository)::

    data_frames : complex, shape (nRx=4, F·M·nTx·N) = (4, 30·255·2·128)
    column index = ((f·M + m)·nTx + t)·N + n          (frame, loop, TX, sample)

i.e. already decoded from the DCA1000 stream; odd MATLAB chirps (t = 0) come
from the first TX, even ones (t = 1) from the second.

Published configuration (``config/get_params_value.m`` and the paper):
startFreq 77 GHz, slope 21.0017 MHz/µs, fs = 4 Msps complex, N = 128,
255 loops, 2 TX (TX0, TX2 of the AWR1843, 2λ apart) × 4 RX (λ/2), Tc = 120 µs
per TX cycle (60 µs per chirp), 30 frames/s. **Not published**: ADC start
time, idle/ramp split, RX gain and the scaling of the stored values (the
repository normalises with ``max_value = 1e4``; stored |x| ≲ 200).

Empirical findings (``docs/real_data/report.md``, scripts in
``python/examples/real_data/``):

* I/Q is not conjugated (most power, and all moving targets, lie in the
  positive range half; range rate from the detection track agrees in sign
  with the Doppler velocity).
* The virtual-array phase is linear across all 8 channels with the default
  TX order ``(0, 1)`` and TX spacing 2λ (the swapped order gives a phase
  jump between channel 3 and 4); see ``check_ordering`` in the examples.
* ``adc_start_time`` is unobservable in this data to first order: in the
  beat model it adds a per-target constant phase 2π·S·τ·t0 and changes the
  Doppler wavelength by S·t0/fc ≈ 0.16 % (t0 = 6 µs). The default 6 µs is
  the TI mmWave-Studio default and an assumption, not a measurement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import RadarConfig

UW_CITATION = ("X. Gao, G. Xing, S. Roy, H. Liu, \"Experiments with mmWave Automotive Radar "
               "Test-bed,\" Asilomar 2019, DOI 10.1109/IEEECONF44664.2019.9048939")

UW_SCENES = {"pms1000": "pedestrian", "bms1000": "cyclist", "cms1000": "car"}


def uw_awr1843_config(adc_start_time: float = 6e-6, idle_time: float = 10e-6,
                      **overrides: Any) -> RadarConfig:
    """RadarConfig for the UW AWR1843 captures (published values, see module doc).

    ``adc_start_time`` and ``idle_time`` are *assumed* (not published); they do
    not change the processed range/Doppler/angle outputs beyond a constant
    phase and a 0.16 % Doppler-scale change (module docstring).
    """
    kw: dict[str, Any] = dict(
        fc=77e9, slope=21.0017e12, fs=4e6, n_samples=128, chirp_period=60e-6,
        n_chirps=255, tx_pos=np.array([0.0, 2.0]), rx_pos=np.array([0.0, 0.5, 1.0, 1.5]),
        adc_start_time=adc_start_time, idle_time=idle_time, frame_period=1.0 / 30.0,
        iq_mode="complex", name="uw_awr1843")
    kw.update(overrides)
    return RadarConfig(**kw)


def uw_decode(data: np.ndarray, cfg: RadarConfig, tx_order: Sequence[int] = (0, 1),
              conj: bool = False) -> np.ndarray:
    """Reshape a ``data_frames`` array (nRx, F·M·nTx·N) into cubes (F, K, M, N).

    ``tx_order[i]`` is the TX index (into ``cfg.tx_pos``) of the i-th chirp of
    each loop; k = t·nRx + r (brief layout). ``conj`` flips the Q sign.
    Trailing columns that do not fill a whole frame are dropped.
    """
    d = np.asarray(data)
    nT, nR, M, N = cfg.n_tx, cfg.n_rx, cfg.n_chirps, cfg.n_samples
    if d.ndim != 2 or d.shape[0] != nR:
        raise ValueError(f"expected data of shape ({nR}, F*M*nTx*N), got {d.shape}")
    if sorted(tx_order) != list(range(nT)):
        raise ValueError("tx_order must be a permutation of range(n_tx)")
    per = M * nT * N
    nf = d.shape[1] // per
    if nf == 0:
        raise ValueError("data shorter than one frame")
    x = d[:, : nf * per].astype(np.complex128)
    if conj:
        x = x.conj()
    x = x.reshape(nR, nf, M, nT, N)                      # (r, f, m, slot, n)
    x = x[:, :, :, np.argsort(tx_order), :]               # slot -> TX index
    return np.ascontiguousarray(x.transpose(1, 3, 0, 2, 4)).reshape(nf, nT * nR, M, N)


def uw_encode(cubes: np.ndarray, cfg: RadarConfig, tx_order: Sequence[int] = (0, 1)) -> np.ndarray:
    """Inverse of :func:`uw_decode` (for tests): (F, K, M, N) → (nRx, F·M·nTx·N)."""
    c = np.asarray(cubes)
    if c.ndim == 3:
        c = c[None]
    nf = c.shape[0]
    nT, nR, M, N = cfg.n_tx, cfg.n_rx, cfg.n_chirps, cfg.n_samples
    x = c.reshape(nf, nT, nR, M, N)[:, list(tx_order)]   # TX index -> slot
    return x.transpose(2, 0, 3, 1, 4).reshape(nR, -1)


def load_uw_mat(path: str | Path, cfg: RadarConfig | None = None, variable: str = "data_frames",
                frames: slice | None = None, tx_order: Sequence[int] = (0, 1),
                conj: bool = False) -> tuple[np.ndarray, RadarConfig]:
    """Load a UW template ``.mat`` → (cubes (F, K, M, N), cfg).

    Reads only ``variable`` with :func:`scipy.io.loadmat` (MATLAB v5; no
    pickle, no object arrays). The file is treated as untrusted: the variable
    must be a 2-D numeric array.
    """
    import scipy.io

    cfg = cfg or uw_awr1843_config()
    m = scipy.io.loadmat(str(path), variable_names=[variable], squeeze_me=False,
                         struct_as_record=True, mat_dtype=False)
    if variable not in m:
        raise KeyError(f"{variable!r} not found in {path}")
    d = m[variable]
    if not isinstance(d, np.ndarray) or d.dtype.kind not in "iufc" or d.ndim != 2:
        raise ValueError(f"{variable!r} must be a 2-D numeric array, got {type(d)} {getattr(d, 'dtype', None)}")
    cubes = uw_decode(d, cfg, tx_order, conj)
    if frames is not None:
        cubes = cubes[frames]
    return cubes, cfg
