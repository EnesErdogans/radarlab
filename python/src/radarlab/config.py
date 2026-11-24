"""Radar configuration (FMCW, TDM-MIMO) and presets.

All quantities are SI (Hz, s, m) unless stated otherwise. Field names follow
``docs/PROJECT_BRIEF.md`` (snake_case here, camelCase in TypeScript / JSON).

Derived quantities (brief, "Binding conventions")::

    λ      = c / fc
    Te     = nTx · Tc                       (effective chirp repetition per TX)
    B_eff  = slope · N / fs                 (bandwidth swept during the ADC window)
    ΔR     = c / (2 · B_eff)
    Rmax   = fs · c / (2 · slope)           (complex IF; half of that for real IF)
    Δv     = λ / (2 · M · Te)
    vmax   = λ / (4 · Te)                   (unambiguous: v ∈ [−vmax, vmax))
    pos_k  = txPos[t] + rxPos[r],  k = t · nRx + r

Notes
-----
``fc`` is the **chirp start frequency** (TI ``startFreq``): the brief's beat
model uses ``fc·τ + slope·τ·t_f`` with ``t_f`` measured from the chirp start,
so ``fc`` must be the instantaneous frequency at ``t = t_{m,t}``. The Doppler
scale therefore uses ``λ = c/fc`` exactly as the brief prescribes; the true
"effective" Doppler wavelength is ``c/(fc + slope·t_f)`` which differs by
``slope·t_f/fc`` (≈0.3 % for the 77 GHz preset) — far below one Doppler bin.
"""
from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

C0: float = 299_792_458.0
"""Speed of light in vacuum [m/s]."""
K_BOLTZMANN: float = 1.380649e-23
"""Boltzmann constant [J/K]."""
T0_KELVIN: float = 290.0
"""Standard noise temperature [K]."""

IqMode = Literal["complex", "real"]


class ConfigWarning(UserWarning):
    """Raised (as a warning) for physically questionable configurations."""


def _as_pos(x: Any, n: int | None = None) -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.ndim != 1:
        raise ValueError("element positions must be 1-D (x positions in wavelengths)")
    if n is not None and a.size != n:
        raise ValueError(f"expected {n} positions, got {a.size}")
    return a


@dataclass
class RadarConfig:
    """FMCW TDM-MIMO radar configuration.

    Parameters
    ----------
    fc : chirp start frequency [Hz].
    slope : chirp slope S [Hz/s].
    fs : ADC sampling rate [Hz] (complex samples/s for ``iq_mode='complex'``).
    n_samples : N, ADC samples per chirp.
    chirp_period : Tc, ramp + idle time of one chirp of one TX [s].
    n_chirps : M, chirps per TX per frame.
    tx_pos, rx_pos : element x-positions in wavelengths (``λ = c/fc``).
    tx_pos_z, rx_pos_z : optional element z-positions in wavelengths (default 0).
    adc_start_time : ADC start relative to chirp start [s].
    idle_time : idle part of ``chirp_period`` [s]; ramp time = Tc − idle.
    frame_period : frame repetition interval [s].
    iq_mode : ``'complex'`` or ``'real'`` IF sampling.
    tx_power_dbm, tx_gain_dbi, rx_gain_dbi, noise_figure_db, system_loss_db :
        radar-equation parameters for ``amplitude_mode='physical'``.
    name : free-text label.
    """

    fc: float = 77e9
    slope: float = 15e12
    fs: float = 10e6
    n_samples: int = 256
    chirp_period: float = 32e-6
    n_chirps: int = 128
    tx_pos: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    rx_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5, 1.0, 1.5]))
    tx_pos_z: np.ndarray | None = None
    rx_pos_z: np.ndarray | None = None
    adc_start_time: float = 0.0
    idle_time: float = 0.0
    frame_period: float = 50e-3
    iq_mode: IqMode = "complex"
    tx_power_dbm: float = 12.0
    tx_gain_dbi: float = 10.0
    rx_gain_dbi: float = 10.0
    noise_figure_db: float = 15.0
    system_loss_db: float = 3.0
    name: str = "custom"

    # ------------------------------------------------------------------ init
    def __post_init__(self) -> None:
        self.tx_pos = _as_pos(self.tx_pos)
        self.rx_pos = _as_pos(self.rx_pos)
        if self.tx_pos_z is not None:
            self.tx_pos_z = _as_pos(self.tx_pos_z, self.tx_pos.size)
        if self.rx_pos_z is not None:
            self.rx_pos_z = _as_pos(self.rx_pos_z, self.rx_pos.size)
        self.n_samples = int(self.n_samples)
        self.n_chirps = int(self.n_chirps)
        if self.iq_mode not in ("complex", "real"):
            raise ValueError("iq_mode must be 'complex' or 'real'")
        for nm in ("fc", "slope", "fs", "chirp_period", "frame_period"):
            if not getattr(self, nm) > 0:
                raise ValueError(f"{nm} must be > 0")
        if self.n_samples < 2 or self.n_chirps < 1:
            raise ValueError("n_samples >= 2 and n_chirps >= 1 required")
        for w in self.validate():
            warnings.warn(w, ConfigWarning, stacklevel=3)

    # ------------------------------------------------------------- derived
    @property
    def n_tx(self) -> int:
        return int(self.tx_pos.size)

    @property
    def n_rx(self) -> int:
        return int(self.rx_pos.size)

    @property
    def n_virtual(self) -> int:
        """K = nTx · nRx."""
        return self.n_tx * self.n_rx

    @property
    def wavelength(self) -> float:
        """λ = c / fc [m]."""
        return C0 / self.fc

    @property
    def te(self) -> float:
        """Te = nTx · Tc [s]: repetition interval of the same TX."""
        return self.n_tx * self.chirp_period

    @property
    def ramp_time(self) -> float:
        return self.chirp_period - self.idle_time

    @property
    def adc_duration(self) -> float:
        return self.n_samples / self.fs

    @property
    def bandwidth_eff(self) -> float:
        """B_eff = slope · N / fs [Hz]."""
        return self.slope * self.n_samples / self.fs

    @property
    def range_resolution(self) -> float:
        """ΔR = c / (2 B_eff) [m] (also the range-bin spacing of an N-point FFT)."""
        return C0 / (2.0 * self.bandwidth_eff)

    @property
    def max_range(self) -> float:
        """Rmax = fs·c/(2·slope) for complex IF, half that for real IF [m]."""
        r = self.fs * C0 / (2.0 * self.slope)
        return r if self.iq_mode == "complex" else 0.5 * r

    @property
    def n_range_bins(self) -> int:
        """Range bins kept after the range FFT (N complex, N/2 real)."""
        return self.n_samples if self.iq_mode == "complex" else self.n_samples // 2

    @property
    def velocity_resolution(self) -> float:
        """Δv = λ / (2 M Te) [m/s]."""
        return self.wavelength / (2.0 * self.n_chirps * self.te)

    @property
    def max_velocity(self) -> float:
        """vmax = λ / (4 Te) [m/s]."""
        return self.wavelength / (4.0 * self.te)

    @property
    def frame_active_time(self) -> float:
        """Duration of the chirp burst, M · Te [s]."""
        return self.n_chirps * self.te

    @property
    def virtual_pos(self) -> np.ndarray:
        """Virtual element x-positions [λ], index k = t·nRx + r, shape (K,)."""
        return (self.tx_pos[:, None] + self.rx_pos[None, :]).reshape(-1)

    @property
    def virtual_pos_z(self) -> np.ndarray:
        """Virtual element z-positions [λ], shape (K,)."""
        tz = self.tx_pos_z if self.tx_pos_z is not None else np.zeros(self.n_tx)
        rz = self.rx_pos_z if self.rx_pos_z is not None else np.zeros(self.n_rx)
        return (tz[:, None] + rz[None, :]).reshape(-1)

    def range_axis(self) -> np.ndarray:
        """Range of each range bin r: R_r = r · ΔR [m]."""
        return np.arange(self.n_range_bins) * self.range_resolution

    def velocity_axis(self) -> np.ndarray:
        """Velocity of each fftshifted Doppler bin d: v_d = (d − M/2) · Δv [m/s]."""
        return (np.arange(self.n_chirps) - self.n_chirps // 2) * self.velocity_resolution

    def chirp_start_times(self, t_frame: float = 0.0) -> np.ndarray:
        """t_{m,t} = t_frame + m·Te + t·Tc, shape (nTx, M) [s]."""
        m = np.arange(self.n_chirps)
        t = np.arange(self.n_tx)
        return t_frame + m[None, :] * self.te + t[:, None] * self.chirp_period

    def fast_times(self) -> np.ndarray:
        """t_f = adcStartTime + n/fs, shape (N,) [s]."""
        return self.adc_start_time + np.arange(self.n_samples) / self.fs

    def noise_power_w(self) -> float:
        """Thermal noise power Pn = k·T0·F·fs [W] (complex sampling)."""
        return K_BOLTZMANN * T0_KELVIN * 10 ** (self.noise_figure_db / 10) * self.fs

    def derived(self) -> dict[str, float]:
        """Dictionary of all derived quantities (for golden vectors / UI)."""
        return {
            "wavelength": self.wavelength,
            "te": self.te,
            "bandwidthEff": self.bandwidth_eff,
            "rangeResolution": self.range_resolution,
            "maxRange": self.max_range,
            "velocityResolution": self.velocity_resolution,
            "maxVelocity": self.max_velocity,
            "nVirtual": float(self.n_virtual),
            "nRangeBins": float(self.n_range_bins),
            "frameActiveTime": self.frame_active_time,
        }

    # ------------------------------------------------------------ validate
    def validate(self, max_range: float | None = None,
                 max_velocity: float | None = None) -> list[str]:
        """Return human-readable warnings about the configuration.

        Checks: ADC window inside the ramp, chirp burst fits in the frame,
        idle time sane, optional range / velocity aliasing for a scenario with
        targets up to ``max_range`` [m] and ``|v| <= max_velocity`` [m/s].
        """
        w: list[str] = []
        if self.idle_time < 0 or self.idle_time >= self.chirp_period:
            w.append("idle_time must be in [0, chirp_period)")
        end = self.adc_start_time + self.adc_duration
        if end > self.ramp_time * (1 + 1e-9):
            w.append(f"ADC window ends at {end*1e6:.2f} us after chirp start but ramp "
                     f"lasts only {self.ramp_time*1e6:.2f} us")
        if self.frame_active_time > self.frame_period * (1 + 1e-9):
            w.append("chirp burst M·Te is longer than frame_period")
        if max_range is not None and max_range > self.max_range:
            w.append(f"targets up to {max_range:.1f} m exceed Rmax={self.max_range:.1f} m "
                     "(beat frequency aliasing)")
        if max_velocity is not None and max_velocity >= self.max_velocity:
            w.append(f"|v| up to {max_velocity:.1f} m/s exceeds vmax={self.max_velocity:.2f} m/s "
                     "(Doppler aliasing)")
        return w

    # ------------------------------------------------------------ serialise
    _CAMEL = {
        "fc": "fc", "slope": "slope", "fs": "fs", "n_samples": "nSamples",
        "chirp_period": "chirpPeriod", "n_chirps": "nChirps", "tx_pos": "txPos",
        "rx_pos": "rxPos", "tx_pos_z": "txPosZ", "rx_pos_z": "rxPosZ",
        "adc_start_time": "adcStartTime", "idle_time": "idleTime",
        "frame_period": "framePeriod", "iq_mode": "iqMode",
        "tx_power_dbm": "txPowerDbm", "tx_gain_dbi": "txGainDbi",
        "rx_gain_dbi": "rxGainDbi", "noise_figure_db": "noiseFigureDb",
        "system_loss_db": "systemLossDb", "name": "name",
    }

    def to_dict(self, camel: bool = True) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (camelCase keys by default)."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if v is None:
                continue
            out[self._CAMEL[f.name] if camel else f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RadarConfig":
        inv = {v: k for k, v in cls._CAMEL.items()}
        kw = {inv.get(k, k): v for k, v in d.items()}
        return cls(**kw)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def replace(self, **kw: Any) -> "RadarConfig":
        """Return a copy with fields replaced (re-validated)."""
        return dataclasses.replace(self, **kw)


# ---------------------------------------------------------------- presets
def automotive_77ghz() -> RadarConfig:
    """2TX × 4RX TDM-MIMO, 8-element λ/2 virtual ULA (positions 0…3.5 λ).

    ΔR ≈ 0.39 m, Rmax ≈ 100 m, vmax ≈ 15.2 m/s, Δv ≈ 0.24 m/s.
    """
    return RadarConfig(
        fc=77e9, slope=15e12, fs=10e6, n_samples=256, chirp_period=32e-6,
        n_chirps=128, tx_pos=np.array([0.0, 2.0]),
        rx_pos=np.array([0.0, 0.5, 1.0, 1.5]), adc_start_time=2e-6,
        idle_time=3e-6, frame_period=50e-3, iq_mode="complex",
        tx_power_dbm=12.0, tx_gain_dbi=10.0, rx_gain_dbi=10.0,
        noise_figure_db=15.0, system_loss_db=3.0, name="automotive_77ghz")


def awr1843_like() -> RadarConfig:
    """TI AWR1843-like 3TX × 4RX geometry (12 virtual channels).

    TX0 at x=0, TX1 at x=1 λ, z=0.5 λ (elevated), TX2 at x=2 λ; RX at 0…1.5 λ.
    TX0+TX2 give an 8-element azimuth ULA (virtual k = 0..3 and 8..11); TX1 adds
    an elevated row (k = 4..7) usable for elevation estimation.
    TDM order is t = 0, 1, 2 (TX0, TX1, TX2).
    """
    return RadarConfig(
        fc=77e9, slope=30e12, fs=10e6, n_samples=256, chirp_period=40e-6,
        n_chirps=64, tx_pos=np.array([0.0, 1.0, 2.0]), tx_pos_z=np.array([0.0, 0.5, 0.0]),
        rx_pos=np.array([0.0, 0.5, 1.0, 1.5]), adc_start_time=4e-6,
        idle_time=7e-6, frame_period=50e-3, iq_mode="complex",
        tx_power_dbm=12.0, tx_gain_dbi=10.0, rx_gain_dbi=10.0,
        noise_figure_db=14.0, system_loss_db=3.0, name="awr1843_like")


def drone_24ghz() -> RadarConfig:
    """24 GHz surveillance radar: ΔR ≈ 1 m (B_eff = 150 MHz), Rmax ≈ 512 m.

    2TX × 4RX (8 virtual ULA), vmax ≈ 26.0 m/s, Δv ≈ 0.41 m/s.
    """
    n, fs = 512, 10e6
    slope = 150e6 / (n / fs)
    return RadarConfig(
        fc=24.0e9, slope=slope, fs=fs, n_samples=n, chirp_period=60e-6,
        n_chirps=128, tx_pos=np.array([0.0, 2.0]),
        rx_pos=np.array([0.0, 0.5, 1.0, 1.5]), adc_start_time=4e-6,
        idle_time=4e-6, frame_period=50e-3, iq_mode="complex",
        tx_power_dbm=20.0, tx_gain_dbi=20.0, rx_gain_dbi=20.0,
        noise_figure_db=10.0, system_loss_db=3.0, name="drone_24ghz")


def micro_doppler_24ghz() -> RadarConfig:
    """1TX × 4RX fast-chirp 24 GHz radar for micro-Doppler.

    Te = Tc = 20 µs → vmax ≈ 156 m/s (blade tips ±100 m/s unaliased),
    frame period = M·Te (gap-free slow time for spectrograms), ΔR ≈ 0.6 m.
    """
    n, fs, m, tc = 64, 5e6, 256, 20e-6
    slope = 250e6 / (n / fs)
    return RadarConfig(
        fc=24.0e9, slope=slope, fs=fs, n_samples=n, chirp_period=tc,
        n_chirps=m, tx_pos=np.array([0.0]), rx_pos=np.array([0.0, 0.5, 1.0, 1.5]),
        adc_start_time=2e-6, idle_time=4e-6, frame_period=m * tc,
        iq_mode="complex", tx_power_dbm=20.0, tx_gain_dbi=15.0, rx_gain_dbi=15.0,
        noise_figure_db=10.0, system_loss_db=3.0, name="micro_doppler_24ghz")


PRESETS = {
    "automotive_77ghz": automotive_77ghz,
    "awr1843_like": awr1843_like,
    "drone_24ghz": drone_24ghz,
    "micro_doppler_24ghz": micro_doppler_24ghz,
}
