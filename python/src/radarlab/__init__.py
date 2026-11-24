"""RadarLab — research-grade FMCW MIMO radar simulation (Python reference).

Conventions are defined in ``docs/PROJECT_BRIEF.md`` and are binding.
"""
from .config import (RadarConfig, C0, automotive_77ghz, awr1843_like, drone_24ghz,
                     micro_doppler_24ghz, PRESETS, ConfigWarning)
from .scene import (Scatterers, Target, PointTarget, Car, Pedestrian, Drone, Bird, ClutterField,
                    static_clutter, Scene)
from .signal import synthesize_cube, amplitude, snr_per_sample_db, complex_noise, Leakage

__version__ = "0.1.0"

from . import dsp, impairments, metrics, pipeline, track  # noqa: E402
from .pipeline import (simulate_frame, process_frame, Simulator, ProcessingParams, SimParams,  # noqa: E402
                       FrameResult)
