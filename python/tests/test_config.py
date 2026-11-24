import json
import warnings

import numpy as np
import pytest

import radarlab as rl
from radarlab.config import C0, ConfigWarning


def test_derived_quantities():
    cfg = rl.automotive_77ghz()
    assert cfg.wavelength == pytest.approx(C0 / 77e9)
    assert cfg.te == pytest.approx(2 * 32e-6)
    assert cfg.bandwidth_eff == pytest.approx(15e12 * 256 / 10e6)
    assert cfg.range_resolution == pytest.approx(C0 / (2 * cfg.bandwidth_eff))
    assert cfg.max_range == pytest.approx(10e6 * C0 / (2 * 15e12))
    assert cfg.velocity_resolution == pytest.approx(cfg.wavelength / (2 * 128 * cfg.te))
    assert cfg.max_velocity == pytest.approx(cfg.wavelength / (4 * cfg.te))
    np.testing.assert_allclose(cfg.virtual_pos, [0, .5, 1, 1.5, 2, 2.5, 3, 3.5])
    tm = cfg.chirp_start_times(1.0)
    assert tm.shape == (2, 128)
    assert tm[1, 3] == pytest.approx(1.0 + 3 * cfg.te + cfg.chirp_period)


def test_real_iq_halves_range():
    c = rl.automotive_77ghz()
    r = c.replace(iq_mode="real")
    assert r.max_range == pytest.approx(c.max_range / 2)
    assert r.n_range_bins == c.n_samples // 2


@pytest.mark.parametrize("name", list(rl.PRESETS))
def test_presets_valid(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        cfg = rl.PRESETS[name]()
    assert cfg.validate() == []


def test_preset_requirements():
    d = rl.drone_24ghz()
    assert d.range_resolution == pytest.approx(1.0, rel=0.01)
    assert d.max_range >= 500
    md = rl.micro_doppler_24ghz()
    assert md.max_velocity > 100
    assert md.frame_period == pytest.approx(md.frame_active_time)
    a = rl.awr1843_like()
    assert a.n_virtual == 12
    assert np.count_nonzero(a.virtual_pos_z) == 4


def test_adc_window_warning():
    with pytest.warns(ConfigWarning, match="ADC window"):
        rl.RadarConfig(n_samples=512, fs=10e6, chirp_period=30e-6)
    cfg = rl.automotive_77ghz()
    assert any("aliasing" in w for w in cfg.validate(max_range=150))
    assert any("Doppler" in w for w in cfg.validate(max_velocity=20))


def test_dict_roundtrip():
    cfg = rl.awr1843_like()
    d = json.loads(cfg.to_json())
    assert "txPosZ" in d and "nSamples" in d
    c2 = rl.RadarConfig.from_dict(d)
    assert c2.derived() == cfg.derived()
    np.testing.assert_array_equal(c2.virtual_pos_z, cfg.virtual_pos_z)
