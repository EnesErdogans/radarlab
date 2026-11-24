import numpy as np
import pytest

import radarlab as rl
from radarlab.io import parse_ti_cfg, read_dca1000, write_dca1000, dca1000_decode, dca1000_encode

TI_CFG = """
% xWR18xx SDK profile (3TX TDM)
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 77 7 4 40 0 0 30 1 256 10000 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 2
frameCfg 0 2 64 0 50 1 0
sensorStart
"""


def test_parse_ti_cfg():
    t = parse_ti_cfg(TI_CFG)
    c = t.config
    assert c.fc == pytest.approx(77e9)
    assert c.slope == pytest.approx(30e12)
    assert c.fs == pytest.approx(10e6)
    assert c.n_samples == 256 and c.n_chirps == 64
    assert c.chirp_period == pytest.approx(47e-6)
    assert c.adc_start_time == pytest.approx(4e-6)
    assert c.frame_period == pytest.approx(50e-3)
    assert t.tx_order == [0, 2, 1]
    np.testing.assert_allclose(c.tx_pos, [0.0, 2.0, 1.0])
    np.testing.assert_allclose(c.tx_pos_z, [0.0, 0.0, 0.5])
    assert t.adc_bits == 16 and t.adc_format == "complex1x" and c.iq_mode == "complex"


def test_parse_partial_rx():
    t = parse_ti_cfg(TI_CFG.replace("channelCfg 15 7 0", "channelCfg 5 1 0")
                     .replace("frameCfg 0 2", "frameCfg 0 0"))
    assert t.rx_enabled == [0, 2]
    np.testing.assert_allclose(t.config.rx_pos, [0.0, 1.0])
    assert t.config.n_tx == 1


@pytest.mark.parametrize("iq", ["complex", "real"])
def test_dca1000_roundtrip(tmp_path, rng, iq):
    cfg = rl.awr1843_like().replace(n_chirps=8, n_samples=32, iq_mode=iq)
    shape = (3, cfg.n_virtual, 8, 32)
    data = rng.integers(-2000, 2000, shape).astype(float)
    if iq == "complex":
        data = data + 1j * rng.integers(-2000, 2000, shape)
    p = tmp_path / "adc_data.bin"
    write_dca1000(p, data, cfg)
    assert p.stat().st_size == data.size * (4 if iq == "complex" else 2)
    back = read_dca1000(p, cfg)
    np.testing.assert_array_equal(back, data)


def test_dca1000_lane_layout():
    """[I0, I1, Q0, Q1] word groups; RX-major samples within a chirp (SWRA581)."""
    cfg = rl.RadarConfig(n_samples=4, n_chirps=1, chirp_period=40e-6, tx_pos=[0], rx_pos=[0, 0.5],
                         frame_period=1e-3)
    words = np.array([1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60, 7, 8, 70, 80], np.int16)
    cube = dca1000_decode(words, cfg)[0]
    np.testing.assert_array_equal(cube[0, 0], [1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j])
    np.testing.assert_array_equal(cube[1, 0], [5 + 50j, 6 + 60j, 7 + 70j, 8 + 80j])
    np.testing.assert_array_equal(dca1000_encode(cube, cfg), words)


def test_sim_to_dca_and_process(tmp_path):
    t = parse_ti_cfg(TI_CFG)
    cfg = t.config.replace(n_chirps=16)
    tg = rl.PointTarget.from_polar(8.0, 10.0, 1.0)
    cube = rl.synthesize_cube(cfg, [tg], noise=True, rng=1)
    q = rl.impairments.adc_quantize(cube * 100, bits=16, full_scale=32768.0)
    write_dca1000(tmp_path / "x.bin", q, cfg)
    back = read_dca1000(tmp_path / "x.bin", cfg)[0]
    res = rl.process_frame(back, cfg)
    assert abs(res.points[0]["range"] - 8.0) < cfg.range_resolution


def test_closed_loop_cfg_sim_dca_process(tmp_path):
    """.cfg -> simulate -> int16 DCA1000 .bin -> read back -> process: identical
    cube codes and the same detections as processing the float cube."""
    t = parse_ti_cfg(TI_CFG)
    cfg = t.config.replace(n_chirps=32)
    tgs = [rl.PointTarget.from_polar(5.0, -25.0, 2.0, rcs=5), rl.PointTarget.from_polar(9.5, 10.0, -1.0, rcs=5),
           rl.PointTarget.from_polar(14.0, 30.0, 0.5, rcs=10)]
    cube = rl.synthesize_cube(cfg, tgs, rng=5, snr_ref_db=25, ref_range=10)
    _, codes = rl.impairments.adc_quantize(cube, bits=16, noise_rms_lsb=64.0, return_codes=True)
    lsb = np.sqrt(0.5) / 64.0
    write_dca1000(tmp_path / "adc_data.bin", codes, cfg)
    back = read_dca1000(tmp_path / "adc_data.bin", cfg)[0]
    np.testing.assert_array_equal(back, codes)
    pp = rl.ProcessingParams(cfar_pfa=1e-6)
    a = rl.process_frame(cube, cfg, pp).points
    b = rl.process_frame(back * lsb, cfg, pp).points
    sa = sorted(zip(a["range_cell"][a["snr_db"] > 20], a["doppler_cell"][a["snr_db"] > 20]))
    sb = sorted(zip(b["range_cell"][b["snr_db"] > 20], b["doppler_cell"][b["snr_db"] > 20]))
    assert sa == sb and len(sa) >= 3
    for tg in tgs:
        R = np.linalg.norm(tg.p0)
        i = int(np.argmin(np.abs(b["range"] - R)))
        assert abs(b["range"][i] - R) < cfg.range_resolution / 2
        assert np.rad2deg(b["azimuth"][i]) == pytest.approx(np.rad2deg(np.arctan2(tg.p0[0], tg.p0[1])), abs=2.0)
