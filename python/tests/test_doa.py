import numpy as np
import pytest

import radarlab as rl
from radarlab import dsp


def _snapshots(pos, thetas_deg, L, snr_db, rng):
    A = dsp.steering_vector(pos, np.deg2rad(np.asarray(thetas_deg)))
    S = (rng.standard_normal((len(thetas_deg), L)) + 1j * rng.standard_normal((len(thetas_deg), L))) / np.sqrt(2)
    return A @ S * 10 ** (snr_db / 20) + rl.complex_noise((len(pos), L), rng)


@pytest.mark.parametrize("method", ["mvdr", "music"])
def test_superresolution(method, rng):
    pos = np.arange(8) * 0.5
    X = _snapshots(pos, [0.0, 9.0], 200, 25, rng)
    grid = np.arange(-60, 60, 0.1)
    p = dsp.doa_spectrum(X, pos, grid, method, n_sources=2, loading=1e-4)
    pk = np.sort(grid[dsp.find_peaks_1d(p, 2)])
    np.testing.assert_allclose(pk, [0.0, 9.0], atol=1.0)
    # conventional beamformer cannot separate 9 deg with an 8-element λ/2 ULA
    pb = dsp.doa_spectrum(X, pos, grid, "bartlett")
    pkb = dsp.find_peaks_1d(pb, 2)
    assert len(pkb) < 2 or pb[pkb[1]] < 0.5 * pb[pkb[0]] or abs(np.diff(grid[pkb]))[0] > 12


def test_elevation_awr1843():
    cfg = rl.awr1843_like()
    tg = rl.PointTarget.from_polar(10.0, 15.0, 0.0, elevation_deg=12.0)
    res = rl.process_frame(rl.synthesize_cube(cfg, [tg], noise=False), cfg)
    p = res.points[0]
    assert np.rad2deg(p["elevation"]) == pytest.approx(12.0, abs=0.5)
    # azimuth FFT measures sinθ·cosφ
    assert np.sin(p["azimuth"]) == pytest.approx(np.sin(np.deg2rad(15)) * np.cos(np.deg2rad(12)), abs=0.01)


def test_tdm_phase_values():
    cfg = rl.awr1843_like()
    ph = dsp.tdm_phase(cfg, 2.0)
    assert ph.shape == (12,)
    np.testing.assert_allclose(ph[:4], 1)
    assert ph[4] == pytest.approx(np.exp(-2j * np.pi * (2 * 2.0 / cfg.wavelength) * cfg.chirp_period))
    assert ph[8] == pytest.approx(np.exp(-2j * np.pi * (2 * 2.0 / cfg.wavelength) * 2 * cfg.chirp_period))
