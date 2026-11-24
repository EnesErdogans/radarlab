import numpy as np
import pytest

from radarlab import dsp


def test_parabolic_offset_exact_for_parabola():
    x0 = 0.3
    y = lambda x: -(x - x0) ** 2
    assert dsp.parabolic_offset(y(-1), y(0), y(1)) == pytest.approx(x0)


def test_connected_clusters_with_doppler_wrap():
    m = np.zeros((20, 16), bool)
    m[5, 0] = m[5, 15] = m[6, 15] = True        # one cluster across the seam
    m[12, 7:9] = True
    P = np.ones(m.shape)
    P[5, 15] = 9
    P[12, 8] = 5
    cl = dsp.cluster_connected(m, P)
    assert len(cl) == 2
    assert (cl[0].range_bin, cl[0].doppler_bin, cl[0].size) == (5, 15, 3)
    cl2 = dsp.cluster_dbscan(m, P, eps=1.5)
    assert sorted(c.size for c in cl2) == [2, 3]


def test_local_maxima():
    P = np.zeros((10, 10))
    P[3, 4] = 2
    P[3, 5] = 1
    P[7, 0] = 3
    pk = dsp.local_maxima(P, P > 0.5)
    assert {tuple(p) for p in pk} == {(3, 4), (7, 0)}


def test_parabolic_offset_non_peak_is_zero():
    """Review m16: same rule as the web engine — no concave fit → offset 0."""
    from radarlab.dsp import parabolic_offset
    assert parabolic_offset(1.0, 0.0, 1.0) == 0.0          # centre is a minimum
    assert parabolic_offset(0.0, 1.0, 2.0) == 0.0          # linear (den = 0)
    assert parabolic_offset(0.0, 2.0, 1.0) == pytest.approx(0.5 * (0 - 1) / (0 - 4 + 1))
