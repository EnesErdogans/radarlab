"""CFAR α formulas and empirical false-alarm rates (exponential noise, K = 1)."""
import numpy as np
import pytest

from radarlab import dsp


def test_ca_alpha_formula():
    for n in (8, 16, 48):
        for pfa in (1e-3, 1e-6):
            assert dsp.ca_alpha(n, pfa) == pytest.approx(n * (pfa ** (-1 / n) - 1))


def test_os_alpha_satisfies_pfa():
    for n in (16, 24, 248):
        k = dsp.os_rank_index(n, 0.75)
        a = dsp.os_alpha(n, 1e-4, k)
        assert dsp.os_pfa(a, n, k) == pytest.approx(1e-4, rel=1e-8)


def test_gamma_exact_matches_fidelity_values():
    # fidelity_requirements §8.2 (MC-verified): K=4, N=16 -> 3.533; K=8, N=24 -> 2.535
    assert dsp.ca_alpha(16, 1e-3, 4) == pytest.approx(3.533, abs=5e-4)
    assert dsp.ca_alpha(24, 1e-3, 8) == pytest.approx(2.535, abs=5e-4)


@pytest.mark.parametrize("kind", ["os", "go", "so"])
def test_numeric_gamma_k_paths_reduce_to_closed_form(kind):
    """The K>1 quadrature formulas evaluated at K=1 equal the closed forms."""
    import scipy.integrate as si
    import scipy.special as sp
    import scipy.stats as st
    n = 16
    if kind == "os":
        k = dsp.os_rank_index(n, 0.75)
        a = dsp.os_alpha(n, 1e-4, k)
        bd = st.beta(k, n - k + 1)
        val = si.quad(lambda u: sp.gammaincc(1, a * sp.gammaincinv(1, u)) * bd.pdf(u), 0, 1, epsrel=1e-11)[0]
    else:
        b = dsp.gs_alpha(n // 2, 1e-4, kind) / (n // 2)
        w = (lambda u: 2 * u) if kind == "go" else (lambda u: 2 * (1 - u))
        val = sum(si.quad(lambda u: sp.gammaincc(1, b * sp.gammaincinv(n // 2, u)) * w(u), x, y, epsrel=1e-11)[0]
                  for x, y in zip([0, .5, .99], [.5, .99, 1]))
    assert val == pytest.approx(1e-4, rel=1e-6)


@pytest.mark.parametrize("kind", ["os", "go", "so"])
def test_empirical_pfa_gamma_k8(kind, rng):
    pfa = 1e-2
    P = rng.gamma(8.0, 1.0, (200, 512))
    res = dsp.cfar_2d(P, (0, 1), (0, 8), pfa, kind, n_channels=8)
    assert res.mask.mean() == pytest.approx(pfa, rel=0.15)


def test_pipeline_default_uses_gamma_k():
    import radarlab as rl
    cfg = rl.automotive_77ghz().replace(n_chirps=32)
    res = rl.process_frame(rl.synthesize_cube(cfg, [], rng=0), cfg)
    assert res.cfar.alpha[100, 0] == pytest.approx(dsp.ca_alpha(int(res.cfar.n_train[100, 0]), 1e-5, 8))


def test_gamma_exact_ca_reduces_to_closed_form():
    assert dsp.ca_alpha(16, 1e-4, 1) == pytest.approx(16 * (1e-4 ** (-1 / 16) - 1))
    # K=4 needs a lower α on the mean (noise averages over channels)
    assert dsp.ca_alpha(16, 1e-4, 4) < dsp.ca_alpha(16, 1e-4, 1)


@pytest.mark.parametrize("kind", ["ca", "os"])
def test_empirical_pfa_2d(kind, rng):
    pfa = 1e-3
    fa, cells = 0, 0
    for _ in range(12):
        P = rng.exponential(1.0, (256, 128))
        res = dsp.cfar_2d(P, (2, 2), (6, 4), pfa, kind)
        fa += res.mask.sum()
        cells += P.size
    assert fa / cells == pytest.approx(pfa, rel=0.15)


@pytest.mark.parametrize("kind", ["go", "so"])
def test_empirical_pfa_go_so_1d(kind, rng):
    pfa = 1e-2
    P = rng.exponential(1.0, (400, 512))       # rows independent, circular axis
    res = dsp.cfar_2d(P, (0, 1), (0, 8), pfa, kind)
    assert res.mask.mean() == pytest.approx(pfa, rel=0.15)


def test_gamma_exact_ca_k4(rng):
    pfa = 1e-3
    P = rng.gamma(4.0, 1.0, (40, 4096))
    res = dsp.cfar_2d(P, (0, 1), (0, 8), pfa, "ca", n_channels=4)
    assert res.mask.mean() == pytest.approx(pfa, rel=0.15)
    res1 = dsp.cfar_2d(P, (0, 1), (0, 8), pfa, "ca", n_channels=1)
    assert res1.mask.mean() < 0.5 * pfa       # exponential α is too conservative for K=4


def test_range_edges_use_available_cells():
    P = np.ones((40, 32))
    res = dsp.cfar_2d(P, (2, 2), (4, 3), 1e-4, "ca")
    full = (2 * 6 + 1) * (2 * 5 + 1) - 25
    assert res.n_train[20, 0] == full
    assert res.n_train[0, 0] < full
    assert res.alpha[0, 0] == pytest.approx(dsp.ca_alpha(int(res.n_train[0, 0]), 1e-4))


def test_masking_os_beats_ca(rng):
    """Closely spaced targets mask each other in CA but not in OS."""
    det = {"ca": 0, "os": 0}
    trials = 200
    for _ in range(trials):
        p = rng.exponential(1.0, 128)
        for i in (50, 54, 58):
            p[i] += 100.0                     # 20 dB targets
        for kind in det:
            r = dsp.cfar_1d(p, guard=1, train=8, pfa=1e-4, kind=kind)
            det[kind] += r.mask[50]
    assert det["os"] / trials > 0.9
    assert det["ca"] / trials < 0.3
