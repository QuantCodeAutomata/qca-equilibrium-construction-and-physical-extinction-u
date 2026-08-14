"""Tests for src/model.py — equilibrium primitives."""

from __future__ import annotations

import numpy as np
import pytest

from src.model import (
    Parameters,
    benchmark_martingale_N,
    benchmark_martingale_R,
    consumption_shares,
    dogmatist_share,
    log_endowment,
    log_sdf,
    log_w,
    log_xi,
    rational_yield,
    simulate_brownian,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def p1() -> Parameters:
    """γ = 1 base parameters."""
    return Parameters(mu=0.02, sigma=0.15, rho=0.02, gamma=1.0, D0=1.0, s0=0.05)


@pytest.fixture()
def p2() -> Parameters:
    """γ = 2 parameters."""
    return Parameters(mu=0.02, sigma=0.15, rho=0.02, gamma=2.0, D0=1.0, s0=0.05)


DT = 0.1


def _sim(T=1.0, n_paths=5, seed=0):
    return simulate_brownian(T=T, dt=DT, n_paths=n_paths, seed=seed)


def _eta_flat(n):
    return np.zeros(n)


def _eta_const(n, val=0.5):
    return np.full(n, val)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def test_w0_derived(self, p1):
        assert p1.w0 == pytest.approx(0.05 / 0.95, rel=1e-12)

    def test_small_s0(self):
        p = Parameters(mu=0.0, sigma=0.1, rho=0.02, gamma=1.0, D0=1.0, s0=1e-6)
        assert p.w0 == pytest.approx(1e-6 / (1 - 1e-6), rel=1e-10)

    def test_positivity(self, p1):
        assert p1.sigma > 0
        assert p1.rho > 0
        assert p1.gamma > 0


# ---------------------------------------------------------------------------
# rational_yield
# ---------------------------------------------------------------------------

class TestRationalYield:
    def test_formula_gamma1(self, p1):
        expected = p1.rho + p1.mu - 0.5 * 1 * 2 * p1.sigma**2
        assert rational_yield(p1) == pytest.approx(expected, rel=1e-14)

    def test_formula_gamma2(self, p2):
        expected = p2.rho + 2 * p2.mu - 0.5 * 2 * 3 * p2.sigma**2
        assert rational_yield(p2) == pytest.approx(expected, rel=1e-14)

    def test_differs_by_gamma(self, p1, p2):
        assert rational_yield(p1) != rational_yield(p2)


# ---------------------------------------------------------------------------
# simulate_brownian
# ---------------------------------------------------------------------------

class TestSimulateBrownian:
    def test_shapes(self):
        t, dB, B = simulate_brownian(T=1.0, dt=0.01, n_paths=50)
        n = round(1.0 / 0.01)
        assert t.shape == (n + 1,)
        assert dB.shape == (n, 50)
        assert B.shape == (n + 1, 50)

    def test_B0_zero(self):
        _, _, B = simulate_brownian(T=1.0, dt=0.05, n_paths=100)
        assert np.all(B[0] == 0.0)

    def test_increments_consistent_with_B(self):
        _, dB, B = simulate_brownian(T=0.5, dt=0.1, n_paths=20)
        assert np.allclose(np.diff(B, axis=0), dB)

    def test_variance(self):
        dt = 0.1
        _, dB, _ = simulate_brownian(T=1.0, dt=dt, n_paths=200_000, seed=0)
        assert np.var(dB) == pytest.approx(dt, rel=0.01)

    def test_reproducibility(self):
        _, a, _ = simulate_brownian(T=1.0, dt=0.1, n_paths=10, seed=42)
        _, b, _ = simulate_brownian(T=1.0, dt=0.1, n_paths=10, seed=42)
        assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# log_endowment
# ---------------------------------------------------------------------------

class TestLogEndowment:
    def test_zero_at_t0(self, p1):
        t, dB, _ = _sim()
        logD = log_endowment(t, dB, DT, p1)
        assert np.allclose(logD[0], 0.0)

    def test_shape(self, p1):
        t, dB, _ = _sim(n_paths=10)
        logD = log_endowment(t, dB, DT, p1)
        assert logD.shape == (len(t), 10)


# ---------------------------------------------------------------------------
# log_xi and log_w
# ---------------------------------------------------------------------------

class TestLogXiLogW:
    def test_xi0_is_zero(self, p1):
        """log ξ_0 = 0."""
        t, dB, _ = _sim()
        eta = _eta_const(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        assert np.allclose(lxi[0], 0.0)

    def test_xi_finite(self, p1):
        t, dB, _ = _sim()
        eta = _eta_const(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        assert np.all(np.isfinite(lxi))

    def test_flat_eta_w_equals_w0(self, p1):
        """With η ≡ 0, ξ_t = 1 so w_t = w_0."""
        t, dB, _ = _sim()
        eta = _eta_flat(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        lw = log_w(lxi, p1.w0, p1.gamma)
        assert np.allclose(np.exp(lw), p1.w0, atol=1e-12)

    def test_w_shape(self, p1):
        t, dB, _ = _sim(n_paths=8)
        eta = _eta_const(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        lw = log_w(lxi, p1.w0, p1.gamma)
        assert lw.shape == (len(t), 8)


# ---------------------------------------------------------------------------
# consumption_shares
# ---------------------------------------------------------------------------

class TestConsumptionShares:
    def test_market_clearing(self, p1):
        """c_R + c_N = D_t (normalised by D_0)."""
        t, dB, _ = _sim(n_paths=15, seed=7)
        eta = _eta_const(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        lw = log_w(lxi, p1.w0, p1.gamma)
        logD = log_endowment(t, dB, DT, p1)
        cR, cN = consumption_shares(lw, logD)
        assert np.allclose(cR + cN, np.exp(logD), rtol=1e-12)

    def test_positivity(self, p1):
        t, dB, _ = _sim(seed=3)
        eta = _eta_const(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        lw = log_w(lxi, p1.w0, p1.gamma)
        logD = log_endowment(t, dB, DT, p1)
        cR, cN = consumption_shares(lw, logD)
        assert np.all(cR > 0) and np.all(cN > 0)


# ---------------------------------------------------------------------------
# log_sdf
# ---------------------------------------------------------------------------

class TestLogSDF:
    def test_pi0_is_one(self, p1):
        t, dB, _ = _sim()
        eta = _eta_const(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        lw = log_w(lxi, p1.w0, p1.gamma)
        logD = log_endowment(t, dB, DT, p1)
        lpi = log_sdf(t, logD, lw, p1)
        assert np.allclose(lpi[0], 0.0, atol=1e-14)

    def test_shape(self, p1):
        n = 8
        t, dB, _ = _sim(n_paths=n)
        eta = _eta_const(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        lw = log_w(lxi, p1.w0, p1.gamma)
        logD = log_endowment(t, dB, DT, p1)
        lpi = log_sdf(t, logD, lw, p1)
        assert lpi.shape == (len(t), n)


# ---------------------------------------------------------------------------
# dogmatist_share
# ---------------------------------------------------------------------------

class TestDogmatistShare:
    def test_range(self, p1):
        t, dB, _ = _sim()
        eta = _eta_const(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        lw = log_w(lxi, p1.w0, p1.gamma)
        s = dogmatist_share(lw)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_flat_eta_equals_s0(self, p1):
        t, dB, _ = _sim()
        eta = _eta_flat(len(t) - 1)
        lxi = log_xi(eta, dB, DT, p1.sigma, p1.gamma)
        lw = log_w(lxi, p1.w0, p1.gamma)
        s = dogmatist_share(lw)
        assert np.allclose(s, p1.s0, atol=1e-12)


# ---------------------------------------------------------------------------
# Benchmark martingales
# ---------------------------------------------------------------------------

class TestBenchmarkMartingales:
    def test_MR_starts_at_one(self, p1):
        t, dB, _ = _sim()
        log_MR = benchmark_martingale_R(t, dB, DT, p1.sigma, p1.gamma)
        assert np.allclose(np.exp(log_MR[0]), 1.0)

    def test_MN_starts_at_one(self, p1):
        t, dB, _ = _sim()
        eta = _eta_const(len(t) - 1)
        log_MN = benchmark_martingale_N(t, eta, dB, DT, p1.sigma, p1.gamma)
        assert np.allclose(np.exp(log_MN[0]), 1.0)

    def test_MR_finite(self, p1):
        t, dB, _ = _sim(T=2.0, seed=99)
        log_MR = benchmark_martingale_R(t, dB, DT, p1.sigma, p1.gamma)
        assert np.all(np.isfinite(log_MR))

    def test_MN_finite(self, p1):
        t, dB, _ = _sim(T=2.0, seed=99)
        eta = _eta_const(len(t) - 1)
        log_MN = benchmark_martingale_N(t, eta, dB, DT, p1.sigma, p1.gamma)
        assert np.all(np.isfinite(log_MN))
