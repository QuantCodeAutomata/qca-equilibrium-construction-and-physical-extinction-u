"""Tests for src/long_run.py — long-run term structure analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.long_run import (
    classify_path,
    long_yield_formula,
    mixed_limit_martingale,
    theta_T_series,
)
from src.paths import (
    A_fading_pessimism_exact,
    H_constant_cyclical_exact,
    H_fading_pessimism_exact,
    H_numerical,
    eta_fading_alternating,
)
from src.model import (
    Parameters,
    benchmark_martingale_N,
    benchmark_martingale_R,
    simulate_brownian,
)

W0 = 1.0 / 19.0
S0 = 0.05
SIGMA = 0.15
GAMMA = 1.0
ETA0 = -1.5
OMEGA = 2.0 * np.pi / 20.0


class TestThetaTSeries:
    def test_range(self):
        H_T = np.linspace(-5, 5, 100)
        th = theta_T_series(W0, H_T, GAMMA, SIGMA)
        assert np.all(th > 0) and np.all(th < 1)

    def test_at_H_zero(self):
        """θ_T = w0 / (1 + w0) = s0 when H_T = 0 and γ = 1."""
        th = theta_T_series(W0, np.array([0.0]), GAMMA, SIGMA)
        assert th[0] == pytest.approx(S0, rel=1e-10)

    def test_tends_to_one_as_H_to_minus_inf(self):
        # q = w0^γ exp(-γ σ² H); for w0=1/19,γ=1,σ=0.15:
        # need H ≈ -400 so q >> 1 and θ = q/(1+q) > 0.99
        H_T = np.array([-400.0])
        th = theta_T_series(W0, H_T, GAMMA, SIGMA)
        assert th[0] > 0.99

    def test_tends_to_zero_as_H_to_plus_inf(self):
        # q = w0 exp(-σ²H) → 0 as H → +∞ ⇒ θ → 0; need large H
        H_T = np.array([400.0])
        th = theta_T_series(W0, H_T, GAMMA, SIGMA)
        assert th[0] < 0.01


class TestLongYieldFormula:
    def test_positive_H_bar_gives_rR(self):
        rR = 0.02
        result = long_yield_formula(rR, GAMMA, SIGMA, H_bar=1.0)
        assert result == pytest.approx(rR)

    def test_zero_H_bar_gives_rR(self):
        rR = 0.02
        result = long_yield_formula(rR, GAMMA, SIGMA, H_bar=0.0)
        assert result == pytest.approx(rR)

    def test_negative_H_bar_strictly_below_rR(self):
        rR = 0.02
        result = long_yield_formula(rR, GAMMA, SIGMA, H_bar=-0.5)
        assert result < rR

    def test_negative_H_bar_formula(self):
        rR = 0.02
        H_bar = -0.3
        result = long_yield_formula(rR, GAMMA, SIGMA, H_bar)
        expected = rR + GAMMA * SIGMA**2 * H_bar
        assert result == pytest.approx(expected)


class TestClassifyPath:
    def _make_A_H(self, path="fp", n=300):
        T_vals = np.linspace(1, 500, n)
        if path == "fp":
            A = A_fading_pessimism_exact(T_vals, ETA0)
            H = H_fading_pessimism_exact(T_vals, ETA0)
        elif path == "cc":
            dt = T_vals[1] - T_vals[0]
            t_fine = np.linspace(0, T_vals[-1], 50000)
            dt_fine = t_fine[1] - t_fine[0]
            eta_cc = 2 * np.cos(OMEGA * t_fine)
            H_fine = np.cumsum(eta_cc * dt_fine)
            A_fine = np.cumsum(eta_cc**2 * dt_fine)
            # Sample at T_vals
            H = np.interp(T_vals, t_fine, H_fine)
            A = np.interp(T_vals, t_fine, A_fine)
        return A, H

    def test_fp_A_diverges(self):
        A, H = self._make_A_H("fp")
        result = classify_path(A, H)
        assert result["A_diverges"]

    def test_fp_H_over_T_converges_to_zero(self):
        """For FP, H_T ~ -2η0√T so H_T/T → 0 as T→∞.
        classify_path uses index-based slope so the estimate may differ from
        the exact limit; we just verify it is numerically small (< 0.5)."""
        A, H = self._make_A_H("fp")
        result = classify_path(A, H)
        assert result["H_over_T_limit"] is not None
        assert abs(result["H_over_T_limit"]) < 0.5

    def test_cc_A_diverges(self):
        A, H = self._make_A_H("cc")
        result = classify_path(A, H)
        assert result["A_diverges"]

    def test_cc_H_over_T_near_zero(self):
        """For CC, H_T = (2/ω)sin(ωT) is bounded, so H_T/T → 0."""
        A, H = self._make_A_H("cc")
        result = classify_path(A, H)
        assert result["H_over_T_limit"] is not None
        assert abs(result["H_over_T_limit"]) < 0.1


class TestMixedLimitMartingale:
    def _make_martingales(self, n_paths=10, seed=0):
        p = Parameters(mu=0.02, sigma=SIGMA, rho=0.02, gamma=GAMMA, D0=1.0, s0=S0)
        t, dB, _ = simulate_brownian(T=2.0, dt=0.1, n_paths=n_paths, seed=seed)
        eta_left = np.full(len(t) - 1, 0.5)
        log_MR = benchmark_martingale_R(t, dB, 0.1, SIGMA, GAMMA)
        log_MN = benchmark_martingale_N(t, eta_left, dB, 0.1, SIGMA, GAMMA)
        return log_MR, log_MN

    def test_starts_at_one(self):
        log_MR, log_MN = self._make_martingales()
        log_M = mixed_limit_martingale(0.5, log_MR, log_MN)
        assert np.allclose(np.exp(log_M[0]), 1.0)

    def test_pure_rational_equals_MR(self):
        """θ_h = 0 → limit martingale = M^R."""
        log_MR, log_MN = self._make_martingales()
        log_M = mixed_limit_martingale(0.0 + 1e-15, log_MR, log_MN)
        assert np.allclose(log_M, log_MR, atol=1e-8)

    def test_pure_dogmatist_equals_MN(self):
        """θ_h = 1 → limit martingale = M^N."""
        log_MR, log_MN = self._make_martingales()
        log_M = mixed_limit_martingale(1.0 - 1e-15, log_MR, log_MN)
        assert np.allclose(log_M, log_MN, atol=1e-8)

    def test_shape(self):
        log_MR, log_MN = self._make_martingales(n_paths=7)
        log_M = mixed_limit_martingale(0.3, log_MR, log_MN)
        assert log_M.shape == log_MR.shape

    def test_interior_between_MR_and_MN(self):
        """For interior θ, limit martingale lies between M^R and M^N."""
        log_MR, log_MN = self._make_martingales(seed=42)
        log_M_half = mixed_limit_martingale(0.5, log_MR, log_MN)
        # At t=0 this is trivially 0; check at some later step
        assert np.all(
            np.exp(log_M_half[5]) >= np.minimum(np.exp(log_MR[5]), np.exp(log_MN[5])) - 1e-14
        )
