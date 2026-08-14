"""Tests for src/approximation.py — general-γ bond-price approximation."""

from __future__ import annotations

import numpy as np
import pytest

from src.approximation import (
    bond_price_approx,
    compute_C_gamma,
    e_tT_bound,
    K_gamma,
    q_tT_general,
    remainder_Rgamma,
)
from src.section4 import bond_price_gamma1, G_0T_fading_pessimism


W0 = 1.0 / 19.0
SIGMA = 0.15
RHO = 0.02
MU = 0.02


class TestRemainderRgamma:
    def test_zero_at_zero(self):
        """R_γ(0) = (1+0)^γ - 1 - 0 = 0."""
        for gamma in [0.5, 2.0, 5.0]:
            r = remainder_Rgamma(0.0, gamma)
            assert r == pytest.approx(0.0, abs=1e-15)

    def test_non_negative_for_gamma_gt1(self):
        """For γ > 1 and x > 0: R_γ(x) ≥ 0 (power-mean inequality)."""
        x = np.logspace(-3, 3, 200)
        for gamma in [2.0, 3.0, 5.0]:
            assert np.all(remainder_Rgamma(x, gamma) >= 0)

    def test_formula_explicit(self):
        x, gamma = 2.0, 2.0
        expected = (1 + x) ** gamma - 1 - x**gamma
        assert remainder_Rgamma(x, gamma) == pytest.approx(expected)

    def test_scalar_array_consistency(self):
        x_arr = np.array([1.0, 2.0, 3.0])
        gamma = 3.0
        r_arr = remainder_Rgamma(x_arr, gamma)
        r_scalars = np.array([remainder_Rgamma(float(xi), gamma) for xi in x_arr])
        assert np.allclose(r_arr, r_scalars)


class TestComputeCGamma:
    def test_positive(self):
        for gamma in [0.5, 2.0, 5.0]:
            C = compute_C_gamma(gamma)
            assert C > 0

    def test_gamma2_is_finite(self):
        C = compute_C_gamma(2.0)
        assert np.isfinite(C)

    def test_large_grid_stable(self):
        """Widening the x-grid should not reduce the constant significantly."""
        x_narrow = np.logspace(-4, 4, 50_000)
        x_wide = np.logspace(-8, 8, 200_000)
        C_narrow = compute_C_gamma(2.0, x_grid=x_narrow)
        C_wide = compute_C_gamma(2.0, x_grid=x_wide)
        # Wide grid should give at least as large a value
        assert C_wide >= C_narrow * 0.9


class TestKGamma:
    def test_equals_2C(self):
        C = compute_C_gamma(2.0)
        assert K_gamma(C) == pytest.approx(2 * C)


class TestQtTGeneral:
    def test_gamma1_matches_section4(self):
        """For γ=1, q_{t,T} = w_t exp(-σ² H_{t,T}) = w_t exp(-G_{t,T}/σ²·σ²)."""
        w_t = W0
        H_tT = 2.0
        gamma = 1.0
        q_gen = q_tT_general(w_t, gamma, SIGMA, H_tT)
        q_sec4 = W0 * np.exp(-SIGMA**2 * H_tT)
        assert q_gen == pytest.approx(q_sec4, rel=1e-12)

    def test_positive(self):
        H = np.linspace(-5, 5, 50)
        for gamma in [0.5, 1.0, 2.0]:
            q = q_tT_general(W0, gamma, SIGMA, H)
            assert np.all(q > 0)

    def test_decreases_with_H(self):
        H = np.array([0.0, 1.0, 2.0])
        q = q_tT_general(W0, 2.0, SIGMA, H)
        assert np.all(np.diff(q) < 0)


class TestBondPriceApprox:
    def test_positive(self):
        T_arr = np.array([5.0, 10.0, 20.0, 30.0])
        gamma = 2.0
        rR = RHO + gamma * MU - 0.5 * gamma * (gamma + 1) * SIGMA**2
        H_arr = T_arr * 0.3
        q = q_tT_general(W0, gamma, SIGMA, H_arr)
        P = bond_price_approx(rR, T_arr, W0, gamma, q)
        assert np.all(P > 0)

    def test_gamma1_matches_section4_formula(self):
        """For γ=1, approx = exact Section-4 formula."""
        T = np.array([10.0, 20.0, 30.0])
        gamma = 1.0
        rR = RHO + MU - SIGMA**2
        G = G_0T_fading_pessimism(T, eta0=-1.5, sigma=SIGMA)
        H = G / SIGMA**2

        q_gen = q_tT_general(W0, gamma, SIGMA, H)
        P_approx = bond_price_approx(rR, T, W0, gamma, q_gen)
        P_section4 = bond_price_gamma1(T, rR, W0, G)
        assert np.allclose(P_approx, P_section4, rtol=1e-12)


class TestETTBound:
    def test_decreases_with_A(self):
        """Bound decays as A_{t,T} grows — the core approximation quality result."""
        gamma = 2.0
        C = compute_C_gamma(gamma)
        K = K_gamma(C)
        A_series = np.array([1.0, 5.0, 10.0, 50.0, 100.0])
        e = e_tT_bound(K, gamma, SIGMA, A_series)
        assert np.all(np.diff(e) < 0)

    def test_positive(self):
        gamma = 2.0
        C = compute_C_gamma(gamma)
        K = K_gamma(C)
        e = e_tT_bound(K, gamma, SIGMA, 5.0)
        assert e > 0

    def test_zero_A_gives_K(self):
        """When A=0, e_{t,T} = K_γ."""
        gamma = 2.0
        C = compute_C_gamma(gamma)
        K = K_gamma(C)
        e = e_tT_bound(K, gamma, SIGMA, 0.0)
        assert e == pytest.approx(K)

    def test_large_A_gives_small_bound(self):
        # Exponent: -γ/(2(1+γ)²) σ² A; for γ=2,σ=0.15 need A≈10000 to reach 1e-5
        gamma = 2.0
        C = compute_C_gamma(gamma)
        K = K_gamma(C)
        e = e_tT_bound(K, gamma, SIGMA, 10_000.0)
        assert e < 1e-5
