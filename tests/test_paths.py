"""Tests for src/paths.py — belief-distortion path primitives."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from src.paths import (
    A_fading_pessimism_exact,
    A_left_riemann,
    A_numerical,
    H_constant_cyclical_exact,
    H_fading_pessimism_exact,
    H_left_riemann,
    H_numerical,
    eta_constant_cyclical,
    eta_fading_alternating,
    eta_fading_pessimism,
)

OMEGA = 2.0 * np.pi / 20.0  # L = 20 years


# ---------------------------------------------------------------------------
# eta functions
# ---------------------------------------------------------------------------

class TestEtaFunctions:
    def test_fp_at_zero(self):
        assert eta_fading_pessimism(0.0, eta0=-1.5) == pytest.approx(-1.5)

    def test_fp_decays_towards_zero(self):
        eta0 = -1.5
        vals = eta_fading_pessimism(np.array([0.0, 1.0, 9.0, 99.0]), eta0)
        assert np.all(np.diff(np.abs(vals)) < 0)

    def test_fa_at_zero(self):
        # eta^FA(0) = 2 sin(0)/sqrt(1) = 0
        assert eta_fading_alternating(0.0, OMEGA) == pytest.approx(0.0, abs=1e-14)

    def test_cc_amplitude(self):
        # eta^CA(0) = 2 cos(0) = 2
        assert eta_constant_cyclical(0.0, OMEGA) == pytest.approx(2.0)

    def test_cc_is_periodic(self):
        """η_CC(t + L) = η_CC(t)."""
        L = 2 * np.pi / OMEGA
        t = np.array([1.0, 5.0, 15.0])
        assert np.allclose(
            eta_constant_cyclical(t, OMEGA),
            eta_constant_cyclical(t + L, OMEGA),
        )

    def test_vectorized_shapes(self):
        t = np.linspace(0, 50, 500)
        for fn, kw in [
            (eta_fading_pessimism, {"eta0": -1.5}),
            (eta_fading_alternating, {"omega": OMEGA}),
            (eta_constant_cyclical, {"omega": OMEGA}),
        ]:
            assert fn(t, **kw).shape == t.shape


# ---------------------------------------------------------------------------
# Exact A_t for fading pessimism
# ---------------------------------------------------------------------------

class TestAFadingPessimism:
    def test_zero_at_t0(self):
        assert A_fading_pessimism_exact(0.0, -1.5) == pytest.approx(0.0)

    def test_monotone_increasing(self):
        t = np.linspace(0, 50, 1000)
        A = A_fading_pessimism_exact(t, -1.5)
        assert np.all(np.diff(A) >= 0)

    def test_diverges_to_infinity(self):
        """A_t = η₀² log(1+t) grows without bound (logarithmically)."""
        # At T=1e8: A = 2.25 * log(1+1e8) ≈ 41.4; still increases without bound
        A_small = A_fading_pessimism_exact(1.0, -1.5)
        A_large = A_fading_pessimism_exact(1e8, -1.5)
        assert A_large > A_small
        assert A_large > 40.0   # confirmed by formula: 2.25*log(1e8) ≈ 41.4

    def test_formula_matches_quadrature(self):
        eta0 = -1.5
        T = 20.0
        quad_val, _ = quad(lambda s: (eta0 / np.sqrt(1 + s)) ** 2, 0, T)
        exact_val = A_fading_pessimism_exact(T, eta0)
        assert exact_val == pytest.approx(quad_val, rel=1e-6)


# ---------------------------------------------------------------------------
# Exact H_t for fading pessimism
# ---------------------------------------------------------------------------

class TestHFadingPessimism:
    def test_zero_at_t0(self):
        assert H_fading_pessimism_exact(0.0, -1.5) == pytest.approx(0.0)

    def test_negative_for_negative_eta0(self):
        t = np.linspace(1, 50, 100)
        H = H_fading_pessimism_exact(t, -1.5)
        assert np.all(H < 0)

    def test_formula_matches_quadrature(self):
        eta0 = -1.5
        T = 30.0
        quad_val, _ = quad(lambda s: eta0 / np.sqrt(1 + s), 0, T)
        exact_val = H_fading_pessimism_exact(T, eta0)
        assert exact_val == pytest.approx(quad_val, rel=1e-6)


# ---------------------------------------------------------------------------
# Exact H_t for constant cyclical
# ---------------------------------------------------------------------------

class TestHConstantCyclical:
    def test_zero_at_t0(self):
        assert H_constant_cyclical_exact(0.0, OMEGA) == pytest.approx(0.0, abs=1e-14)

    def test_bounded(self):
        t = np.linspace(0, 200, 10000)
        H = H_constant_cyclical_exact(t, OMEGA)
        assert np.all(np.abs(H) <= 2.0 / OMEGA + 1e-10)

    def test_formula_matches_quadrature(self):
        T = 15.0
        quad_val, _ = quad(lambda s: 2 * np.cos(OMEGA * s), 0, T)
        exact_val = H_constant_cyclical_exact(T, OMEGA)
        assert exact_val == pytest.approx(quad_val, rel=1e-6)


# ---------------------------------------------------------------------------
# Numerical integrals (trapezoid and left-Riemann)
# ---------------------------------------------------------------------------

class TestNumericalIntegrals:
    def _setup(self, T=10.0, n=10000):
        t = np.linspace(0, T, n + 1)
        dt = T / n
        eta = eta_fading_pessimism(t, eta0=-1.5)
        return t, eta, dt

    def test_A_numerical_starts_at_zero(self):
        t, eta, dt = self._setup()
        A = A_numerical(t, eta, dt)
        assert A[0] == pytest.approx(0.0)

    def test_A_numerical_matches_exact(self):
        T = 10.0
        t, eta, dt = self._setup(T=T)
        A_num = A_numerical(t, eta, dt)
        A_exact = A_fading_pessimism_exact(T, -1.5)
        assert A_num[-1] == pytest.approx(A_exact, rel=1e-4)

    def test_H_numerical_starts_at_zero(self):
        t, eta, dt = self._setup()
        H = H_numerical(t, eta, dt)
        assert H[0] == pytest.approx(0.0)

    def test_H_numerical_matches_exact(self):
        T = 10.0
        t, eta, dt = self._setup(T=T)
        H_num = H_numerical(t, eta, dt)
        H_exact = H_fading_pessimism_exact(T, -1.5)
        assert H_num[-1] == pytest.approx(H_exact, rel=1e-4)

    def test_left_riemann_A_shape(self):
        n = 100
        eta_left = eta_fading_pessimism(np.linspace(0, 10, n), eta0=-1.5)
        A = A_left_riemann(eta_left, dt=0.1)
        assert A.shape == (n + 1,)

    def test_left_riemann_H_shape(self):
        n = 100
        eta_left = eta_fading_pessimism(np.linspace(0, 10, n), eta0=-1.5)
        H = H_left_riemann(eta_left, dt=0.1)
        assert H.shape == (n + 1,)

    def test_left_riemann_A_starts_at_zero(self):
        eta_left = np.ones(50)
        A = A_left_riemann(eta_left, dt=0.1)
        assert A[0] == pytest.approx(0.0)

    def test_left_riemann_H_constant_eta(self):
        """With η ≡ c, H_T = c * T exactly."""
        c = 0.7
        n = 1000
        dt = 0.01
        eta_left = np.full(n, c)
        H = H_left_riemann(eta_left, dt=dt)
        assert H[-1] == pytest.approx(c * n * dt, rel=1e-12)
