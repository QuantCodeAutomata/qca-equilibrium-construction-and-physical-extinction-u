"""Tests for src/section4.py — γ=1 closed-form bond pricing."""

from __future__ import annotations

import numpy as np
import pytest

from src.section4 import (
    G_0T_constant_cyclical,
    G_0T_fading_pessimism,
    G_0T_numerical,
    bond_price_gamma1,
    bond_price_premium,
    bond_vol_gamma1,
    forward_growth_shift_gamma1,
    forward_rate_spread_gamma1,
    q_0T,
    theta_0T,
    yield_spread_gamma1,
)
from src.paths import eta_fading_pessimism


W0 = 1.0 / 19.0   # s0 = 5%
S0 = 0.05
SIGMA = 0.15
ETA0 = -1.5
OMEGA = 2.0 * np.pi / 20.0

# Rational yield for γ=1, mu=0.02, rho=0.02, sigma=0.15:
# r^R = 0.02 + 0.02 - 0.5*1*2*0.15^2 = 0.04 - 0.0225 = 0.0175
RHO = 0.02
MU = 0.02
RR = RHO + MU - 0.5 * 1 * 2 * SIGMA**2   # = 0.0175


class TestQ0T:
    def test_equals_w0_when_G_zero(self):
        assert q_0T(W0, 0.0) == pytest.approx(W0)

    def test_decreases_with_positive_G(self):
        q1 = q_0T(W0, 1.0)
        q2 = q_0T(W0, 2.0)
        assert q2 < q1 < W0

    def test_always_positive(self):
        G = np.linspace(-5, 5, 100)
        assert np.all(q_0T(W0, G) > 0)


class TestTheta0T:
    def test_range(self):
        """θ must lie in (0, 1)."""
        G = np.linspace(-3, 3, 100)
        q = q_0T(W0, G)
        th = theta_0T(q)
        assert np.all(th > 0) and np.all(th < 1)

    def test_equals_w_over_1pw_when_G0(self):
        q = q_0T(W0, 0.0)
        th = theta_0T(q)
        assert th == pytest.approx(W0 / (1 + W0), rel=1e-12)

    def test_formula_direct(self):
        q = 0.3
        assert theta_0T(q) == pytest.approx(0.3 / 1.3, rel=1e-12)


class TestBondPriceGamma1:
    def test_G0_price_equals_rational(self):
        """With G = 0, q = w0, bond price = exp(-rR*T)."""
        T = 10.0
        P = bond_price_gamma1(T, RR, W0, 0.0)
        expected = np.exp(-RR * T)
        assert P == pytest.approx(expected, rel=1e-12)

    def test_pessimism_raises_price(self):
        """Fading pessimism (η0 < 0) → G < 0 → q > w0 → bond price above rational."""
        T = 20.0
        G = G_0T_fading_pessimism(T, ETA0, SIGMA)
        P_het = bond_price_gamma1(T, RR, W0, G)
        P_rat = np.exp(-RR * T)
        assert P_het > P_rat

    def test_positive(self):
        T = np.linspace(1, 50, 100)
        G = G_0T_fading_pessimism(T, ETA0, SIGMA)
        P = bond_price_gamma1(T, RR, W0, G)
        assert np.all(P > 0)


class TestYieldSpreadGamma1:
    def test_zero_spread_when_G_zero(self):
        T = np.array([5.0, 10.0, 20.0])
        spread = yield_spread_gamma1(T, W0, np.zeros_like(T))
        assert np.allclose(spread, 0.0, atol=1e-12)

    def test_negative_spread_under_pessimism(self):
        """Fading pessimism → q > w0 → log(1+w0) < log(1+q) → spread < 0."""
        T = np.array([10.0, 20.0, 30.0])
        G = G_0T_fading_pessimism(T, ETA0, SIGMA)
        spread = yield_spread_gamma1(T, W0, G)
        assert np.all(spread < 0)

    def test_converges_to_zero(self):
        """Spread should converge to 0 as T → ∞ for fading pessimism."""
        T = np.array([50.0, 100.0, 200.0, 500.0])
        G = G_0T_fading_pessimism(T, ETA0, SIGMA)
        spread = yield_spread_gamma1(T, W0, G)
        # |spread| should decrease with T
        assert np.all(np.diff(np.abs(spread)) < 0)


class TestForwardRateSpread:
    def test_zero_eta_gives_zero_spread(self):
        assert forward_rate_spread_gamma1(0.0, 0.0, SIGMA, 0.3) == pytest.approx(0.0)

    def test_sign_follows_eta_theta(self):
        """Negative η → negative forward spread."""
        result = forward_rate_spread_gamma1(0.0, ETA0, SIGMA, 0.1)
        assert result < 0


class TestBondVol:
    def test_non_negative(self):
        T = np.linspace(1, 50, 50)
        G = G_0T_fading_pessimism(T, ETA0, SIGMA)
        th = theta_0T(q_0T(W0, G))
        vol = bond_vol_gamma1(ETA0, SIGMA, th, S0)
        assert np.all(vol >= 0)


class TestForwardGrowthShift:
    def test_zero_eta_gives_zero(self):
        assert forward_growth_shift_gamma1(0.0, SIGMA, 0.3) == pytest.approx(0.0)

    def test_sign_follows_eta(self):
        assert forward_growth_shift_gamma1(ETA0, SIGMA, 0.2) < 0


class TestG0T:
    def test_fp_zero_at_T0(self):
        assert G_0T_fading_pessimism(0.0, ETA0, SIGMA) == pytest.approx(0.0)

    def test_cc_zero_at_T0(self):
        assert G_0T_constant_cyclical(0.0, OMEGA, SIGMA) == pytest.approx(0.0, abs=1e-14)

    def test_fp_matches_numerical(self):
        T = 20.0
        dt = 1e-4
        t_grid = np.arange(0, T + dt, dt)
        eta_vals = eta_fading_pessimism(t_grid, ETA0)
        G_num = G_0T_numerical(t_grid, eta_vals, SIGMA, dt)
        G_exact = G_0T_fading_pessimism(T, ETA0, SIGMA)
        assert G_num[-1] == pytest.approx(G_exact, rel=1e-4)


class TestBondPricePremium:
    def test_zero_when_G_zero(self):
        T = 10.0
        prem = bond_price_premium(T, RR, W0, 0.0)
        assert prem == pytest.approx(0.0, abs=1e-12)

    def test_positive_under_pessimism(self):
        T = 20.0
        G = G_0T_fading_pessimism(T, ETA0, SIGMA)
        prem = bond_price_premium(T, RR, W0, G)
        assert prem > 0
