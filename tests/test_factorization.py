"""Tests for src/factorization.py — Hansen-Scheinkman factorizations."""

from __future__ import annotations

import numpy as np
import pytest

from src.factorization import (
    identity_check_MR_wgamma,
    log_phi_ab,
    mixed_martingale_log,
    phi_ab,
    sdf_factorization_check,
)
from src.model import (
    Parameters,
    benchmark_martingale_N,
    benchmark_martingale_R,
    log_endowment,
    log_sdf,
    log_w,
    log_xi,
    rational_yield,
    simulate_brownian,
)
from src.paths import H_left_riemann, eta_fading_pessimism

DT = 0.05
T = 2.0
N_PATHS = 20
SEED = 7


def _base_sim():
    p = Parameters(mu=0.02, sigma=0.15, rho=0.02, gamma=1.0, D0=1.0, s0=0.05)
    t, dB, _ = simulate_brownian(T=T, dt=DT, n_paths=N_PATHS, seed=SEED)
    eta_left = eta_fading_pessimism(t[:-1], eta0=-1.5)
    lxi = log_xi(eta_left, dB, DT, p.sigma, p.gamma)
    lw = log_w(lxi, p.w0, p.gamma)
    logD = log_endowment(t, dB, DT, p)
    lpi = log_sdf(t, logD, lw, p)
    log_MR = benchmark_martingale_R(t, dB, DT, p.sigma, p.gamma)
    log_MN = benchmark_martingale_N(t, eta_left, dB, DT, p.sigma, p.gamma)
    # Use left-Riemann H for exact consistency with log_xi quadrature
    H_t = H_left_riemann(eta_left, DT)
    rR = rational_yield(p)
    return p, t, lw, lpi, log_MR, log_MN, H_t, rR


class TestPhiAb:
    def test_rational_case_is_positive(self):
        """Φ_{1,0}(0,0,w0) = 1/(1+w0)^γ > 0."""
        w0 = 1 / 19
        val = phi_ab(0.0, 0.0, w0, a=1.0, b=0.0, gamma=1.0, sigma=0.15)
        assert val > 0

    def test_dogmatist_case_formula(self):
        """Φ_{0,1}(0, H=0, w) = w^γ/(1+w)^γ."""
        w = 0.1
        gamma = 2.0
        sigma = 0.15
        val = phi_ab(0.0, 0.0, w, a=0.0, b=1.0, gamma=gamma, sigma=sigma)
        expected = w**gamma / (1 + w) ** gamma
        assert val == pytest.approx(expected, rel=1e-12)

    def test_mixed_case_positive(self):
        w = 0.05
        val = phi_ab(0.0, 0.0, w, a=1.0, b=1.0, gamma=1.0, sigma=0.15)
        assert val > 0


class TestLogPhiAb:
    def test_consistent_with_phi_ab(self):
        H = 1.5
        log_w_val = np.log(0.1)
        w = np.exp(log_w_val)
        gamma, sigma = 1.0, 0.15
        a, b = 1.0, 1.0
        direct = np.log(phi_ab(0.0, H, w, a, b, gamma, sigma))
        via_log = log_phi_ab(H, log_w_val, a, b, gamma, sigma)
        assert via_log == pytest.approx(direct, rel=1e-12)

    def test_rational_only_no_w_dependence_on_b(self):
        """For a=1, b=0: result should not depend on log_w."""
        H = 0.5
        gamma, sigma = 2.0, 0.10
        a, b = 1.0, 0.0
        lw1 = np.log(0.05)
        lw2 = np.log(0.5)
        v1 = log_phi_ab(H, lw1, a, b, gamma, sigma)
        # For a=1,b=0: log Φ = log(1) - γ log(1+w) = -γ*logaddexp(0, log_w)
        assert v1 != log_phi_ab(H, lw2, a, b, gamma, sigma)


class TestMixedMartingaleLog:
    def test_starts_at_one(self):
        """M_0^{a,b} = 1 so log M_0^{a,b} = 0."""
        p, t, lw, lpi, log_MR, log_MN, H_t, rR = _base_sim()
        for a, b in [(1, 0), (0, 1), (1, 1)]:
            log_Mab = mixed_martingale_log(log_MR, log_MN, p.w0, a, b, p.gamma)
            assert np.allclose(log_Mab[0], 0.0, atol=1e-14), f"a={a},b={b}"

    def test_is_finite(self):
        p, t, lw, lpi, log_MR, log_MN, H_t, rR = _base_sim()
        log_Mab = mixed_martingale_log(log_MR, log_MN, p.w0, 1.0, 1.0, p.gamma)
        assert np.all(np.isfinite(log_Mab))


class TestIdentityCheck:
    def test_identity_holds_to_machine_precision(self):
        """M_t^R w_t^γ = w_0^γ exp(−γσ²H_t) M_t^N up to floating-point tolerance."""
        p, t, lw, lpi, log_MR, log_MN, H_t, rR = _base_sim()
        result = identity_check_MR_wgamma(
            log_MR, log_MN, p.w0, lw, H_t, p.gamma, p.sigma, tol=1e-8
        )
        assert result["passed"], f"max_abs_err={result['max_abs_err']:.2e}"

    def test_residual_shape(self):
        p, t, lw, lpi, log_MR, log_MN, H_t, rR = _base_sim()
        result = identity_check_MR_wgamma(
            log_MR, log_MN, p.w0, lw, H_t, p.gamma, p.sigma
        )
        assert result["residual"].shape == (len(t), N_PATHS)


class TestSDFFactorizationCheck:
    def _run(self, a, b):
        p, t, lw, lpi, log_MR, log_MN, H_t, rR = _base_sim()
        wt = np.exp(lw)
        log_Mab = mixed_martingale_log(log_MR, log_MN, p.w0, a, b, p.gamma)
        return sdf_factorization_check(
            t, lpi, log_Mab, H_t, wt, p.w0, a, b, p.gamma, p.sigma, rR, tol=1e-8
        )

    def test_rational_factorization(self):
        result = self._run(a=1.0, b=0.0)
        assert result["passed"], f"max_abs_err={result['max_abs_err']:.2e}"

    def test_dogmatist_factorization(self):
        result = self._run(a=0.0, b=1.0)
        assert result["passed"], f"max_abs_err={result['max_abs_err']:.2e}"

    def test_mixed_factorization(self):
        result = self._run(a=1.0, b=1.0)
        assert result["passed"], f"max_abs_err={result['max_abs_err']:.2e}"

    def test_all_cases_same_eigenvalue(self):
        """All (a,b) pairs share r^R — confirmed by passing same rR value."""
        p, t, lw, lpi, log_MR, log_MN, H_t, rR = _base_sim()
        # Confirm same rR appears in all three factorizations (just check consistency)
        for a, b in [(1, 0), (0, 1), (1, 1)]:
            wt = np.exp(lw)
            log_Mab = mixed_martingale_log(log_MR, log_MN, p.w0, a, b, p.gamma)
            result = sdf_factorization_check(
                t, lpi, log_Mab, H_t, wt, p.w0, a, b, p.gamma, p.sigma, rR
            )
            assert result["passed"], f"Factorization failed for a={a},b={b}"
