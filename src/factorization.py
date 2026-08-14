"""Hansen-Scheinkman factorizations for the Arrow-Debreu SDF (Experiment 2).

The SDF decomposes as

    π_t = exp(−r_R t) · M_t^{a,b} · Φ_{a,b}(0, H_0, w_0) / Φ_{a,b}(t, H_t, w_t)

where

    Φ_{a,b}(t, H_t, w_t) = [a + b exp(γ σ² H_t) w_t^γ] / (1 + w_t)^γ

and M_t^{a,b} is the (a, b)-mixed martingale

    M_t^{a,b} = [a M_t^R + b w_0^γ M_t^N] / (a + b w_0^γ).

The companion identity

    M_t^R · w_t^γ = w_0^γ · exp(−γ σ² H_t) · M_t^N

follows from the definitions of M_t^R and M_t^N.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .model import Parameters, rational_yield


# ---------------------------------------------------------------------------
# Eigenfunction Φ_{a,b}
# ---------------------------------------------------------------------------


def phi_ab(
    s: float | np.ndarray,
    H_s: float | np.ndarray,
    w: float | np.ndarray,
    a: float,
    b: float,
    gamma: float,
    sigma: float,
) -> float | np.ndarray:
    """Evaluate the eigenfunction Φ_{a,b}(s, H_s, w).

    The family of eigenfunctions indexed by (a, b) is defined as

        Φ_{a,b}(s, H_s, w) = [a + b exp(γ σ² H_s) w^γ] / (1 + w)^γ

    At t = 0, H_0 = 0 so Φ_{a,b}(0, 0, w_0) = (a + b w_0^γ) / (1 + w_0)^γ.

    Parameters
    ----------
    s : float or np.ndarray
        Time index (kept for API completeness; the formula depends only on
        H_s and w, not s directly).
    H_s : float or np.ndarray
        Level integral H_t = ∫₀ˢ η_u du evaluated at time s.
    w : float or np.ndarray
        Dogmatist-to-rationalist wealth ratio w_s.
    a : float
        Rationalist weight in the mixture (a ≥ 0).
    b : float
        Dogmatist weight in the mixture (b ≥ 0).
    gamma : float
        CRRA coefficient γ.
    sigma : float
        Endowment volatility σ.

    Returns
    -------
    float or np.ndarray
        Φ_{a,b}(s, H_s, w).
    """
    numerator = a + b * np.exp(gamma * sigma**2 * H_s) * w**gamma
    denominator = (1.0 + w) ** gamma
    return numerator / denominator


def log_phi_ab(
    H_s: float | np.ndarray,
    log_w: float | np.ndarray,
    a: float,
    b: float,
    gamma: float,
    sigma: float,
) -> float | np.ndarray:
    """Numerically stable log of Φ_{a,b} using log-sum-exp.

    Parameters
    ----------
    H_s : float or np.ndarray
        Level integral H_t at evaluation time.
    log_w : float or np.ndarray
        log w_t (log dogmatist-to-rationalist wealth ratio).
    a : float
        Rationalist mixture weight (a ≥ 0).
    b : float
        Dogmatist mixture weight (b ≥ 0).
    gamma : float
        CRRA coefficient γ.
    sigma : float
        Endowment volatility σ.

    Returns
    -------
    float or np.ndarray
        log Φ_{a,b}(H_s, w).
    """
    log_a = np.log(a) if a > 0 else -np.inf
    log_b_term = np.log(b) + gamma * sigma**2 * H_s + gamma * log_w if b > 0 else -np.inf
    log_numerator = np.logaddexp(log_a, log_b_term)
    log_denominator = gamma * np.logaddexp(0.0, log_w)
    return log_numerator - log_denominator


# ---------------------------------------------------------------------------
# Mixed martingale M^{a,b}
# ---------------------------------------------------------------------------


def mixed_martingale_log(
    log_MR: np.ndarray,
    log_MN: np.ndarray,
    w0: float,
    a: float,
    b: float,
    gamma: float,
) -> np.ndarray:
    """Compute log M_t^{a,b}, the (a, b)-mixed martingale in log form.

    The mixture is defined as

        M_t^{a,b} = [a M_t^R + b w_0^γ M_t^N] / (a + b w_0^γ)

    so that

        log M_t^{a,b} = log(a exp(log M_t^R) + b w_0^γ exp(log M_t^N))
                        − log(a + b w_0^γ)

    The numerator log-sum-exp is evaluated in a numerically stable way.

    Parameters
    ----------
    log_MR : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^R (from
        :func:`src.model.benchmark_martingale_R`).
    log_MN : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^N (from
        :func:`src.model.benchmark_martingale_N`).
    w0 : float
        Initial wealth ratio w_0.
    a : float
        Rationalist weight a ≥ 0.
    b : float
        Dogmatist weight b ≥ 0.
    gamma : float
        CRRA coefficient γ.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^{a,b}.
    """
    log_a = np.log(a) if a > 0 else -np.inf
    log_bwgamma = np.log(b) + gamma * np.log(w0) if b > 0 else -np.inf
    log_normaliser = np.logaddexp(log_a, log_bwgamma)

    log_num = np.logaddexp(log_a + log_MR, log_bwgamma + log_MN)
    return log_num - log_normaliser


# ---------------------------------------------------------------------------
# Numerical verification checks
# ---------------------------------------------------------------------------


def sdf_factorization_check(
    t_grid: np.ndarray,
    log_pi: np.ndarray,
    log_Mab: np.ndarray,
    H_t: np.ndarray,
    w_t: np.ndarray,
    w0: float,
    a: float,
    b: float,
    gamma: float,
    sigma: float,
    rR: float,
    tol: float = 1e-8,
) -> dict:
    """Verify the Hansen-Scheinkman SDF factorization numerically.

    Checks that the maximum absolute error of

        log π_t − [−r_R t + log M_t^{a,b}
                   + log Φ_{a,b}(0, 0, w_0)
                   − log Φ_{a,b}(t, H_t, w_t)]

    is below ``tol`` (default 1e-8).

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid.
    log_pi : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log π_t.
    log_Mab : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^{a,b}.
    H_t : np.ndarray
        Shape ``(n_steps + 1,)``.  Level integral H_t (deterministic).

        .. note::
            For machine-precision residuals (< 1e-8) this must be computed
            with the same left-endpoint Riemann quadrature used by
            :func:`src.model.log_xi`.  Use
            :func:`src.paths.H_left_riemann` with ``eta_vals_left =
            eta_vals[:-1]``.  Passing the trapezoid-rule output of
            :func:`src.paths.H_numerical` introduces an O(Δt) discrepancy.
    w_t : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Wealth ratio w_t = exp(log w_t).
    w0 : float
        Initial wealth ratio.
    a, b : float
        Mixture weights.
    gamma : float
        CRRA coefficient.
    sigma : float
        Endowment volatility.
    rR : float
        Rational yield r_R.
    tol : float, optional
        Tolerance for the maximum absolute residual.

    Returns
    -------
    dict
        Keys:

        * ``'residual'``    – np.ndarray of shape ``(n_steps+1, n_paths)``.
        * ``'max_abs_err'`` – float, maximum absolute residual.
        * ``'passed'``      – bool, True if max_abs_err < tol.
    """
    log_w_t = np.log(w_t)
    # Φ_{a,b}(0, H_0=0, w_0)
    log_phi0 = log_phi_ab(0.0, np.log(w0), a, b, gamma, sigma)
    # Φ_{a,b}(t, H_t, w_t)
    log_phi_t = log_phi_ab(H_t[:, None], log_w_t, a, b, gamma, sigma)

    rhs = -rR * t_grid[:, None] + log_Mab + log_phi0 - log_phi_t
    residual = log_pi - rhs
    max_abs_err = float(np.max(np.abs(residual)))
    return {
        "residual": residual,
        "max_abs_err": max_abs_err,
        "passed": max_abs_err < tol,
    }


def identity_check_MR_wgamma(
    log_MR: np.ndarray,
    log_MN: np.ndarray,
    w0: float,
    log_w: np.ndarray,
    H_t: np.ndarray,
    gamma: float,
    sigma: float,
    tol: float = 1e-8,
) -> dict:
    """Verify the identity M_t^R · w_t^γ = w_0^γ exp(−γ σ² H_t) M_t^N.

    In log form the identity is

        log M_t^R + γ log w_t = γ log w_0 − γ σ² H_t + log M_t^N.

    This follows from the definitions of M_t^R, M_t^N, and log w_t.

    Parameters
    ----------
    log_MR : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^R.
    log_MN : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^N.
    w0 : float
        Initial wealth ratio w_0.
    log_w : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log w_t.
    H_t : np.ndarray
        Shape ``(n_steps + 1,)``.  Level integral H_t.

        .. note::
            For machine-precision residuals (< 1e-8) pass H_t from
            :func:`src.paths.H_left_riemann`, which uses the same
            left-endpoint Riemann quadrature as the martingale functions.
            The trapezoid-rule variant introduces an O(Δt) mismatch.
    gamma : float
        CRRA coefficient γ.
    sigma : float
        Endowment volatility σ.
    tol : float, optional
        Tolerance for the maximum absolute residual.

    Returns
    -------
    dict
        Keys:

        * ``'residual'``    – np.ndarray of shape ``(n_steps+1, n_paths)``.
        * ``'max_abs_err'`` – float.
        * ``'passed'``      – bool.
    """
    lhs = log_MR + gamma * log_w
    rhs = gamma * np.log(w0) - gamma * sigma**2 * H_t[:, None] + log_MN
    residual = lhs - rhs
    max_abs_err = float(np.max(np.abs(residual)))
    return {
        "residual": residual,
        "max_abs_err": max_abs_err,
        "passed": max_abs_err < tol,
    }
