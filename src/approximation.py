"""General-γ bond-pricing approximation and error bounds (Experiment 5).

For general γ > 0 the exact bond price requires a conditional expectation that
has no closed form.  This module implements the approximation

    P̂(t, T) = exp(−r_R (T−t)) · (1 + w_t)^{−γ} · (1 + q_{t,T})

where q_{t,T} = w_t^γ exp(−γ σ² H_{t,T}), and derives a non-asymptotic
L¹ error bound:

    |P(t,T)/P(0,T) − P̂(t,T)/P̂(0,T)| ≤ e_{t,T}

    e_{t,T} = K_γ · exp(−γ / (2(1+γ)²) · σ² A_{t,T})

The constant K_γ = 2 C_γ where C_γ is obtained by numerically maximising
the normalised remainder R_γ(x) = (1+x)^γ − 1 − x^γ.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .model import Parameters, simulate_brownian, log_endowment, log_xi, log_w, log_sdf
from .paths import compute_path_objects


# ---------------------------------------------------------------------------
# Approximation objects
# ---------------------------------------------------------------------------


def q_tT_general(
    w_t: float | np.ndarray,
    gamma: float,
    sigma: float,
    H_tT: float | np.ndarray,
) -> float | np.ndarray:
    """Dogmatist's generalised pricing weight q_{t,T} for arbitrary γ.

    Defined as

        q_{t,T} = w_t^γ exp(−γ σ² H_{t,T})

    For γ = 1 this reduces to the Section-4 quantity q_{t,T} = w_t e^{−G_{t,T}}.

    Parameters
    ----------
    w_t : float or np.ndarray
        Wealth ratio w_t (or array over paths/maturities).
    gamma : float
        CRRA coefficient γ.
    sigma : float
        Endowment volatility σ.
    H_tT : float or np.ndarray
        Level integral H_{t,T} = ∫_t^T η_s ds.

    Returns
    -------
    float or np.ndarray
        q_{t,T} = w_t^γ exp(−γ σ² H_{t,T}).
    """
    return w_t**gamma * np.exp(-gamma * sigma**2 * H_tT)


def bond_price_approx(
    rR: float,
    T_minus_t: float | np.ndarray,
    w_t: float | np.ndarray,
    gamma: float,
    q_tT: float | np.ndarray,
) -> float | np.ndarray:
    """Approximate zero-coupon bond price P̂(t, T) for general γ.

    The approximation is

        P̂(t, T) = exp(−r_R (T−t)) · (1 + w_t)^{−γ} · (1 + q_{t,T})

    which coincides with the exact formula at γ = 1 (Section 4).

    Parameters
    ----------
    rR : float
        Rational yield r_R.
    T_minus_t : float or np.ndarray
        Time-to-maturity T − t (scalar or array over maturities).
    w_t : float or np.ndarray
        Wealth ratio w_t (scalar or array over paths).
    gamma : float
        CRRA coefficient γ.
    q_tT : float or np.ndarray
        Generalised pricing weight from :func:`q_tT_general`.  Must be
        broadcast-compatible with ``w_t`` and ``T_minus_t``.

    Returns
    -------
    float or np.ndarray
        Approximate bond price P̂(t, T).
    """
    return np.exp(-rR * T_minus_t) * (1.0 + w_t) ** (-gamma) * (1.0 + q_tT)


# ---------------------------------------------------------------------------
# Error bound machinery
# ---------------------------------------------------------------------------


def remainder_Rgamma(
    x: float | np.ndarray,
    gamma: float,
) -> float | np.ndarray:
    """Remainder R_γ(x) = (1 + x)^γ − 1 − x^γ in the power-function expansion.

    This quantity measures the failure of the approximation (1+x)^γ ≈ 1 + x^γ
    and appears naturally in the bond-price error analysis.

    Parameters
    ----------
    x : float or np.ndarray
        Non-negative argument x ≥ 0.
    gamma : float
        Exponent γ > 0.

    Returns
    -------
    float or np.ndarray
        R_γ(x) = (1 + x)^γ − 1 − x^γ.
    """
    return (1.0 + x) ** gamma - 1.0 - x**gamma


def compute_C_gamma(
    gamma: float,
    x_grid: np.ndarray | None = None,
) -> float:
    """Numerically compute the normalisation constant C_γ for the error bound.

    C_γ is the supremum over x > 0 of

        |R_γ(x)| / [x^{γ/(1+γ)} + x^{γ²/(1+γ)}]

    where R_γ(x) = (1+x)^γ − 1 − x^γ.

    Parameters
    ----------
    gamma : float
        CRRA coefficient γ > 0, γ ≠ 1.
    x_grid : np.ndarray or None, optional
        Grid of positive x values over which to maximise.  Defaults to a
        log-spaced grid from 10⁻⁶ to 10⁶ with 100 000 points.

    Returns
    -------
    float
        C_γ = sup_{x>0} |R_γ(x)| / [x^{γ/(1+γ)} + x^{γ²/(1+γ)}].
    """
    if x_grid is None:
        x_grid = np.logspace(-6, 6, 100_000)

    exp1 = gamma / (1.0 + gamma)
    exp2 = gamma**2 / (1.0 + gamma)

    numerator = np.abs(remainder_Rgamma(x_grid, gamma))
    denominator = x_grid**exp1 + x_grid**exp2

    # Avoid division by zero at x = 0 (denominator → 0 faster than numerator)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(denominator > 0.0, numerator / denominator, 0.0)

    return float(np.max(ratio))


def K_gamma(C_gamma: float) -> float:
    """Error-bound prefactor K_γ = 2 C_γ.

    Parameters
    ----------
    C_gamma : float
        The constant C_γ from :func:`compute_C_gamma`.

    Returns
    -------
    float
        K_γ = 2 C_γ.
    """
    return 2.0 * C_gamma


def e_tT_bound(
    K_gam: float,
    gamma: float,
    sigma: float,
    A_tT: float | np.ndarray,
) -> float | np.ndarray:
    """Non-asymptotic error bound e_{t,T} on the normalised bond-price error.

    The bound is

        e_{t,T} = K_γ · exp(−γ / (2(1+γ)²) · σ² · A_{t,T})

    where A_{t,T} = ∫_t^T η_s² ds is the quadratic-variation integral over
    the remaining horizon.  The bound decays to zero as A_{t,T} → ∞ (which
    happens when A_T diverges).

    Parameters
    ----------
    K_gam : float
        Error-bound prefactor K_γ = 2 C_γ.
    gamma : float
        CRRA coefficient γ.
    sigma : float
        Endowment volatility σ.
    A_tT : float or np.ndarray
        Quadratic-variation integral A_{t,T} = ∫_t^T η_s² ds.

    Returns
    -------
    float or np.ndarray
        Error bound e_{t,T}.
    """
    exponent = -gamma / (2.0 * (1.0 + gamma) ** 2) * sigma**2 * A_tT
    return K_gam * np.exp(exponent)


# ---------------------------------------------------------------------------
# Monte Carlo estimators
# ---------------------------------------------------------------------------


def exact_bond_price_mc(
    t: float,
    T: float,
    w_t: float,
    log_pi_t: float,
    t_grid: np.ndarray,
    eta_func,
    params: Parameters,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimate of the exact zero-coupon bond price E^P[π_T/π_t | F_t].

    Simulates ``n_paths`` independent continuations of the Brownian motion
    from t to T, evaluates π_T on each path, and returns the sample mean of
    π_T / π_t.

    Parameters
    ----------
    t : float
        Current time t ≥ 0.
    T : float
        Maturity T > t.
    w_t : float
        Realised wealth ratio w_t at time t (single scalar, representing a
        fixed conditioning event).
    log_pi_t : float
        Realised log π_t at time t.
    t_grid : np.ndarray
        Full time grid from 0 to some horizon ≥ T (used only to infer dt).
    eta_func : callable
        Belief-distortion function t ↦ η_t.
    params : Parameters
        Economy parameters.
    n_paths : int, optional
        Number of Monte Carlo paths (default 10 000).
    seed : int or None, optional
        RNG seed.

    Returns
    -------
    float
        Monte Carlo estimate of P(t, T) = E^P[π_T / π_t | F_t].
    """
    dt = float(t_grid[1] - t_grid[0])
    tau = T - t
    if tau <= 0:
        raise ValueError("T must be strictly greater than t.")

    # Simulate Brownian increments from t to T
    t_fut, dB_fut, B_fut = simulate_brownian(tau, dt, n_paths, seed=seed)

    # η values on the future grid [t, T] using left endpoints
    t_abs = t + t_fut[:-1]                         # (n_steps_fut,)
    eta_fut = eta_func(t_abs)                       # (n_steps_fut,)

    # Log-endowment increment from t to T: log(D_T/D_t) = log(D_T/D0) - log(D_t/D0)
    # We only need the increment, which is (mu - 0.5*sigma^2)*tau + sigma*B_fut[-1]
    drift_inc = (params.mu - 0.5 * params.sigma**2) * tau
    log_DT_over_Dt = drift_inc + params.sigma * B_fut[-1]  # (n_paths,)

    # log ξ increment from t to T
    stoch_fut = np.cumsum(eta_fut[:, None] * dB_fut, axis=0)[-1]  # (n_paths,)
    A_fut = float(np.sum(eta_fut**2 * dt))
    log_xi_inc = params.sigma * stoch_fut - 0.5 * params.sigma**2 * A_fut  # (n_paths,)

    # log w_T: log w_T = log w_t + (1/gamma) * log_xi_inc
    log_wT = np.log(w_t) + (1.0 / params.gamma) * log_xi_inc  # (n_paths,)

    # log π_T - log π_t using the SDF formula relative to time t
    log_one_plus_wT = np.logaddexp(0.0, log_wT)
    log_one_plus_wt = np.log1p(w_t)
    log_pi_T_minus_pi_t = (
        -params.rho * tau
        - params.gamma * log_DT_over_Dt
        + params.gamma * (log_one_plus_wT - log_one_plus_wt)
    )

    pi_ratio = np.exp(log_pi_T_minus_pi_t)
    return float(np.mean(pi_ratio))


def l1_forward_density_error_mc(
    t: float,
    T: float,
    w_t: float,
    t_grid: np.ndarray,
    eta_func,
    params: Parameters,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimate of the L¹ forward-density approximation error.

    Computes E^P[|Z_t^T − Ẑ_t^T|] where

        Z_t^T  = π_t P(t,T) / P(0,T)        (exact forward density)
        Ẑ_t^T = π_t P̂(t,T) / P̂(0,T)       (approximate forward density)

    evaluated at time t using n_paths Monte Carlo paths for the future
    (t to T) continuation.

    Parameters
    ----------
    t : float
        Current time t ≥ 0.
    T : float
        Maturity T > t.
    w_t : float
        Realised wealth ratio w_t (conditioning value).
    t_grid : np.ndarray
        Full time grid from 0 to some horizon ≥ T.
    eta_func : callable
        Belief-distortion function t ↦ η_t.
    params : Parameters
        Economy parameters.
    n_paths : int, optional
        Number of Monte Carlo paths (default 10 000).
    seed : int or None, optional
        RNG seed.

    Returns
    -------
    float
        Monte Carlo estimate of E^P[|Z_t^T − Ẑ_t^T|].
    """
    from .model import rational_yield

    dt = float(t_grid[1] - t_grid[0])
    tau = T - t
    if tau <= 0:
        raise ValueError("T must be strictly greater than t.")

    rR = rational_yield(params)
    gamma = params.gamma
    sigma = params.sigma

    # Simulate future paths
    t_fut, dB_fut, B_fut = simulate_brownian(tau, dt, n_paths, seed=seed)
    t_abs = t + t_fut[:-1]
    eta_fut = eta_func(t_abs)
    H_tT = float(np.sum(eta_fut * dt))
    A_tT = float(np.sum(eta_fut**2 * dt))

    drift_inc = (params.mu - 0.5 * sigma**2) * tau
    log_DT_over_Dt = drift_inc + sigma * B_fut[-1]            # (n_paths,)

    stoch_fut = np.cumsum(eta_fut[:, None] * dB_fut, axis=0)[-1]  # (n_paths,)
    log_xi_inc = sigma * stoch_fut - 0.5 * sigma**2 * A_tT

    log_wT = np.log(w_t) + (1.0 / gamma) * log_xi_inc
    log_one_plus_wT = np.logaddexp(0.0, log_wT)
    log_one_plus_wt = np.log1p(w_t)

    log_pi_T_minus_pi_t = (
        -params.rho * tau
        - gamma * log_DT_over_Dt
        + gamma * (log_one_plus_wT - log_one_plus_wt)
    )
    pi_ratio = np.exp(log_pi_T_minus_pi_t)        # (n_paths,), π_T/π_t per path

    # Exact bond price (MC estimate from these same paths)
    P_exact = float(np.mean(pi_ratio))

    # Approximate bond price
    q_approx = q_tT_general(w_t, gamma, sigma, H_tT)
    P_approx = bond_price_approx(rR, tau, w_t, gamma, q_approx)

    # Z_t^T per path: each path gives (π_T/π_t) / E[(π_T/π_t)] (normalised)
    # We compare on a per-path basis the contribution to the forward density
    Z_exact = pi_ratio / P_exact if P_exact > 0 else pi_ratio
    Z_approx_val = float(P_approx / P_exact) if P_exact > 0 else 1.0

    l1_error = float(np.mean(np.abs(Z_exact - Z_approx_val)))
    return l1_error
