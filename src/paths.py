"""Deterministic belief-distortion path specifications and their path integrals.

Each η-path is a deterministic function t ↦ η_t that encodes the dogmatist's
belief distortion relative to the rational measure.  The two key derived
quantities are:

    A_t = ∫₀ᵗ η_s² ds   (quadratic-variation integral)
    H_t = ∫₀ᵗ η_s  ds   (level integral / cumulative drift)

Exact closed forms are provided where available; numerical trapezoid
approximations are always available as a fallback.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid


# ---------------------------------------------------------------------------
# η-path functions
# ---------------------------------------------------------------------------


def eta_fading_pessimism(t: float | np.ndarray, eta0: float) -> float | np.ndarray:
    """Fading-pessimism belief path η_t = η₀ / √(1 + t).

    The distortion starts at η₀ and decays algebraically to zero.  For η₀ < 0
    this represents persistent but fading pessimism; for η₀ > 0, optimism.

    Parameters
    ----------
    t : float or np.ndarray
        Time(s) at which to evaluate the path (t ≥ 0).
    eta0 : float
        Initial belief distortion η₀.

    Returns
    -------
    float or np.ndarray
        η_t = η₀ / √(1 + t).
    """
    return eta0 / np.sqrt(1.0 + t)


def eta_fading_alternating(t: float | np.ndarray, omega: float) -> float | np.ndarray:
    """Fading-alternating belief path η_t = 2 sin(ω t) / √(1 + t).

    The distortion oscillates with frequency ω and decaying amplitude.  The
    sign of η_t alternates, producing episodes of optimism and pessimism.

    Parameters
    ----------
    t : float or np.ndarray
        Time(s) at which to evaluate the path (t ≥ 0).
    omega : float
        Angular frequency ω > 0.

    Returns
    -------
    float or np.ndarray
        η_t = 2 sin(ω t) / √(1 + t).
    """
    return 2.0 * np.sin(omega * t) / np.sqrt(1.0 + t)


def eta_constant_cyclical(t: float | np.ndarray, omega: float) -> float | np.ndarray:
    """Constant-amplitude cyclical belief path η_t = 2 cos(ω t).

    The distortion oscillates with fixed amplitude 2 and frequency ω.  Unlike
    the fading paths the quadratic variation A_t grows without bound.

    Parameters
    ----------
    t : float or np.ndarray
        Time(s) at which to evaluate the path (t ≥ 0).
    omega : float
        Angular frequency ω > 0.

    Returns
    -------
    float or np.ndarray
        η_t = 2 cos(ω t).
    """
    return 2.0 * np.cos(omega * t)


# ---------------------------------------------------------------------------
# Exact analytic path integrals
# ---------------------------------------------------------------------------


def A_fading_pessimism_exact(t: float | np.ndarray, eta0: float) -> float | np.ndarray:
    """Exact quadratic-variation integral for the fading-pessimism path.

    With η_t = η₀ / √(1 + t),

        A_t = ∫₀ᵗ η_s² ds = η₀² ∫₀ᵗ 1/(1+s) ds = η₀² log(1 + t).

    Parameters
    ----------
    t : float or np.ndarray
        Upper integration limit (t ≥ 0).
    eta0 : float
        Initial belief distortion η₀.

    Returns
    -------
    float or np.ndarray
        A_t = η₀² log(1 + t).
    """
    return eta0**2 * np.log1p(t)


def H_fading_pessimism_exact(t: float | np.ndarray, eta0: float) -> float | np.ndarray:
    """Exact level integral for the fading-pessimism path.

    With η_t = η₀ / √(1 + t),

        H_t = ∫₀ᵗ η_s ds = η₀ ∫₀ᵗ (1+s)^{−1/2} ds = 2 η₀ (√(1+t) − 1).

    Parameters
    ----------
    t : float or np.ndarray
        Upper integration limit (t ≥ 0).
    eta0 : float
        Initial belief distortion η₀.

    Returns
    -------
    float or np.ndarray
        H_t = 2 η₀ (√(1 + t) − 1).
    """
    return 2.0 * eta0 * (np.sqrt(1.0 + t) - 1.0)


def H_constant_cyclical_exact(t: float | np.ndarray, omega: float) -> float | np.ndarray:
    """Exact level integral for the constant-cyclical path.

    With η_t = 2 cos(ω t),

        H_t = ∫₀ᵗ 2 cos(ω s) ds = (2/ω) sin(ω t).

    Parameters
    ----------
    t : float or np.ndarray
        Upper integration limit (t ≥ 0).
    omega : float
        Angular frequency ω > 0.

    Returns
    -------
    float or np.ndarray
        H_t = (2/ω) sin(ω t).
    """
    return (2.0 / omega) * np.sin(omega * t)


# ---------------------------------------------------------------------------
# Numerical path integrals (cumulative trapezoid)
# ---------------------------------------------------------------------------


def A_numerical(
    t_grid: np.ndarray,
    eta_vals: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Numerically compute A_t = ∫₀ᵗ η_s² ds via the cumulative trapezoid rule.

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Uniform time grid from 0 to T.
    eta_vals : np.ndarray
        Shape ``(n_steps + 1,)``.  Belief-distortion values η_{t_i} on the
        full time grid.
    dt : float
        Uniform time-step size Δt (used as the grid spacing).

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1,)``.  Cumulative integral A_t starting at A_0 = 0.
    """
    return cumulative_trapezoid(eta_vals**2, dx=dt, initial=0.0)


def H_numerical(
    t_grid: np.ndarray,
    eta_vals: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Numerically compute H_t = ∫₀ᵗ η_s ds via the cumulative trapezoid rule.

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Uniform time grid from 0 to T.
    eta_vals : np.ndarray
        Shape ``(n_steps + 1,)``.  Belief-distortion values η_{t_i} on the
        full time grid.
    dt : float
        Uniform time-step size Δt.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1,)``.  Cumulative integral H_t starting at H_0 = 0.
    """
    return cumulative_trapezoid(eta_vals, dx=dt, initial=0.0)


def A_left_riemann(eta_vals_left: np.ndarray, dt: float) -> np.ndarray:
    """Compute A_t = ∫₀ᵗ η_s² ds via the left-endpoint Riemann sum.

    Uses the same discretisation as :func:`src.model.log_xi` and
    :func:`src.model.benchmark_martingale_N`, so the result is exactly
    consistent with those functions.  Pass the output to
    :func:`src.factorization.identity_check_MR_wgamma` to obtain
    machine-precision residuals.

    Parameters
    ----------
    eta_vals_left : np.ndarray
        Shape ``(n_steps,)``.  Belief-distortion values η at the *left*
        endpoint of each interval (i.e. ``t_grid[:-1]``).
    dt : float
        Uniform time-step size Δt.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1,)``.  A_t starting at A_0 = 0.
    """
    return np.concatenate([[0.0], np.cumsum(eta_vals_left**2 * dt)])


def H_left_riemann(eta_vals_left: np.ndarray, dt: float) -> np.ndarray:
    """Compute H_t = ∫₀ᵗ η_s ds via the left-endpoint Riemann sum.

    Uses the same discretisation as :func:`src.model.log_xi` and
    :func:`src.model.benchmark_martingale_N`, so the result is exactly
    consistent with those functions.  Pass the output to
    :func:`src.factorization.identity_check_MR_wgamma` to obtain
    machine-precision residuals.

    Parameters
    ----------
    eta_vals_left : np.ndarray
        Shape ``(n_steps,)``.  Belief-distortion values η at the *left*
        endpoint of each interval (i.e. ``t_grid[:-1]``).
    dt : float
        Uniform time-step size Δt.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1,)``.  H_t starting at H_0 = 0.
    """
    return np.concatenate([[0.0], np.cumsum(eta_vals_left * dt)])


# ---------------------------------------------------------------------------
# Composite path-object builder
# ---------------------------------------------------------------------------


def compute_path_objects(
    t_grid: np.ndarray,
    eta_func: Callable[[np.ndarray], np.ndarray],
    dt: float,
    method: str = "numerical",
    **kwargs,
) -> dict[str, np.ndarray]:
    """Evaluate a belief-distortion path and its derived integrals on a grid.

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid from 0 to T.
    eta_func : callable
        Function ``t ↦ η_t`` accepting a scalar or array of times and
        returning the corresponding belief distortions.
    dt : float
        Uniform time-step size Δt.
    method : {'numerical', 'exact'}
        How to compute A_t and H_t.

        * ``'numerical'``: cumulative trapezoid of η² and η (always valid).
        * ``'exact'``: the callable must accept ``t_grid`` and the additional
          keyword arguments ``A_func`` and ``H_func`` (each a callable
          ``t ↦ scalar``).  Use this for closed-form paths to avoid
          discretisation error.
    **kwargs
        Additional keyword arguments forwarded to the exact integral
        functions when ``method='exact'``.  Expected keys:

        * ``A_func`` – callable computing A_t analytically.
        * ``H_func`` – callable computing H_t analytically.

    Returns
    -------
    dict
        Keys:

        * ``'eta'`` – np.ndarray of shape ``(n_steps + 1,)``.  Belief-
          distortion values η_{t_i}.
        * ``'A'``   – np.ndarray of shape ``(n_steps + 1,)``.  Quadratic-
          variation integral A_t.
        * ``'H'``   – np.ndarray of shape ``(n_steps + 1,)``.  Level
          integral H_t.
    """
    eta_vals = eta_func(t_grid)

    if method == "exact":
        A_func: Callable[[np.ndarray], np.ndarray] = kwargs["A_func"]
        H_func: Callable[[np.ndarray], np.ndarray] = kwargs["H_func"]
        A_vals = A_func(t_grid)
        H_vals = H_func(t_grid)
    else:
        A_vals = A_numerical(t_grid, eta_vals, dt)
        H_vals = H_numerical(t_grid, eta_vals, dt)

    return {"eta": eta_vals, "A": A_vals, "H": H_vals}
