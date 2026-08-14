"""Long-run yield and forward-rate analysis (Experiment 4).

Studies the asymptotic behaviour of the term structure as maturity T → ∞.
Key results depend on the limiting behaviour of

    A_T = ∫₀ᵀ η_s² ds   and   H_T = ∫₀ᵀ η_s ds.

When H̄ = lim_{T→∞} H_T / T exists the long bond yield converges to

    r_∞ = r_R + γ σ² min(H̄, 0).

The module also provides the forward-density process

    Z_t^T = π_t · P(t,T) / P(0,T)

whose limit as T → ∞ (when it exists) characterises the long-run pricing
measure (Hansen-Scheinkman recovered measure).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Term-structure helpers
# ---------------------------------------------------------------------------


def theta_T_series(
    w0: float,
    H_T_series: np.ndarray,
    gamma: float,
    sigma: float,
) -> np.ndarray:
    """Compute the dogmatist's pricing share θ_T for a series of maturities.

    For general γ,

        q_{0,T} = w_0^γ exp(−γ σ² H_T),    θ_T = q_{0,T} / (1 + q_{0,T}).

    Parameters
    ----------
    w0 : float
        Initial wealth ratio w_0.
    H_T_series : np.ndarray
        Shape ``(n_maturities,)``.  Level integral H_T evaluated at each
        maturity.
    gamma : float
        CRRA coefficient γ.
    sigma : float
        Endowment volatility σ.

    Returns
    -------
    np.ndarray
        Shape ``(n_maturities,)``.  Dogmatist pricing share θ_T ∈ (0, 1).
    """
    q = w0**gamma * np.exp(-gamma * sigma**2 * H_T_series)
    return q / (1.0 + q)


def long_yield_formula(
    rR: float,
    gamma: float,
    sigma: float,
    H_bar: float,
) -> float:
    """Long-bond yield r_∞ when the ergodic limit H̄ = lim H_T / T exists.

        r_∞ = r_R + γ σ² min(H̄, 0)

    For η-paths that are on average optimistic (H̄ > 0) the long yield
    equals r_R; for pessimistic paths (H̄ < 0) it is strictly below r_R.

    Parameters
    ----------
    rR : float
        Rational yield r_R.
    gamma : float
        CRRA coefficient γ.
    sigma : float
        Endowment volatility σ.
    H_bar : float
        Limit H̄ = lim_{T→∞} H_T / T (or best available estimate).

    Returns
    -------
    float
        Long-bond yield r_∞.
    """
    return rR + gamma * sigma**2 * min(H_bar, 0.0)


def classify_path(
    A_T_large: np.ndarray,
    H_T_large: np.ndarray,
    tol: float = 1e-3,
) -> dict:
    """Classify an η-path based on the long-run behaviour of A_T and H_T.

    Uses the last few values of A_T and H_T at large maturities to infer
    asymptotic behaviour.

    Parameters
    ----------
    A_T_large : np.ndarray
        Shape ``(m,)``.  Values of A_T at a sequence of large maturities
        T_1 < T_2 < … < T_m.
    H_T_large : np.ndarray
        Shape ``(m,)``.  Values of H_T at the same maturities.
    tol : float, optional
        Tolerance for classifying quantities as zero (default 1e-3).

    Returns
    -------
    dict
        Keys:

        * ``'A_diverges'``          – bool. True if A_T appears to diverge.
        * ``'H_limit'``             – float or None. Estimated lim H_T if
          H_T appears to converge; None otherwise.
        * ``'H_over_T_limit'``      – float or None. Estimated lim H_T / T
          (the time-average H̄).
        * ``'long_forward_exists'`` – bool. True when A_T diverges (a
          sufficient condition for the long forward rate to exist and equal
          r_∞).
        * ``'long_bond_exists'``    – bool. True when H_T / T converges to
          a finite limit H̄ (long-bond yield = r_∞).
    """
    # Increments of A_T at the tail
    dA = np.diff(A_T_large[-min(10, len(A_T_large)):])
    A_diverges = bool(np.mean(dA) > tol)

    # Estimate H_T convergence: check whether the last values are nearly
    # constant (absolute convergence) or growing linearly (rate convergence).
    H_tail = H_T_large[-min(10, len(H_T_large)):]
    dH = np.diff(H_tail)
    H_converged = bool(np.max(np.abs(dH)) < tol)
    H_limit = float(H_tail[-1]) if H_converged else None

    # Estimate H_T / T at the last point
    # Approximate by fitting a line to the last 20% of the series
    n = len(H_T_large)
    tail_start = max(0, int(0.8 * n))
    H_tail_full = H_T_large[tail_start:]
    # Construct matching T values (we do not have them directly, so we use
    # relative indices normalised by their own spacing)
    T_indices = np.arange(len(H_tail_full), dtype=float) + (tail_start + 1)
    if len(T_indices) >= 2:
        slope = np.polyfit(T_indices, H_tail_full, 1)[0]
        H_over_T_limit: float | None = float(slope)
    else:
        H_over_T_limit = None

    long_forward_exists = A_diverges
    long_bond_exists = (H_over_T_limit is not None) and (
        np.isfinite(H_over_T_limit)
    )

    return {
        "A_diverges": A_diverges,
        "H_limit": H_limit,
        "H_over_T_limit": H_over_T_limit,
        "long_forward_exists": long_forward_exists,
        "long_bond_exists": long_bond_exists,
    }


# ---------------------------------------------------------------------------
# Limit martingale and normalised bond price
# ---------------------------------------------------------------------------


def mixed_limit_martingale(
    theta_h: float,
    log_MR: np.ndarray,
    log_MN: np.ndarray,
) -> np.ndarray:
    """Compute the log of the (θ_h)-mixed limit martingale.

    Defined as the convex combination

        log[(1 − θ_h) M_t^R + θ_h M_t^N]

    evaluated in a numerically stable way via log-sum-exp.

    Parameters
    ----------
    theta_h : float
        Long-run dogmatist pricing weight θ_h ∈ (0, 1).
    log_MR : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^R.
    log_MN : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^N.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log[(1 − θ_h) M^R + θ_h M^N].
    """
    log_w_R = np.log(1.0 - theta_h)
    log_w_N = np.log(theta_h)
    # log-sum-exp of two terms along a new axis
    arr = np.stack([log_w_R + log_MR, log_w_N + log_MN], axis=0)
    return logsumexp(arr, axis=0)


def normalized_bond_price(
    t_idx: int,
    T_maturities: np.ndarray,
    t_grid: np.ndarray,
    log_pi: np.ndarray,
    log_pi_t: np.ndarray,
    log_w: np.ndarray,
    w0: float,
    gamma: float,
    sigma: float,
    H_t_to_T: np.ndarray,
    A_t_to_T: np.ndarray,
) -> np.ndarray:
    """Compute the normalised bond price P(t, T) / P(0, T) for γ = 1.

    Uses the closed-form γ = 1 bond prices

        P(0, T) = exp(−r_R T) (1 + q_{0,T}) / (1 + w_0)
        P(t, T) = exp(−r_R (T−t)) (1 + q_{t,T}) / (1 + w_t)

    where q_{t,T} = w_t exp(−σ² H_{t,T}) and w_t = exp(log_w[t_idx]).

    Parameters
    ----------
    t_idx : int
        Index of the current time t in ``t_grid``.
    T_maturities : np.ndarray
        Shape ``(n_T,)``.  Maturities T > t.
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Full time grid.
    log_pi : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Full log π path (unused for the
        closed-form γ = 1 formula but kept for API consistency).
    log_pi_t : np.ndarray
        Shape ``(n_paths,)``.  log π_t at the current time (slice of log_pi).
    log_w : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Full log w path.
    w0 : float
        Initial wealth ratio w_0.
    gamma : float
        CRRA coefficient (must be 1 for the closed form).
    sigma : float
        Endowment volatility σ.
    H_t_to_T : np.ndarray
        Shape ``(n_T,)``.  H_{t,T} = ∫_t^T η_s ds for each maturity T.
    A_t_to_T : np.ndarray
        Shape ``(n_T,)``.  A_{t,T} = ∫_t^T η_s² ds (included for future
        extension; not used in the γ = 1 formula).

    Returns
    -------
    np.ndarray
        Shape ``(n_T, n_paths)``.  P(t, T) / P(0, T) for each maturity and
        path.
    """
    t = t_grid[t_idx]
    w_t = np.exp(log_w[t_idx])                   # (n_paths,)

    # q_{0,T} for each maturity (uses full H from 0 to T)
    # We approximate H_{0,T} = H_{0,t} + H_{t,T}
    H_0_t = float(np.mean(log_w[t_idx] - np.log(w0)))  # rough: use exact if available
    # For the ratio we only need (1+q_{t,T})/(1+q_{0,T})*(1+w_0)/(1+w_t)
    # Use the provided H_t_to_T directly
    q_tT = w_t[None, :] * np.exp(-sigma**2 * H_t_to_T[:, None])  # (n_T, n_paths)
    q_0T = w0 * np.exp(-sigma**2 * (H_0_t + H_t_to_T[:, None]))  # (n_T, n_paths)

    P_tT = (1.0 + q_tT) / (1.0 + w_t[None, :])          # (n_T, n_paths)
    P_0T = (1.0 + q_0T) / (1.0 + w0)                    # (n_T, n_paths)

    return P_tT / P_0T


def forward_density_process(
    t_idx: int,
    T_idx: int,
    t_grid: np.ndarray,
    log_pi: np.ndarray,
    P_0T: np.ndarray,
    P_t_T: np.ndarray,
) -> np.ndarray:
    """Compute the forward-density process Z_t^T = π_t · P(t,T) / P(0,T).

    Z^T is a P-martingale that represents the change of measure from P to
    the T-forward measure Q^T.  Its limit as T → ∞ (when it exists) is the
    long-run pricing kernel.

    Parameters
    ----------
    t_idx : int
        Index of the evaluation time t in ``t_grid``.
    T_idx : int
        Index of the maturity T in ``t_grid`` (T > t).
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid.
    log_pi : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log π path.
    P_0T : np.ndarray
        Shape ``(n_paths,)`` or scalar.  P(0, T).
    P_t_T : np.ndarray
        Shape ``(n_paths,)``.  P(t, T) for each path.

    Returns
    -------
    np.ndarray
        Shape ``(n_paths,)``.  Z_t^T = π_t · P(t, T) / P(0, T).
    """
    pi_t = np.exp(log_pi[t_idx])      # (n_paths,)
    return pi_t * P_t_T / P_0T
