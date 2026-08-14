"""Arrow-Debreu two-investor economy: parameters, equilibrium objects, and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class Parameters:
    """Economy parameters for the two-investor Arrow-Debreu model.

    Attributes
    ----------
    mu : float
        Drift of the log-endowment process under the rational measure P.
    sigma : float
        Volatility of the log-endowment process (σ > 0).
    rho : float
        Common time-discount rate (ρ > 0).
    gamma : float
        Common CRRA risk-aversion coefficient (γ > 0).
    D0 : float
        Initial endowment level (D_0 > 0).
    s0 : float
        Initial wealth share of the dogmatist investor (0 < s_0 < 1).
    w0 : float
        Dogmatist-to-rationalist initial wealth ratio w_0 = s_0 / (1 − s_0).
        Derived automatically from *s0* and not accepted as a constructor
        argument.
    """

    mu: float
    sigma: float
    rho: float
    gamma: float
    D0: float
    s0: float
    w0: float = field(init=False)

    def __post_init__(self) -> None:
        self.w0 = self.s0 / (1.0 - self.s0)


# ---------------------------------------------------------------------------
# Deterministic pricing primitives
# ---------------------------------------------------------------------------


def rational_yield(params: Parameters) -> float:
    """Equilibrium short rate in the pure-rationalist (homogeneous) economy.

    Under the rational measure P the instantaneous riskless rate is constant
    and given by:

        r_R = ρ + γ μ − ½ γ (γ + 1) σ²

    Parameters
    ----------
    params : Parameters
        Economy parameters.

    Returns
    -------
    float
        The rational yield r_R.
    """
    return (
        params.rho
        + params.gamma * params.mu
        - 0.5 * params.gamma * (params.gamma + 1.0) * params.sigma**2
    )


# ---------------------------------------------------------------------------
# Path-level simulation helpers
# ---------------------------------------------------------------------------


def simulate_brownian(
    T: float,
    dt: float,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate standard Brownian motion paths on a uniform time grid.

    Parameters
    ----------
    T : float
        Terminal time (T > 0).
    dt : float
        Uniform time-step size.
    n_paths : int
        Number of independent sample paths.
    seed : int or None, optional
        Random-number-generator seed for reproducibility.

    Returns
    -------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid ``[0, dt, 2·dt, …, T]``.
    dB : np.ndarray
        Shape ``(n_steps, n_paths)``.  IID increments B_{t+dt} − B_t drawn
        from N(0, dt).
    B_t : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Cumulative path with B_0 = 0.
    """
    rng = np.random.default_rng(seed)
    n_steps = round(T / dt)
    t_grid = np.linspace(0.0, T, n_steps + 1)
    dB = rng.normal(0.0, np.sqrt(dt), size=(n_steps, n_paths))
    B_t = np.vstack([np.zeros(n_paths), np.cumsum(dB, axis=0)])
    return t_grid, dB, B_t


def log_endowment(
    t_grid: np.ndarray,
    dB: np.ndarray,
    dt: float,
    params: Parameters,
) -> np.ndarray:
    """Simulate the normalised log-endowment log(D_t / D_0) on a time grid.

    The endowment follows

        d log D_t = (μ − ½ σ²) dt + σ dB_t

    so that log(D_t / D_0) = (μ − ½ σ²) t + σ B_t.

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid (see :func:`simulate_brownian`).
    dB : np.ndarray
        Shape ``(n_steps, n_paths)``.  Brownian increments.
    dt : float
        Time-step size (not used directly but kept for API consistency).
    params : Parameters
        Economy parameters.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log(D_t / D_0) for each path and
        time point, starting at 0 (t = 0).
    """
    n_paths = dB.shape[1]
    B_t = np.vstack([np.zeros(n_paths), np.cumsum(dB, axis=0)])
    drift = (params.mu - 0.5 * params.sigma**2) * t_grid[:, None]
    return drift + params.sigma * B_t


def log_xi(
    eta_vals: np.ndarray,
    dB: np.ndarray,
    dt: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """Compute the log-likelihood ratio log ξ_t = log(dQ/dP)|_{F_t}.

    The dogmatist's Radon-Nikodym derivative ξ_t is the Doléans-Dadé
    exponential of σ η_t dB_t:

        log ξ_t = σ ∫₀ᵗ η_s dB_s − ½ σ² ∫₀ᵗ η_s² ds
                = σ Σᵢ η_i ΔB_i − ½ σ² A_t

    where A_t = Σᵢ η_i² Δt is approximated by a left-endpoint Riemann sum.

    Parameters
    ----------
    eta_vals : np.ndarray
        Shape ``(n_steps,)``.  Belief-distortion path η evaluated at the left
        endpoint of each time interval (i.e. at ``t_grid[:-1]``).
    dB : np.ndarray
        Shape ``(n_steps, n_paths)``.  Brownian increments.
    dt : float
        Uniform time-step size (used for the Riemann sum of A_t).
    sigma : float
        Endowment volatility σ.
    gamma : float
        CRRA coefficient.  Included for API consistency; log ξ does not
        depend on γ.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log ξ_t for each path and time
        point.  Row 0 is identically 0 (log ξ_0 = 0).
    """
    n_paths = dB.shape[1]
    stochastic = np.cumsum(eta_vals[:, None] * dB, axis=0)   # (n_steps, n_paths)
    A_t = np.cumsum(eta_vals**2 * dt)                        # (n_steps,)
    log_xi_t = sigma * stochastic - 0.5 * sigma**2 * A_t[:, None]
    return np.vstack([np.zeros(n_paths), log_xi_t])


def log_w(
    log_xi_arr: np.ndarray,
    w0: float,
    gamma: float,
) -> np.ndarray:
    """Compute the log dogmatist-to-rationalist wealth ratio log w_t.

    In equilibrium the individual consumption allocations satisfy

        w_t = c_{N,t} / c_{R,t} = w_0 · ξ_t^{1/γ}

    so that log w_t = log w_0 + (1/γ) log ξ_t.

    Parameters
    ----------
    log_xi_arr : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Output of :func:`log_xi`.
    w0 : float
        Initial wealth ratio w_0 = s_0 / (1 − s_0).
    gamma : float
        CRRA risk-aversion coefficient γ.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log w_t for each path.
    """
    return np.log(w0) + (1.0 / gamma) * log_xi_arr


def consumption_shares(
    log_w_arr: np.ndarray,
    log_D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute individual consumption levels (in units of D_0).

    Market clearing gives

        c_{R,t} = D_t / (1 + w_t),    c_{N,t} = w_t · D_t / (1 + w_t)

    where D_t / D_0 = exp(log_D).

    Parameters
    ----------
    log_w_arr : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Log wealth ratio log w_t.
    log_D : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Normalised log-endowment
        log(D_t / D_0) (output of :func:`log_endowment`).

    Returns
    -------
    c_R : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Rationalist consumption
        D_t / (D_0 (1 + w_t)).
    c_N : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Dogmatist consumption
        w_t · D_t / (D_0 (1 + w_t)).
    """
    log_D_minus_log1pw = log_D - np.logaddexp(0.0, log_w_arr)
    c_R = np.exp(log_D_minus_log1pw)
    c_N = np.exp(log_w_arr + log_D_minus_log1pw)
    return c_R, c_N


def log_sdf(
    t_grid: np.ndarray,
    log_D: np.ndarray,
    log_w_arr: np.ndarray,
    params: Parameters,
) -> np.ndarray:
    """Compute the log state-price density (SDF) log π_t.

    The equilibrium SDF from the rationalist's first-order condition is

        log π_t = −ρ t − γ log(D_t/D_0) + γ log((1 + w_t)/(1 + w_0))

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid.
    log_D : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log(D_t / D_0).
    log_w_arr : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log w_t.
    params : Parameters
        Economy parameters.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log π_t.  Row 0 is identically 0
        (π_0 = 1 by normalisation).
    """
    log_one_plus_wt = np.logaddexp(0.0, log_w_arr)      # log(1 + w_t)
    log_one_plus_w0 = np.log1p(params.w0)               # log(1 + w_0), scalar
    return (
        -params.rho * t_grid[:, None]
        - params.gamma * log_D
        + params.gamma * (log_one_plus_wt - log_one_plus_w0)
    )


def dogmatist_share(log_w_arr: np.ndarray) -> np.ndarray:
    """Compute the dogmatist's consumption (wealth) share s_t = w_t / (1 + w_t).

    Equivalently s_t = σ(log w_t) where σ is the logistic sigmoid.  The
    numerically stable form avoids overflow for large |log w_t|:

        s_t = 1 / (1 + exp(−log w_t))

    Parameters
    ----------
    log_w_arr : np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log w_t.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  Dogmatist share s_t ∈ (0, 1).
    """
    return expit(log_w_arr)


# ---------------------------------------------------------------------------
# Benchmark martingales
# ---------------------------------------------------------------------------


def benchmark_martingale_R(
    t_grid: np.ndarray,
    dB: np.ndarray,
    dt: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """Compute log M_t^R, the martingale factor in the rationalist's SDF.

    In the pure-rationalist economy the SDF decomposes as

        π_t^R = exp(−r_R t) · M_t^R

    where

        log M_t^R = −γ σ B_t − ½ γ² σ² t

    M^R is the Doléans-Dadé exponential of −γ σ dB_t.

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid.
    dB : np.ndarray
        Shape ``(n_steps, n_paths)``.  Brownian increments.
    dt : float
        Time-step size (kept for API consistency).
    sigma : float
        Endowment volatility σ.
    gamma : float
        CRRA coefficient γ.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^R.  Row 0 is 0.
    """
    n_paths = dB.shape[1]
    B_t = np.vstack([np.zeros(n_paths), np.cumsum(dB, axis=0)])
    return -gamma * sigma * B_t - 0.5 * gamma**2 * sigma**2 * t_grid[:, None]


def benchmark_martingale_N(
    t_grid: np.ndarray,
    eta_vals: np.ndarray,
    dB: np.ndarray,
    dt: float,
    sigma: float,
    gamma: float,
) -> np.ndarray:
    """Compute log M_t^N, the martingale factor in the dogmatist's SDF.

    In the pure-dogmatist economy the SDF decomposes as

        π_t^N = exp(−r_R t) · M_t^N · Φ_N(0) / Φ_N(t)

    where M_t^N is the Doléans-Dadé exponential of σ (η_t − γ) dB_t:

        log M_t^N = σ Σᵢ (η_i − γ) ΔB_i − ½ σ² Σᵢ (η_i − γ)² Δt

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid.
    eta_vals : np.ndarray
        Shape ``(n_steps,)``.  Belief-distortion path η at left endpoints.
    dB : np.ndarray
        Shape ``(n_steps, n_paths)``.  Brownian increments.
    dt : float
        Uniform time-step size.
    sigma : float
        Endowment volatility σ.
    gamma : float
        CRRA coefficient γ.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1, n_paths)``.  log M_t^N.  Row 0 is 0.
    """
    n_paths = dB.shape[1]
    eta_m_gamma = eta_vals - gamma                                   # (n_steps,)
    stochastic = np.cumsum(eta_m_gamma[:, None] * dB, axis=0)       # (n_steps, n_paths)
    quad = np.cumsum(eta_m_gamma**2 * dt)                           # (n_steps,)
    log_MN_t = sigma * stochastic - 0.5 * sigma**2 * quad[:, None]
    return np.vstack([np.zeros(n_paths), log_MN_t])
