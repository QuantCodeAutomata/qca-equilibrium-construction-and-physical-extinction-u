"""Exact closed-form bond-pricing formulas for the γ = 1 (log-utility) case.

When both investors have logarithmic utility (γ = 1) the equilibrium SDF and
bond prices admit explicit closed forms because the dogmatist's wealth ratio
w_t = w_0 ξ_t is a martingale under P.

Zero-coupon bond price (maturity T, current time 0):

    P(0, T) = exp(−r_R T) · (1 + q_{0,T}) / (1 + w_0)

where

    q_{0,T} = w_0 exp(−G_{0,T}),    G_{0,T} = σ² H_{0,T} = σ² ∫₀ᵀ η_s ds.

See the companion paper, Section 4, for derivations.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .model import Parameters
from .paths import H_fading_pessimism_exact, H_constant_cyclical_exact, H_numerical


# ---------------------------------------------------------------------------
# Core γ = 1 pricing objects
# ---------------------------------------------------------------------------


def q_0T(w0: float, G_0T: float | np.ndarray) -> float | np.ndarray:
    """Dogmatist's pricing-measure weight at maturity T.

    Defined as

        q_{0,T} = w_0 exp(−G_{0,T})

    where G_{0,T} = σ² H_{0,T} summarises the cumulative effect of the
    belief distortion on the term structure.

    Parameters
    ----------
    w0 : float
        Initial wealth ratio w_0 = s_0 / (1 − s_0).
    G_0T : float or np.ndarray
        Integrated belief-drift G_{0,T} = σ² ∫₀ᵀ η_s ds.

    Returns
    -------
    float or np.ndarray
        q_{0,T} = w_0 exp(−G_{0,T}).
    """
    return w0 * np.exp(-G_0T)


def theta_0T(q: float | np.ndarray) -> float | np.ndarray:
    """Dogmatist's forward-pricing share θ_{0,T} = q / (1 + q).

    Parameters
    ----------
    q : float or np.ndarray
        The quantity q_{0,T} from :func:`q_0T`.

    Returns
    -------
    float or np.ndarray
        θ_{0,T} = q / (1 + q) ∈ (0, 1).
    """
    return q / (1.0 + q)


def bond_price_gamma1(
    T: float | np.ndarray,
    rR: float,
    w0: float,
    G_0T: float | np.ndarray,
) -> float | np.ndarray:
    """Zero-coupon bond price P(0, T) for γ = 1.

        P(0, T) = exp(−r_R T) · (1 + q_{0,T}) / (1 + w_0)

    Parameters
    ----------
    T : float or np.ndarray
        Maturity (T > 0).
    rR : float
        Rational yield r_R = ρ + μ − σ².
    w0 : float
        Initial wealth ratio w_0.
    G_0T : float or np.ndarray
        Integrated belief-drift G_{0,T}.

    Returns
    -------
    float or np.ndarray
        P(0, T).
    """
    return np.exp(-rR * T) * (1.0 + q_0T(w0, G_0T)) / (1.0 + w0)


def yield_spread_gamma1(
    T: float | np.ndarray,
    w0: float,
    G_0T: float | np.ndarray,
) -> float | np.ndarray:
    """Yield spread y(0, T) − r_R for a zero-coupon bond with γ = 1.

    The continuously compounded yield is −log P(0,T)/T, so the spread
    relative to the rational yield r_R is

        y(0,T) − r_R = [log(1 + w_0) − log(1 + q_{0,T})] / T.

    A positive spread arises when q_{0,T} < w_0, i.e. G_{0,T} > 0
    (average optimism); a negative spread when G_{0,T} < 0 (pessimism).

    Parameters
    ----------
    T : float or np.ndarray
        Maturity (T > 0).
    w0 : float
        Initial wealth ratio w_0.
    G_0T : float or np.ndarray
        Integrated belief-drift G_{0,T}.

    Returns
    -------
    float or np.ndarray
        Yield spread (positive = above rational, negative = below).
    """
    return (np.log1p(w0) - np.log1p(q_0T(w0, G_0T))) / T


def forward_rate_spread_gamma1(
    t: float | np.ndarray,
    eta_t: float | np.ndarray,
    sigma: float,
    theta_tT: float | np.ndarray,
) -> float | np.ndarray:
    """Instantaneous forward-rate spread δ_t · θ_{t,T} for γ = 1.

    The forward rate at time t for delivery at T satisfies

        f(t, T) − r_R = σ² η_T · θ_{t,T}

    where δ_T = σ² η_T is the instantaneous belief-drift contribution and
    θ_{t,T} = q_{t,T} / (1 + q_{t,T}) is the dogmatist's pricing share.

    Parameters
    ----------
    t : float or np.ndarray
        Observation time (for API completeness; not used in the formula).
    eta_t : float or np.ndarray
        Belief distortion η evaluated at maturity T.
    sigma : float
        Endowment volatility σ.
    theta_tT : float or np.ndarray
        Dogmatist's pricing share θ_{t,T}.

    Returns
    -------
    float or np.ndarray
        Forward-rate spread σ² η_T · θ_{t,T}.
    """
    return sigma**2 * eta_t * theta_tT


def bond_vol_gamma1(
    eta0: float,
    sigma: float,
    theta_0T: float | np.ndarray,
    s0: float,
) -> float | np.ndarray:
    """Bond return volatility for γ = 1 (instantaneous).

    The return volatility of the bond P(0, T) with respect to the Brownian
    shock at time 0 is

        vol(0, T) = |η₀ σ (θ_{0,T} − s_0)|

    where s_0 = w_0 / (1 + w_0) is the dogmatist's initial wealth share.

    Parameters
    ----------
    eta0 : float
        Initial belief distortion η_0.
    sigma : float
        Endowment volatility σ.
    theta_0T : float or np.ndarray
        Dogmatist's pricing share θ_{0,T} = q_{0,T} / (1 + q_{0,T}).
    s0 : float
        Dogmatist's initial wealth share s_0.

    Returns
    -------
    float or np.ndarray
        Bond return volatility |η₀ σ (θ_{0,T} − s_0)|.
    """
    return np.abs(eta0 * sigma * (theta_0T - s0))


def forward_growth_shift_gamma1(
    eta0: float,
    sigma: float,
    theta_0T: float | np.ndarray,
) -> float | np.ndarray:
    """Forward expected-growth shift due to heterogeneous beliefs (γ = 1).

    The contribution of the dogmatist's initial belief to the forward
    risk-adjusted expected growth rate is

        Δ_growth(0, T) = σ² η₀ θ_{0,T}.

    Parameters
    ----------
    eta0 : float
        Initial belief distortion η₀.
    sigma : float
        Endowment volatility σ.
    theta_0T : float or np.ndarray
        Dogmatist's pricing share θ_{0,T}.

    Returns
    -------
    float or np.ndarray
        σ² η₀ θ_{0,T}.
    """
    return sigma**2 * eta0 * theta_0T


# ---------------------------------------------------------------------------
# G_{0,T} constructors for specific η-paths
# ---------------------------------------------------------------------------


def G_0T_fading_pessimism(
    T: float | np.ndarray,
    eta0: float,
    sigma: float,
) -> float | np.ndarray:
    """G_{0,T} = σ² H_{0,T} for the fading-pessimism path η_t = η₀/√(1+t).

    Uses the exact formula H_{0,T} = 2 η₀ (√(1+T) − 1).

    Parameters
    ----------
    T : float or np.ndarray
        Maturity.
    eta0 : float
        Initial belief distortion η₀.
    sigma : float
        Endowment volatility σ.

    Returns
    -------
    float or np.ndarray
        G_{0,T} = σ² · 2 η₀ (√(1+T) − 1).
    """
    return sigma**2 * H_fading_pessimism_exact(T, eta0)


def G_0T_constant_cyclical(
    T: float | np.ndarray,
    omega: float,
    sigma: float,
) -> float | np.ndarray:
    """G_{0,T} = σ² H_{0,T} for the constant-cyclical path η_t = 2 cos(ω t).

    Uses the exact formula H_{0,T} = (2/ω) sin(ω T).

    Parameters
    ----------
    T : float or np.ndarray
        Maturity.
    omega : float
        Angular frequency ω > 0.
    sigma : float
        Endowment volatility σ.

    Returns
    -------
    float or np.ndarray
        G_{0,T} = σ² (2/ω) sin(ω T).
    """
    return sigma**2 * H_constant_cyclical_exact(T, omega)


def G_0T_numerical(
    t_grid: np.ndarray,
    eta_vals: np.ndarray,
    sigma: float,
    dt: float,
) -> np.ndarray:
    """G_{0,T} = σ² H_{0,T} computed via the cumulative trapezoid rule.

    Parameters
    ----------
    t_grid : np.ndarray
        Shape ``(n_steps + 1,)``.  Time grid from 0 to T.
    eta_vals : np.ndarray
        Shape ``(n_steps + 1,)``.  Belief-distortion values on the time grid.
    sigma : float
        Endowment volatility σ.
    dt : float
        Uniform time-step size Δt.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1,)``.  G_{0,t} = σ² H_{0,t} for each grid point.
    """
    return sigma**2 * H_numerical(t_grid, eta_vals, dt)


# ---------------------------------------------------------------------------
# Bond price premium and long-maturity objects
# ---------------------------------------------------------------------------


def bond_price_premium(
    T: float | np.ndarray,
    rR: float,
    w0: float,
    G_0T: float | np.ndarray,
) -> float | np.ndarray:
    """Bond price premium over the rational bond price for γ = 1.

    The premium is defined as

        premium(0, T) = P(0, T) / exp(−r_R T) − 1
                      = (1 + q_{0,T}) / (1 + w_0) − 1.

    A positive premium means the bond trades above the rational price (the
    dogmatist is willing to pay extra for safe claims).

    Parameters
    ----------
    T : float or np.ndarray
        Maturity.
    rR : float
        Rational yield r_R.
    w0 : float
        Initial wealth ratio w_0.
    G_0T : float or np.ndarray
        Integrated belief-drift G_{0,T}.

    Returns
    -------
    float or np.ndarray
        Bond price premium (1 + q_{0,T}) / (1 + w_0) − 1.
    """
    return (1.0 + q_0T(w0, G_0T)) / (1.0 + w0) - 1.0


def future_consumption_value_N(
    T_start: float,
    params: Parameters,
    upper: float = 2000.0,
    n: int = 50_000,
) -> float:
    """Dogmatist's share of total wealth at time T_start, for γ = 1.

    Computes

        W_N(T_start) / (W_R + W_N)

    where W_i(T_start) = E^P[∫_{T_start}^{∞} π_t c_{i,t} dt] is the
    continuation wealth of investor i.

    For γ = 1 the expectations simplify deterministically.  Since ξ_t is a
    P-martingale, E^P[w_t] = w_0 for all t, giving

        E^P[π_t c_{N,t}] = exp(−ρ t) D_0 · w_0 / (1 + w_0)
        E^P[π_t D_t]     = exp(−ρ t) D_0

    The ratio is therefore

        W_N / (W_R + W_N) = [w_0 / (1+w_0)] · ∫_{T}^{upper} exp(−ρ t) dt
                            / ∫_{0}^{upper}   exp(−ρ t) dt

    evaluated numerically on a fine grid.

    Parameters
    ----------
    T_start : float
        Time from which to start the future consumption stream.
    params : Parameters
        Economy parameters.  Only ``rho``, ``w0``, and ``s0`` are used.
    upper : float, optional
        Truncation of the infinite horizon (default 2000).
    n : int, optional
        Number of equally spaced quadrature points on [T_start, upper].

    Returns
    -------
    float
        Dogmatist's continuation-wealth share W_N(T_start) / (W_R + W_N).
    """
    rho = params.rho
    w0 = params.w0

    # Deterministic integrand: exp(-rho*t), normalised by D0 which cancels.
    t_num = np.linspace(T_start, upper, n)
    t_den = np.linspace(0.0, upper, n)
    dt_num = t_num[1] - t_num[0]
    dt_den = t_den[1] - t_den[0]

    disc_num = np.exp(-rho * t_num)
    disc_den = np.exp(-rho * t_den)

    # Trapezoid integrals
    integral_num = np.trapz(disc_num, dx=dt_num)
    integral_den = np.trapz(disc_den, dx=dt_den)

    return (w0 / (1.0 + w0)) * integral_num / integral_den
