"""Exp 1: Equilibrium Construction and Physical Extinction under Time-Varying Beliefs.

Validates Proposition 1: the dogmatist consumption share s_t = w_t/(1+w_t)
converges to 0 a.s. if and only if A_t -> infinity, and
    log(w_t) / A_t  -> -sigma^2 / (2 gamma)  a.s.

Three Section-4 belief paths are tested:
  - Fading pessimism (FP):      eta_t = eta0 / sqrt(1+t),  eta0 = -1.5
  - Fading alternating (FA):    eta_t = 2 sin(omega t)/sqrt(1+t)
  - Constant-amplitude cyclical (CA): eta_t = 2 cos(omega t)

All results saved to results/.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import (
    Parameters,
    benchmark_martingale_N,
    benchmark_martingale_R,
    consumption_shares,
    dogmatist_share,
    log_endowment,
    log_sdf,
    log_w,
    log_xi,
    rational_yield,
    simulate_brownian,
)
from src.paths import (
    A_fading_pessimism_exact,
    A_left_riemann,
    A_numerical,
    H_constant_cyclical_exact,
    H_fading_pessimism_exact,
    H_left_riemann,
    H_numerical,
    compute_path_objects,
    eta_constant_cyclical,
    eta_fading_alternating,
    eta_fading_pessimism,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MU = 0.02
SIGMA = 0.15
RHO = 0.02
GAMMA = 2.0
D0 = 1.0
S0 = 0.05
T_SIM = 300.0      # years
DT = 1.0 / 52.0   # weekly steps
N_PATHS = 10
SEED = 42

L = 20.0
OMEGA = 2.0 * np.pi / L

ETA0_FP = -1.5


# ---------------------------------------------------------------------------
# Helper: run one scenario
# ---------------------------------------------------------------------------

def run_scenario(
    path_label: str,
    eta_func,
    t_grid: np.ndarray,
    dB: np.ndarray,
    dt: float,
    params: Parameters,
    A_exact_func=None,
    H_exact_func=None,
) -> dict:
    """Compute equilibrium objects for a given deterministic belief path.

    Parameters
    ----------
    path_label : str
        Human-readable name of the scenario.
    eta_func : callable
        t -> eta_t (vectorized).
    t_grid : np.ndarray
        Time grid, shape (N,).
    dB : np.ndarray
        Brownian increments, shape (N-1, n_paths).
    dt : float
        Step size.
    params : Parameters
        Economy parameters.
    A_exact_func : callable or None
        Optional exact formula for A_t(t_grid) -> array of shape (N,).
    H_exact_func : callable or None
        Optional exact formula for H_t(t_grid) -> array of shape (N,).

    Returns
    -------
    dict
        Contains arrays: A_t, H_t, log_w_t, s_t, ratio, c_R, c_N, log_D, log_pi.
    """
    N = len(t_grid)
    n_paths = dB.shape[1]

    # Deterministic path objects
    eta_vals = eta_func(t_grid)
    if A_exact_func is not None:
        A_t = A_exact_func(t_grid)
    else:
        A_t = A_numerical(t_grid, eta_vals, dt)  # shape (N,) with A_0 = 0

    if H_exact_func is not None:
        H_t = H_exact_func(t_grid)
    else:
        H_t = H_numerical(t_grid, eta_vals, dt)  # shape (N,) with H_0 = 0

    # Stochastic objects (per path)
    # log_xi uses left-endpoint Riemann sum: pass eta at left endpoints.
    # log_xi already returns shape (N, n_paths) with row 0 = 0.
    eta_left = eta_vals[:-1]  # shape (N-1,)
    log_xi_full = log_xi(eta_left, dB, dt, params.sigma, params.gamma)  # (N, n_paths)

    log_w_paths = log_w(log_xi_full, params.w0, params.gamma)  # (N, n_paths)
    s_t_paths = dogmatist_share(log_w_paths)  # (N, n_paths)

    # Ratio log(w_t)/A_t (avoid division by zero at t=0)
    ratio = np.full_like(log_w_paths, np.nan)
    safe = A_t > 1e-10
    ratio[safe, :] = log_w_paths[safe, :] / A_t[safe, np.newaxis]

    # Consumption shares (using endowment).
    # log_endowment expects dB increments, shape (N-1, n_paths); it builds B_t internally.
    log_D_paths = log_endowment(t_grid, dB, dt, params)  # (N, n_paths)
    c_R_paths, c_N_paths = consumption_shares(log_w_paths, log_D_paths)

    log_pi_paths = log_sdf(t_grid, log_D_paths, log_w_paths, params)

    # Validation: c_R + c_N = D_t
    D_t = np.exp(log_D_paths)
    sum_c = c_R_paths + c_N_paths
    max_clearing_err = np.max(np.abs(sum_c - D_t))
    print(f"  [{path_label}] max market-clearing error: {max_clearing_err:.2e}")

    # pi_0 = 1 check
    pi0_err = np.max(np.abs(log_pi_paths[0, :]))
    print(f"  [{path_label}] max |log_pi_0|:              {pi0_err:.2e}")

    return {
        "label": path_label,
        "t_grid": t_grid,
        "A_t": A_t,
        "H_t": H_t,
        "log_w": log_w_paths,
        "s_t": s_t_paths,
        "ratio": ratio,
        "c_R": c_R_paths,
        "c_N": c_N_paths,
        "log_D": log_D_paths,
        "log_pi": log_pi_paths,
        "A_T_final": A_t[-1],
        "H_T_final": H_t[-1],
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_scenario(res: dict, sigma: float, gamma: float, save_path: str) -> None:
    """Save a 2x2 summary plot for a belief-path scenario."""
    t = res["t_grid"]
    A_t = res["A_t"]
    s_t = res["s_t"]
    ratio = res["ratio"]
    theoretical_limit = -(sigma**2) / (2.0 * gamma)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(res["label"], fontsize=13)

    # (a) A_t
    ax = axes[0, 0]
    ax.plot(t, A_t, "b-", lw=2)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"$A_t$")
    ax.set_title(r"Total disagreement $A_t = \int_0^t \eta_s^2 ds$")
    ax.grid(True, alpha=0.3)

    # (b) s_t paths
    ax = axes[0, 1]
    for j in range(s_t.shape[1]):
        ax.plot(t, s_t[:, j], alpha=0.6, lw=0.8)
    ax.axhline(0.0, color="k", lw=1.5, ls="--", label=r"$s_t \to 0$")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"$s_t = w_t/(1+w_t)$")
    ax.set_title("Dogmatist consumption share")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) log(w_t)/A_t
    ax = axes[1, 0]
    # Start from first non-nan index
    start = np.where(~np.isnan(ratio[:, 0]))[0]
    if len(start):
        start = start[0]
        for j in range(ratio.shape[1]):
            ax.plot(t[start:], ratio[start:, j], alpha=0.6, lw=0.8)
    ax.axhline(
        theoretical_limit,
        color="r",
        lw=2,
        ls="--",
        label=rf"$-\sigma^2/(2\gamma) = {theoretical_limit:.4f}$",
    )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"$\log(w_t) / A_t$")
    ax.set_title("Asymptotic rate (Proposition 1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) H_t
    ax = axes[1, 1]
    ax.plot(t, res["H_t"], "g-", lw=2)
    ax.axhline(0.0, color="k", lw=1, ls=":")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"$H_t = \int_0^t \eta_s ds$")
    ax.set_title(r"Net disagreement $H_t$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """Run all three belief-path scenarios and save results."""
    params = Parameters(mu=MU, sigma=SIGMA, rho=RHO, gamma=GAMMA, D0=D0, s0=S0)
    rR = rational_yield(params)
    theoretical_limit = -(SIGMA**2) / (2.0 * GAMMA)

    print(f"Parameters: gamma={GAMMA}, sigma={SIGMA}, rho={RHO}")
    print(f"Rational yield r_R = {rR:.6f}")
    print(f"Theoretical asymptotic limit: -sigma^2/(2*gamma) = {theoretical_limit:.6f}")
    print(f"Simulation: T={T_SIM}yr, dt=1/52, n_paths={N_PATHS}\n")

    t_grid, dB, _ = simulate_brownian(T_SIM, DT, N_PATHS, seed=SEED)

    summary_rows = []
    scenarios = []

    # ---- Fading Pessimism ----
    print("Scenario 1: Fading Pessimism (eta0=-1.5)")
    res_FP = run_scenario(
        "Fading Pessimism (eta0=-1.5)",
        lambda t: eta_fading_pessimism(t, ETA0_FP),
        t_grid,
        dB,
        DT,
        params,
        A_exact_func=lambda t: A_fading_pessimism_exact(t, ETA0_FP),
        H_exact_func=lambda t: H_fading_pessimism_exact(t, ETA0_FP),
    )
    scenarios.append(res_FP)

    # ---- Fading Alternating ----
    print("Scenario 2: Fading Alternating (omega=2pi/20)")
    res_FA = run_scenario(
        "Fading Alternating (omega=2pi/20)",
        lambda t: eta_fading_alternating(t, OMEGA),
        t_grid,
        dB,
        DT,
        params,
    )
    scenarios.append(res_FA)

    # ---- Constant-Amplitude Cyclical ----
    print("Scenario 3: Constant-Amplitude Cyclical (omega=2pi/20)")
    res_CA = run_scenario(
        "Constant-Amplitude Cyclical (omega=2pi/20)",
        lambda t: eta_constant_cyclical(t, OMEGA),
        t_grid,
        dB,
        DT,
        params,
        H_exact_func=lambda t: H_constant_cyclical_exact(t, OMEGA),
    )
    scenarios.append(res_CA)

    # ---- Summary statistics ----
    for res in scenarios:
        s_T = res["s_t"][-1, :]
        ratio_final = res["ratio"]
        # Get last non-nan ratio
        valid_rows = ~np.isnan(ratio_final[-100, :])
        r_vals = ratio_final[-100, valid_rows] if valid_rows.any() else np.array([np.nan])
        summary_rows.append(
            {
                "path_type": res["label"],
                "A_T_final": res["A_T_final"],
                "H_T_final": res["H_T_final"],
                "mean_s_T": float(np.mean(s_T)),
                "mean_log_w_over_A_T": float(np.mean(r_vals)),
                "theoretical_limit": theoretical_limit,
            }
        )

    # ---- Plots ----
    for res, fname in zip(
        scenarios,
        ["exp1_FP_paths.png", "exp1_FA_paths.png", "exp1_CA_paths.png"],
    ):
        plot_scenario(res, SIGMA, GAMMA, os.path.join(RESULTS_DIR, fname))

    # ---- Summary table ----
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp1_summary.csv"), index=False)
    print("\n=== Exp 1 Summary ===")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    main()
