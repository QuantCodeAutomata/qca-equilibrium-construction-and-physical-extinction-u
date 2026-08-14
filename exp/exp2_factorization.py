"""Exp 2: Candidate Hansen-Scheinkman Factorizations and Benchmark Martingale Structure.

Validates Lemma 2: for every (a,b) >= 0 with a+b > 0, the pair
    Phi_{a,b}(s, w)  and  M_t^{a,b}
constitutes an exact Hansen-Scheinkman factorization of the state-price density
with common eigenvalue r^R.

Three candidate pairs are tested:
  - Rational benchmark:   (a, b) = (1, 0)
  - Dogmatist benchmark:  (a, b) = (0, 1)
  - Interior mixture:     (a, b) = (1, 1)

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
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.factorization import (
    identity_check_MR_wgamma,
    log_phi_ab,
    mixed_martingale_log,
    phi_ab,
    rational_yield,
    sdf_factorization_check,
)
from src.model import (
    Parameters,
    benchmark_martingale_N,
    benchmark_martingale_R,
    dogmatist_share,
    log_endowment,
    log_sdf,
    log_w,
    log_xi,
    simulate_brownian,
)
from src.paths import (
    A_fading_pessimism_exact,
    H_fading_pessimism_exact,
    H_left_riemann,
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
T_SIM = 50.0
DT = 1.0 / 52.0
N_PATHS = 5
SEED = 123
ETA0 = -1.5

AB_PAIRS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """Run factorization validation for all (a,b) pairs."""
    params = Parameters(mu=MU, sigma=SIGMA, rho=RHO, gamma=GAMMA, D0=D0, s0=S0)
    rR = rational_yield(params)
    print(f"r_R = {rR:.6f}")

    t_grid, dB, _ = simulate_brownian(T_SIM, DT, N_PATHS, seed=SEED)
    N = len(t_grid)

    # Deterministic path objects (left-endpoint for consistency with stochastic integrals)
    eta_vals = eta_fading_pessimism(t_grid, ETA0)
    eta_left = eta_vals[:-1]
    A_t_left = np.concatenate([[0.0], np.cumsum(eta_left**2 * DT)])  # shape (N,)
    H_t_left = np.concatenate([[0.0], np.cumsum(eta_left * DT)])     # shape (N,)

    # Stochastic objects.
    # log_xi already returns (N, n_paths) with row 0 = 0; no manual prepend needed.
    log_xi_full = log_xi(eta_left, dB, DT, SIGMA, GAMMA)   # (N, n_paths)
    log_w_paths = log_w(log_xi_full, params.w0, GAMMA)      # (N, n_paths)
    w_t_paths = np.exp(log_w_paths)

    # log_endowment expects dB increments (N-1, n_paths); it builds B_t internally.
    log_D_paths = log_endowment(t_grid, dB, DT, params)     # (N, n_paths)
    log_pi_paths = log_sdf(t_grid, log_D_paths, log_w_paths, params)

    log_MR = benchmark_martingale_R(t_grid, dB, DT, SIGMA, GAMMA)   # (N, n_paths)
    log_MN = benchmark_martingale_N(t_grid, eta_left, dB, DT, SIGMA, GAMMA)

    # ---- Identity check: M_t^R * w_t^gamma == w0^gamma * exp(-gamma*sigma^2*H_t) * M_t^N ----
    id_check = identity_check_MR_wgamma(
        log_MR, log_MN, params.w0, log_w_paths, H_t_left, GAMMA, SIGMA
    )
    print(f"\nIdentity check M^R * w^gamma == w0^gamma * exp(-gamma*sigma^2*H) * M^N:")
    print(f"  max absolute error: {id_check['max_abs_err']:.2e}")
    print(f"  passes tol=1e-8:    {id_check['passed']}")

    # ---- Per (a,b) factorization checks ----
    summary_rows = []
    log_Mab_all = {}

    for a, b in AB_PAIRS:
        label = f"({a:.0f},{b:.0f})"
        log_Mab = mixed_martingale_log(log_MR, log_MN, params.w0, a, b, GAMMA)
        log_Mab_all[label] = log_Mab

        check = sdf_factorization_check(
            t_grid,
            log_pi_paths,
            log_Mab,
            H_t_left,
            w_t_paths,
            params.w0,
            a,
            b,
            GAMMA,
            SIGMA,
            rR,
            tol=1e-6,
        )
        print(f"\nFactorization check (a,b)={label}:")
        print(f"  max abs error:   {check['max_abs_err']:.2e}")
        print(f"  passes tol=1e-6: {check['passed']}")

        summary_rows.append(
            {
                "(a,b)": label,
                "max_identity_error": id_check["max_abs_err"],
                "max_factorization_error": check["max_abs_err"],
                "passes": check["passed"],
            }
        )

    # ---- Semigroup Monte Carlo check for (1,0) ----
    tau = 5.0
    n_mc = 2000
    t0_idx = int(1.0 / DT)  # t=1 year
    t0 = t_grid[t0_idx]
    # Use path 0 for the conditioning state
    w_t0 = w_t_paths[t0_idx, 0]
    H_t0 = H_t_left[t0_idx]

    rng = np.random.default_rng(999)
    dB_future = rng.normal(0.0, np.sqrt(DT), size=(int(tau / DT), n_mc))
    t_future = np.linspace(t0 + DT, t0 + tau, int(tau / DT))
    # eta at future left endpoints
    t_future_left = np.linspace(t0, t0 + tau - DT, int(tau / DT))
    eta_future = eta_fading_pessimism(t_future_left, ETA0)
    H_future = H_t0 + np.concatenate([[0.0], np.cumsum(eta_future * DT)])  # (n_future_steps+1,)
    # log_xi from t0 to t0+tau.
    # log_xi already returns (n_steps+1, n_mc) with row 0 = 0; no manual prepend needed.
    log_xi_fut = log_xi(eta_future, dB_future, DT, SIGMA, GAMMA)   # (n_steps+1, n_mc)
    log_w_fut = log_w(log_xi_fut, w_t0, GAMMA)                      # (n_steps+1, n_mc)

    # phi_{1,0}(s, H_s, w_s) = 1/(1+w_s)^gamma
    H_at_tau = H_future[-1]
    phi_t0 = phi_ab(t0, H_t0, w_t0, 1.0, 0.0, GAMMA, SIGMA)
    phi_tau_vals = np.array([
        phi_ab(t0 + tau, H_at_tau, float(np.exp(log_w_fut[-1, j])), 1.0, 0.0, GAMMA, SIGMA)
        for j in range(n_mc)
    ])
    # Price ratio via log_sdf difference.
    # log_endowment expects dB increments (N-1, n_mc); pass dB_future directly.
    t_fut_full = np.concatenate([[t0], t_future])
    log_D_fut = log_endowment(
        t_fut_full,
        dB_future,
        DT,
        Parameters(mu=MU, sigma=SIGMA, rho=RHO, gamma=GAMMA, D0=float(np.exp(log_D_paths[t0_idx, 0])), s0=S0),
    )
    # Use a fresh parameters with D0 = D_{t0} so log_sdf gives the *ratio* pi_tau/pi_{t0}
    # pi_t = exp(-rho*t)*(D_t/D0)^(-gamma)*((1+w_t)/(1+w0))^gamma
    # pi_tau/pi_t0 ratio:
    pi_ratio = np.exp(
        -RHO * tau
        - GAMMA * (log_D_fut[-1, :] - log_D_fut[0, :])
        + GAMMA * (np.log1p(np.exp(log_w_fut[-1, :])) - np.log1p(w_t0))
    )
    # Semigroup: E[pi_tau/pi_t0 * phi(t0+tau, H_tau, w_tau)] = exp(-rR*tau)*phi(t0, H_t0, w_t0)
    lhs_mc = float(np.mean(pi_ratio * phi_tau_vals))
    rhs_exact = float(np.exp(-rR * tau) * phi_t0)
    sg_rel_err = abs(lhs_mc - rhs_exact) / abs(rhs_exact)
    print(f"\nSemigroup check (a,b)=(1,0), tau={tau}yr:")
    print(f"  MC estimate: {lhs_mc:.6f}")
    print(f"  Exact rhs:   {rhs_exact:.6f}")
    print(f"  Relative error: {sg_rel_err:.4f}")
    summary_rows[0]["semigroup_rel_error"] = sg_rel_err

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Benchmark Martingales (path 0)", fontsize=12)

    ax = axes[0]
    ax.plot(t_grid, log_MR[:, 0], label=r"$\log M_t^R$", color="blue")
    ax.set_xlabel("Time (years)"); ax.set_ylabel(r"log value")
    ax.set_title(r"Rational martingale $\log M_t^R$")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_grid, log_MN[:, 0], label=r"$\log M_t^N$", color="orange")
    ax.set_xlabel("Time (years)")
    ax.set_title(r"Dogmatist martingale $\log M_t^N$")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for (a, b), label in zip(AB_PAIRS, log_Mab_all.keys()):
        ax.plot(t_grid, log_Mab_all[label][:, 0], label=rf"$\log M_t^{{{label}}}$")
    ax.set_xlabel("Time (years)")
    ax.set_title(r"Mixed martingales $\log M_t^{a,b}$")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_martingales.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Factorization errors plot
    fig, axes = plt.subplots(1, len(AB_PAIRS), figsize=(15, 5))
    fig.suptitle("SDF Factorization Relative Errors", fontsize=12)
    for idx, (a, b) in enumerate(AB_PAIRS):
        label = f"({a:.0f},{b:.0f})"
        log_Mab = log_Mab_all[label]
        log_phi_0 = log_phi_ab(H_t_left[0], log_w_paths[0, :], a, b, GAMMA, SIGMA)
        log_phi_t = log_phi_ab(H_t_left[:, np.newaxis], log_w_paths, a, b, GAMMA, SIGMA)  # (N, n_paths)
        log_lhs = log_pi_paths
        log_rhs = -rR * t_grid[:, np.newaxis] + log_Mab + log_phi_0[np.newaxis, :] - log_phi_t
        rel_err = np.abs(np.exp(log_lhs) - np.exp(log_rhs)) / (np.abs(np.exp(log_lhs)) + 1e-300)
        axes[idx].semilogy(t_grid[1:], rel_err[1:, 0], lw=1.5)
        axes[idx].set_title(rf"$(a,b)={label}$")
        axes[idx].set_xlabel("Time (years)")
        axes[idx].set_ylabel("Rel. error")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_factorization_errors.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp2_summary.csv"), index=False)
    print("\n=== Exp 2 Summary ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
