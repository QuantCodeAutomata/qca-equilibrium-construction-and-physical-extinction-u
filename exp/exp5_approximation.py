"""Exp 5: General-gamma Finite-Maturity Approximation and Error-Bound Validation.

Validates Theorem 3(i): for gamma != 1, the bond-price approximation
    P_t^T ≈ exp(-r^R*(T-t)) * (1+w_t)^(-gamma) * (1+q_{t,T})
converges to the exact price as A_{t,T} grows, with errors bounded by
    e_{t,T} = K_gamma * exp{-[gamma/(2*(1+gamma)^2)] * sigma^2 * A_{t,T}}

Uses Monte Carlo for exact bond prices and forward-density errors.
Tested for gamma in {0.5, 2.0, 5.0} under fading pessimism.
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

from src.approximation import (
    bond_price_approx,
    compute_C_gamma,
    e_tT_bound,
    exact_bond_price_mc,
    l1_forward_density_error_mc,
    q_tT_general,
)
from src.model import Parameters, rational_yield
from src.paths import A_fading_pessimism_exact, eta_fading_pessimism

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MU = 0.02
SIGMA = 0.15
RHO = 0.02
D0 = 1.0
S0 = 0.05
W0 = S0 / (1.0 - S0)
ETA0 = -1.5

GAMMA_VALUES = [0.5, 2.0, 5.0]
T0 = 0.0          # evaluation at t=0
W_T0 = W0
MATURITIES = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
N_MC = 4000
DT = 1.0 / 52.0
SEED = 777

# Minimal t_grid used only to pass dt; exact_bond_price_mc reads t_grid[1]-t_grid[0]
_T_GRID_REF = np.array([0.0, DT])


# ---------------------------------------------------------------------------
# Run one (gamma, T) combination
# ---------------------------------------------------------------------------

def run_one(gamma: float, T: float, params: Parameters, seed_offset: int) -> dict:
    """Compute exact price, approx price, errors, and bound for one case.

    Parameters
    ----------
    gamma : float
        CRRA risk-aversion coefficient.
    T : float
        Maturity in years.
    params : Parameters
        Economy parameters (should have gamma matching the arg).
    seed_offset : int
        Added to base SEED for reproducible Monte Carlo draws per run.

    Returns
    -------
    dict
        Results including exact price, approx price, relative error, and bound.
    """
    from scipy.integrate import quad as _quad

    rR = rational_yield(params)
    A_tT = float(A_fading_pessimism_exact(np.array([T]), ETA0)[0])
    H_tT, _ = _quad(lambda t: float(eta_fading_pessimism(np.asarray(t), ETA0)), 0.0, T)

    # q_{0,T} and approximation
    q = float(q_tT_general(W_T0, gamma, SIGMA, H_tT))
    approx = float(bond_price_approx(rR, T - T0, W_T0, gamma, q))

    # Exact price via MC (log_pi_t=0 since t=0 so pi_0=1)
    exact = exact_bond_price_mc(
        T0,
        T,
        W_T0,
        0.0,          # log_pi_t (not used in function body)
        _T_GRID_REF,
        lambda t: eta_fading_pessimism(np.asarray(t), ETA0),
        params,
        N_MC,
        SEED + seed_offset,
    )

    rel_err = abs(exact - approx) / (abs(exact) + 1e-300)

    # Error bound
    C_gam = compute_C_gamma(gamma)
    Kg = 2.0 * C_gam
    e_val = float(e_tT_bound(Kg, gamma, SIGMA, A_tT))

    # L1 forward density error (only if e_tT < 1)
    l1_err = np.nan
    bound_2e = np.nan
    if e_val < 1.0:
        l1_err = float(
            l1_forward_density_error_mc(
                T0,
                T,
                W_T0,
                _T_GRID_REF,
                lambda t: eta_fading_pessimism(np.asarray(t), ETA0),
                params,
                N_MC // 2,
                SEED + seed_offset + 50,
            )
        )
        bound_2e = 2.0 * e_val / (1.0 - e_val)

    return {
        "gamma": gamma,
        "T": T,
        "A_tT": A_tT,
        "H_tT": H_tT,
        "q_tT": q,
        "exact_price": exact,
        "approx_price": approx,
        "rel_error": rel_err,
        "e_tT": e_val,
        "l1_fwd_error": l1_err,
        "bound_2e": bound_2e,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """Run all gamma/maturity combinations and save results."""
    all_rows = []
    seed_offset = 0

    for gamma in GAMMA_VALUES:
        params = Parameters(mu=MU, sigma=SIGMA, rho=RHO, gamma=gamma, D0=D0, s0=S0)
        rR = rational_yield(params)
        print(f"\n=== gamma = {gamma}, r_R = {rR:.6f} ===")

        for T in MATURITIES:
            row = run_one(gamma, T, params, seed_offset)
            all_rows.append(row)
            seed_offset += 1
            bound_str = f"{row['bound_2e']:.4f}" if not np.isnan(row["bound_2e"]) else "N/A (e≥1)"
            l1_str = f"{row['l1_fwd_error']:.4f}" if not np.isnan(row["l1_fwd_error"]) else "N/A"
            print(
                f"  T={T:5.1f}  A={row['A_tT']:.3f}  q={row['q_tT']:.4f}"
                f"  exact={row['exact_price']:.6f}  approx={row['approx_price']:.6f}"
                f"  rel_err={row['rel_error']:.4e}  e_tT={row['e_tT']:.4f}"
                f"  L1={l1_str}  bound={bound_str}"
            )

    df = pd.DataFrame(all_rows)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Exp 5: Approximation Errors vs. A_{t,T} by gamma", fontsize=12)

    for i, gamma in enumerate(GAMMA_VALUES):
        sub = df[df["gamma"] == gamma]
        ax = axes[i]
        ax.semilogy(sub["A_tT"], sub["rel_error"], "o-b", lw=1.5, label="Relative error |exact/approx - 1|")
        ax.semilogy(sub["A_tT"], sub["e_tT"], "s--r", lw=1.5, label=r"Bound $e_{t,T}$")
        valid = ~sub["bound_2e"].isna()
        if valid.any():
            ax.semilogy(sub["A_tT"][valid], sub["bound_2e"][valid], "^:g", lw=1.5, label=r"$2e/(1-e)$")
        ax.set_xlabel(r"$A_{t,T}$")
        ax.set_ylabel("Error")
        ax.set_title(rf"$\gamma = {gamma}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_approx_errors.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Summary table ----
    df_show = df[["gamma", "T", "A_tT", "q_tT", "exact_price", "approx_price",
                  "rel_error", "e_tT", "l1_fwd_error", "bound_2e"]].copy()
    df_show = df_show.round(6)
    df_show.to_csv(os.path.join(RESULTS_DIR, "exp5_approx_summary.csv"), index=False)

    print("\n=== Exp 5 Summary ===")
    print(df_show.to_string(index=False))

    return df


if __name__ == "__main__":
    main()
