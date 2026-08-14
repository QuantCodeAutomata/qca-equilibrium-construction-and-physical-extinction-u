"""Exp 4: Long-Run Pricing Law, Long Bond Existence, and Separation from Long Yield.

Validates Theorem 3(ii)-(iii): physical extinction alone does not guarantee
convergence of long-run pricing objects.

Three canonical belief paths:
  FP:  fading pessimism   → A_T → ∞, H_T → -∞  → long-forward measure + long bond EXIST
  FA:  fading alternating → A_T → ∞, H_T → h∈R  → long-forward measure + long bond EXIST (interior mix)
  CA:  constant-amplitude → A_T → ∞, H_T oscillates → NO long-forward measure or long bond

In all cases H_T/T → 0, so long yield = r^R (separation from long-forward measure).
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import (
    Parameters,
    benchmark_martingale_N,
    benchmark_martingale_R,
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
    H_constant_cyclical_exact,
    H_fading_pessimism_exact,
    eta_constant_cyclical,
    eta_fading_alternating,
    eta_fading_pessimism,
)
from src.long_run import (
    classify_path,
    long_yield_formula,
    mixed_limit_martingale,
    normalized_bond_price,
    theta_T_series,
)
from src.section4 import (
    G_0T_constant_cyclical,
    G_0T_fading_pessimism,
    q_0T,
    theta_0T,
    yield_spread_gamma1,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MU = 0.02
SIGMA = 0.15
RHO = 0.02
GAMMA = 1.0      # use log utility for closed-form comparison; classification works for any gamma
D0 = 1.0
S0 = 0.05
W0 = S0 / (1.0 - S0)
L = 20.0
OMEGA = 2.0 * np.pi / L
ETA0 = -1.5

MATURITY_LADDER = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0])


# ---------------------------------------------------------------------------
# Compute A_T and H_T over ladder for each path
# ---------------------------------------------------------------------------

def compute_AT_HT(
    T_vals: np.ndarray,
    eta_func,
    A_exact=None,
    H_exact=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute A_T and H_T at each maturity in T_vals."""
    if A_exact is not None:
        A = A_exact(T_vals)
    else:
        A = np.array([quad(lambda t: eta_func(t)**2, 0.0, T)[0] for T in T_vals])

    if H_exact is not None:
        H = H_exact(T_vals)
    else:
        H = np.array([quad(lambda t: eta_func(t), 0.0, T)[0] for T in T_vals])

    return A, H


# ---------------------------------------------------------------------------
# Long yield curve from section4 formulas (gamma=1)
# ---------------------------------------------------------------------------

def long_yield_curve(
    T_vals: np.ndarray,
    w0: float,
    G_0T: np.ndarray,
    rR: float,
) -> np.ndarray:
    """Compute y_0^T for each T using the exact gamma=1 bond price."""
    prices = np.array([
        float(np.exp(-rR * T) * (1.0 + q_0T(w0, g)) / (1.0 + w0))
        for T, g in zip(T_vals, G_0T)
    ])
    yields = -np.log(prices) / T_vals
    return yields


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    params = Parameters(mu=MU, sigma=SIGMA, rho=RHO, gamma=GAMMA, D0=D0, s0=S0)
    rR = rational_yield(params)
    print(f"r_R = {rR:.6f}\n")

    # -----------------------------------------------------------------------
    # 1. Asymptotic classification
    # -----------------------------------------------------------------------
    T_dense = np.logspace(0.1, np.log10(500), 200)

    # FP
    A_fp, H_fp = compute_AT_HT(
        T_dense,
        lambda t: eta_fading_pessimism(np.asarray(t), ETA0),
        A_exact=lambda T: A_fading_pessimism_exact(T, ETA0),
        H_exact=lambda T: H_fading_pessimism_exact(T, ETA0),
    )
    # FA
    A_fa = np.array([quad(lambda t: (2.0 * np.sin(OMEGA * t) / np.sqrt(1.0 + t))**2, 0.0, T)[0] for T in T_dense])
    H_fa = np.array([quad(lambda t: 2.0 * np.sin(OMEGA * t) / np.sqrt(1.0 + t), 0.0, T)[0] for T in T_dense])
    # CA
    A_ca = np.array([T * 2.0 for T in T_dense])   # int_0^T 4*cos^2 dt = 2*T analytically
    H_ca = H_constant_cyclical_exact(T_dense, OMEGA)

    print("=== Asymptotic Classification ===")
    print(f"FP:  A_T(500) = {A_fp[-1]:.2f}, H_T(500) = {H_fp[-1]:.2f}, H_T/T(500) = {H_fp[-1]/500:.4f}")
    print(f"FA:  A_T(500) = {A_fa[-1]:.2f}, H_T(500) = {H_fa[-1]:.2f}, H_T/T(500) = {H_fa[-1]/500:.4f}")
    print(f"CA:  A_T(500) = {A_ca[-1]:.2f}, H_T(500) = {H_ca[-1]:.2f}, H_T/T(500) = {H_ca[-1]/500:.4f}")

    # -----------------------------------------------------------------------
    # 2. theta_T convergence — deterministic time-zero diagnostic (gamma=1)
    # -----------------------------------------------------------------------
    T_ladder = MATURITY_LADDER

    G_fp_ladder = G_0T_fading_pessimism(T_ladder, ETA0, SIGMA)
    G_ca_ladder = G_0T_constant_cyclical(T_ladder, OMEGA, SIGMA)
    G_fa_ladder = np.array([
        SIGMA**2 * quad(lambda t: 2.0 * np.sin(OMEGA * t) / np.sqrt(1.0 + t), 0.0, T)[0]
        for T in T_ladder
    ])

    theta_fp = theta_0T(q_0T(W0, G_fp_ladder)) * 100.0
    theta_fa = theta_0T(q_0T(W0, G_fa_ladder)) * 100.0
    theta_ca = theta_0T(q_0T(W0, G_ca_ladder)) * 100.0

    print("\n=== theta_T series (% valuation weight) ===")
    df_theta = pd.DataFrame({
        "T": T_ladder,
        "theta_FP (%)": np.round(theta_fp, 4),
        "theta_FA (%)": np.round(theta_fa, 4),
        "theta_CA (%)": np.round(theta_ca, 4),
    })
    print(df_theta.to_string(index=False))

    # -----------------------------------------------------------------------
    # 3. Long yield curve
    # -----------------------------------------------------------------------
    T_long = np.linspace(1.0, 320.0, 500)
    G_fp_long = G_0T_fading_pessimism(T_long, ETA0, SIGMA)
    G_ca_long = G_0T_constant_cyclical(T_long, OMEGA, SIGMA)
    G_fa_long = np.array([
        SIGMA**2 * quad(lambda t: 2.0 * np.sin(OMEGA * t) / np.sqrt(1.0 + t), 0.0, T)[0]
        for T in T_long
    ])

    y_fp = long_yield_curve(T_long, W0, G_fp_long, rR) * 100.0
    y_fa = long_yield_curve(T_long, W0, G_fa_long, rR) * 100.0
    y_ca = long_yield_curve(T_long, W0, G_ca_long, rR) * 100.0
    y_inf_theory = rR * 100.0   # bar_eta = 0 for all three paths since H_T/T -> 0

    print(f"\nTheoretical long yield y^inf = r^R + gamma*sigma^2*min(bar_eta,0) = {y_inf_theory:.4f}%")
    print(f"Numerical y_0^T(FP, T=320) = {y_fp[-1]:.4f}%")
    print(f"Numerical y_0^T(FA, T=320) = {y_fa[-1]:.4f}%")
    print(f"Numerical y_0^T(CA, T=320) = {y_ca[-1]:.4f}%")

    # -----------------------------------------------------------------------
    # 4. Existence classification table
    # -----------------------------------------------------------------------
    # FP: A_T → ∞, H_T → -∞  → long-forward measure exists (dogmatist benchmark)
    # FA: A_T → ∞, H_T → finite limit  → long-forward measure exists (interior mix)
    # CA: A_T → ∞, H_T oscillates, no limit → long-forward measure does NOT exist
    # All: H_T/T → 0 → long yield = r^R

    classification_rows = [
        {
            "Path": "FP (fading pessimism)",
            "A_T→∞": True,
            "H_T limit": "−∞",
            "H_T/T→": 0.0,
            "Long-fwd measure exists": True,
            "Long bond exists": True,
            "Long yield = r^R": True,
            "Limiting measure": "Dogmatist benchmark",
        },
        {
            "Path": "FA (fading alternating)",
            "A_T→∞": True,
            "H_T limit": "h∈R (finite)",
            "H_T/T→": 0.0,
            "Long-fwd measure exists": True,
            "Long bond exists": True,
            "Long yield = r^R": True,
            "Limiting measure": "Interior mixture",
        },
        {
            "Path": "CA (constant-amplitude cyclical)",
            "A_T→∞": True,
            "H_T limit": "oscillates (no limit)",
            "H_T/T→": 0.0,
            "Long-fwd measure exists": False,
            "Long bond exists": False,
            "Long yield = r^R": True,
            "Limiting measure": "Does not exist",
        },
    ]
    df_class = pd.DataFrame(classification_rows)
    print("\n=== Long-Run Existence Classification ===")
    print(df_class.to_string(index=False))

    # -----------------------------------------------------------------------
    # 5. Plots
    # -----------------------------------------------------------------------

    # (a) theta_T convergence
    T_theta_dense = np.linspace(1.0, 320.0, 500)
    G_fp_d = G_0T_fading_pessimism(T_theta_dense, ETA0, SIGMA)
    G_ca_d = G_0T_constant_cyclical(T_theta_dense, OMEGA, SIGMA)
    G_fa_d = np.array([
        SIGMA**2 * quad(lambda t: 2.0 * np.sin(OMEGA * t) / np.sqrt(1.0 + t), 0.0, T)[0]
        for T in T_theta_dense
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Long-Run Analysis: theta_T and Yield Curves", fontsize=12)

    ax = axes[0]
    ax.plot(T_theta_dense, theta_0T(q_0T(W0, G_fp_d)) * 100, "b-", lw=1.5, label="FP → dogmatist (100%)")
    ax.plot(T_theta_dense, theta_0T(q_0T(W0, G_fa_d)) * 100, "g-", lw=1.5, label="FA → interior limit")
    ax.plot(T_theta_dense, theta_0T(q_0T(W0, G_ca_d)) * 100, "r-", lw=1.5, label="CA → oscillates (no limit)")
    ax.axhline(S0 * 100, color="k", ls="--", lw=1.0, label=rf"$s_0={S0*100}\%$")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel(r"$\theta_{0,T}$ (%)")
    ax.set_title(r"Valuation weight $\theta_{0,T}$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(T_long, y_fp, "b-", lw=1.5, label="FP")
    ax.plot(T_long, y_fa, "g-", lw=1.5, label="FA")
    ax.plot(T_long, y_ca, "r-", lw=1.5, label="CA")
    ax.axhline(y_inf_theory, color="k", ls="--", lw=1.5, label=rf"$y^\infty = r^R = {y_inf_theory:.2f}\%$")
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel(r"Yield $y_{0,T}$ (%)")
    ax.set_title(r"Yield curve — all three paths converge to $r^R$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_long_run.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # (b) A_T and H_T diagnostic
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.semilogy(T_dense, A_fp, "b-", lw=1.5, label="FP")
    ax.semilogy(T_dense, A_fa, "g-", lw=1.5, label="FA")
    ax.semilogy(T_dense, A_ca, "r-", lw=1.5, label="CA")
    ax.set_xlabel("T (years)"); ax.set_ylabel(r"$A_T$ (log scale)")
    ax.set_title(r"Total disagreement $A_T \to \infty$ for all paths")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(T_dense, H_fp, "b-", lw=1.5, label="FP → −∞")
    ax.plot(T_dense, H_fa, "g-", lw=1.5, label="FA → finite limit")
    ax.plot(T_dense, H_ca, "r-", lw=1.5, label="CA → oscillates")
    ax.axhline(0.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("T (years)"); ax.set_ylabel(r"$H_T$")
    ax.set_title(r"Net disagreement $H_T$ — divergent, convergent, oscillatory")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_AT_HT.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Save tables ----
    df_theta.to_csv(os.path.join(RESULTS_DIR, "exp4_theta_series.csv"), index=False)
    df_class.to_csv(os.path.join(RESULTS_DIR, "exp4_classification.csv"), index=False)
    print("\nAll exp4 results saved.")
    return df_class


if __name__ == "__main__":
    main()
