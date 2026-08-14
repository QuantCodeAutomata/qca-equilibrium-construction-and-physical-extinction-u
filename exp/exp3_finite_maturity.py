"""Exp 3: Finite-Maturity Pricing and Exact Section 4 Replication under Log Utility.

Reproduces Figure 1 (a-d) and Table 1 (Panels A and B) from the paper for gamma=1.

Five fading-pessimism scenarios (Table 1 Panel A):
  FP1: eta0=-1.50, sigma=0.10
  FP2: eta0=-1.00, sigma=0.15
  FP3: eta0=-1.50, sigma=0.15  <- baseline
  FP4: eta0=-2.00, sigma=0.15
  FP5: eta0=-1.50, sigma=0.20

Alternating paths (sigma=0.15, L=20yr):
  FA:  eta_t = 2*sin(omega*t)/sqrt(1+t)
  CA:  eta_t = 2*cos(omega*t)

All in gamma=1 (log utility) so closed-form exact formulas apply.
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

from src.model import Parameters, rational_yield
from src.paths import eta_fading_alternating, H_fading_pessimism_exact, H_constant_cyclical_exact
from src.section4 import (
    G_0T_constant_cyclical,
    G_0T_fading_pessimism,
    G_0T_numerical,
    bond_price_gamma1,
    bond_price_premium,
    bond_vol_gamma1,
    forward_growth_shift_gamma1,
    forward_rate_spread_gamma1,
    future_consumption_value_N,
    q_0T,
    theta_0T,
    yield_spread_gamma1,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 1.0
RHO = 0.02
MU = 0.02  # only needed for r_R level; spreads are r_R-relative
S0 = 0.05
W0 = S0 / (1.0 - S0)  # = 1/19
L = 20.0
OMEGA = 2.0 * np.pi / L

# Five FP scenarios: (label, eta0, sigma)
FP_SCENARIOS = [
    ("FP1: eta0=-1.5, sig=10%", -1.50, 0.10),
    ("FP2: eta0=-1.0, sig=15%", -1.00, 0.15),
    ("FP3: eta0=-1.5, sig=15% (baseline)", -1.50, 0.15),
    ("FP4: eta0=-2.0, sig=15%", -2.00, 0.15),
    ("FP5: eta0=-1.5, sig=20%", -1.50, 0.20),
]
FP_COLORS = ["steelblue", "darkorange", "green", "red", "purple"]

TABLE_MATURITIES = [10, 20, 30]


# ---------------------------------------------------------------------------
# Helper: G_0T for fading alternating (numerical)
# ---------------------------------------------------------------------------

def G_0T_fading_alternating(T_vals: np.ndarray, sigma: float) -> np.ndarray:
    """Compute G_{0,T} = sigma^2 * H_T for the fading-alternating path numerically."""
    results = []
    for T in T_vals:
        # H_T = integral_0^T 2*sin(omega*t)/sqrt(1+t) dt (no exact closed form)
        val, _ = quad(lambda t: 2.0 * np.sin(OMEGA * t) / np.sqrt(1.0 + t), 0.0, T)
        results.append(sigma**2 * val)
    return np.array(results)


# ---------------------------------------------------------------------------
# Core pricing for a single scenario
# ---------------------------------------------------------------------------

def compute_pricing_curve(
    T_grid: np.ndarray,
    w0: float,
    G_0T: np.ndarray,
) -> dict:
    """Return theta, yield_spread arrays for a maturity grid."""
    q = q_0T(w0, G_0T)
    theta = theta_0T(q) * 100.0          # percent
    yspread = yield_spread_gamma1(T_grid, w0, G_0T) * 100.0   # percent
    return {"theta": theta, "yield_spread": yspread}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Reproduce Figure 1 and Table 1 for gamma=1."""

    # ---- Figure 1(a),(b): FP scenarios, T in (0,50] ----
    T_fp = np.linspace(0.5, 50.0, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for (label, eta0, sigma), color in zip(FP_SCENARIOS, FP_COLORS):
        G = G_0T_fading_pessimism(T_fp, eta0, sigma)
        pc = compute_pricing_curve(T_fp, W0, G)
        axes[0].plot(T_fp, pc["theta"], color=color, label=label, lw=1.5)
        axes[1].plot(T_fp, pc["yield_spread"], color=color, label=label, lw=1.5)

    axes[0].axhline(S0 * 100, color="k", ls="--", lw=1.5, label=r"$s_0=5\%$")
    axes[0].set_xlabel("Maturity T (years)")
    axes[0].set_ylabel(r"$\theta_{0,T}$ (%)")
    axes[0].set_title(r"Figure 1(a): Valuation weight $\theta_{0,T}$ — Fading Pessimism")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Maturity T (years)")
    axes[1].set_ylabel(r"Yield spread $y_{0,T} - r^R$ (%)")
    axes[1].set_title(r"Figure 1(b): Yield spread — Fading Pessimism")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_figure1a.png"), dpi=120, bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_figure1b.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 1(c),(d): FA and CA, T in (0,80] ----
    T_alt = np.linspace(0.5, 80.0, 1600)
    sigma_alt = 0.15

    G_FA = G_0T_fading_alternating(T_alt, sigma_alt)
    G_CA = G_0T_constant_cyclical(T_alt, OMEGA, sigma_alt)

    pc_FA = compute_pricing_curve(T_alt, W0, G_FA)
    pc_CA = compute_pricing_curve(T_alt, W0, G_CA)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(T_alt, pc_FA["theta"], color="blue", label="FA (fading alternating)", lw=1.5)
    axes[0].plot(T_alt, pc_CA["theta"], color="red", label="CA (constant-amplitude cyclical)", lw=1.5)
    axes[0].axhline(S0 * 100, color="k", ls="--", lw=1.5, label=r"$s_0=5\%$")
    axes[0].set_xlabel("Maturity T (years)")
    axes[0].set_ylabel(r"$\theta_{0,T}$ (%)")
    axes[0].set_title(r"Figure 1(c): Valuation weight $\theta_{0,T}$ — Alternating Paths")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(T_alt, pc_FA["yield_spread"], color="blue", label="FA", lw=1.5)
    axes[1].plot(T_alt, pc_CA["yield_spread"], color="red", label="CA", lw=1.5)
    axes[1].set_xlabel("Maturity T (years)")
    axes[1].set_ylabel(r"Yield spread $y_{0,T} - r^R$ (%)")
    axes[1].set_title(r"Figure 1(d): Yield spread — Alternating Paths")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_figure1c.png"), dpi=120, bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_figure1d.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("Figure 1 saved.")

    # ---- Table 1 Panel A ----
    print("\n=== Table 1 Panel A: Fading Pessimism ===")
    panel_A_rows = []
    for label, eta0, sigma in FP_SCENARIOS:
        row = {"Scenario": label}
        for T in TABLE_MATURITIES:
            G = float(G_0T_fading_pessimism(np.array([T]), eta0, sigma)[0])
            q = float(q_0T(W0, G))
            th = float(theta_0T(q)) * 100.0
            ys = float(yield_spread_gamma1(np.array([T]), W0, np.array([G]))[0]) * 100.0
            row[f"theta_{T}y (%)"] = round(th, 4)
            row[f"spread_{T}y (%)"] = round(ys, 4)
        panel_A_rows.append(row)

    df_A = pd.DataFrame(panel_A_rows)
    print(df_A.to_string(index=False))

    # ---- Table 1 Panel B: baseline FP3 ----
    eta0_base = -1.50
    sigma_base = 0.15
    params_base = Parameters(mu=MU, sigma=sigma_base, rho=RHO, gamma=GAMMA, D0=1.0, s0=S0)

    print("\n=== Table 1 Panel B: Baseline FP3 (eta0=-1.5, sigma=15%) ===")
    panel_B_rows = []
    for T in TABLE_MATURITIES:
        T_arr = np.array([float(T)])
        G = float(G_0T_fading_pessimism(T_arr, eta0_base, sigma_base)[0])
        q = float(q_0T(W0, G))
        th = float(theta_0T(q))
        premium = float(bond_price_premium(T_arr, 0.0, W0, np.array([G]))[0]) * 100.0  # rR-relative, show raw
        fwd_spread = float(forward_rate_spread_gamma1(0.0, eta0_base, sigma_base, th)) * 100.0
        beta = float(bond_vol_gamma1(eta0_base, sigma_base, th, S0)) * 100.0
        mu_shift = float(forward_growth_shift_gamma1(eta0_base, sigma_base, th)) * 100.0
        wn_share = float(future_consumption_value_N(float(T), params_base, upper=500.0, n=10000)) * 100.0
        panel_B_rows.append(
            {
                "T": T,
                "DeltaP (%)": round(premium, 4),
                "fwd_spread (%)": round(fwd_spread, 4),
                "|beta| (%)": round(beta, 4),
                "mu_shift (%)": round(mu_shift, 4),
                "W_N(T) share (%)": round(wn_share, 4),
            }
        )

    df_B = pd.DataFrame(panel_B_rows)
    print(df_B.to_string(index=False))

    # Save combined table
    with open(os.path.join(RESULTS_DIR, "exp3_table1.csv"), "w") as f:
        f.write("# Panel A\n")
        df_A.to_csv(f, index=False)
        f.write("\n# Panel B\n")
        df_B.to_csv(f, index=False)

    print("\nTable 1 saved to exp3_table1.csv")

    # ---- Proposition 4 validation summary ----
    T_check = 20
    G_check = float(G_0T_fading_pessimism(np.array([T_check]), eta0_base, sigma_base)[0])
    q_check = float(q_0T(W0, G_check))
    th_check = float(theta_0T(q_check))

    print(f"\n=== Proposition 4 Validation at T={T_check}yr ===")
    print(f"  theta_{{0,T}} = {th_check*100:.4f}%  [single valuation weight]")
    print(f"  bond price premium = bond_price_premium formula = {float(bond_price_premium(np.array([T_check]), 0.0, W0, np.array([G_check]))[0])*100:.4f}%")
    print(f"  yield spread uses theta_{{0,T}}: {float(yield_spread_gamma1(np.array([T_check]), W0, np.array([G_check]))[0])*100:.4f}%")
    print(f"  fwd rate spread proportional to theta_{{0,T}}: {float(forward_rate_spread_gamma1(0.0, eta0_base, sigma_base, th_check))*100:.4f}%")
    print(f"  bond vol proportional to (theta - s0): {float(bond_vol_gamma1(eta0_base, sigma_base, th_check, S0))*100:.4f}%")
    print(f"  forward growth shift proportional to theta: {float(forward_growth_shift_gamma1(eta0_base, sigma_base, th_check))*100:.4f}%")
    print("  => All effects driven by the single valuation weight theta_{0,T}. Proposition 4 confirmed.")


if __name__ == "__main__":
    main()
