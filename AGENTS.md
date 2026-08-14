# AGENTS.md — Arrow-Debreu Equilibrium Project

## Project Overview

Research project studying Arrow-Debreu equilibrium prices in a two-investor economy
with time-varying, deterministic belief distortions (dogmatist vs rationalist).  All
research code lives in `src/`; experiment scripts go in `exp/`; generated artefacts
go in `results/`.

---

## Module Map

| File | Purpose |
|------|---------|
| `src/model.py` | Core: `Parameters` dataclass, simulation (`simulate_brownian`), equilibrium objects (SDF, wealth ratio, consumption shares, benchmark martingales) |
| `src/paths.py` | Deterministic η-path functions and their integrals (exact and numerical) |
| `src/factorization.py` | Hansen-Scheinkman SDF factorization and numerical identity checks |
| `src/section4.py` | Closed-form γ = 1 bond-pricing formulas |
| `src/long_run.py` | Long-run yield/forward-rate analysis and forward-density process |
| `src/approximation.py` | General-γ bond-price approximation and non-asymptotic error bounds |

---

## Key Mathematical Conventions

### Sign / normalisation
- `log_D` everywhere means **log(D_t / D_0)**, not log(D_t).  Row 0 of every
  `(n_steps+1, n_paths)` array is 0 at t=0.
- `log_pi[0]` is identically 0 (π_0 = 1 by normalisation).
- `log_xi[0]` is identically 0 (ξ_0 = 1).

### Array shapes
All simulation outputs are `(n_steps+1, n_paths)` — time index first, paths second.
`t_grid` has shape `(n_steps+1,)`.  Deterministic arrays (η, A, H) have shape
`(n_steps+1,)` for the full grid or `(n_steps,)` for left-endpoint values.

### η left-endpoint convention
`log_xi`, `benchmark_martingale_N`, and `benchmark_martingale_R` all take
`eta_vals` of shape `(n_steps,)` representing η at **left endpoints**
`t_grid[:-1]`.  Pass `path['eta'][:-1]` when using `compute_path_objects` output.

---

## Numerical Consistency Critical Point

**Quadrature mismatch**: the stochastic integrals in `log_xi` and
`benchmark_martingale_N` use **left-endpoint Riemann sums** internally.
The path integrals `H_t`, `A_t` in `compute_path_objects` use the **trapezoid
rule** (more accurate for general use).  These are O(Δt) apart.

Consequence:
- Passing trapezoid-rule H_t to `identity_check_MR_wgamma` or
  `sdf_factorization_check` gives residuals ~O(Δt) (e.g. 7e-6 for dt=0.05).
- Passing left-endpoint H_t from `H_left_riemann(eta_left, dt)` gives
  machine-precision residuals (~4e-16).

**Always use `H_left_riemann` / `A_left_riemann` for identity/factorization checks.**

---

## Rational Yield Formula

    r_R = ρ + γ μ − ½ γ(γ+1) σ²

For the default parameters (μ=0.02, σ=0.02, ρ=0.01, γ=1): r_R = 0.0296.

---

## Section 4 (γ = 1) Bond Price Derivation Sketch

Under P, E^P[w_t] = w_0 for all t (w_t = w_0 ξ_t is a martingale).  This gives:

    P(0,T) = exp(−r_R T) · (1 + q_{0,T}) / (1 + w_0)
    q_{0,T} = w_0 exp(−σ² H_{0,T})

Yield spread = [log(1+w_0) − log(1+q_{0,T})] / T.
A pessimistic dogmatist (η < 0, H_{0,T} < 0) pushes q > w_0, raising bond prices
and *lowering* yields.

---

## Long-Run Yield

When lim_{T→∞} H_T/T = H̄ exists:

    r_∞ = r_R + γ σ² min(H̄, 0)

Paths with H̄ > 0 (on-average optimism) have r_∞ = r_R.
Paths with H̄ < 0 (on-average pessimism) have r_∞ < r_R.

---

## Error Bound (General γ)

    e_{t,T} = K_γ · exp(−γ / (2(1+γ)²) · σ² · A_{t,T})

where K_γ = 2 C_γ and C_γ = sup_{x>0} |R_γ(x)| / [x^{γ/(1+γ)} + x^{γ²/(1+γ)}].
For γ = 2: C_2 = 1 exactly (achieved at x = 1, R_2(1) = 2, denominator = 2).

---

## Dependencies

Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7.

Install: `pip install numpy scipy matplotlib`

---

## Testing

Run the full test suite with `python -m pytest tests/ -v`.
Quick smoke test: `python -c "import src.model; print('OK')"`.
