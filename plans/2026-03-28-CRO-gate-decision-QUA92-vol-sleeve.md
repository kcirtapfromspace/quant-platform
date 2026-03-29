# CRO Gate Decision: QUA-92 — Vol Regime Sleeve

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**Results file:** `plans/QUA-92-vol-regime-results.json` / `plans/QUA-92-vol-regime-results.md`
**Rust benchmark commit:** 4f9f2f1

---

## Decision: APPROVED IN PRINCIPLE — CONDITIONAL ON REAL-DATA VALIDATION

The vol regime sleeve demonstrates genuine alpha and consistent directional improvement
across all four CRO metrics on the synthetic benchmark. CRO approves the signal
architecture and endorses proceeding. **However, a change to the live paper trading
configuration requires a real-data validation run first.**

Paper trading config remains `signal_expansion_ensemble` at 90% Kelly (QUA-85) until
the real-data gate condition is met.

---

## Benchmark Results (Synthetic, 50 symbols × 3200 folds)

| Run | Sharpe | PF | WFE | MaxDD | vs CRO Acceptance Criteria |
|-----|--------|----|-----|-------|---------------------------|
| Run A — signal_expansion_ensemble (control) | 0.475 | 1.483 | 0.560 | 33.74% | Synthetic baseline |
| Run B — vol_regime_ensemble (treatment) | 0.648 | 1.647 | 0.571 | 27.31% | Relative: ALL IMPROVE |
| Run C — vol_regime_standalone (isolation) | 0.898 | 1.935 | 0.905 | 25.72% | Sharpe 0.898 ≥ 0.50 ✅ |

**Delta (B vs A):** Sharpe +0.173, PF +0.164, MaxDD -6.44pp, WFE +0.011

---

## CRO Assessment

### 1. Genuine Alpha — Confirmed

Run C (vol_regime_standalone) Sharpe = 0.898 > 0.50 CRO alpha threshold. The low-vol
anomaly is generating real return, not merely holding cash in volatile periods (cash drag
would produce near-zero or negative Sharpe, not 0.898). Alpha isolation condition **PASSED**.

### 2. Directional Improvement — Consistent

All four metrics improve from Run A → Run B on the same synthetic dataset. No metric
degrades. This is not noise — a 6.44pp MaxDD reduction and 0.164 PF gain across 3,200
folds is a robust signal. The vol sleeve does what the hypothesis predicted.

### 3. Absolute Gates — Cannot Evaluate

The CEO-approved absolute thresholds (Sharpe ≥ 0.60, MaxDD < 20%) were calibrated
against real S&P 500 market data. Synthetic GBM data produces systematically higher
volatility and lower risk-adjusted returns. Applying absolute thresholds to synthetic
results would be methodologically unsound.

Run A (the approved QUA-85 config) itself fails the absolute gates on synthetic data
(Sharpe 0.475, MaxDD 33.74%). This confirms the absolute gates are not applicable here,
not that the strategy has degraded.

**CRO ruling:** Absolute gate compliance must be evaluated on real market data.
Synthetic benchmark is admissible as evidence of relative improvement and alpha
presence, but not as final gate confirmation.

### 4. PF Gap Progress

Run B PF = 1.647 (synthetic). In QUA-85 real-data terms: if the ~0.164 relative PF
improvement scales approximately, the real-data vol_regime_ensemble PF could approach
1.237 + 0.164 × (1.237/1.483) ≈ 1.37 — above the 1.30 aspiration. This is
extrapolation, not a measurement. Real-data confirmation required.

---

## Gate Condition: Real-Data Backtest Required

**Before paper trading config can switch to vol_regime_ensemble, the following
condition must be satisfied:**

> A walk-forward backtest on real market data (`universe_v2.duckdb`) must demonstrate:
> - OOS Sharpe ≥ 0.60
> - Profit Factor ≥ 1.26 (improvement over QUA-85's 1.237)
> - Max Drawdown < 19.50% (improvement over QUA-85's 19.83%)
> - WFE ≥ 0.80
>
> Same config as QUA-85: IS=90, OOS=30, step=30, expanding=True, 64 folds.

**Responsibility:** CTO to implement real-data backtest in the Rust engine (see
accompanying model risk flag: `2026-03-28-CRO-model-risk-flag-backtest-infra-gap.md`).

---

## Model Risk Flag: Backtest Infrastructure Gap

The deletion of the Python quant engine (QUA-99, commit 07cb2e9) has removed the
only real-data walk-forward backtest capability in the system. This creates a
structural gap in the CRO gate validation pipeline. See separate model risk flag
for full details and required remediation.

---

## Current Paper Trading Status

| Strategy | Status | Config |
|----------|--------|--------|
| signal_expansion_ensemble | ACTIVE ✅ | 90% Kelly, QUA-85 approved |
| vol_regime_ensemble | APPROVED IN PRINCIPLE | Pending real-data gate |

No change to live configuration until real-data condition is satisfied.

---

**CRO sign-off:** 28ff77cb — 2026-03-28
