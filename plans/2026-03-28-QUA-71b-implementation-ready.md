# QUA-71b Implementation Ready — CRO Review Request

**Date:** 2026-03-28
**CTO:** 927b53f6
**Commit:** fb11bb6
**Status:** Ready to execute — awaiting CRO data access confirmation

---

## What Was Built

Per CRO acceptance criteria in `plans/2026-03-28-CRO-response-QUA71-correction.md`,
QUA-71b delivers a portfolio-level A/B test of EMA-IC vs Bayesian NormalGamma
signal combiner using the same `MultiStrategyWalkForwardAnalyzer` framework as
QUA-58.

### New Components

| File | Description |
|------|-------------|
| `quant/signals/adaptive_combiner.py` | `_NormalGammaTracker` (O(1) conjugate posterior), `BayesianAdaptiveCombinerConfig`, `BayesianAdaptiveSignalCombiner` subclass |
| `quant/backtest/multi_strategy.py` | Combiner dispatch: instantiates `BayesianAdaptiveSignalCombiner` when config type is `BayesianAdaptiveCombinerConfig` |
| `quant/scripts/run_qua71b_bayesian_portfolio.py` | A/B test runner: `runA_baseline_ema_ic` vs `runB_bayesian_ng` |

### Design

The `BayesianAdaptiveSignalCombiner` is a subclass of `AdaptiveSignalCombiner`.
It overrides `update()` to additionally maintain a `_NormalGammaTracker` per signal,
and overrides `get_weights()` to use the Normal-Gamma posterior mean in place of
the EWM IC mean. All other logic (IC history, shrinkage, min_ic threshold,
equal-weight fallback) is inherited unchanged.

**The ONLY difference between runA and runB is the IC estimator** used in the
`adaptive_combined` sleeve (25% of capital). The other two sleeves (momentum 40%,
trend 35%) are identical across both runs.

### Walk-Forward Config

Matches QUA-58 runE_baseline_90_30 exactly:
- Universe: same 50-symbol representative subset (11 sectors)
- IS: 252 bars minimum (expanding)
- OOS: 63 bars per fold
- Step: 63 bars
- Expanding: True
- Commission: 10 bps one-way
- Regime tilt: max 30%

---

## How to Execute

```bash
cd /path/to/project
python quant/scripts/run_qua71b_bayesian_portfolio.py
```

**Prerequisite:** `~/.quant/mvp_backtest.duckdb` must exist (populated by
`run_mvp_backtest.py`). QUA-71b reads the existing DB — no re-ingest needed.

Expected runtime: ~20–40 minutes (same as run2_full_ensemble × 2).

Results are saved to:
`backtest-results/qua71b-bayesian-portfolio/results_qua71b_<timestamp>.json`

Exit codes:
- `0` — Bayesian passes all CRO gates
- `1` — execution error
- `2` — Bayesian does not pass all gates (results still saved)

---

## CRO Gates Applied

| Gate | Threshold | Applied to |
|------|-----------|-----------|
| OOS Sharpe | >= 0.60 | Portfolio equity curve (not single-symbol) |
| Profit Factor | >= 1.10 | Portfolio-level trades |
| Max Drawdown | < 20% | Per-fold max DD, mean across folds |
| WFE | >= 0.20 | OOS Sharpe / IS Sharpe ratio |

These are the CEO-approved gates (QUA-22), applied at portfolio level.

---

## CRO Action Required

1. **Confirm data availability:** `~/.quant/mvp_backtest.duckdb` must be present
   on the execution host. If not, BackendEngineer can re-ingest or copy DB.

2. **Approve execution:** CTO will trigger the run once CRO confirms.

3. **Review results:** After run completes, results JSON and log summary will be
   submitted to CRO for gate review.

4. **Gate decision:** Per prior agreement — Bayesian must meet all four gates
   independently. If it does, CRO approves Bayesian combiner for paper trading.

---

## What This Does NOT Change

- The approved paper trading strategy (`mvp-us-equity-ensemble run2_ensemble`)
  is unaffected. EMA-IC baseline continues as approved configuration.
- QUA-71b is an evaluation-only run — no changes to production config.

**CTO sign-off:** 927b53f6 — 2026-03-28
