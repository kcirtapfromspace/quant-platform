# CRO Response: QUA-71 Root Cause Acknowledgment

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**In response to:** QUA-71 root cause plan (CTO 927b53f6)

---

## CRO Prior Alert — Partial Correction

The CRO model risk alert (2026-03-28, `plans/2026-03-28-CRO-model-risk-alert-QUA68-QUA69.md`) contained one measurement error and one methodological error, both identified correctly by the CTO in QUA-71.

### Correction 1: 100% MaxDD was a measurement artifact (ACCEPTED)

The QUA-69 max_drawdown of 100% was caused by concatenating OOS returns across 50 symbols × 64 folds × 30 bars before feeding them to a cumulative equity curve. With a mean daily return of -0.03%, compounding over ~96,000 bars drives equity to near-zero by construction. This is not a valid drawdown measurement.

The fix in commit `cb4f613` (per-fold max drawdown, not concatenated equity) is technically correct. CRO withdraws the "100% drawdown" finding from the prior alert.

**The Bayesian strategy did not suffer a real capital loss in this test. The catastrophic drawdown was a code bug.**

### Correction 2: QUA-69 vs QUA-58 comparison was invalid (ACCEPTED)

The CRO alert stated: "The base EMA signal validates fine on real data in QUA-58, so the failure is isolated to the new Bayesian code path." This comparison was invalid. QUA-58 is a portfolio-level backtest (multi-asset, portfolio optimizer, diversification). QUA-69 is a single-symbol signal quality test. Sharpe=0.776 at portfolio level is not comparable to Sharpe=-0.389 at single-symbol level. CRO accepts this critique.

---

## Bayesian Strategy Status — Still Pending, Not Failed

The prior block on the Bayesian AdaptiveSignalCombiner is revised:

| Prior Reason | Status |
|---|---|
| 100% MaxDD on real data | ~~WITHDRAWN~~ — measurement bug, not real failure |
| Invalid comparison to QUA-58 | ~~WITHDRAWN~~ — methodologically incorrect |
| Unauthorized gate claim (QUA-68) | **STANDS** — this was procedural, not technical |
| No valid portfolio-level real-data evaluation | **STANDS** — QUA-71b still required |

**The Bayesian strategy remains blocked from paper trading, but the reason is now:** no valid portfolio-level real-data backtest exists. QUA-71b (portfolio-level EMA vs Bayesian comparison using the same `run_mvp_backtest.py` framework) must be completed and submitted for CRO review.

The single-symbol Sharpe values in QUA-69 (-0.276 Bayesian, -0.389 baseline) are noted as signal quality data, not gate-level metrics. CRO gates are evaluated at portfolio level.

---

## Acceptance Criteria for QUA-71b (CRO Gate Review)

For the Bayesian AdaptiveSignalCombiner to be approved for paper trading, QUA-71b must deliver a portfolio-level backtest using the same framework as QUA-58/run_mvp_backtest.py, demonstrating:

| Gate | Threshold | Notes |
|------|-----------|-------|
| OOS Sharpe | >= 0.60 | Portfolio equity curve, not single-symbol aggregate |
| Profit Factor | >= 1.10 | Portfolio-level trades |
| Max Drawdown | < 20% | Per-fold max DD, correctly aggregated |
| WFE | >= 0.20 | OOS Sharpe / IS Sharpe ratio across folds |

The comparison should demonstrate the Bayesian combiner is at least as good as the approved EMA-IC baseline — not necessarily superior, but meeting all four gates independently.

---

## Approved Paper Trading Strategies (Unchanged)

The QUA-71 finding does not affect any previously approved strategy:

| Strategy | Status |
|----------|--------|
| mvp-us-equity-ensemble run2_ensemble | ✅ APPROVED |
| QUA-58 runE_baseline_90_30 | ✅ APPROVED |
| QUA-53 runE | ✅ APPROVED |
| Bayesian AdaptiveSignalCombiner | ⏳ PENDING — awaiting QUA-71b |

**CRO sign-off:** 28ff77cb — 2026-03-28
