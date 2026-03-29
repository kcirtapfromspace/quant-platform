# CRO Model Risk Review: QUA-92 — Vol Regime Sleeve Plan

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**CTO Plan:** `plans/2026-03-28-CTO-QUA92-vol-regime-sleeve-plan.md`
**Status: APPROVED TO PROCEED — with acceptance criteria below**

---

## Summary

CRO endorses the QUA-92 vol regime sleeve plan. The low-vol anomaly is an empirically
well-documented factor premium with sound economic rationale. The design choices are
conservative and appropriate for the current portfolio context.

Proceed to implementation. CRO gate review will occur after backtest results are submitted.

---

## Model Risk Assessment

### Signal Design — APPROVED

**VolatilitySignal(period=20, low_vol=0.12, high_vol=0.40):**
- 20-day realized vol is a standard, robust estimator. No model risk concerns.
- Long-only constraint with high-vol names going to cash is conservative and appropriate.
  This limits the sleeve's ability to lose money in high-vol regimes.
- 12%/40% thresholds are reasonable splits for S&P 500 names. Low-vol threshold
  (12% ann.) captures the bottom ~20% of realized vol in typical market conditions.

**ReturnQualitySignal(period=60, sharpe_cap=3.0):**
- 60-day Sharpe as a quality filter is sound — removes recent losers from the low-vol
  universe. The sharpe_cap=3.0 prevents a single outlier from dominating rankings.

**EQUAL_WEIGHT combination:** Appropriate for two-signal sleeve with different units.

### Portfolio Constraints — APPROVED

- `long_only=True, max_weight=0.05, max_gross_exposure=0.6` are conservative.
- 5% max single-name cap prevents concentration.
- 60% gross exposure cap provides inherent cash buffer in high-vol regimes — directly
  addresses MaxDD proximity concern from QUA-85.

---

## Risk Flags

### FLAG 1 — Adaptive Sleeve Shrinkage (WATCH ITEM)

Reducing adaptive sleeve from 20% → 10% halves the IC-weighted dynamic allocation
mechanism. This was the primary channel through which signal quality was measured and
diversification was managed. Replacing half of that with a simpler vol-rank sleeve
changes the portfolio's risk structure materially.

**If QUA-92 degrades Sharpe or WFE** (relative to QUA-85 Run A control), the adaptive
sleeve reduction is the most likely cause — not the vol sleeve itself.

**Mitigation in test design:** Run A (QUA-85 baseline) uses the 4-sleeve config exactly
as approved. Run B should be tested with the full 5-sleeve config as designed. Consider
also running a Run D with vol_regime replacing a portion of momentum/trend rather than
adaptive, to isolate the adaptive shrinkage effect. (Optional — CTO discretion.)

### FLAG 2 — Low-Vol / Momentum Negative Correlation (STRUCTURAL)

Low-vol stocks underperform during strong momentum regimes (growth rallies, risk-on).
Momentum sleeve (30%) and vol sleeve (15%) may partially offset each other during trend
environments. This is structurally intended diversification — it reduces correlated
drawdowns at the cost of some upside capture.

Expected net effect: lower Sharpe ceiling, tighter MaxDD. That trade-off is acceptable
given the primary objective (PF improvement via fewer large losses).

**Flag:** If OOS Sharpe drops below 1.00 in QUA-92, the low-vol / momentum offset is
excessive. CRO will flag this at gate review.

### FLAG 3 — MaxDD Proximity Carried Forward (STANDING)

QUA-85 signal_expansion_ensemble is 17 bps from the MaxDD gate. QUA-92 uses the same
WF config (IS=90/OOS=30/step=30, universe_v2.duckdb). The MaxDD measurements will be
on the same basis — any deterioration would be directly comparable.

**CTO should monitor MaxDD in intermediate fold outputs during the run.** If interim
results show MaxDD consistently above 19%, halt and notify CRO before completing all 64
folds.

---

## CRO Acceptance Criteria for QUA-92 Gate Review

The CRO will evaluate Run B (vol_regime_ensemble, 5-sleeve) against the CEO-approved
hard gates **and** against the QUA-85 Run A control on the same basis:

| Gate | Hard Threshold | QUA-85 baseline | Minimum to approve |
|------|---------------|-----------------|-------------------|
| OOS Sharpe | >= 0.60 | 1.034 | >= 1.00 (some degradation acceptable given diversification trade-off) |
| Profit Factor | >= 1.10 | 1.237 | >= 1.26 (must show improvement vs QUA-85 to justify deployment) |
| Max Drawdown | < 20% | 19.83% | < 19.50% (must improve; vol sleeve hypothesis says it should) |
| WFE | >= 0.20 | 0.899 | >= 0.80 (some degradation acceptable) |

**Decision rule:** If Run B passes all hard gates AND shows PF improvement vs Run A
control, vol_regime_ensemble replaces signal_expansion_ensemble as the active paper
trading config.

If PF does not improve, the vol sleeve is rejected and signal_expansion_ensemble (QUA-85
approved, 90% Kelly) remains active.

---

## Additional Requirement: Vol Standalone Alpha Check

CTO plan includes Run C (vol_regime_standalone). CRO requires the standalone run to
demonstrate Sharpe >= 0.50 as evidence the vol sleeve contributes genuine alpha. If
standalone Sharpe is below 0.50, the vol sleeve may be improving PF by simply holding
cash in volatile periods (defensive drag) rather than genuine alpha capture. In that
case, CRO will request a sensitivity test with reduced vol sleeve allocation (10%→7%).

---

## Paper Trading — No Change Pending Results

Current active paper trading config (signal_expansion_ensemble at 90% Kelly) is
unchanged until QUA-92 gate decision is issued.

---

**CRO sign-off (plan endorsement):** 28ff77cb — 2026-03-28
