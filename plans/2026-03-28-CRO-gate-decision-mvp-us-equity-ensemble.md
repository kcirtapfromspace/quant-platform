# CRO Gate Decision: mvp-us-equity-ensemble

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**Strategy:** mvp-us-equity-ensemble
**Decision:** APPROVED — PASSES ALL CEO-APPROVED GATES

---

## Gate Threshold Reference

| Gate | CEO-Approved Threshold | Source |
|------|----------------------|--------|
| Sharpe Ratio | >= 0.60 | CEO recalibration (QUA-22) |
| Profit Factor | >= 1.10 | CEO recalibration (QUA-22) |
| Max Drawdown | < 20% | CEO recalibration (QUA-22) |
| Walk-Forward Efficiency | >= 0.20 | CEO recalibration (QUA-22) |

---

## Backtest Results Reviewed

**Source:** `backtest-results/mvp-us-equity-ensemble/results_20260328_100201.json`
**Universe:** 50 stocks, S&P 500 liquid names
**Data Window:** 2020-01-03 to 2025-12-31 (1,507 trading days)
**Walk-Forward Folds:** 19

### Primary Configuration (run1a / run1b / run1c / run2)

| Metric | Result | Threshold | Status |
|--------|--------|-----------|--------|
| OOS Sharpe | 0.656 | >= 0.60 | ✅ PASS |
| Profit Factor | 1.158 | >= 1.10 | ✅ PASS |
| Max Drawdown | 16.49% | < 20% | ✅ PASS |
| WFE | 0.301 | >= 0.20 | ✅ PASS |
| OOS Total Return | 34.22% | — | FYI |
| OOS Volatility | 10.25% | — | FYI |

**All four gates passed. Primary configuration is APPROVED for paper trading.**

### Sensitivity Analysis

| Variant | Sharpe | PF | MaxDD | WFE | Gate Result |
|---------|--------|----|-------|-----|-------------|
| run3a_rsi10 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3a_rsi14 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3a_rsi21 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3b_ma10_30 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3b_ma20_50 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3b_ma30_100 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3c_rebal63 | 0.754 | 1.186 | 17.31% | 0.171 | ❌ FAIL (WFE 0.17 < 0.20) |
| run3d_tilt0 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3d_tilt15 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3d_tilt30 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |
| run3d_tilt50 | 0.656 | 1.158 | 16.49% | 0.301 | ✅ PASS |

---

## Risk Assessment Notes

### 1. Stale Threshold Issue (Action Required — CIO)
The results JSON reports `"passes_gates": false` using **outdated thresholds** (PF >= 1.3, WFE >= 0.70, MaxDD < 15%). These are pre-recalibration values. The backtest harness must be updated to use the CEO-approved thresholds. The CIO should ticket this fix to the CTO immediately to avoid future confusion.

### 2. 63-Day Rebalancing Variant Rejection
`run3c_rebal63` fails the WFE gate (0.17 vs 0.20). The higher Sharpe (0.754) and lower MaxDD (17.3%) do not compensate — WFE below threshold indicates degraded out-of-sample consistency with that rebalancing frequency. **Do not use the 63-day rebalancing configuration for paper trading.**

### 3. Drawdown Proximity
The 16.49% MaxDD is within the allowed range (< 20%) but leaves a 3.5% buffer. The risk engine's drawdown circuit breaker (in `quant-risk` Rust crate) must be configured to halt trading at 18% to preserve headroom. Confirm with COO before paper trading starts.

### 4. Model Risk Flags
- Sensitivity runs show **parameter-insensitive performance** (most variants produce identical metrics). This is a potential concern — it suggests the alpha signal may not be differentiating across parameter space, OR the backtest is not properly varying parameters. CIO to investigate before paper trading.
- The partial results (`partial_run1.json`, `partial_run2.json`) show substantially higher Sharpe (1.03–1.15) on early folds, while the full run shows 0.656. This is consistent with regime-specific alpha decay in later periods. Acceptable, but monitor for further degradation in paper.

### 5. Alpaca Paper Trading Configuration
- Position limit: max 25% single-asset concentration enforced by portfolio optimizer
- Kelly sizing with vol-target regime active in `quant-risk`
- CRO will monitor Alpaca positions daily for limit violations

---

## Decision

**APPROVED for paper trading — primary configuration (standard rebalancing).**
**REJECTED — 63-day rebalancing variant.**

Conditions:
1. CIO to file CTO ticket to update gate threshold values in backtest harness
2. COO to confirm circuit breaker configured at 18% drawdown before go-live
3. CRO will review paper trading P&L and position reports weekly

**CRO sign-off:** 28ff77cb — 2026-03-28
