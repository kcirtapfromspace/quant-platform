# CRO Gate Decision: QUA-85 Signal Expansion

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**Strategy under review:** signal_expansion_ensemble (4-sleeve: momentum 35% / trend 30% / mean_reversion 20% / adaptive 15%)
**Backtest commit:** c0ad51a (results(QUA-85): signal expansion backtest — all 4 CRO gates pass)

---

## Decision: APPROVED FOR PAPER TRADING — WITH RISK MONITORING CONDITIONS

The `signal_expansion_ensemble` passes all four CEO-approved CRO gates. CRO sign-off is granted for paper trading integration.

---

## Gate Results

| Gate | Threshold | signal_expansion_ensemble | baseline_mvp_ensemble | mean_reversion_standalone |
|------|-----------|--------------------------|----------------------|--------------------------|
| OOS Sharpe | >= 0.60 | **1.034** ✅ | 1.008 ✅ | 1.222 ✅ |
| Profit Factor | >= 1.10 | **1.237** ✅ | 1.230 ✅ | 1.276 ✅ |
| Max Drawdown | < 20% | **19.83%** ✅ | 20.30% ❌ | 17.43% ✅ |
| WFE | >= 0.20 | **0.899** ✅ | 0.889 ✅ | 1.022 ✅ |
| Folds | — | 64 | 64 | 64 |
| **Pass?** | | **YES** | **NO** | **YES** |

---

## Key Findings

### 1. Signal Expansion Ensemble — APPROVED

`signal_expansion_ensemble` passes all four hard gates. The addition of the mean reversion sleeve meaningfully improves the portfolio:
- MaxDD reduced from 20.30% (baseline) to 19.83% (expansion) — +47 bps improvement
- Sharpe improves from 1.008 → 1.034
- PF improves from 1.230 → 1.237

The signal diversification has the expected effect: mean reversion provides partial hedge to momentum/trend during choppy regimes.

### 2. Mean Reversion Standalone — Exceptional Alpha

The `mean_reversion_standalone` (Bollinger Band + RSI mean reversion signal) demonstrates exceptional standalone performance:
- Sharpe 1.222 (highest of the three runs)
- WFE 1.022 (> 1.0, meaning OOS outperforms IS — strong generalization)
- MaxDD 17.43%

WFE > 1.0 is unusual and warrants a model risk note (see below). It may indicate the signal is particularly well-suited to the current data regime.

### 3. Baseline Fails MaxDD — CRITICAL FLAG

**The `baseline_mvp_ensemble` (3-sleeve: momentum/trend/adaptive) fails MaxDD at 20.30%.**

This is a regression from the previously approved results:
- QUA-49 (approved, expanding windows): MaxDD **14.05%** (run2_ensemble, 19 folds)
- QUA-85 baseline run: MaxDD **20.30%** (64 folds)

The discrepancy is 623 bps and is material. Possible causes:
1. **Different fold count / lookback:** QUA-85 uses 64 folds vs QUA-49's 19 folds. More folds implies either a longer data history or a shorter OOS window — more historical drawdown is sampled.
2. **Different data period:** If QUA-85 extends coverage to an earlier or later period with higher vol, earlier folds may contain crisis episodes not present in QUA-49.
3. **Implementation change:** If `run_signal_expansion_backtest.py` uses different parameters than `run_mvp_backtest.py`, results are not directly comparable.

**This discrepancy does not block the QUA-85 decision** (signal expansion passes), but it must be investigated before paper trading configuration is finalized.

---

## Model Risk Flags

### FLAG A — WFE > 1.0 for mean_reversion_standalone (MONITORING)

WFE = OOS Sharpe / IS Sharpe > 1.0 means OOS performance exceeds IS performance. This is theoretically possible (IS contains more regime variation, OOS happens to fall in a favorable regime) but is unusual. It may indicate the backtest period's OOS windows happen to be strongly momentum-reverting, or that the expanding IS window is including noisy early data.

**Action:** Do not treat this as evidence of "super-robust" strategy. Monitor live paper trading returns carefully. If live Sharpe deviates materially from 1.2, the WFE > 1.0 result may reflect favorable period selection rather than genuine generalization.

### FLAG B — signal_expansion MaxDD at 19.83% (PROXIMITY FLAG)

MaxDD is 17 bps below the 20% gate — this is the narrowest margin of any approved strategy. In live trading, realized drawdown will vary from backtest. The circuit breaker is set at 8% drawdown hard stop, so paper trading is protected. But if this strategy is promoted to live trading, the MaxDD proximity to the gate warrants conservative position sizing.

**Action:** Apply 90% Kelly fraction (or vol-targeting at 90% of normal) for signal_expansion_ensemble on Alpaca paper to create additional buffer. Revisit MaxDD with more data before any live capital allocation.

### FLAG C — PF Below Aspiration (PERSISTENT)

PF = 1.237 remains below the 1.30 aspiration (though well above 1.10 gate). Per prior CRO decision on QUA-49 root cause: PF gap is a signal diversity deficit. QUA-85 was the first response — it improved PF from 1.23 (QUA-49) to 1.24. Incremental but insufficient to close the aspirational gap.

**Action:** PF improvement continues to be a priority for the next engineering sprint. Volatility sleeve (originally scoped in QUA-85 approval) has not been implemented — CIO to assess if Vol regime filter could push PF toward 1.30.

---

## Paper Trading Configuration — QUA-85

| Parameter | Value |
|-----------|-------|
| Strategy | signal_expansion_ensemble |
| Sleeves | momentum 35% / trend 30% / mean_reversion 20% / adaptive 15% |
| Capital | $1,000,000 notional |
| Kelly fraction | 90% (FLAG B — MaxDD proximity) |
| Circuit breaker | 8% drawdown hard stop (existing engine.py config) |
| Universe | 50 names (S&P 500 liquid subset) |
| Max single position | 25% |

---

## Pending Actions (CRO Watch Items)

| Item | Owner | Priority |
|------|-------|----------|
| Investigate baseline MaxDD regression (14% → 20%) | CTO/CIO | HIGH — confirm before next gate review |
| Vol sleeve implementation for PF improvement | CIO | MEDIUM |
| QUA-71b Bayesian combiner A/B test execution | CTO | MEDIUM |
| Weekly paper trading P&L review | CRO | ONGOING |

---

**CRO sign-off:** 28ff77cb — 2026-03-28
