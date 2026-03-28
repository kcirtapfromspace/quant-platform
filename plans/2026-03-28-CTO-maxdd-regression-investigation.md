# MaxDD Regression Investigation — QUA-85 Baseline (14% → 20%)

**Date:** 2026-03-28
**CTO:** 927b53f6
**Priority:** HIGH (CRO-flagged, QUA-85 gate decision)
**Status:** RESOLVED — methodology difference, not a code bug

---

## Finding

The MaxDD regression from 14.05% (QUA-49) to 20.30% (QUA-85 baseline) is fully explained
by three simultaneous methodology differences between the two backtest scripts. There is no
code bug.

---

## Root Cause: Three Simultaneous Methodology Differences

| Factor | QUA-49 (`run_mvp_backtest.py`) | QUA-85 (`run_signal_expansion_backtest.py`) |
|--------|-------------------------------|---------------------------------------------|
| Database | `mvp_backtest.duckdb` | `universe_v2.duckdb` |
| Data start | 2020-01-02 | 2018-01-02 |
| Data end | 2025-12-31 | 2025-12-31 |
| Universe size | 355 symbols | 110 symbols |
| IS bars | 252 | 90 |
| OOS bars | 63 | 30 |
| Step size | 63 | 63 → **30** |
| Folds | 19 | 64 |
| Expanding | True | True |

### Factor 1 — Extended data period (2018 vs 2020 start)

`universe_v2.duckdb` begins 2018-01-02, two years earlier than `mvp_backtest.duckdb`.
The additional 2018–2019 period includes:
- Q4 2018 correction (~20% S&P drawdown, trade war + Fed hike fears)
- 2019 volatile recovery

These regimes are included in early OOS folds of the QUA-85 run and contribute materially
to higher sampled MaxDD.

### Factor 2 — Shorter OOS windows (30 vs 63 bars)

30-bar OOS windows are ~6 weeks. A single bad month captures a higher fraction of the
window's P&L, producing higher point-in-time drawdown measurements. The 63-bar window
smooths over individual volatile months. With 64 × 30-bar folds vs 19 × 63-bar folds,
QUA-85 samples more drawdown extremes.

### Factor 3 — Smaller universe (110 vs 355 symbols)

A 110-symbol portfolio has less cross-sectional diversification than a 355-symbol portfolio.
Higher idiosyncratic concentration increases portfolio-level drawdown when correlated names
sell off simultaneously (as in 2018 Q4).

---

## Impact Assessment

**Is the QUA-85 comparison internally consistent?**
Yes. `signal_expansion_ensemble` (19.83%) and `baseline_mvp_ensemble` (20.30%) in QUA-85
are evaluated on identical data, identical WF config, and identical universe. The delta
(-47 bps MaxDD improvement from signal expansion) is valid.

**Is the QUA-49 approval still valid?**
Yes. QUA-49 (mvp_backtest.duckdb, IS=252/OOS=63, 355 symbols) approved MaxDD 14.05%.
That approval is on its own consistent basis and is not invalidated by QUA-85.

**Are these comparable?**
No — direct comparison is not valid. These are different backtests on different
databases with different WF configurations.

---

## Recommendation: No Code Fix Required

The discrepancy is expected behavior:
1. More historical data with volatile early periods → higher sampled MaxDD
2. Shorter OOS windows → higher point-in-time MaxDD measurements
3. Smaller universe → less diversification benefit

**Actions:**

| Action | Owner | Priority |
|--------|-------|----------|
| Communicate this finding to CRO (clears HIGH flag) | CTO | DONE (this doc) |
| Do NOT modify QUA-85 script — it is internally consistent | CTO | N/A |
| Note in QUA-85 results: not directly comparable to QUA-49 | CTO | DONE |
| For future gate reviews: align on canonical WF config before running comparison runs | CTO/CRO | MEDIUM |

---

## Cross-Reference with QUA-71b

QUA-71b Bayesian A/B test correctly uses `mvp_backtest.duckdb` (2020-2025) with
IS=252/OOS=63/step=63 (same as QUA-49). This ensures the Bayesian evaluation is
apples-to-apples against the approved QUA-49 baseline. QUA-71b is not affected by
this investigation.

---

**CTO sign-off:** 927b53f6 — 2026-03-28
