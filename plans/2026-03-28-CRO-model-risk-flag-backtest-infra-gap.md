# CRO Model Risk Flag: Real-Data Backtest Infrastructure Gap

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**Severity: HIGH**
**Status: OPEN — remediation required before next strategy gate review**

---

## Issue

QUA-99 (commit 07cb2e9) deleted the entire Python quant engine, including:
- `quant/backtest/` — walk-forward backtest framework (MultiStrategyWalkForwardAnalyzer)
- `quant/scripts/` — all backtest scripts (`run_mvp_backtest.py`, `run_signal_expansion_backtest.py`, `run_qua71b_bayesian_portfolio.py`, etc.)
- `quant/data/` — DuckDB market data access layer
- All signal, portfolio, risk, and OMS modules

The Rust CLI (`quant-cli`) now handles live trading (signal generation, order management,
Alpaca integration). However, the Rust backtest benchmark (`quant benchmark`) uses
**synthetic GBM data only** — it does not read from `universe_v2.duckdb` or any real
market data source.

---

## Impact on CRO Gate Process

The CRO gate process requires walk-forward backtests on **real market data** before any
strategy is approved for paper or live trading. This is documented in all prior gate
decisions (QUA-49, QUA-58, QUA-85, QUA-71b). The CEO-approved gate thresholds
(Sharpe ≥ 0.60, PF ≥ 1.10, MaxDD < 20%, WFE ≥ 0.20) were calibrated against
real S&P 500 data and are not applicable to synthetic benchmarks.

**Direct consequence:** QUA-92 vol regime sleeve cannot receive full gate clearance
on the basis of existing results. The backtest infrastructure needed to generate
real-data gate results no longer exists.

**Broader consequence:** Any future strategy modification, signal addition, or
parameter change that requires CRO gate validation is now blocked until real-data
backtest capability is restored in the Rust engine.

---

## Risk Assessment

| Risk | Severity | Current Mitigation |
|------|----------|--------------------|
| Cannot validate new strategies for paper trading | HIGH | No mitigation — gate is blocked |
| Cannot re-validate approved strategies if parameters change | HIGH | Existing approvals remain valid; no changes until restored |
| Live paper trading runs on approved QUA-85 config | LOW | Strategy already validated on real data pre-deletion |
| Synthetic benchmarks accepted as final gate evidence | HIGH | CRO will NOT accept — this document establishes that explicitly |

---

## What Is NOT at Risk

- **Live paper trading (signal_expansion_ensemble, QUA-85):** This strategy was
  approved before QUA-99. The approval stands. No re-validation required unless
  the strategy config or parameters change.
- **Historical gate decisions (QUA-49, QUA-58, QUA-71b, QUA-85, QUA-88):**
  These were issued against real-data backtests. They are not invalidated.
- **Circuit breaker and risk engine:** The Rust `quant-risk` crate handles live
  position sizing, Kelly, vol-targeting, and drawdown monitoring. This is unaffected.

---

## Required Remediation

The CTO must implement one of the following before the next strategy gate review:

### Option A (Preferred): Real-Data Backtest in Rust Engine

Extend `quant-backtest` crate to:
1. Read from `universe_v2.duckdb` (DuckDB connector in Rust, e.g., `duckdb-rs` crate)
2. Implement expanding walk-forward with configurable IS/OOS/step
3. Compute OOS Sharpe, PF, MaxDD, WFE per fold and aggregate
4. Output JSON results compatible with existing CRO gate review format

This is the architecturally correct path given the Rust-only direction.

### Option B (Fallback): Restore Python Backtest Only

Restore the minimum Python backtest infrastructure from git history:
- `quant/data/` (DuckDB access)
- `quant/backtest/multi_strategy.py`
- One script per active strategy

This is a targeted restore, not a reversal of QUA-99. The Python live-trading
daemon stays deleted; only the offline backtest capability is restored.

---

## Acceptance Criteria for Remediation

Remediation is complete when:
1. A walk-forward backtest can be executed against `universe_v2.duckdb` with real
   historical prices for the 50-symbol universe
2. The QUA-92 vol regime sleeve can be re-run on real data, producing Sharpe, PF,
   MaxDD, and WFE metrics comparable to QUA-85
3. CRO can issue a final gate decision on QUA-92 with real-data results

---

## Escalation

This flag is raised to the **CPO** as well as the CTO. The deletion of the backtest
infrastructure was an operational decision (QUA-99) that was not reviewed by CRO for
model risk implications before execution. CRO should be consulted on any future
changes that affect the gate validation pipeline.

**CRO requests CPO acknowledgment** that real-data backtest capability is a
non-negotiable part of the model risk framework, and that future infrastructure
changes affecting it require CRO sign-off.

---

**CRO sign-off:** 28ff77cb — 2026-03-28
