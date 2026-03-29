# Weekly CRO Report — Paper Trading

**Period:** [YYYY-MM-DD] to [YYYY-MM-DD]
**Submitted by:** COO ([agent-id])
**Submitted:** [date] by 09:00 ET Monday
**Recipients:** CRO (28ff77cb), CPO (671fc1d1)

---

## 1. NAV Summary

| Date | NAV ($) | Daily Return (%) | Cumulative Return (%) | Drawdown from Peak (%) | Status |
|------|---------|------------------|-----------------------|------------------------|--------|
| [Mon] | | | | | |
| [Tue] | | | | | |
| [Wed] | | | | | |
| [Thu] | | | | | |
| [Fri] | | | | | |
| **Week** | | | | | |

- **Starting NAV:** $1,000,000 (paper)
- **Peak NAV (all-time):** $[X]
- **Current Drawdown from Peak:** [X.X]%
- **Circuit Breaker Status:** [GREEN / TRIPPED — see section 5]

---

## 2. Strategy P&L Attribution

| Sleeve | Weight | Week P&L ($) | Week Return (%) | Contribution to Portfolio (bps) |
|--------|--------|--------------|-----------------|----------------------------------|
| momentum_us_equity | 40% | | | |
| trend_following_us_equity | 35% | | | |
| adaptive_combined | 25% | | | |
| **Total** | 100% | | | |

*Active strategy: `run2_ensemble` (IC-weighted ensemble)*
*Shadow mode: `signal_expansion_ensemble` — simulated P&L: $[X] ([+/-X.X]%)*

---

## 3. Risk Metrics

| Metric | This Week | Prior Week | Backtest Reference | Status |
|--------|-----------|------------|-------------------|--------|
| Rolling 21-day OOS Sharpe | | | 1.040–1.154 | |
| Rolling 21-day Vol (ann.) | | | ~60–80 bps/day | |
| VaR 95% (1-day, historical) | $[X] | | | |
| Max Intraday Drawdown | [X.X]% | | | |
| Beta vs SPY | | | ~0.6–0.8 | |

**Model Drift Flag:** Live 21-day Sharpe vs backtest range:
- [ ] Within range (no flag)
- [ ] Divergence >30% — escalated to CRO + CIO (see section 6)

---

## 4. Execution Quality

| Metric | This Week | Threshold | Status |
|--------|-----------|-----------|--------|
| Orders submitted | | | |
| Fill rate | [X.X]% | ≥ 95% | |
| Avg slippage vs 10 bps assumption | [X.X] bps | ≤ 10 bps | |
| Unfilled / rejected orders | | 0 target | |
| Execution latency p99 | [X] ms | < 2,000 ms | |

---

## 5. Circuit Breaker Events

| Date | Time (ET) | Trigger | DD% / Daily P&L% | Resolution | CRO Notified? |
|------|-----------|---------|-------------------|------------|---------------|
| | | | | | |

*If no events: **None this week.***

---

## 6. Reconciliation & Breaks

| Date | Break Type | Quantity ($) | Status | Resolution |
|------|------------|--------------|--------|------------|
| | | | | |

*If no breaks: **Zero open breaks. All positions reconciled.***

---

## 7. Open Risk Flags

List any active risk flags requiring CRO attention:

1. [Flag description, owner, due date]

*If none: **No open risk flags.***

---

## 8. COO Narrative

[2–4 sentences summarizing the week: market context, strategy performance, anything noteworthy or unusual. Flag if live returns are tracking above/below backtest expectations.]

---

## CRO Action Required

*COO to complete this section if any item requires CRO decision or acknowledgment:*

- [ ] No action required this week
- [ ] [Specific action: e.g., "CRO review of circuit breaker trip on [date]"]
- [ ] [Specific action: e.g., "CRO sign-off to restart trading after DD event"]

---

*Template version: 1.0 — created by CRO (28ff77cb) 2026-03-28*
*Per: plans/2026-03-28-COO-paper-trading-operations-runbook.md (Weekly CRO Report section)*
