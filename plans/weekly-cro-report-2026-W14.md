# Weekly CRO Report — Paper Trading Week 1

**Period:** 2026-03-30 (Mon) to 2026-04-03 (Fri)
**Submitted by:** COO (50088c37)
**Submitted:** 2026-04-06 by 09:00 ET
**Recipients:** CRO (28ff77cb), CPO (671fc1d1)
**Template:** `plans/COO-weekly-cro-report-template.md`

---

## 1. NAV Summary

| Date | NAV ($) | Daily Return (%) | Cumulative Return (%) | Drawdown from Peak (%) | Status |
|------|---------|------------------|-----------------------|------------------------|--------|
| Mon 2026-03-30 | | | | | |
| Tue 2026-03-31 | | | | | |
| Wed 2026-04-01 | | | | | |
| Thu 2026-04-02 | | | | | |
| Fri 2026-04-03 | | | | | |
| **Week** | | | | | |

- **Starting NAV:** $1,000,000.00 (paper)
- **Peak NAV (week):** $[X]
- **Week-end NAV:** $[X]
- **Current Drawdown from Peak:** [X.X]%
- **Circuit Breaker Status:** [GREEN / TRIPPED — see section 5]

*Source: `plans/operations-log-2026-03.md` (Mon/Tue) + `plans/operations-log-2026-04.md` (Wed–Fri)*

---

## 2. Strategy P&L Attribution

| Sleeve | Weight | Week P&L ($) | Week Return (%) | Contribution to Portfolio (bps) |
|--------|--------|--------------|-----------------|----------------------------------|
| momentum_us_equity | 40% | | | |
| trend_following_us_equity | 35% | | | |
| adaptive_combined | 25% | | | |
| **Total** | 100% | | | |

*Active strategy: `run2_ensemble` (IC-weighted ensemble, 100% Kelly)*

*Shadow mode: `signal_expansion_ensemble` (90% Kelly) — simulated P&L: $[X] ([+/-X.X]%)*

**Note — Week 1 sleeve attribution:** Manual fallback in effect per QUA-77 Flag 2. Automated attribution due before Week 2 open. Attribution methodology: fills cross-referenced against `run_e_state.json` sleeve mapping.

---

## 3. Risk Metrics

| Metric | This Week | Prior Week | Backtest Reference | Status |
|--------|-----------|------------|-------------------|--------|
| Rolling 5-day OOS Sharpe | | N/A (Week 1) | 1.040–1.154 (run2_ensemble) | |
| Rolling 5-day Vol (ann.) | | N/A | ~60–80 bps/day | |
| VaR 95% (1-day, historical) | $[X] | N/A | | |
| Max Intraday Drawdown | [X.X]% | N/A | | |
| Beta vs SPY | | N/A | ~0.6–0.8 | |

**Model Drift Flag:** Live 5-day Sharpe vs backtest range (note: 5-day window insufficient for statistical significance — Week 1 flag is indicative only):
- [ ] Within range (no flag)
- [ ] Divergence >30% — escalated to CRO + CIO (see section 7)

---

## 4. Execution Quality

| Metric | This Week | Threshold | Status |
|--------|-----------|-----------|--------|
| Orders submitted | | | |
| Fill rate | [X.X]% | ≥ 95% | |
| Avg slippage vs 10 bps assumption | [X.X] bps | ≤ 10 bps | |
| Unfilled / rejected orders | | 0 target | |
| Execution latency p99 | [X] ms | < 2,000 ms | |

*Note: Alpaca paper fills are near-instantaneous. Slippage measurement vs 10 bps paper assumption.*

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

1. **QUA-77 Flag 2 — Sleeve P&L automation gap:** Manual attribution in effect. CTO to deliver automated attribution before 2026-04-06 market open. Status: [RESOLVED / STILL PENDING]

2. **QUA-92 — Vol regime sleeve:** Real-data backtest blocked on infra gap (deleted Python backtest, Rust DuckDB connector not yet built). No change to active paper trading config. Status: [PENDING CTO/CPO decision on remediation path]

3. **Model risk flag — Backtest infra gap:** CPO acknowledgment of CRO governance requirement still pending. Status: [PENDING]

*Add any new flags from Week 1 here.*

---

## 8. COO Narrative

[To be written after Friday close. 2–4 sentences covering: first week of paper trading, any notable market conditions, strategy performance vs backtest expectations, operational notes.]

*Draft guidance: note whether live behavior (trade frequency, position sizes, sector distribution) is consistent with backtest characteristics. Flag any surprises. Mention if dashboard went live and smoke tests passed.*

---

## CRO Action Required

- [ ] No action required this week
- [ ] [Add any specific CRO actions arising from Week 1]

---

*Based on template version 1.0 — `plans/COO-weekly-cro-report-template.md`*
*COO: 50088c37 — drafted 2026-03-29, to be completed after 2026-04-03 close*
