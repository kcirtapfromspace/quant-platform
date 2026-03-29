# Circuit Breaker Response Playbook

**Author:** COO (50088c37)
**Date:** 2026-03-28
**Last updated:** 2026-03-28 (session 3 — CRO-required daily P&L reset clarification per QUA-79 sign-off)
**Scope:** Alpaca paper trading — drawdown and daily P&L halt events
**Status:** ACTIVE — CRO-APPROVED (QUA-79)

---

## Circuit Breaker Architecture

Two independent triggers halt all new order submission:

| Trigger | Threshold | Source |
|---------|-----------|--------|
| Portfolio drawdown (from peak) | ≥ 8% | `QUANT_DD_CIRCUIT_BREAKER=0.08` in `.env` |
| Daily P&L | ≤ -3% | `QUANT_DAILY_PNL_HALT=-0.03` in `.env` |

Both are enforced by `quant.risk.circuit_breaker.DrawdownCircuitBreaker` and the risk engine pre-order checks in `quant/risk/engine.py`. When tripped, the engine returns `approved=False` for all orders.

---

## Drawdown Monitoring Levels

| Level | Threshold | Status |
|-------|-----------|--------|
| Circuit Breaker (hard stop) | DD ≥ 8% OR daily P&L ≤ -3% | **HALT** |
| Yellow | DD ≥ 10% | Monitor — CRO brief |
| Orange | DD ≥ 15% | Urgent — COO + CRO + CPO call |
| Red | DD ≥ 20% | Strategy suspended |

Note: Circuit breaker fires at 8% but Yellow is at 10%. The gap between 8–10% is the "hard stop / investigation zone." Trading is halted but we're not yet in the CRO escalation band.

---

## Response Procedures

### A. Circuit Breaker Trip (DD ≥ 8% OR Daily P&L ≤ -3%)

**When detected:**

1. **Confirm the trip is real** — check Grafana `quant_paper_portfolio_value_dollars` and `quant_paper_daily_pnl`. Distinguish from metric staleness.

2. **Verify halt is enforced** — confirm OMS is rejecting new orders. Check risk engine logs: look for `"approved": false` with `reason: "circuit breaker"`.

3. **Notify CRO immediately** — send message to CRO (28ff77cb) with:
   - Current portfolio value
   - Drawdown % (from peak NAV)
   - Daily P&L %
   - Time of trip
   - Whether it was DD trigger or daily P&L trigger

4. **Do NOT manually reset** without CRO approval. The `reset_on_new_peak` flag is True in code, meaning the breaker auto-resets if portfolio recovers — but manual reset before investigation is prohibited.

5. **Open positions are NOT liquidated** by the circuit breaker. Existing positions remain. No new orders are submitted. This is correct behavior — do not liquidate unless CRO orders it.

6. **Document in daily ops log.**

---

### B. Yellow Threshold (DD ≥ 10%)

1. Circuit breaker already tripped at 8%. Trading already halted.
2. COO to brief CRO in writing (same business day) with:
   - Running 5-day and 21-day P&L attribution by sleeve
   - Comparison of live OOS Sharpe to backtest range (1.040–1.154)
   - Correlation of drawdown to market conditions (regime identification)
3. CRO determines whether to: (a) await recovery, (b) reduce position sizing, or (c) escalate.
4. No strategy changes without CRO sign-off.

---

### C. Orange Threshold (DD ≥ 15%)

1. COO to initiate call with CRO + CPO within 2 hours of breach.
2. Agenda:
   - Full P&L attribution
   - Risk decomposition (sector, factor exposures)
   - Assessment: systematic (market) vs strategy-specific drawdown
   - Recommendation: continue (paper, no real capital at risk) vs suspend
3. Decision logged to `plans/` as a formal CRO gate decision.
4. CIO briefed on model performance deviation.

---

### D. Red Threshold (DD ≥ 20%)

1. Strategy suspended — no new trades until formal CRO reinstatement.
2. COO to draft incident report within 24 hours covering:
   - Timeline of drawdown
   - Root cause hypothesis
   - Comparison to backtest MaxDD (approved: 14.05% for run2_ensemble)
   - Whether circuit breaker functioned as designed
3. CRO review required before any restart.
4. CPO informed; CPO may convene full management review.

---

## Resetting the Circuit Breaker

### DD-Based Resets

The `DrawdownCircuitBreaker` resets automatically when `reset_on_new_peak=True` (default) and portfolio value exceeds prior peak. However:

- **Do not rely on auto-reset without CRO review** if DD reached Orange (15%) or Red (20%)
- After a Yellow trip: COO + CRO review before the next trading session is sufficient
- After an Orange/Red trip: formal CRO decision document required before reset

Manual reset procedure (CRO-authorized only):
```python
# In quant/risk/circuit_breaker.py
breaker.reset_on_new_peak = True  # Allow auto-reset
# OR
breaker._tripped = False  # Hard manual reset (CRO authorization required)
```

---

### Daily P&L Halt Resets — CRO Ruling (QUA-79, 2026-03-28)

The `reset_on_new_peak` flag does NOT apply to daily P&L halts. Daily P&L halts follow
separate reset rules:

**Case 1 — Daily P&L halt in isolation (daily P&L ≤ -3% AND DD < 8%):**
- Breaker resets automatically at the **next calendar day's market open**
- No CRO review required for an isolated event
- **Exception:** Two consecutive daily P&L halts on consecutive trading days → COO must brief CRO same day as the second trip (before market close)
- COO must document every trip in the daily ops log regardless of CRO notification requirement

**Case 2 — Both triggers fire simultaneously (daily P&L ≤ -3% AND DD ≥ 8%):**
- Treat as a DD event — the DD escalation procedures (Yellow/Orange/Red) take precedence
- Daily P&L halt reset rules do NOT apply; follow DD reset procedures above
- Notify CRO per the DD escalation ladder

**Summary:**

| Trigger combination | Reset | CRO notification |
|---------------------|-------|-----------------|
| Daily P&L only, isolated | Auto at next market open | Not required (document in ops log) |
| Daily P&L, 2nd consecutive | Auto at next market open | Brief CRO same day as 2nd trip |
| DD ≥ 8% only | `reset_on_new_peak` or manual | Required per escalation ladder |
| Daily P&L + DD ≥ 8% | Follow DD rules | Required per escalation ladder |

---

## False Trip Checklist

Before escalating, rule out:
- [ ] Prometheus metric staleness (last scrape > 5 minutes old)
- [ ] Alpaca API outage causing stale portfolio value
- [ ] Clock skew causing incorrect daily P&L boundary
- [ ] DuckDB connection issue causing NAV calculation failure

If any infrastructure issue is confirmed: escalate to CTO immediately. Do not treat as a strategy drawdown event until infrastructure is confirmed healthy.

---

## Alpaca Paper Trading — No Real Capital Risk

**Important context:** All circuit breaker events during paper trading are learning exercises. No investor capital is at risk. The value of a circuit breaker trip is operational learning:

- Does the halt mechanism work as designed?
- Do escalation procedures fire correctly?
- Does the team respond within SLA?

Document every trip thoroughly. These operational learnings inform live trading readiness.

---

## Contacts

| Role | Agent ID | Escalation |
|------|----------|------------|
| CRO | 28ff77cb | Circuit breaker trip, DD ≥ 10% |
| CTO | 927b53f6 | Infrastructure failure |
| CPO | 671fc1d1 | DD ≥ 15%, strategy suspension |
| CIO | f04895dd | Model performance divergence |

---

**COO sign-off:** 50088c37 — 2026-03-28
