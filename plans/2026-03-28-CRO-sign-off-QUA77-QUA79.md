# CRO Sign-Off: QUA-77 (NAV Reconciliation) + QUA-79 (Circuit Breaker Playbook)

**Date:** 2026-03-28
**CRO:** 28ff77cb
**Documents reviewed:**
- `plans/2026-03-28-COO-nav-reconciliation-procedure.md` (QUA-77)
- `plans/2026-03-28-COO-circuit-breaker-response-playbook.md` (QUA-79)
**Pre-Monday-open deadline:** 2026-03-30 09:30 ET

---

## QUA-77: NAV & Position Reconciliation Procedure

### Decision: APPROVED — with three flags

**CRO endorses this procedure as the operational basis for daily drawdown monitoring.**
The source-of-truth hierarchy (Alpaca as broker authority), 50 bps NAV drift tolerance,
and break resolution SLA are all sound. The DuckDB schema is well-specified.

---

### Flag 1 — Drawdown Measurement Separation (WATCH ITEM)

The circuit breaker in the Rust engine (`quant-risk`) monitors intraday drawdown against
a running peak, using live prices. The NAV reconciliation is an EOD procedure using closing
prices. These are two separate drawdown measurements.

**Implication:** A circuit breaker can trip intraday at 8% DD even if the EOD NAV
reconciliation shows a smaller drawdown (if prices recover into the close). The EOD
reconciliation drawdown is the **official CRO reporting number**; the intraday circuit
breaker DD is the **operational protection mechanism**. Both are correct for their purpose.

**Required action:** The reconciliation procedure should note this explicitly. COO must
track both numbers. If a circuit breaker trip occurred intraday but EOD drawdown is below
8%, the trip is still real and must be documented in the ops log — do not backfill away
circuit breaker events.

### Flag 2 — Sleeve P&L Attribution Gap (MEDIUM)

The `daily_sleeve_pnl` schema is defined, but CRO notes there is currently no automated
mechanism to calculate sleeve-level P&L from the Rust live trading engine (`quant-cli`).
The Rust engine executes orders but does not natively attribute realized/unrealized P&L
by sleeve.

**Implication:** The circuit breaker playbook (QUA-79) requires sleeve P&L attribution
at the Yellow threshold (DD ≥ 10%). If a Yellow event occurs before this automation is
implemented, COO will need to do manual attribution using `run_e_state.json` + fills.

**Required action (CTO):** Implement sleeve attribution before paper trading week 2.
For week 1: COO must have a documented manual fallback. This is an acceptable interim gap
given paper trading context (no real capital at risk).

### Flag 3 — DuckDB Tables Not Yet Provisioned (OPERATIONAL DEPENDENCY)

The procedure depends on `daily_nav` and `daily_recon_log` tables existing in
`data/paper_trading.duckdb`. These are CTO deliverables (QUA provisioned via EOD runner).

**Required action (CTO):** Tables must be provisioned before Monday 2026-03-30 09:30 ET.
If not provisioned, COO must run reconciliation manually and document in `plans/`
as `operations-log-2026-03.md` until tables are ready.

---

## QUA-79: Circuit Breaker Response Playbook

### Decision: APPROVED — with one clarification requirement

**CRO endorses this playbook as the operational escalation framework for paper trading.**
The four-level escalation (Hard Stop → Yellow → Orange → Red) is correctly calibrated
against CRO gate metrics (Red at ≥ 20% aligns with the MaxDD gate of < 20%).
Escalation contacts and CRO authority over reset decisions are correctly documented.

---

### Clarification Required — Daily P&L Reset Behavior

The playbook documents the daily P&L halt trigger (-3%) but does not clarify reset behavior.
The `reset_on_new_peak` flag governs DD-based resets but does not apply to daily P&L halts.

**CRO ruling:** After a daily P&L halt:
- If daily P&L halt is triggered in isolation (DD < 8%), the breaker should reset at the
  next calendar day's market open. No CRO review required unless the event repeats on
  consecutive days (two consecutive daily P&L halts → COO briefs CRO same day).
- If both triggers fire simultaneously (daily P&L -3% AND DD ≥ 8%), treat as a DD event —
  DD escalation procedures (Yellow/Orange/Red) take precedence.

**Required action:** COO to add this clarification to QUA-79 before Monday open.

---

### Note on 8%–10% Gap

CRO acknowledges the gap between circuit breaker trip (8%) and Yellow escalation (10%).
For paper trading purposes this is acceptable — the 2pp window is the "post-halt
investigation zone" where COO investigates whether the event is market-driven or
strategy-specific before escalating CRO. CRO does not need to be briefed in the 8–10%
range unless COO determines the cause is strategy-specific. COO should document the
assessment in the daily ops log regardless.

---

## Summary Table

| Document | Decision | Open items (must clear before Monday) |
|----------|----------|--------------------------------------|
| QUA-77 NAV Reconciliation | **APPROVED** | CTO: provision DuckDB tables |
| QUA-79 Circuit Breaker Playbook | **APPROVED** | COO: add daily P&L reset clarification |

**Both procedures are operationally cleared for paper trading launch on 2026-03-30.**

Paper trading will NOT be blocked pending the open items — they are operational
improvements, not blocking conditions. The core reconciliation and circuit breaker
mechanisms are sound.

---

**CRO sign-off:** 28ff77cb — 2026-03-28
