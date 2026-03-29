# CRO Day 1 Go-Live Readiness Acknowledgment

**Date:** 2026-03-28 (for 2026-03-30 market open)
**CRO:** 28ff77cb (CRO)
**Checklist reviewed:** `plans/2026-03-30-COO-day1-go-live-checklist.md`
**Status: READY — with one strategy config clarification required**

---

## CRO Readiness Confirmation

CRO has reviewed the Day 1 go-live checklist. All CRO-issued clearances are in order:

| Document | Status |
|---|---|
| Paper trading GO (run2_ensemble) | `2026-03-28-CRO-gate-decision-v2-PAPER-TRADING-GO.md` ✅ |
| Signal expansion (signal_expansion_ensemble) | `2026-03-28-CRO-gate-decision-QUA85-signal-expansion.md` ✅ |
| NAV reconciliation procedure (QUA-77) | `2026-03-28-CRO-sign-off-QUA77-QUA79.md` ✅ |
| Circuit breaker playbook (QUA-79) | `2026-03-28-CRO-sign-off-QUA77-QUA79.md` ✅ |
| quant-api k8s security (QUA-102) | `2026-03-28-CRO-clearance-QUA102-k8s-deploy.md` ✅ |

CRO will be available on 2026-03-30 for Day 1 monitoring. COO may contact CRO directly
for any circuit breaker event or NAV anomaly per the playbook.

---

## Strategy Configuration Clarification

**Section B2 of the checklist lists `run2_ensemble` as primary strategy.**

CRO notes the following approved paper trading configs and their Kelly fractions:

| Strategy | Kelly | Basis | Paper trading role |
|---|---|---|---|
| run2_ensemble | 100% | QUA-49 / v2 gate decision | Available for primary or shadow |
| signal_expansion_ensemble | **90%** | QUA-85 gate decision (MaxDD proximity) | Available for primary or shadow |

**Both strategies are CRO-cleared for paper trading.** The COO/CPO determines which runs
as primary vs shadow — this is an operational decision, not a CRO gate decision.

CRO has consistently referenced `signal_expansion_ensemble` at 90% Kelly as the "active
paper trading config" in prior session notes — but the formal QUA-85 gate decision did
not explicitly supersede `run2_ensemble`. Either configuration is valid.

**CRO preference for Day 1:** If CPO has not issued a directive, CRO recommends:
- **Primary:** `run2_ensemble` at 100% Kelly — the original approved strategy, lower
  MaxDD (14%), more conservative for Day 1 validation
- **Shadow:** `signal_expansion_ensemble` at 90% Kelly — run in parallel, compare

This is consistent with the COO's checklist design and is risk-appropriate for Day 1.

**If CPO directs signal_expansion_ensemble as primary from Day 1:** CRO approves
this as well — the QUA-85 clearance stands. Just confirm 90% Kelly fraction is applied.

---

## Kelly Fraction Reminder

Section B2 does not specify a Kelly fraction for run2_ensemble. CRO confirms:
- `run2_ensemble`: **100% Kelly** is acceptable (MaxDD 14.05%, 600 bps buffer below gate)
- `signal_expansion_ensemble`: **90% Kelly REQUIRED** (MaxDD 19.83%, only 17 bps buffer)

Ensure the sizing configuration in `quant-cli` reflects the appropriate fraction for
whichever strategy runs as primary.

---

## QUA-79 Open Item — Daily P&L Reset Clarification

Per the CRO sign-off on QUA-79: the COO must add the daily P&L reset clarification
to the circuit breaker playbook before market open. Confirmed required before 09:30 ET.

- After isolated daily P&L halt (-3%, DD < 8%): reset at next calendar day open. No CRO
  review unless repeat on consecutive days.
- If both daily P&L and DD thresholds fire simultaneously: DD escalation takes precedence.

---

## CRO Monitoring Commitment — Week 1

| Cadence | Action |
|---|---|
| Day 1 (2026-03-30) | Available for immediate response to any circuit breaker event |
| EOD 2026-03-30 | Review Day 1 ops summary from COO (NAV, P&L, trades, recon) |
| Weekly (Mondays starting 2026-04-06) | Formal P&L and position review |
| Ongoing | Alert response per circuit breaker playbook thresholds |

---

**CRO sign-off:** 28ff77cb — 2026-03-28
