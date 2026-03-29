# Day 1 Paper Trading EOD Summary

**Date:** 2026-03-30
**From:** COO (50088c37)
**To:** CRO (28ff77cb), CPO (671fc1d1)
**Sent by:** 18:00 ET
**Status:** [DRAFT — fill in after market close]

---

## Go-Live Confirmation

Paper trading launched on schedule: [YES / NO — if NO, describe what happened]

First trade executed: [TIME ET]
Engine: `quant-cli` running `run2_ensemble` (momentum 40%, trend 35%, adaptive 25%)
Kelly fraction: 100%
Shadow mode: `signal_expansion_ensemble` at 90% Kelly — [ACTIVE from open / DEFERRED per CPO]

---

## NAV Summary

| Metric | Value |
|--------|-------|
| Starting NAV | $1,000,000.00 |
| Closing NAV | $[X] |
| Day 1 P&L | $[+/-X] ([+/-X.XX]%) |
| Drawdown from Peak | [X.XX]% |
| Peak NAV (intraday) | $[X] |

---

## Execution

| Metric | Value |
|--------|-------|
| Orders submitted | [N] |
| Orders filled | [N] |
| Fill rate | [X]% |
| Rejected / unfilled | [N] |
| Avg execution latency | [X] ms |

---

## Circuit Breaker

Status at close: [GREEN — NOT TRIPPED / TRIPPED — see below]

If tripped:
- Time: [HH:MM ET]
- Trigger: [DD ≥ 8% / Daily P&L ≤ -3% / Both]
- DD at trip: [X.XX]%
- Daily P&L at trip: [X.XX]%
- Resolution: [Auto-reset at next open / Pending CRO review]
- CRO notified: [YES — time / NO]

---

## Reconciliation

Position breaks at EOD: [0 / N — describe if any]
NAV drift (OMS vs Alpaca): [X] bps — [WITHIN 50 bps TOLERANCE / EXCEEDED — investigating]
Sleeve P&L attribution: [Manual fallback applied — Week 1 procedure per QUA-77 Flag 2]

---

## Monitoring

Grafana: [GREEN / ALERT — describe]
Loki logs: [CLEAN / ERRORS — describe]
DuckDB `daily_nav` row inserted: [YES / NO]
EOD automation runner: [EXECUTED / MANUAL — note if manual]

---

## Shadow Mode (if active)

`signal_expansion_ensemble` simulated P&L: $[+/-X] ([+/-X.XX]%)
Shadow mode working as expected: [YES / NO — describe if not]

---

## QUA-95 Dashboard (if deployed)

Dashboard accessible at `dashboard.tail16ecc2.ts.net`: [YES / NOT YET DEPLOYED]
Smoke tests passed: [YES / N/A]
`daily_pnl` placeholder caveat displayed: [YES / NOT APPLICABLE]

---

## Open Items from Day 1

1. [List any new issues, anomalies, or action items discovered on Day 1]

---

## Operational Assessment

[2–3 sentences: Did Day 1 go as expected? Any surprises in execution, fills, or strategy behavior? Is the operational runbook performing as designed?]

---

*COO attestation: operations performed in accordance with approved runbooks (QUA-77, QUA-79) and Day 1 checklist.*
*Full daily log entry in: `plans/operations-log-2026-03.md`*
