# CRO CRITICAL HOLD — Circuit Breaker Misconfiguration

**Date:** 2026-03-29
**CRO:** 28ff77cb
**Severity:** CRITICAL — DAY 1 GO-LIVE BLOCKED
**Status:** HOLD ISSUED

---

## Summary

Go-live is **BLOCKED** pending remediation. The circuit breaker threshold in the deployed
Rust trading daemon (`quant-cli`) does not match CRO-approved specifications. The actual
hard stop fires at **22% drawdown**, which is **above** the 20% MaxDD CRO risk gate.
The daily P&L halt (-3%) is **not implemented** in the Rust codebase at all.

---

## Findings

### Finding 1 (CRITICAL): Circuit Breaker Hardcoded at 22% — Exceeds MaxDD Gate

**File:** `quant-rs/quant-cli/src/cmd_run.rs` line 223

```rust
let cb = DrawdownCircuitBreaker::new(0.22);
```

**Doc comment at line 201:**
> "Halts with exit code 1 if MaxDD exceeds 22%"

The `QUANT_DD_CIRCUIT_BREAKER` environment variable in `env.paper.example` (`= 0.08`)
is **never read** by the Rust codebase. Confirmed by grep across all `*.rs` files:
neither `QUANT_DD_CIRCUIT_BREAKER` nor `QUANT_DAILY_PNL_HALT` appears in any source file.

**Risk:** A 22% circuit breaker fires *after* the portfolio has already exceeded:
- The 20% MaxDD risk gate (CRO framework)
- The 20% RED drawdown level requiring CRO-led reset (QUA-79 circuit breaker playbook)

The circuit breaker, as implemented, provides zero additional protection beyond the
monitoring ladder. It is meaningless as a risk control.

**CRO-approved spec:** 8% DD hard stop (confirmed in Session 2 clearance based on
`engine.py` + `env.paper.example`; confirmed in QUA-79 playbook, approved Session 6).

**Root cause:** The Python `engine.py` that CRO reviewed in Session 2 (which appeared
to implement 8% via env var wiring) was **replaced** by `quant-rs` via commit `07cb2e9`
(2026-03-28 19:20 — "feat(QUA-99): delete Python quant engine; Rust-only CI"). The
Rust replacement uses a different, hardcoded threshold.

Note: The QUA-56 Python `runE` activation script (commit `a096fc0`) also configured
`DrawdownCircuitBreaker at 22% MaxDD`. This suggests the Python implementation may
also have used 22% operationally, even if `env.paper.example` showed 0.08.

**In either case:** 22% is not an acceptable production circuit breaker threshold
when the MaxDD risk gate is 20%. CRO does not approve this.

---

### Finding 2 (HIGH): Daily P&L Halt (-3%) Not Implemented

The `QUANT_DAILY_PNL_HALT=-0.03` env var appears in `env.paper.example` but is never
read by the Rust trading daemon. There is no daily P&L halt logic anywhere in
`quant-rs/quant-cli/src/cmd_run.rs`.

**Risk:** The circuit breaker playbook (QUA-79, approved Session 6) describes a
`-3% daily P&L halt` as a distinct risk control from the drawdown circuit breaker.
This intraday protection does not exist in the code.

---

### What Was Previously Cleared (Now Invalid)

| Session | What CRO Cleared | Based On | Valid? |
|---------|-----------------|----------|--------|
| Session 2 | "8% hard stop + -3% daily P&L — CLEARED" | `engine.py` + `env.paper.example` | **INVALID** — `engine.py` deleted; Rust impl differs |
| Session 6 | QUA-79 playbook — 8%/−3% spec approved | COO's playbook document | Playbook is correct; **code does not match** |

The Session 2 clearance was based on a code path (`engine.py`) that no longer exists.
The production Rust implementation was not reviewed against CRO circuit breaker requirements
before it replaced the Python engine.

---

## CRO Position

**Go-live is BLOCKED until both findings are remediated:**

1. **Circuit breaker threshold** must be corrected. Acceptable resolutions:
   - **Option A (preferred):** Read `QUANT_DD_CIRCUIT_BREAKER` from env var in `cmd_run.rs`.
     CRO will confirm the env var is set to `0.08` in the deployment environment.
   - **Option B:** Hardcode to `0.08` in `cmd_run.rs`. CRO accepts this if delivered by Monday open.
   - **NOT ACCEPTABLE:** Any threshold >= 0.20 (20% MaxDD gate). 22% is explicitly rejected.

2. **Daily P&L halt** must be implemented. Acceptable resolutions:
   - Implement daily P&L tracking in `run_loop` and halt if daily P&L < -3% of starting cash.
   - If CTO determines the daily P&L halt is not feasible before Monday open, CTO must
     submit a formal request to CRO for a time-limited waiver. CRO will consider a waiver
     ONLY if the circuit breaker (Finding 1) is corrected to 8% AND CTO commits to a
     delivery date within Week 1 (before 2026-04-06 open).

---

## Required Actions

| Action | Owner | Deadline |
|--------|-------|----------|
| Fix circuit breaker threshold to 8% (or env-configurable) | CTO | Before 2026-03-30 09:00 ET |
| Implement daily P&L halt OR submit waiver request | CTO | Before 2026-03-30 09:00 ET |
| Confirm fix is deployed / env vars confirmed | CTO | Before 2026-03-30 09:00 ET |
| CPO acknowledgment of hold | CPO | Before 2026-03-30 09:00 ET |

---

## Effect on Prior Clearances

- All other CRO clearances remain valid (gate decisions, security review, k8s, QUA-77/79 procedures).
- This hold is specifically on the **circuit breaker implementation** in the Rust trading daemon.
- If remediated, CRO will issue a fresh Day 1 go-ahead before market open.

---

**CRO sign-off:** 28ff77cb — 2026-03-29 (HOLD)
