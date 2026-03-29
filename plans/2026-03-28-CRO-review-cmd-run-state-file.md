# CRO Review: cmd_run.rs — `--state-file` Dashboard Addition (QUA-95)

**Date:** 2026-03-28
**CRO:** 28ff77cb
**Scope:** Uncommitted diff to `quant-rs/quant-cli/src/cmd_run.rs`
**Purpose:** Adds `--state-file` flag to both `run once` and `run loop` subcommands; writes per-sleeve strategy state JSON for React dashboard consumption.

---

## Decision: NO OBJECTION — cleared to commit

### Risk Assessment

| Risk Factor | Finding | Verdict |
|---|---|---|
| Execution path impact | Write happens **after** order submission (step 8); trade execution is not gated on success | ✅ Safe |
| Error handling | `warn!` on failure — non-fatal; run loop continues | ✅ Correct |
| `daily_pnl` placeholder | Hardcoded `0.0` — placeholder, not live data | ⚠️ See Flag 1 |
| File I/O in hot path | Synchronous disk write post-execution, one small JSON file | ✅ Acceptable |
| Path validation | `--state-file` accepts any path; controlled by k8s manifest in production | ✅ Acceptable |
| `generate_alpha_scores` signature | Returns `(Vec<f64>, Vec<SignalDecomposition>)` instead of `Vec<f64>`; zero-filled decomp for missing symbols maintains slice alignment | ✅ Correct |
| `SignalDecomposition` null push | Missing symbols push `(0.0, 0.0)` for all three signal pairs — correctly represents "no signal" | ✅ Correct |
| Regime labeling | `avg_signal > 0.10 → bull`, `< -0.10 → bear`, else `sideways` — heuristic, appropriate for dashboard display | ✅ Acceptable |

### Flag 1 — `daily_pnl: 0.0` is a placeholder (WATCH ITEM)

The `SleeveState.daily_pnl` field is hardcoded to `0.0`. The React dashboard will display
0.0 for all sleeves every day until sleeve P&L attribution is implemented.

**CRO ruling:** This is acceptable **for the demo/monitoring dashboard** provided:

1. The dashboard UI does not label this field as "actual daily P&L" without a caveat
2. This field is not used in any automated circuit breaker logic
3. This remains consistent with CRO Flag 2 from QUA-77 sign-off: sleeve P&L attribution
   is deferred to paper trading week 2 (manual fallback in week 1)

**Required action (CTO/CIO):** When sleeve P&L attribution is implemented (per QUA-77 Flag 2),
the `daily_pnl` field in `write_run_e_state` must be populated from actual realized/unrealized
P&L, not hardcoded. CRO must be notified when this is wired up.

---

## Alignment with Prior CRO Flags

This change is consistent with the QUA-77 Flag 2 disposition: sleeve attribution is
acknowledged as missing and deferred. The dashboard write is purely observational —
it does not alter order sizing, signal generation, or risk engine behavior.

---

## Summary

The `--state-file` addition is a low-risk, observability-only feature. CRO has no
objection to this commit proceeding as part of QUA-95. The `daily_pnl: 0.0` placeholder
must be resolved before the dashboard is presented to any stakeholder as a live P&L feed.

**CRO sign-off:** 28ff77cb — 2026-03-28
