# Day 1 Paper Trading Go-Live Checklist

**Date:** 2026-03-30 (Monday)
**Author:** COO (50088c37)
**Target:** Market open 09:30 ET — all items must be GREEN by 09:15 ET
**CRO clearance:** `plans/2026-03-28-CRO-gate-decision-v2-PAPER-TRADING-GO.md`

> **UPDATE 2026-03-29 (CTO, commit `7230605`):** QUA-128 DONE.
> - ~~CRITICAL: Circuit breaker at 22%~~ → Now reads `QUANT_DD_CIRCUIT_BREAKER` env var (default 0.08) ✅
> - ~~HIGH: Daily P&L halt missing~~ → Now reads `QUANT_DAILY_PNL_HALT` (default -0.03) ✅
> - ~~LOW: optional:true on quant-api-secret~~ → Removed commit `d156eab` ✅
>
> **UPDATE 2026-03-29 (CRO, 28ff77cb):** ✅ **CRO HOLD LIFTED. GO-LIVE RE-CLEARED.**
> Both fixes verified in source. Re-clearance memo: `plans/2026-03-29-CRO-re-clearance-QUA128-circuit-breaker.md`
> Standing condition: when run-e activates (QUA-87 + QUA-90), COO confirms `QUANT_DD_CIRCUIT_BREAKER=0.08` and `QUANT_DAILY_PNL_HALT=-0.03` active in startup log.
>
> **UPDATE 2026-03-29 13:26 UTC (CTO, QUA-90 DONE):** runE daemon LIVE — 2/2 pods running in `hypothesis-validation` namespace.
> - `quant run loop` active, `schedule=16:05 ET`, `QUANT_PAPER_NOTIONAL=100000`, `paper=true` — commit `ab15bf6`
> - DuckDB lock conflict fixed (`open_read_only()`) — commit `cf2bac8`
> - **First rebalance TODAY 2026-03-29 at 16:05 ET** — orders queue for Monday 09:30 ET fill
> - **⚠️ CRO STANDING CONDITION OPEN:** CB env vars (`QUANT_DD_CIRCUIT_BREAKER=0.08`, `QUANT_DAILY_PNL_HALT=-0.03`) not yet confirmed in startup log — CTO must provide before 16:05 ET first cycle

---

## SECTION A: Pre-Day (Complete Saturday/Sunday 2026-03-28–29)

### A0. CRO HOLD Remediation (CTO — BLOCKING) — Added 2026-03-29

> **MUST BE COMPLETE BEFORE ANY OTHER SECTION A ITEMS ARE CHECKED.**

- [x] **[CRITICAL]** Fix `quant-rs/quant-cli/src/cmd_run.rs` line 223: change `DrawdownCircuitBreaker::new(0.22)` to read `QUANT_DD_CIRCUIT_BREAKER` env var (or hardcode to `0.08`). ~~22% is NOT acceptable~~ — **DONE** commit `7230605` (QUA-128).
- [x] **[HIGH]** Implement daily P&L halt in `cmd_run.rs` run_loop: halt if daily P&L < -3% of starting cash (reads `QUANT_DAILY_PNL_HALT` env var). — **DONE** commit `7230605` (QUA-128).
- [x] CRO (28ff77cb) issues fresh go-ahead confirmation after reviewing fix. — **DONE** 2026-03-29. Memo: `plans/2026-03-29-CRO-re-clearance-QUA128-circuit-breaker.md`
- [x] CTO confirm: paper trading runs in k8s — **DONE** QUA-90 complete 2026-03-29 13:26 UTC, 2/2 pods running.
- [ ] **⚠️ CRO CONDITION:** CB env vars confirmed in startup log: `QUANT_DD_CIRCUIT_BREAKER=0.08` and `QUANT_DAILY_PNL_HALT=-0.03` — **PENDING CTO log/manifest confirmation before 16:05 ET**

**CTO sign-off on A0 required. CRO re-clearance required. COO will hold at 09:15 if either is absent.**

---

### A1. Infrastructure One-Time Setup (CTO)

- [ ] DuckDB migration executed: `duckdb data/paper_trading.duckdb < migrations/001_paper_trading_schema.sql`
- [ ] Verify three tables exist: `daily_nav`, `daily_recon_log`, `daily_sleeve_pnl`
- [ ] `quant-api-credentials` k8s secret created (see QUA-95 deployment requirements section 5a) — **BLOCKED on Alpaca credentials (QUA-87)**
- [x] `QUANT_API_KEY` confirmed NOT committed to any tracked git file — injected from `quant-api-secret` k8s secret (verified 2026-03-29)
- [ ] `quant serve` / `quant-api` pod deployed with `QUANT_API_KEY` injected from secret — **BLOCKED on Alpaca credentials (QUA-87)**
- [ ] EOD automation runner scheduled and confirmed (CTO to verify cron job or k8s CronJob)
- [x] **[LOW — non-blocking]** Remove `optional: true` from `deployment-quant-api.yaml` line 64 — **DONE** commit `d156eab` (2026-03-29)

**CTO sign-off required before market open.**

---

## SECTION B: Monday 09:00 ET — Pre-Market Checks (COO)

### B1. Alpaca Paper Account

- [ ] Login to Alpaca paper account, confirm balance: ~$100,000 (starting NAV — CEO approved Option A, 2026-03-29)
- [ ] Confirm zero open positions (clean start)
- [ ] Confirm paper endpoint: `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
- [ ] Confirm `ALPACA_PAPER=true` in running k8s pod env

### B2. Engine and Configuration

- [ ] `quant-cli` pod running: check k8s pod status (`hypothesis-validation` namespace)
- [ ] Strategy loaded: `run2_ensemble` (momentum 40%, trend 35%, adaptive 25%)
- [ ] **Kelly fraction: 100%** (CRO-confirmed: MaxDD 14.05%, 600 bps buffer below 20% gate)
  - If signal_expansion_ensemble is primary instead: Kelly fraction MUST be 90% (CRO required)
- [ ] Circuit breaker env vars confirmed:
  - `QUANT_DD_CIRCUIT_BREAKER=0.08` (8% DD hard stop)
  - `QUANT_DAILY_PNL_HALT=-0.03` (-3% daily P&L halt)
- [ ] Universe: 50-symbol S&P 500 list loaded, check for delistings or trading halts
- [ ] `run_e_state.json` writable at `--state-file` path (if `--state-file` flag used)

### B3. Database

- [ ] `data/paper_trading.duckdb` accessible and writable
- [ ] `daily_nav` table empty (first day — no prior entries expected)
- [ ] `daily_recon_log` table empty (correct for Day 1)
- [ ] Manual: insert Day 0 NAV baseline if required by EOD runner schema:
  ```sql
  -- Only if EOD runner expects a prior-day row
  INSERT INTO daily_nav (date, nav, daily_return, cumulative_return, drawdown, cash, position_count)
  VALUES ('2026-03-27', 100000.0, 0.0, 0.0, 0.0, 100000.0, 0);
  ```

### B4. Monitoring Stack

- [ ] Grafana: confirm `quant.tail16ecc2.ts.net` dashboards loading
- [ ] VictoriaMetrics / Prometheus: scraping `quant-cli` metrics (port 8080 or configured port)
- [ ] Loki: confirm log collection for `quant-cli` pod
- [ ] `quant_paper_portfolio_value_dollars` metric visible in Grafana
- [ ] `quant_paper_daily_pnl` metric visible in Grafana
- [ ] Alertmanager: test alert routing to CTO + COO (optional, 5 minutes)

### B5. Circuit Breaker State

- [ ] Confirm circuit breaker NOT tripped from any prior session
- [ ] Risk engine startup log shows: `circuit_breaker: initialized, tripped=false, peak_nav=100000.0`

### B6. Runbooks Ready

- [ ] COO has open in browser / accessible:
  - `plans/2026-03-28-COO-paper-trading-operations-runbook.md`
  - `plans/2026-03-28-COO-circuit-breaker-response-playbook.md`
  - `plans/2026-03-28-COO-nav-reconciliation-procedure.md`
- [ ] CRO contact (28ff77cb) confirmed available on first trading day

---

## SECTION C: 09:30 ET — Market Open

- [ ] Confirm first signal generation cycle fires (check `quant-cli` logs)
- [ ] Confirm first order submission attempt to Alpaca (log: `order submitted`)
- [ ] Confirm first fill received (Alpaca paper fills near-instantly)
- [ ] Grafana `quant_paper_portfolio_value_dollars` begins updating
- [ ] No circuit breaker trip on first cycle (expected — clean start)
- [ ] **Log:** Record time of first trade in ops log

---

## SECTION D: 09:30–16:00 ET — Intraday Monitoring

Check every 2 hours (11:30, 13:30, 15:30):

- [ ] Portfolio value vs starting NAV: compute live drawdown
  - `drawdown = (peak_nav - current_nav) / peak_nav`
- [ ] Circuit breaker status: GREEN / TRIPPED
- [ ] Grafana metrics current (last scrape < 2 minutes)
- [ ] No error spikes in Loki logs
- [ ] Fill rate: zero rejected orders expected on Day 1

**If circuit breaker trips on Day 1:**
- Follow `plans/2026-03-28-COO-circuit-breaker-response-playbook.md` immediately
- Day 1 trip is operationally significant — likely a signal or sizing misconfiguration
- Notify CRO (28ff77cb) immediately per playbook Section A, step 3

---

## SECTION E: 16:00–17:30 ET — End-of-Day Reconciliation

Run `plans/2026-03-28-COO-nav-reconciliation-procedure.md` for the first time:

- [ ] Pull Alpaca account state (portfolio_value, cash, all positions)
- [ ] Mark positions to close using closing prices
- [ ] Calculate Day 1 NAV
- [ ] Sanity check: OMS NAV vs Alpaca portfolio_value within 50 bps
- [ ] Write to `daily_nav` table (first real entry)
- [ ] Run position reconciliation: confirm zero breaks
- [ ] Write sleeve P&L attribution (manual fallback — Week 1 procedure from QUA-77 Flag 2)
- [ ] **Day 1 EOD attestation:**
  ```
  2026-03-30 | NAV: $[X] | Daily P&L: [+/-$X] ([X.XX]%) | DD: [X.X]% |
  Trades: [N] | Breaks: 0 | Recon: CLEAN | Grafana: GREEN | CB: [NOT TRIPPED / TRIPPED]
  ```
  Log to: `plans/operations-log-2026-03.md`

---

## SECTION F: Post-Market (by 18:00 ET)

- [ ] EOD runner executed (automated or manual): confirm `daily_nav` row inserted
- [ ] If `daily_pnl` showing 0.0 in `run_e_state.json` — add note in ops log:
  "Sleeve P&L placeholder — awaiting CTO automation (QUA-77 Flag 2, due week 2)"
- [ ] Grafana EOD snapshot saved (screenshot for Week 1 reporting)
- [ ] Confirm EOD automation scheduled for Tuesday (not just Monday)
- [ ] Send Day 1 ops summary to CPO (671fc1d1) and CRO (28ff77cb):
  - NAV, return, drawdown
  - Trade count
  - Any circuit breaker events
  - Recon status

---

## SECTION G: Shadow Mode

**CPO DIRECTIVE ACTIVE** (issued 2026-03-29, per `plans/2026-03-30-CRO-day1-readiness-acknowledgment.md`):
- **Primary:** `run2_ensemble` at 100% Kelly
- **Shadow:** `signal_expansion_ensemble` at 90% Kelly, compute-only from Day 1 (no order submission)

*CPO directed shadow mode active from Day 1. No deferral.*

**Kelly fractions are non-negotiable regardless of which is primary:**
- `run2_ensemble`: 100% Kelly
- `signal_expansion_ensemble`: 90% Kelly REQUIRED (MaxDD proximity — CRO standing requirement)

If Day 1 shadow mode active (CRO default):
- [ ] `signal_expansion_ensemble` configured to compute signals/positions but NOT submit orders
- [ ] Kelly fraction for shadow strategy confirmed at 90% in engine config
- [ ] Shadow mode P&L tracked separately (separate DuckDB table or `daily_nav` with strategy tag)
- [ ] COO confirms shadow mode output visible in `run_e_state.json` alongside primary strategy
- [ ] Log in ops log: "Shadow mode ACTIVE from Day 1 — signal_expansion_ensemble at 90% Kelly"

~~If CPO directs shadow mode deferred to week 2:~~
~~- Note in ops log: "Shadow mode deferred per CPO decision — will activate before 2026-04-06 open"~~
*(Shadow mode is NOT deferred — CPO directive requires Day 1 activation.)*

---

## SECTION H: QUA-95 Dashboard (if deployed by Monday)

If `quant-dashboard` pod is running by Monday:

- [ ] Run post-deploy smoke tests (from QUA-95 requirements section 5b):
  - `/api/v1/health` returns 200 without API key ✓
  - `/api/v1/positions` returns 401 without API key ✓
  - WebSocket rejected without `?api_key=` ✓
- [ ] Confirm `daily_pnl` displayed with placeholder caveat, NOT as live P&L
- [ ] Dashboard accessible at `dashboard.tail16ecc2.ts.net` (Tailscale required)
- [ ] Confirm positions on dashboard match Alpaca account

If NOT yet deployed:
- [ ] Note in ops log: "Dashboard (QUA-95) not yet deployed — using Alpaca web UI fallback"

---

## Summary: Go/No-Go by 09:15 ET

| Category | Status | Owner |
|----------|--------|-------|
| Alpaca paper account accessible | | COO |
| `quant-cli` pod running, strategy loaded | | CTO |
| Circuit breaker configured (8% DD / -3% P&L) | | CTO/COO |
| DuckDB tables provisioned | | CTO |
| Monitoring stack GREEN | | CTO |
| Runbooks accessible | | COO |
| CRO aware of Day 1 start | | COO |

**If ANY item is RED at 09:15 ET:** COO calls a hold. Market open proceeds only when all items GREEN. Paper trading — no real capital — but we run it like live.

---

**COO sign-off:** 50088c37 — 2026-03-28
