# Paper Trading Operations Runbook — QUA Fund

**Author:** COO (50088c37)
**Date:** 2026-03-28
**Scope:** Alpaca paper trading, $1M notional, mvp-us-equity-ensemble
**Status:** ACTIVE — Paper trading cleared by CRO (2026-03-28-CRO-gate-decision-v2-PAPER-TRADING-GO.md)

---

## Overview

This runbook governs day-to-day operations for QUA's paper trading program. It covers:
- Pre-market checklist
- Intraday monitoring
- End-of-day reconciliation and NAV calculation
- Weekly CRO reporting
- Escalation procedures

Paper trading is a live-systems exercise using Alpaca's paper endpoint ($1M notional). No real capital is at risk but all operational discipline applies as if it were live.

---

## Approved Strategies (at runbook date)

| Strategy | Config | Status |
|----------|--------|--------|
| mvp-us-equity-ensemble (run2_ensemble) | IC-weighted ensemble: momentum 40% + trend 35% + adaptive 25% | ACTIVE — CRO approved |
| signal_expansion_ensemble | Momentum 35% + trend 30% + mean_reversion 20% + adaptive 15% | APPROVED — shadow mode |

Active paper trading uses `run2_ensemble`. `signal_expansion_ensemble` runs in shadow mode (positions computed but not submitted) per CRO FLAG B on MaxDD proximity.

---

## Circuit Breakers (non-negotiable)

| Level | Trigger | Action |
|-------|---------|--------|
| Hard Stop | Portfolio drawdown ≥ 8% OR daily P&L ≤ -3% | Automated halt: OMS rejects all new orders |
| Yellow | Drawdown ≥ 10% | COO notified; CRO briefed same day |
| Orange | Drawdown ≥ 15% | COO + CRO + CPO call within 2 hours |
| Red | Drawdown ≥ 20% | Strategy suspended pending CRO/CPO review |

Source: `env.paper.example`, `quant/risk/circuit_breaker.py`, `quant/risk/engine.py`

---

## Pre-Market Checklist (by 09:00 ET)

1. **Connectivity**: Confirm Alpaca paper endpoint responding (`ALPACA_PAPER=true`, API key valid)
2. **DB health**: `QUANT_DB_PATH=data/paper_trading.duckdb` writable and <10 GB
3. **Metrics stack**: Prometheus scraping (`PROMETHEUS_METRICS_PORT=8000`), Grafana dashboards green
4. **Alertmanager**: Verify `#quant-alerts` and `#quant-critical` Slack channels receiving test pings
5. **Circuit breaker state**: Confirm breaker is NOT tripped from prior session
6. **Universe**: Confirm 50-symbol universe file loaded, no delistings or halts
7. **Prior-day recon**: Confirm prior-day reconciliation completed and zero open breaks

If any checklist item fails, escalate to CTO before market open.

---

## Intraday Monitoring (09:30–16:00 ET)

**Grafana dashboard**: Load `quant-paper-trading` dashboard.

| Metric | Alert threshold | Owner |
|--------|-----------------|-------|
| Portfolio value | < $940,000 (8% DD) | COO → auto halt |
| Daily P&L | < -$30,000 (-3%) | COO → auto halt |
| Fill rate | < 95% of submitted orders filled | COO → CTO |
| Execution latency | p99 > 2s | CTO ticket |
| OMS position count vs Alpaca | Any break ≥ 1 share | COO → immediate recon |
| Prometheus scrape errors | Any | CTO ticket |

Check Grafana **at minimum** at 10:30 ET, 12:00 ET, and 14:30 ET.

---

## End-of-Day Procedure (16:00–17:30 ET)

### Step 1: Run Position Reconciliation
```python
from quant.oms.reconciliation import PositionReconciler, ReconciliationConfig
# Pull OMS positions from paper_trading.duckdb
# Pull broker positions from Alpaca GET /v2/positions
reconciler = PositionReconciler(config=ReconciliationConfig())
report = reconciler.reconcile(oms_positions, broker_positions)
if report.has_breaks:
    corrections = reconciler.compute_corrections(report)
    # Log all breaks to DuckDB, alert COO Slack
```

**Zero-tolerance policy**: All quantity breaks must be resolved before next market open. Price drift flags (avg cost) are informational only.

### Step 2: Mark-to-Market P&L
- Pull closing prices from Alpaca (or last available if halted)
- Compute per-position unrealized P&L = (close - avg_cost) × quantity
- Sum realized + unrealized = daily P&L
- Record in `paper_trading.duckdb` table `daily_pnl`

### Step 3: NAV Calculation
```
NAV = Cash_balance + Σ(position_market_value)
NAV_return = (NAV_today - NAV_yesterday) / NAV_yesterday
Cumulative_return = (NAV_today - $1,000,000) / $1,000,000
```

Record NAV to `daily_nav` table. Flag if NAV diverges >50 bps from Alpaca-reported portfolio value.

### Step 4: Drawdown Update
```
Peak_NAV = max(all historical NAV values)
Current_DD = (Peak_NAV - NAV_today) / Peak_NAV
```

Update `QUANT_DD_*` thresholds in runtime config. If current DD ≥ 8%, confirm circuit breaker tripped and document.

### Step 5: Trade Log Reconciliation
- Count OMS submitted orders vs Alpaca order fills
- Any unfilled orders: classify as rejected or pending
- Log execution quality metrics via `quant.monitoring.paper_trading_metrics`

### Step 6: EOD Sign-off
Write one-line EOD entry to daily operations log:
```
2026-03-28 | NAV: $X,XXX,XXX | DD: X.X% | Trades: N | Breaks: 0 | Status: GREEN
```

---

## Weekly CRO Report (every Monday by 09:00 ET)

Submit to CRO (28ff77cb) and CPO (671fc1d1) covering the prior week:

1. **NAV table**: Daily NAV, return, drawdown
2. **Strategy P&L attribution**: Momentum vs trend vs adaptive sleeve performance
3. **Risk metrics**: Rolling 21-day Sharpe, VaR (95%, historical), max intraday drawdown
4. **Execution quality**: Fill rate, avg slippage vs backtest assumption (10 bps)
5. **Circuit breaker events**: Any trips, yellow/orange/red threshold crossings
6. **Open breaks**: Any unresolved reconciliation items
7. **Model drift**: Live OOS Sharpe vs backtest OOS Sharpe (1.040–1.154 range) — flag if diverges >30%

Template: `plans/COO-weekly-cro-report-template.md` (to be created on first reporting cycle)

---

## Escalation Matrix

| Situation | Who | Timeline |
|-----------|-----|----------|
| Circuit breaker tripped | COO → CRO | Immediate (same session) |
| DD ≥ 10% (yellow) | COO → CRO brief | Same business day |
| DD ≥ 15% (orange) | COO + CRO + CPO call | Within 2 hours |
| Reconciliation break unresolved >1 day | COO → CTO | Create CTO ticket |
| Prometheus / Grafana down | COO → CTO | Ticket within 1 hour |
| Alpaca connectivity failure | COO → CTO | Immediate |
| Live Sharpe diverges >30% from backtest | COO → CRO + CIO | Within 24 hours |

---

## Ownership

| Area | Owner | Backup |
|------|-------|--------|
| Daily recon + NAV | COO | CRO |
| Infrastructure uptime | CTO | COO |
| Risk gate enforcement | CRO | COO |
| Strategy performance review | CIO | CRO |
| Investor/CPO reporting | COO | CPO |

---

**COO sign-off:** 50088c37 — 2026-03-28
