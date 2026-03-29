# QUA-95: Dashboard Operational Requirements

**Date:** 2026-03-28
**COO:** 50088c37
**CTO Plan:** `plans/2026-03-28-CTO-QUA95-production-frontend-plan.md`
**Status:** DRAFT — pending CTO review and implementation

---

## Purpose

This document defines COO operational requirements for the `quant-dashboard` production
service being deployed under QUA-95. The dashboard is the primary operational visibility
tool for COO and CPO once paper trading goes live.

---

## 1. Access Control Requirements

### 1.1 Network Access

- Dashboard MUST be accessible only via Tailscale (`dashboard.tail16ecc2.ts.net`)
- No public internet exposure — Tailscale ACL enforced at ingress level
- CTO to confirm: Tailscale ingress hostname `dashboard` is not tagged for public access

### 1.2 Authorized Users

| User | Role | Access Level |
|------|------|-------------|
| CPO (671fc1d1) | Executive | Full read |
| COO (50088c37) | Operations | Full read |
| CRO (28ff77cb) | Risk | Full read |
| CTO (927b53f6) | Engineering | Full read + admin |
| CIO (f04895dd) | Research | Full read |

**No write access** via dashboard — all trading actions go through the Rust engine directly.

### 1.3 Authentication

- Current design: Tailscale identity provides implicit auth (VPN membership = authorized)
- **Requirement:** CTO to confirm no additional user-facing auth is planned for paper trading phase
- For live trading phase: COO will require session-based auth + audit log of dashboard access

---

## 2. Data Access Boundaries

### 2.1 Alpaca Credentials

- `alpaca-credentials` secret already exists in `hypothesis-validation` namespace
- Dashboard uses `ALPACA_PAPER=true` enforced at pod level — **no live trading API calls possible**
- Secret is read-only from the dashboard container — no order submission endpoints in Express API

### 2.2 run-e-state PVC

- PVC mounted read-only at `/app/data` — COO confirms this is acceptable
- Dashboard reads `run_e_state.json` for strategy state display
- **Requirement:** CTO to confirm PVC mount mode is `readOnly: true` in deployment.yaml

### 2.3 No Direct DuckDB Access

- Dashboard reads from Alpaca API and `run_e_state.json` only
- Does NOT connect directly to DuckDB operational tables
- COO reconciliation pipeline remains separate from dashboard data path

---

## 3. Monitoring Requirements

### 3.1 Uptime Monitoring

CTO to add a Prometheus/Grafana uptime probe for the dashboard service:

- **Endpoint:** `https://dashboard.tail16ecc2.ts.net/api/health` (or equivalent healthcheck)
- **Check interval:** 60 seconds
- **Alert threshold:** Service down for > 5 minutes
- **Alert recipients:** CTO (primary), COO (secondary)

### 3.2 Grafana Dashboard Panel

Request CTO add a panel to the existing Grafana monitoring board showing:
- Dashboard pod uptime/restarts
- API endpoint response time (p95)
- WebSocket connection count (active live quote streams)

### 3.3 Loki Log Aggregation

- Dashboard Express server logs MUST be collected by Loki
- Log format: structured JSON (timestamp, level, endpoint, status_code, latency_ms)
- Retention: same as other services (default cluster retention)

---

## 4. Incident Response

### 4.1 Dashboard Down

**Impact:** Loss of operational visibility. Trading continues unaffected (dashboard is read-only).

**Response:**
1. COO notified via Grafana alert
2. COO checks Loki logs for error cause
3. COO creates CTO task: restart pod / investigate if auto-restart fails
4. If down > 30 minutes during trading hours: COO falls back to Alpaca web UI + Grafana metrics
5. CTO resolves and notifies COO

**SLA (paper trading phase):** Best effort. Not a trading-critical service.
**SLA (live trading phase):** 99% uptime during trading hours (09:30–16:00 ET).

### 4.2 Stale Data on Dashboard

**Symptoms:** Dashboard shows outdated positions or P&L vs Alpaca actual.

**Response:**
1. COO compares dashboard data vs Alpaca paper account directly
2. If delta > $100 (paper) or > $1,000 (live): escalate to CTO as data pipeline bug
3. Document discrepancy in reconciliation log (`paper_trading.reconciliation_breaks`)

### 4.3 Alpaca API Credential Failure

**Symptoms:** Portfolio page shows errors, no positions data.

**Response:**
1. CTO rotates `alpaca-credentials` secret
2. CTO performs pod rollout to pick up new secret
3. COO confirms data resumes within 10 minutes

---

## 5. Deployment Approval Gate (COO)

Before CTO deploys QUA-95 to production cluster, COO requires confirmation of:

| Requirement | Owner | Status |
|-------------|-------|--------|
| `ALPACA_PAPER=true` enforced at pod spec | CTO | Pending |
| PVC mounted `readOnly: true` | CTO | Pending |
| Tailscale ingress NOT public | CTO | Pending |
| Grafana uptime probe configured | CTO | Pending |
| Loki log collection confirmed | CTO | Pending |
| Health check endpoint available | CTO (WS-1) | Pending |

COO sign-off on deployment: pending above checklist completion.

---

## 6. Operational Runbook Update

Once QUA-95 is deployed, COO will update the paper trading operations runbook
(`plans/2026-03-28-COO-paper-trading-operations-runbook.md`) to add:
- Dashboard URL for daily operational use
- Fallback procedures when dashboard is unavailable
- Dashboard data cross-check as part of EOD reconciliation routine

---

**COO sign-off (requirements draft):** 50088c37 — 2026-03-28
