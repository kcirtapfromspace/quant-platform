# Daily NAV & Position Reconciliation Procedure

**Author:** COO (50088c37)
**Date:** 2026-03-28
**Frequency:** Every trading day by 17:30 ET
**Status:** ACTIVE

---

## Purpose

Produce a definitive daily NAV and confirm zero position breaks between the OMS
(`data/paper_trading.duckdb`) and Alpaca broker state. This is the operational
basis for all performance reporting and CRO drawdown monitoring.

---

## Source of Truth Hierarchy

| Data | Source of Truth |
|------|----------------|
| Position quantities | Alpaca (broker) |
| Average cost basis | Alpaca (broker) |
| Cash balance | Alpaca GET /v2/account → `cash` |
| Closing prices | Alpaca latest trade / prior close |
| Historical NAV | DuckDB `daily_nav` table (OMS) |
| Order fills | Alpaca GET /v2/orders (filled) |

---

## Step-by-Step: End-of-Day NAV Calculation

### 1. Pull Alpaca Account State (16:05 ET — after close)

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest

client = TradingClient(api_key, secret_key, paper=True)
account = client.get_account()
positions = client.get_all_positions()

cash_balance = float(account.cash)
portfolio_value = float(account.portfolio_value)  # Alpaca's own calculation
```

### 2. Mark Positions to Close

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

data_client = StockHistoricalDataClient(api_key, secret_key)
symbols = [p.symbol for p in positions]
latest = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbols))

position_values = {}
for pos in positions:
    close_price = float(latest[pos.symbol].price)
    qty = float(pos.qty)
    market_value = close_price * qty
    unrealized_pnl = (close_price - float(pos.avg_entry_price)) * qty
    position_values[pos.symbol] = {
        "qty": qty,
        "avg_cost": float(pos.avg_entry_price),
        "close_price": close_price,
        "market_value": market_value,
        "unrealized_pnl": unrealized_pnl,
    }
```

### 3. Calculate NAV

```python
total_market_value = sum(p["market_value"] for p in position_values.values())
nav = cash_balance + total_market_value

# Sanity check vs Alpaca's own portfolio_value (should be within 0.5%)
alpaca_nav = portfolio_value
nav_drift_bps = abs(nav - alpaca_nav) / alpaca_nav * 10000
if nav_drift_bps > 50:
    alert("NAV DRIFT: OMS NAV={nav:.2f} vs Alpaca={alpaca_nav:.2f} ({nav_drift_bps:.0f} bps)")
```

### 4. Compute Returns and Drawdown

```python
# From DuckDB
prior_nav = db.execute("SELECT nav FROM daily_nav ORDER BY date DESC LIMIT 1").fetchone()[0]
starting_nav = 1_000_000.0  # per QUA-22

daily_return = (nav - prior_nav) / prior_nav
cumulative_return = (nav - starting_nav) / starting_nav

# Drawdown
peak_nav = db.execute("SELECT MAX(nav) FROM daily_nav").fetchone()[0]
peak_nav = max(peak_nav or starting_nav, nav)  # account for first day
current_drawdown = (peak_nav - nav) / peak_nav
```

### 5. Write to DuckDB

```sql
INSERT INTO daily_nav (date, nav, daily_return, cumulative_return, drawdown, cash, position_count)
VALUES (CURRENT_DATE, ?, ?, ?, ?, ?, ?)
```

---

## Position Reconciliation

Run using `quant.oms.reconciliation.PositionReconciler`. Broker is always source of truth.

### Break Classification

| Type | Definition | Action |
|------|-----------|--------|
| `MATCHED` | OMS qty = broker qty ± 1e-6 | No action |
| `QUANTITY_BREAK` | Both have position, qty differs | Apply SET_QUANTITY correction same day |
| `PHANTOM` | OMS has position, broker does not | Apply REMOVE_POSITION correction same day |
| `MISSING` | Broker has position, OMS does not | Apply CREATE_POSITION correction same day |

### Break Resolution SLA

- **Quantity break / Phantom / Missing**: Must resolve before next market open.
- If cause is unclear: escalate to CTO. Do NOT paper over with corrections until root cause known.
- Document every break in `daily_recon_log` DuckDB table with: date, symbol, break_type, oms_qty, broker_qty, resolution, resolved_by.

### Average Cost Drift

If `price_drift_pct > 1%` on a matched position:
- Flag in daily log (informational)
- Investigate if persistent >3 days (possible corporate action or split not reflected in OMS)
- Escalate to CTO if systematic

---

## DuckDB Schema (required tables)

```sql
-- NAV history
CREATE TABLE IF NOT EXISTS daily_nav (
    date          DATE PRIMARY KEY,
    nav           DOUBLE NOT NULL,
    daily_return  DOUBLE,
    cumulative_return DOUBLE,
    drawdown      DOUBLE,
    cash          DOUBLE,
    position_count INTEGER,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reconciliation log
CREATE TABLE IF NOT EXISTS daily_recon_log (
    id            INTEGER PRIMARY KEY,
    recon_date    DATE NOT NULL,
    symbol        VARCHAR,
    break_type    VARCHAR,  -- MATCHED / QUANTITY_BREAK / PHANTOM / MISSING
    oms_qty       DOUBLE,
    broker_qty    DOUBLE,
    qty_diff      DOUBLE,
    price_drift_pct DOUBLE,
    resolution    VARCHAR,  -- 'auto_corrected' / 'manual' / 'pending'
    resolved_by   VARCHAR,
    notes         TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily P&L attribution by sleeve
CREATE TABLE IF NOT EXISTS daily_sleeve_pnl (
    date          DATE NOT NULL,
    sleeve        VARCHAR NOT NULL,  -- 'momentum' / 'trend' / 'adaptive'
    pnl           DOUBLE,
    weight        DOUBLE,
    PRIMARY KEY (date, sleeve)
);
```

Schema migration: CTO to provision these tables on `data/paper_trading.duckdb` before go-live.

---

## Automated vs Manual

| Task | Automated? | Who |
|------|-----------|-----|
| Pull Alpaca positions | Yes (runner.py EOD job) | CTO infra |
| NAV calculation | Yes (runner.py EOD job) | CTO infra |
| DB write | Yes (runner.py EOD job) | CTO infra |
| Break detection | Yes (recon module) | CTO infra |
| Break resolution | Manual review required | COO |
| Escalation if unresolved | Manual | COO |

CTO action required: Confirm EOD jobs are scheduled in the paper trading runner. If not, COO will run manually until automation is in place.

---

## Attestation Log Format

Daily COO attestation (logged to `plans/operations-log-YYYY-MM.md`):

```
YYYY-MM-DD | NAV: $X,XXX,XXX | Daily P&L: +/-$X,XXX (X.XX%) | DD: X.X% |
Trades: N | Breaks: 0 | Recon: CLEAN | Grafana: GREEN | CB: NOT TRIPPED
```

---

**COO sign-off:** 50088c37 — 2026-03-28
