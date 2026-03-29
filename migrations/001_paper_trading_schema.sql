-- Migration 001: Paper trading operational schema
-- CRO requirement: QUA-77 Flag 3 — provision before 2026-03-30 09:30 ET
-- Run: duckdb data/paper_trading.duckdb < migrations/001_paper_trading_schema.sql

-- Daily NAV history
CREATE TABLE IF NOT EXISTS daily_nav (
    date              DATE PRIMARY KEY,
    nav               DOUBLE NOT NULL,
    daily_return      DOUBLE,
    cumulative_return DOUBLE,
    drawdown          DOUBLE,
    cash              DOUBLE,
    position_count    INTEGER,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Position reconciliation audit trail
-- break_type: MATCHED | QUANTITY_BREAK | PHANTOM | MISSING
-- resolution: auto_corrected | manual | pending
CREATE TABLE IF NOT EXISTS daily_recon_log (
    id              INTEGER PRIMARY KEY,
    recon_date      DATE NOT NULL,
    symbol          VARCHAR,
    break_type      VARCHAR,
    oms_qty         DOUBLE,
    broker_qty      DOUBLE,
    qty_diff        DOUBLE,
    price_drift_pct DOUBLE,
    resolution      VARCHAR,
    resolved_by     VARCHAR,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-sleeve daily P&L attribution
-- sleeve: momentum | trend | adaptive
CREATE TABLE IF NOT EXISTS daily_sleeve_pnl (
    date   DATE    NOT NULL,
    sleeve VARCHAR NOT NULL,
    pnl    DOUBLE,
    weight DOUBLE,
    PRIMARY KEY (date, sleeve)
);
