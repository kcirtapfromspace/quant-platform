"""Initial schema — OHLCV, orders, positions, backtest_runs.

Creates core tables and configures ohlcv_bars as a TimescaleDB hypertable.

Revision ID: 0001
Revises:
Create Date: 2026-03-27
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # ohlcv_bars — TimescaleDB hypertable
    # ------------------------------------------------------------------
    op.create_table(
        "ohlcv_bars",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("interval", sa.String(8), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("adj_close", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "ts", "interval", name="uq_ohlcv_symbol_ts_interval"),
    )
    op.create_index("ix_ohlcv_symbol_interval_ts", "ohlcv_bars", ["symbol", "interval", "ts"])
    op.create_index("ix_ohlcv_bars_ts", "ohlcv_bars", ["ts"])

    # Convert to TimescaleDB hypertable (no-op on plain PostgreSQL without TimescaleDB)
    op.execute(
        "SELECT create_hypertable('ohlcv_bars', 'ts', if_not_exists => TRUE) "
        "WHERE EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
    )

    # ------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------
    op.create_table(
        "orders",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("order_id", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("order_type", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("limit_price", sa.Float(), nullable=True),
        sa.Column("filled_quantity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_fill_price", sa.Float(), nullable=True),
        sa.Column("strategy_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("order_id"),
    )
    op.create_index("ix_orders_symbol_created", "orders", ["symbol", "created_at"])
    op.create_index("ix_orders_status", "orders", ["status"])

    # ------------------------------------------------------------------
    # positions
    # ------------------------------------------------------------------
    op.create_table(
        "positions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("avg_cost", sa.Float(), nullable=False),
        sa.Column("realized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol"),
    )

    # ------------------------------------------------------------------
    # backtest_runs
    # ------------------------------------------------------------------
    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("start_date", sa.String(10), nullable=False),
        sa.Column("end_date", sa.String(10), nullable=False),
        sa.Column("universe", sa.Text(), nullable=False),
        sa.Column("total_return", sa.Numeric(10, 6), nullable=True),
        sa.Column("sharpe_ratio", sa.Numeric(10, 6), nullable=True),
        sa.Column("max_drawdown", sa.Numeric(10, 6), nullable=True),
        sa.Column("result_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    op.create_index("ix_backtest_strategy_date", "backtest_runs", ["strategy_id", "created_at"])


def downgrade() -> None:
    op.drop_table("backtest_runs")
    op.drop_table("positions")
    op.drop_table("orders")
    op.drop_table("ohlcv_bars")
