"""SQLAlchemy ORM models for PostgreSQL / TimescaleDB.

TimescaleDB hypertables are created via Alembic migrations using raw SQL
(``op.execute("SELECT create_hypertable(...)")``) since SQLAlchemy does not
natively know about TimescaleDB-specific DDL.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class OHLCVBar(Base):
    """Daily / intraday OHLCV bars.

    Stored in a TimescaleDB hypertable partitioned by ``ts``.
    The migration creates the hypertable after the table is created.
    """

    __tablename__ = "ohlcv_bars"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    interval: Mapped[str] = mapped_column(String(8), nullable=False)  # e.g. "1d", "1h"
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        UniqueConstraint("symbol", "ts", "interval", name="uq_ohlcv_symbol_ts_interval"),
        Index("ix_ohlcv_symbol_interval_ts", "symbol", "interval", "ts"),
    )


class Order(Base):
    """Persisted order records from the OMS."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)   # buy / sell
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)  # market/limit
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    filled_quantity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    strategy_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        Index("ix_orders_symbol_created", "symbol", "created_at"),
        Index("ix_orders_status", "status"),
    )


class Position(Base):
    """Current position snapshot per symbol."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    avg_cost: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


class BacktestRun(Base):
    """Backtest run metadata — results are stored as parquet in MinIO."""

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    start_date: Mapped[str] = mapped_column(String(10), nullable=False)  # ISO date
    end_date: Mapped[str] = mapped_column(String(10), nullable=False)
    universe: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    total_return: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    result_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # MinIO path
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (Index("ix_backtest_strategy_date", "strategy_id", "created_at"),)
