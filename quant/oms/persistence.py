"""OMS state persistence — SQLite write-through store.

Persists orders, fills, positions, and portfolio snapshots so that the OMS
can recover its full state after a process restart.  Uses SQLite in WAL mode
for safe concurrent reads/writes from a single process.

Design:
  * Write-through: every OMS mutation (submit, fill, cancel) persists
    synchronously before the call returns.
  * Recovery: ``load_*`` methods reconstruct the in-memory state on startup.
  * Schema auto-migration: tables are created if missing via ``_ensure_schema``.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from quant.oms.models import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)

# ISO-8601 format used for all stored timestamps.
_DT_FMT = "%Y-%m-%dT%H:%M:%S.%f%z"


def _dt_to_str(dt: datetime) -> str:
    return dt.strftime(_DT_FMT)


def _str_to_dt(s: str) -> datetime:
    return datetime.strptime(s, _DT_FMT)


def _opt_dt_to_str(dt: datetime | None) -> str | None:
    return _dt_to_str(dt) if dt is not None else None


def _opt_str_to_dt(s: str | None) -> datetime | None:
    return _str_to_dt(s) if s is not None else None


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    quantity        REAL    NOT NULL,
    order_type      TEXT    NOT NULL,
    limit_price     REAL,
    stop_price      REAL,
    time_in_force   TEXT    NOT NULL,
    broker_order_id TEXT,
    status          TEXT    NOT NULL,
    filled_quantity REAL    NOT NULL DEFAULT 0.0,
    avg_fill_price  REAL    NOT NULL DEFAULT 0.0,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    strategy_id     TEXT,
    sector          TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id          TEXT PRIMARY KEY,
    order_id         TEXT NOT NULL,
    broker_order_id  TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    side             TEXT NOT NULL,
    quantity         REAL NOT NULL,
    price            REAL NOT NULL,
    filled_at        TEXT NOT NULL,
    commission       REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_broker_order_id ON fills(broker_order_id);

CREATE TABLE IF NOT EXISTS positions (
    symbol       TEXT PRIMARY KEY,
    quantity     REAL NOT NULL,
    avg_cost     REAL NOT NULL,
    market_price REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT    NOT NULL,
    cash                 REAL    NOT NULL,
    peak_portfolio_value REAL    NOT NULL
);
"""


class SQLiteStateStore:
    """SQLite-backed OMS state store.

    Args:
        db_path: Path to the SQLite database file.  Use ``":memory:"`` for
            an ephemeral in-memory store (useful for tests).
    """

    def __init__(self, db_path: str | Path = "oms_state.db") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _ensure_schema(self) -> None:
        with self._cursor() as cur:
            cur.executescript(_SCHEMA_SQL)

    # ── Orders ─────────────────────────────────────────────────────────────

    def save_order(self, order: Order) -> None:
        """Insert or update an order (upsert)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO orders (
                    id, symbol, side, quantity, order_type, limit_price,
                    stop_price, time_in_force, broker_order_id, status,
                    filled_quantity, avg_fill_price, created_at, updated_at,
                    strategy_id, sector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    broker_order_id = excluded.broker_order_id,
                    status          = excluded.status,
                    filled_quantity = excluded.filled_quantity,
                    avg_fill_price  = excluded.avg_fill_price,
                    updated_at      = excluded.updated_at
                """,
                (
                    order.id,
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.order_type.value,
                    order.limit_price,
                    order.stop_price,
                    order.time_in_force.value,
                    order.broker_order_id,
                    order.status.value,
                    order.filled_quantity,
                    order.avg_fill_price,
                    _dt_to_str(order.created_at),
                    _dt_to_str(order.updated_at),
                    order.strategy_id,
                    order.sector,
                ),
            )

    def load_orders(self) -> list[Order]:
        """Load all orders from the database."""
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT id, symbol, side, quantity, order_type, limit_price, "
                "stop_price, time_in_force, broker_order_id, status, "
                "filled_quantity, avg_fill_price, created_at, updated_at, "
                "strategy_id, sector FROM orders"
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def load_active_orders(self) -> list[Order]:
        """Load only orders in non-terminal states."""
        active_states = (
            OrderStatus.PENDING.value,
            OrderStatus.SUBMITTED.value,
            OrderStatus.ACCEPTED.value,
            OrderStatus.PARTIALLY_FILLED.value,
        )
        placeholders = ",".join("?" for _ in active_states)
        with self._cursor() as cur:
            rows = cur.execute(
                f"SELECT id, symbol, side, quantity, order_type, limit_price, "
                f"stop_price, time_in_force, broker_order_id, status, "
                f"filled_quantity, avg_fill_price, created_at, updated_at, "
                f"strategy_id, sector FROM orders WHERE status IN ({placeholders})",
                active_states,
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    @staticmethod
    def _row_to_order(row: tuple) -> Order:
        return Order(
            symbol=row[1],
            side=OrderSide(row[2]),
            quantity=row[3],
            order_type=OrderType(row[4]),
            limit_price=row[5],
            stop_price=row[6],
            time_in_force=TimeInForce(row[7]),
            id=row[0],
            broker_order_id=row[8],
            status=OrderStatus(row[9]),
            filled_quantity=row[10],
            avg_fill_price=row[11],
            created_at=_str_to_dt(row[12]),
            updated_at=_str_to_dt(row[13]),
            strategy_id=row[14],
            sector=row[15],
        )

    # ── Fills ──────────────────────────────────────────────────────────────

    def save_fill(self, fill: Fill) -> None:
        """Insert a fill (fills are immutable; duplicates are ignored)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR IGNORE INTO fills (
                    fill_id, order_id, broker_order_id, symbol, side,
                    quantity, price, filled_at, commission
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.fill_id,
                    fill.order_id,
                    fill.broker_order_id,
                    fill.symbol,
                    fill.side.value,
                    fill.quantity,
                    fill.price,
                    _dt_to_str(fill.filled_at),
                    fill.commission,
                ),
            )

    def load_fills(self) -> list[Fill]:
        """Load all fills from the database."""
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT fill_id, order_id, broker_order_id, symbol, side, "
                "quantity, price, filled_at, commission FROM fills"
            ).fetchall()
        return [self._row_to_fill(r) for r in rows]

    def load_fills_for_order(self, order_id: str) -> list[Fill]:
        """Load fills belonging to a specific OMS order."""
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT fill_id, order_id, broker_order_id, symbol, side, "
                "quantity, price, filled_at, commission "
                "FROM fills WHERE order_id = ?",
                (order_id,),
            ).fetchall()
        return [self._row_to_fill(r) for r in rows]

    @staticmethod
    def _row_to_fill(row: tuple) -> Fill:
        return Fill(
            fill_id=row[0],
            order_id=row[1],
            broker_order_id=row[2],
            symbol=row[3],
            side=OrderSide(row[4]),
            quantity=row[5],
            price=row[6],
            filled_at=_str_to_dt(row[7]),
            commission=row[8],
        )

    # ── Positions ──────────────────────────────────────────────────────────

    def save_position(self, position: Position) -> None:
        """Insert or update a position (upsert)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO positions (symbol, quantity, avg_cost, market_price)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity     = excluded.quantity,
                    avg_cost     = excluded.avg_cost,
                    market_price = excluded.market_price
                """,
                (position.symbol, position.quantity, position.avg_cost, position.market_price),
            )

    def delete_position(self, symbol: str) -> None:
        """Remove a closed position."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

    def load_positions(self) -> dict[str, Position]:
        """Load all open positions keyed by symbol."""
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT symbol, quantity, avg_cost, market_price FROM positions"
            ).fetchall()
        return {
            row[0]: Position(
                symbol=row[0], quantity=row[1], avg_cost=row[2], market_price=row[3]
            )
            for row in rows
        }

    # ── Snapshots ──────────────────────────────────────────────────────────

    def save_snapshot(self, cash: float, peak_portfolio_value: float) -> None:
        """Save a portfolio state snapshot (for circuit breaker recovery)."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO snapshots (timestamp, cash, peak_portfolio_value) "
                "VALUES (?, ?, ?)",
                (
                    _dt_to_str(datetime.now(timezone.utc)),
                    cash,
                    peak_portfolio_value,
                ),
            )

    def load_latest_snapshot(self) -> dict | None:
        """Load the most recent snapshot, or None if none exist.

        Returns:
            Dict with keys ``timestamp``, ``cash``, ``peak_portfolio_value``,
            or None.
        """
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT timestamp, cash, peak_portfolio_value "
                "FROM snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "timestamp": _str_to_dt(row[0]),
            "cash": row[1],
            "peak_portfolio_value": row[2],
        }

    # ── Maintenance ────────────────────────────────────────────────────────

    def purge_terminal_orders(self, before: datetime) -> int:
        """Delete terminal orders (and their fills) older than *before*.

        Useful for keeping the database lean over long-running deployments.
        Returns the number of orders deleted.
        """
        terminal_states = (
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
        )
        placeholders = ",".join("?" for _ in terminal_states)
        ts = _dt_to_str(before)
        with self._cursor() as cur:
            # Find matching order IDs first
            order_ids = [
                row[0]
                for row in cur.execute(
                    f"SELECT id FROM orders "
                    f"WHERE status IN ({placeholders}) AND updated_at < ?",
                    (*terminal_states, ts),
                ).fetchall()
            ]
            if not order_ids:
                return 0
            id_placeholders = ",".join("?" for _ in order_ids)
            cur.execute(
                f"DELETE FROM fills WHERE order_id IN ({id_placeholders})",
                order_ids,
            )
            cur.execute(
                f"DELETE FROM orders WHERE id IN ({id_placeholders})",
                order_ids,
            )
        return len(order_ids)
