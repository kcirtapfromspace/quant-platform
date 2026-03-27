"""Alpaca broker adapter — paper and live trading via alpaca-py SDK.

Supports:
  - REST order submission and cancellation
  - WebSocket streaming for real-time fill events
  - Both paper (ALPACA_PAPER=true) and live endpoints

Requires: alpaca-py >= 0.8.0
Install: pip install alpaca-py
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from quant.oms.broker import BrokerAdapter, BrokerError
from quant.oms.models import Fill, Order, OrderSide, OrderType, Position, TimeInForce

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        StopLimitOrderRequest,
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaSide,
        TimeInForce as AlpacaTIF,
        OrderStatus as AlpacaOrderStatus,
    )
    from alpaca.trading.stream import TradingStream
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_TIF_MAP = {
    TimeInForce.DAY: "day",
    TimeInForce.GTC: "gtc",
    TimeInForce.IOC: "ioc",
    TimeInForce.FOK: "fok",
}


class AlpacaAdapter(BrokerAdapter):
    """Alpaca execution adapter (paper + live).

    Args:
        api_key: Alpaca API key ID. Falls back to ALPACA_API_KEY env var.
        secret_key: Alpaca secret key. Falls back to ALPACA_SECRET_KEY env var.
        paper: When True, uses Alpaca paper trading endpoint. Falls back to
            ALPACA_PAPER env var ("true"/"1"). Defaults to True for safety.
        stream_fills: When True, starts a WebSocket stream for real-time fills.

    Usage::

        adapter = AlpacaAdapter()  # reads from env vars
        with adapter:
            broker_id = adapter.submit_order(order)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
        stream_fills: bool = True,
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")

        if paper is None:
            env_paper = os.environ.get("ALPACA_PAPER", "true").lower()
            paper = env_paper in ("true", "1", "yes")
        self._paper = paper
        self._stream_fills = stream_fills

        self._client: Optional[object] = None
        self._stream: Optional[object] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required for AlpacaAdapter. "
                "Install with: pip install alpaca-py"
            )
        if not self._api_key or not self._secret_key:
            raise BrokerError(
                "Alpaca API credentials not set. "
                "Provide api_key/secret_key or set ALPACA_API_KEY/ALPACA_SECRET_KEY."
            )
        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )
        self._connected = True
        logger.info(
            "AlpacaAdapter: connected (paper={})", self._paper
        )

        if self._stream_fills:
            self._start_fill_stream()

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=5)
            self._stream_thread = None
        self._client = None
        self._connected = False
        logger.info("AlpacaAdapter: disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Order operations ──────────────────────────────────────────────────

    def submit_order(self, order: Order) -> str:
        self._require_connected()
        try:
            request = self._build_order_request(order)
            response = self._client.submit_order(request)
            broker_id = str(response.id)
            logger.info(
                "AlpacaAdapter: submitted {} {} qty={} broker_id={}",
                order.symbol,
                order.side.value,
                order.quantity,
                broker_id,
            )
            return broker_id
        except Exception as exc:
            raise BrokerError(f"Alpaca order submission failed: {exc}") from exc

    def cancel_order(self, broker_order_id: str) -> bool:
        self._require_connected()
        try:
            self._client.cancel_order_by_id(broker_order_id)
            logger.info("AlpacaAdapter: cancelled broker_order_id={}", broker_order_id)
            return True
        except Exception as exc:
            logger.warning("AlpacaAdapter: cancel failed for {} — {}", broker_order_id, exc)
            return False

    # ── Query operations ──────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[Position]:
        self._require_connected()
        try:
            pos = self._client.get_open_position(symbol)
            return Position(
                symbol=symbol,
                quantity=float(pos.qty),
                avg_cost=float(pos.avg_entry_price),
                market_price=float(pos.current_price or pos.avg_entry_price),
            )
        except Exception:
            return None

    def get_fills(self, broker_order_id: str) -> list[Fill]:
        self._require_connected()
        try:
            activities = self._client.get_activities(
                activity_types=["FILL"],
                order_id=broker_order_id,
            )
            fills = []
            for act in activities:
                fills.append(
                    Fill(
                        order_id="",  # not known here; OMS sets via broker_id map
                        broker_order_id=broker_order_id,
                        fill_id=str(act.id),
                        symbol=str(act.symbol),
                        side=OrderSide.BUY if act.side == "buy" else OrderSide.SELL,
                        quantity=float(act.qty),
                        price=float(act.price),
                        filled_at=act.transaction_time,
                    )
                )
            return fills
        except Exception as exc:
            logger.warning("AlpacaAdapter: get_fills failed for {} — {}", broker_order_id, exc)
            return []

    def get_account_cash(self) -> float:
        self._require_connected()
        try:
            account = self._client.get_account()
            return float(account.cash)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch account cash: {exc}") from exc

    # ── Internal helpers ──────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerError("AlpacaAdapter is not connected. Call connect() first.")

    def _build_order_request(self, order: Order):
        tif = _TIF_MAP.get(order.time_in_force, "day")
        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
            )
        elif order.order_type == OrderType.LIMIT:
            return LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
            )
        elif order.order_type == OrderType.STOP:
            return StopOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                stop_price=order.stop_price,
            )
        elif order.order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )
        raise BrokerError(f"Unsupported order type: {order.order_type}")

    def _start_fill_stream(self) -> None:
        """Start a background thread running the Alpaca WebSocket fill stream."""
        if not _ALPACA_AVAILABLE:
            return
        self._stream = TradingStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=self._paper,
        )

        async def _on_trade_update(data):
            event = getattr(data, "event", None)
            if event not in ("fill", "partial_fill"):
                return
            order_obj = getattr(data, "order", None)
            if order_obj is None:
                return
            try:
                fill = Fill(
                    order_id="",  # resolved by OMS via broker_id map
                    broker_order_id=str(order_obj.id),
                    fill_id=str(uuid.uuid4()),
                    symbol=str(order_obj.symbol),
                    side=OrderSide.BUY if order_obj.side == "buy" else OrderSide.SELL,
                    quantity=float(order_obj.filled_qty or 0),
                    price=float(order_obj.filled_avg_price or 0),
                    filled_at=_utcnow(),
                )
                self._notify_fill(fill)
            except Exception:
                logger.exception("AlpacaAdapter: error processing fill stream event")

        self._stream.subscribe_trade_updates(_on_trade_update)

        def _run_stream():
            try:
                self._stream.run()
            except Exception:
                logger.exception("AlpacaAdapter: fill stream exited with error")

        self._stream_thread = threading.Thread(
            target=_run_stream, daemon=True, name="alpaca-fill-stream"
        )
        self._stream_thread.start()
        logger.info("AlpacaAdapter: fill stream started")
