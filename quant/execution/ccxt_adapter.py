"""CCXT broker adapter — crypto exchange execution via the ccxt library.

Supports any CCXT-compatible exchange (Binance, Coinbase, Kraken, etc.).
Uses async CCXT for non-blocking I/O with asyncio.

Requires: ccxt >= 4.0.0
Install: pip install ccxt
"""
from __future__ import annotations

import asyncio
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from quant.oms.broker import BrokerAdapter, BrokerError
from quant.oms.models import Fill, Order, OrderSide, OrderType, Position

try:
    import ccxt.async_support as ccxt_async
    _CCXT_AVAILABLE = True
except ImportError:
    _CCXT_AVAILABLE = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CCXTAdapter(BrokerAdapter):
    """Crypto exchange adapter powered by ccxt.

    All order operations run synchronously by executing async calls on a
    dedicated event loop running in a background thread.

    Args:
        exchange_id: ccxt exchange identifier (e.g. "binance", "coinbasepro").
            Falls back to CCXT_EXCHANGE env var.
        api_key: Exchange API key. Falls back to CCXT_API_KEY env var.
        secret: Exchange API secret. Falls back to CCXT_SECRET env var.
        sandbox: When True, use the exchange's sandbox/testnet endpoint.
            Falls back to CCXT_SANDBOX env var. Defaults to True for safety.
        exchange_config: Additional kwargs forwarded to the ccxt exchange
            constructor (e.g. {"password": "..."} for some exchanges).

    Usage::

        adapter = CCXTAdapter(exchange_id="binance", sandbox=True)
        with adapter:
            broker_id = adapter.submit_order(order)
    """

    def __init__(
        self,
        exchange_id: Optional[str] = None,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        sandbox: Optional[bool] = None,
        exchange_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._exchange_id = exchange_id or os.environ.get("CCXT_EXCHANGE", "binance")
        self._api_key = api_key or os.environ.get("CCXT_API_KEY", "")
        self._secret = secret or os.environ.get("CCXT_SECRET", "")

        if sandbox is None:
            env_sb = os.environ.get("CCXT_SANDBOX", "true").lower()
            sandbox = env_sb in ("true", "1", "yes")
        self._sandbox = sandbox
        self._exchange_config = exchange_config or {}

        self._exchange: Optional[object] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return
        if not _CCXT_AVAILABLE:
            raise ImportError(
                "ccxt is required for CCXTAdapter. Install with: pip install ccxt"
            )

        # Start a dedicated event loop in a background thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="ccxt-event-loop"
        )
        self._loop_thread.start()

        exchange_class = getattr(ccxt_async, self._exchange_id, None)
        if exchange_class is None:
            raise BrokerError(f"Unknown ccxt exchange: {self._exchange_id!r}")

        self._exchange = exchange_class(
            {
                "apiKey": self._api_key,
                "secret": self._secret,
                **self._exchange_config,
            }
        )
        if self._sandbox:
            self._exchange.set_sandbox_mode(True)

        # Load markets synchronously (required before trading)
        self._run_async(self._exchange.load_markets())
        self._connected = True
        logger.info(
            "CCXTAdapter: connected to {} (sandbox={})",
            self._exchange_id,
            self._sandbox,
        )

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._exchange is not None:
            try:
                self._run_async(self._exchange.close())
            except Exception:
                pass
            self._exchange = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)
            self._loop = None
        self._connected = False
        logger.info("CCXTAdapter: disconnected from {}", self._exchange_id)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Order operations ──────────────────────────────────────────────────

    def submit_order(self, order: Order) -> str:
        self._require_connected()
        side = "buy" if order.side == OrderSide.BUY else "sell"
        order_type = self._map_order_type(order.order_type)
        params: dict = {}

        price = None
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            price = order.limit_price
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            params["stopPrice"] = order.stop_price

        try:
            response = self._run_async(
                self._exchange.create_order(
                    symbol=order.symbol,
                    type=order_type,
                    side=side,
                    amount=order.quantity,
                    price=price,
                    params=params,
                )
            )
            broker_id = str(response["id"])
            logger.info(
                "CCXTAdapter: submitted {} {} {} qty={} broker_id={}",
                self._exchange_id,
                order.symbol,
                side,
                order.quantity,
                broker_id,
            )
            return broker_id
        except Exception as exc:
            raise BrokerError(f"CCXT order submission failed: {exc}") from exc

    def cancel_order(self, broker_order_id: str) -> bool:
        self._require_connected()
        try:
            # ccxt cancel_order requires symbol on some exchanges — we store it
            # in the order; fall back to fetching the order first
            self._run_async(
                self._exchange.cancel_order(broker_order_id)
            )
            logger.info("CCXTAdapter: cancelled broker_order_id={}", broker_order_id)
            return True
        except Exception as exc:
            logger.warning("CCXTAdapter: cancel failed for {} — {}", broker_order_id, exc)
            return False

    # ── Query operations ──────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[Position]:
        self._require_connected()
        try:
            positions = self._run_async(self._exchange.fetch_positions([symbol]))
            for pos in positions:
                if pos.get("symbol") == symbol and pos.get("contracts", 0) != 0:
                    qty = float(pos.get("contracts", 0))
                    if pos.get("side") == "short":
                        qty = -qty
                    return Position(
                        symbol=symbol,
                        quantity=qty,
                        avg_cost=float(pos.get("entryPrice") or 0),
                        market_price=float(pos.get("markPrice") or pos.get("entryPrice") or 0),
                    )
        except Exception:
            pass
        # Spot balance fallback
        try:
            base = symbol.split("/")[0]
            balance = self._run_async(self._exchange.fetch_balance())
            qty = float(balance.get("free", {}).get(base, 0))
            if qty > 0:
                ticker = self._run_async(self._exchange.fetch_ticker(symbol))
                price = float(ticker.get("last") or 0)
                return Position(symbol=symbol, quantity=qty, market_price=price)
        except Exception:
            pass
        return None

    def get_fills(self, broker_order_id: str) -> list[Fill]:
        self._require_connected()
        try:
            trades = self._run_async(self._exchange.fetch_order_trades(broker_order_id))
            fills = []
            for t in trades:
                fills.append(
                    Fill(
                        order_id="",
                        broker_order_id=broker_order_id,
                        fill_id=str(t.get("id", uuid.uuid4())),
                        symbol=str(t.get("symbol", "")),
                        side=OrderSide.BUY if t.get("side") == "buy" else OrderSide.SELL,
                        quantity=float(t.get("amount", 0)),
                        price=float(t.get("price", 0)),
                        filled_at=datetime.fromtimestamp(
                            t.get("timestamp", 0) / 1000, tz=timezone.utc
                        ),
                        commission=float(t.get("fee", {}).get("cost", 0)),
                    )
                )
            return fills
        except Exception as exc:
            logger.warning("CCXTAdapter: get_fills failed — {}", exc)
            return []

    def get_account_cash(self) -> float:
        self._require_connected()
        try:
            balance = self._run_async(self._exchange.fetch_balance())
            # Return USDT/USD free balance as a proxy for cash
            for quote in ("USDT", "USD", "BUSD", "USDC"):
                cash = balance.get("free", {}).get(quote)
                if cash is not None:
                    return float(cash)
            return float(balance.get("total", {}).get("USDT", 0.0))
        except Exception as exc:
            raise BrokerError(f"Failed to fetch CCXT balance: {exc}") from exc

    # ── Helpers ───────────────────────────────────────────────────────────

    def _run_async(self, coro):
        """Run a coroutine on the dedicated event loop and return the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerError("CCXTAdapter is not connected. Call connect() first.")

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
        }
        return mapping.get(order_type, "market")
