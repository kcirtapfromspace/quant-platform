"""Interactive Brokers adapter via ib_insync.

This adapter is provided as a scaffolded implementation for future use.
ib_insync requires a running TWS (Trader Workstation) or IB Gateway instance.

Requires: ib_insync >= 0.9.86
Install: pip install ib_insync

Connection prereqs:
  - TWS or IB Gateway running on the configured host/port
  - API connections enabled in TWS (Global Configuration → API → Settings)
  - Client ID must be unique per connection (no two processes share the same ID)
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
    import ib_insync as ibi
    _IB_AVAILABLE = True
except ImportError:
    _IB_AVAILABLE = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_TIF_MAP = {
    TimeInForce.DAY: "DAY",
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
}


class IBAdapter(BrokerAdapter):
    """Interactive Brokers execution adapter via ib_insync.

    Args:
        host: TWS/IB Gateway host (default "127.0.0.1"). Falls back to IB_HOST.
        port: TWS/IB Gateway port (default 7497 paper / 7496 live).
            Falls back to IB_PORT.
        client_id: Unique integer client ID for this connection.
            Falls back to IB_CLIENT_ID (default 1).
        paper: When True, expects a paper trading port (7497). Defaults to True.
            Falls back to IB_PAPER env var.

    Usage::

        adapter = IBAdapter(host="127.0.0.1", port=7497)
        with adapter:
            broker_id = adapter.submit_order(order)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        paper: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self._host = host or os.environ.get("IB_HOST", "127.0.0.1")
        self._client_id = client_id or int(os.environ.get("IB_CLIENT_ID", "1"))

        if paper is None:
            env_paper = os.environ.get("IB_PAPER", "true").lower()
            paper = env_paper in ("true", "1", "yes")
        self._paper = paper

        default_port = 7497 if paper else 7496
        if port is not None:
            self._port = port
        else:
            self._port = int(os.environ.get("IB_PORT", str(default_port)))

        self._ib: Optional[object] = None
        self._connected = False

        # broker_order_id (str) → ib Trade object
        self._trades: dict = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return
        if not _IB_AVAILABLE:
            raise ImportError(
                "ib_insync is required for IBAdapter. "
                "Install with: pip install ib_insync"
            )
        self._ib = ibi.IB()
        try:
            self._ib.connect(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
            )
        except Exception as exc:
            raise BrokerError(
                f"IB connect failed ({self._host}:{self._port}): {exc}"
            ) from exc

        self._ib.orderStatusEvent += self._on_order_status
        self._connected = True
        logger.info(
            "IBAdapter: connected to {}:{} clientId={} paper={}",
            self._host,
            self._port,
            self._client_id,
            self._paper,
        )

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
        self._connected = False
        logger.info("IBAdapter: disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    # ── Order operations ──────────────────────────────────────────────────

    def submit_order(self, order: Order) -> str:
        self._require_connected()
        contract = self._build_contract(order.symbol)
        ib_order = self._build_ib_order(order)

        try:
            trade = self._ib.placeOrder(contract, ib_order)
            # perm_id is assigned asynchronously; use orderId as broker reference
            broker_id = str(trade.order.orderId)
            self._trades[broker_id] = trade
            logger.info(
                "IBAdapter: submitted {} {} qty={} broker_id={}",
                order.symbol,
                order.side.value,
                order.quantity,
                broker_id,
            )
            return broker_id
        except Exception as exc:
            raise BrokerError(f"IB order submission failed: {exc}") from exc

    def cancel_order(self, broker_order_id: str) -> bool:
        self._require_connected()
        trade = self._trades.get(broker_order_id)
        if trade is None:
            logger.warning("IBAdapter: cancel_order — unknown broker_id={}", broker_order_id)
            return False
        try:
            self._ib.cancelOrder(trade.order)
            logger.info("IBAdapter: cancel requested for broker_id={}", broker_order_id)
            return True
        except Exception as exc:
            logger.warning("IBAdapter: cancel failed for {} — {}", broker_order_id, exc)
            return False

    # ── Query operations ──────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[Position]:
        self._require_connected()
        try:
            positions = self._ib.positions()
            for pos in positions:
                if pos.contract.symbol == symbol:
                    avg_cost = float(pos.avgCost) if pos.avgCost else 0.0
                    return Position(
                        symbol=symbol,
                        quantity=float(pos.position),
                        avg_cost=avg_cost,
                        market_price=avg_cost,  # current price requires a market data sub
                    )
        except Exception as exc:
            logger.warning("IBAdapter: get_position failed — {}", exc)
        return None

    def get_fills(self, broker_order_id: str) -> list[Fill]:
        self._require_connected()
        fills = []
        try:
            executions = self._ib.executions()
            for exec_obj in executions:
                if str(exec_obj.orderId) == broker_order_id:
                    fills.append(
                        Fill(
                            order_id="",
                            broker_order_id=broker_order_id,
                            fill_id=exec_obj.execId,
                            symbol=exec_obj.contract.symbol,
                            side=OrderSide.BUY if exec_obj.side == "BOT" else OrderSide.SELL,
                            quantity=float(exec_obj.shares),
                            price=float(exec_obj.price),
                            filled_at=_utcnow(),
                        )
                    )
        except Exception as exc:
            logger.warning("IBAdapter: get_fills failed — {}", exc)
        return fills

    def get_account_cash(self) -> float:
        self._require_connected()
        try:
            account_values = self._ib.accountValues()
            for av in account_values:
                if av.tag == "CashBalance" and av.currency == "BASE":
                    return float(av.value)
            # Fallback: NetLiquidation
            for av in account_values:
                if av.tag == "NetLiquidation":
                    return float(av.value)
        except Exception as exc:
            raise BrokerError(f"IB get_account_cash failed: {exc}") from exc
        return 0.0

    # ── Helpers ───────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerError("IBAdapter is not connected. Call connect() first.")

    def _build_contract(self, symbol: str):
        """Build an IB Stock contract for US equities."""
        contract = ibi.Stock(symbol, "SMART", "USD")
        return contract

    def _build_ib_order(self, order: Order):
        """Translate an OMS Order to an ib_insync Order object."""
        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        tif = _TIF_MAP.get(order.time_in_force, "DAY")

        if order.order_type == OrderType.MARKET:
            return ibi.MarketOrder(action=action, totalQuantity=order.quantity, tif=tif)
        elif order.order_type == OrderType.LIMIT:
            return ibi.LimitOrder(
                action=action,
                totalQuantity=order.quantity,
                lmtPrice=order.limit_price,
                tif=tif,
            )
        elif order.order_type == OrderType.STOP:
            return ibi.StopOrder(
                action=action,
                totalQuantity=order.quantity,
                stopPrice=order.stop_price,
                tif=tif,
            )
        elif order.order_type == OrderType.STOP_LIMIT:
            return ibi.StopLimitOrder(
                action=action,
                totalQuantity=order.quantity,
                lmtPrice=order.limit_price,
                stopPrice=order.stop_price,
                tif=tif,
            )
        raise BrokerError(f"Unsupported order type: {order.order_type}")

    def _on_order_status(self, trade) -> None:
        """IB order status callback — fires fills on 'Filled' status."""
        if trade.orderStatus.status != "Filled":
            return
        fills_list = trade.fills
        for fill in fills_list:
            try:
                oms_fill = Fill(
                    order_id="",
                    broker_order_id=str(trade.order.orderId),
                    fill_id=fill.execution.execId,
                    symbol=trade.contract.symbol,
                    side=OrderSide.BUY if fill.execution.side == "BOT" else OrderSide.SELL,
                    quantity=float(fill.execution.shares),
                    price=float(fill.execution.price),
                    filled_at=_utcnow(),
                    commission=float(fill.commissionReport.commission)
                    if fill.commissionReport
                    else 0.0,
                )
                self._notify_fill(oms_fill)
            except Exception:
                logger.exception("IBAdapter: error converting fill event")
