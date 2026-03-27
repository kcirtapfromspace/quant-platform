"""Abstract BrokerAdapter interface — all execution adapters must implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from quant.oms.models import Fill, Order, Position


class BrokerAdapter(ABC):
    """Abstract base class for all broker/exchange execution adapters.

    Implementations must be thread-safe and idempotent where possible.
    Each adapter is responsible for:
      - translating OMS Order objects into broker-specific API calls
      - normalizing broker responses back into OMS Fill objects
      - notifying the OMS of fill events via the registered callback

    Usage::

        adapter = AlpacaAdapter(api_key="...", secret_key="...")
        adapter.connect()
        adapter.register_fill_callback(oms.on_fill)
        broker_id = adapter.submit_order(order)
    """

    def __init__(self) -> None:
        self._fill_callback: Optional[Callable[[Fill], None]] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker API.

        Should be idempotent — calling connect() twice must not raise.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down connections and clean up resources."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True when the adapter has an active broker connection."""

    # ── Order operations ──────────────────────────────────────────────────

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order to the broker.

        Args:
            order: The OMS order to submit.

        Returns:
            The broker-assigned order ID string.

        Raises:
            BrokerError: If the broker rejects or fails to accept the order.
        """

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """Request cancellation of an open order.

        Args:
            broker_order_id: The broker-side order ID returned by submit_order.

        Returns:
            True if the cancellation was accepted; False if it was too late
            (e.g. already filled) or the order could not be found.
        """

    # ── Query operations ──────────────────────────────────────────────────

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Fetch the current position for *symbol* from the broker.

        Returns None if there is no open position for the symbol.
        """

    @abstractmethod
    def get_fills(self, broker_order_id: str) -> list[Fill]:
        """Retrieve all fills for a broker order ID.

        Returns an empty list if the order has no fills or does not exist.
        """

    @abstractmethod
    def get_account_cash(self) -> float:
        """Return the current cash / buying-power balance."""

    # ── Fill streaming ────────────────────────────────────────────────────

    def register_fill_callback(self, callback: Callable[[Fill], None]) -> None:
        """Register a callback that fires when a fill event arrives.

        The OMS calls this to hook its own on_fill handler so it can update
        internal state in real time.

        Args:
            callback: Callable accepting a single :class:`Fill` argument.
        """
        self._fill_callback = callback

    def _notify_fill(self, fill: Fill) -> None:
        """Dispatch a fill event to the registered callback (if any)."""
        if self._fill_callback is not None:
            self._fill_callback(fill)

    # ── Context manager support ───────────────────────────────────────────

    def __enter__(self) -> "BrokerAdapter":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()


class BrokerError(Exception):
    """Raised when the broker returns an error or rejects a request."""
