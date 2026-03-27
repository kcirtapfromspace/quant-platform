"""Abstract base classes for market data sources."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import date
from typing import Sequence


@dataclass(frozen=True, slots=True)
class OHLCVRecord:
    """A single OHLCV bar for a symbol on a given date."""

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float


class DataSource(abc.ABC):
    """Abstract market data source.

    Implementations must return OHLCVRecord sequences for a list of symbols
    over a date range. The pipeline layer handles dedup and storage.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of this data source."""

    @abc.abstractmethod
    def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> list[OHLCVRecord]:
        """Fetch OHLCV bars for *symbols* between *start* and *end* (inclusive).

        Args:
            symbols: Ticker symbols to fetch (e.g. ["AAPL", "MSFT"]).
            start: First date to include.
            end: Last date to include.

        Returns:
            List of OHLCVRecord, sorted by (symbol, date).
        """
