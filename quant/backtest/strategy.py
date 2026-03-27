"""Abstract strategy interface for the backtesting engine."""
from __future__ import annotations

import abc

import pandas as pd


class Strategy(abc.ABC):
    """Abstract base class for trading strategies.

    Subclasses implement ``generate_signals`` to produce a position series from
    OHLCV data.  All signal generation must use only information available at the
    time of the bar — the engine shifts signals forward by one period before
    computing returns, but the responsibility for not peeking at future prices
    within the strategy itself rests with the implementor.
    """

    @property
    def name(self) -> str:
        """Human-readable strategy name (defaults to class name)."""
        return self.__class__.__name__

    @abc.abstractmethod
    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Return a position signal for each bar in *ohlcv*.

        Args:
            ohlcv: DataFrame with a DatetimeIndex (or date index) and at least
                the columns ``open``, ``high``, ``low``, ``close``,
                ``adj_close``, ``volume``.  Rows are in ascending date order.

        Returns:
            A ``pd.Series`` with the same index as *ohlcv* containing position
            weights.  Conventional values: ``1`` (fully long), ``0`` (flat),
            ``-1`` (fully short).  Fractional values are accepted for
            position sizing.

        Important:
            The signal at row *t* will be applied to the **close-to-close
            return from t to t+1**.  Do not use ``ohlcv`` data beyond row *t*
            when computing the signal for row *t*.
        """
