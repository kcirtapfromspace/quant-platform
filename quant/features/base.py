"""Abstract base class for all feature computations."""
from __future__ import annotations

import abc

import pandas as pd


class BaseFeature(abc.ABC):
    """Abstract feature that computes a named time-series from OHLCV data.

    All features receive a pandas DataFrame with the canonical OHLCV schema
    (symbol, date, open, high, low, close, volume, adj_close) sorted ascending
    by date. They return a pd.Series indexed by date.

    Feature names must be unique within a FeatureRegistry.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique, human-readable feature identifier (e.g. ``rsi_14``)."""

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute the feature from a single-symbol OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame sorted ascending by ``date`` with columns
                [symbol, date, open, high, low, close, volume, adj_close].

        Returns:
            pd.Series indexed by ``date`` (same index as *df*), dtype float64.
            Values at warm-up positions should be NaN.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
