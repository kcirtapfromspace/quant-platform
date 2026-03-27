"""Feature engine: compute named features from stored OHLCV data.

Orchestrates: store query -> cache lookup -> feature compute -> cache store.

Optionally converts output to polars Series when polars is installed and
``backend="polars"`` is requested.

Usage
-----
from quant.features import FeatureEngine, DEFAULT_REGISTRY, InMemoryFeatureCache
from quant.data.storage.duckdb import MarketDataStore

store = MarketDataStore("/data/market.duckdb")
engine = FeatureEngine(store, DEFAULT_REGISTRY, cache=InMemoryFeatureCache())

results = engine.compute(
    symbols=["AAPL", "MSFT"],
    features=["returns", "rsi_14", "macd_12_26"],
    start=date(2024, 1, 1),
    end=date(2024, 12, 31),
)
# results["AAPL"]["rsi_14"] -> pd.Series
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

import pandas as pd
from loguru import logger

from quant.data.storage.duckdb import MarketDataStore
from quant.features.base import BaseFeature
from quant.features.cache import FeatureCache
from quant.features.registry import FeatureRegistry

if TYPE_CHECKING:
    try:
        import polars as pl  # type: ignore[import]
        _PolarsSeriesType = pl.Series
    except ImportError:
        _PolarsSeriesType = None

FeatureSeries = Union[pd.Series, "pl.Series"]


def _to_polars(series: pd.Series) -> "pl.Series":
    """Convert a pandas Series to polars Series. Raises ImportError if polars not installed."""
    try:
        import polars as pl  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "polars backend requires the 'polars' package. "
            "Install it with: pip install polars"
        ) from exc
    return pl.Series(name=str(series.name or ""), values=series.to_numpy())


class FeatureEngine:
    """Compute named features for one or more symbols over a date range.

    Args:
        store: MarketDataStore providing OHLCV bars.
        registry: FeatureRegistry mapping names to BaseFeature instances.
        cache: Optional FeatureCache to avoid recomputation. Default None.
    """

    def __init__(
        self,
        store: MarketDataStore,
        registry: FeatureRegistry,
        cache: Optional[FeatureCache] = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._cache = cache

    def compute(
        self,
        symbols: Sequence[str],
        features: Sequence[str],
        start: date,
        end: date,
        backend: Literal["pandas", "polars"] = "pandas",
    ) -> dict[str, dict[str, FeatureSeries]]:
        """Compute features for each symbol in [start, end].

        Args:
            symbols: Ticker symbols (case-insensitive).
            features: Feature names; must all be registered in the registry.
            start: First date (inclusive).
            end: Last date (inclusive).
            backend: Output Series type — ``"pandas"`` (default) or ``"polars"``.

        Returns:
            Nested dict ``{symbol: {feature_name: Series}}``.
            Symbols with no stored data are omitted.

        Raises:
            KeyError: If any feature name is not registered.
        """
        # Validate feature names upfront
        for name in features:
            self._registry.get(name)  # raises KeyError if missing

        results: dict[str, dict[str, FeatureSeries]] = {}

        for raw_symbol in symbols:
            symbol = raw_symbol.upper()
            df = self._store.query(symbol, start, end)

            if df.empty:
                logger.warning(
                    "No OHLCV data for {} in [{}, {}] — skipping.",
                    symbol,
                    start,
                    end,
                )
                continue

            df = df.sort_values("date").reset_index(drop=True)
            results[symbol] = {}

            for feat_name in features:
                series = self._compute_one(symbol, feat_name, df, start, end)
                if backend == "polars":
                    series = _to_polars(series)
                results[symbol][feat_name] = series

        return results

    def compute_dataframe(
        self,
        symbol: str,
        features: Sequence[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Convenience method: return all features as a single DataFrame.

        Columns are the feature names; index is date from the OHLCV store.

        Args:
            symbol: Single ticker symbol.
            features: Feature names to compute.
            start: First date (inclusive).
            end: Last date (inclusive).

        Returns:
            DataFrame with feature columns, indexed by row position (date column present).
            Returns empty DataFrame if no data.
        """
        result = self.compute([symbol], features, start, end, backend="pandas")
        if symbol.upper() not in result:
            return pd.DataFrame()

        sym_result = result[symbol.upper()]
        df = self._store.query(symbol.upper(), start, end).sort_values("date").reset_index(drop=True)
        out = df[["date"]].copy()
        for feat_name, series in sym_result.items():
            out[feat_name] = series.values
        return out

    def _compute_one(
        self,
        symbol: str,
        feat_name: str,
        df: pd.DataFrame,
        start: date,
        end: date,
    ) -> pd.Series:
        if self._cache is not None:
            key = self._cache.make_key(symbol, feat_name, start, end)
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("Cache hit: {}/{}", symbol, feat_name)
                return cached

        feat: BaseFeature = self._registry.get(feat_name)
        series = feat.compute(df)
        logger.debug("Computed {}/{} ({} values)", symbol, feat_name, len(series))

        if self._cache is not None:
            self._cache.set(key, series)

        return series
