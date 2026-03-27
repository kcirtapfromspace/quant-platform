"""Built-in technical and statistical features.

All features operate on a single-symbol OHLCV DataFrame sorted ascending by
date and return a pd.Series indexed by date.

Warm-up periods produce NaN (e.g. RSI needs at least *period* rows before the
first valid value).

All hot-path computations are delegated to the Rust extension ``quant_rs``
(built via maturin from ``quant-rs/``).  The extension must be installed:

    cd quant-rs && maturin develop --release
"""
from __future__ import annotations

import pandas as pd

import quant_rs as _qrs

from quant.features.base import BaseFeature
from quant.features.registry import FeatureRegistry


# ---------------------------------------------------------------------------
# Return-based features
# ---------------------------------------------------------------------------

class Returns(BaseFeature):
    """Simple period returns: (close[t] - close[t-1]) / close[t-1]."""

    @property
    def name(self) -> str:
        return "returns"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.returns(df["close"].tolist())
        return pd.Series(values, index=df.index, name=self.name)


class LogReturns(BaseFeature):
    """Log returns: log(close[t] / close[t-1])."""

    @property
    def name(self) -> str:
        return "log_returns"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.log_returns(df["close"].tolist())
        return pd.Series(values, index=df.index, name=self.name)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

class RollingMean(BaseFeature):
    """Rolling mean of close price over *period* bars."""

    def __init__(self, period: int = 20) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period

    @property
    def name(self) -> str:
        return f"rolling_mean_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.rolling_mean(df["close"].tolist(), self._period)
        return pd.Series(values, index=df.index, name=self.name)


class RollingStd(BaseFeature):
    """Rolling standard deviation of close price over *period* bars."""

    def __init__(self, period: int = 20) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self._period = period

    @property
    def name(self) -> str:
        return f"rolling_std_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.rolling_std(df["close"].tolist(), self._period)
        return pd.Series(values, index=df.index, name=self.name)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class RSI(BaseFeature):
    """Relative Strength Index.

    Uses EWM smoothing with ``alpha = 1 / period`` seeded from the first
    difference, matching ``pandas.ewm(alpha=1/period, adjust=False)``.
    """

    def __init__(self, period: int = 14) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self._period = period

    @property
    def name(self) -> str:
        return f"rsi_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.rsi(df["close"].tolist(), self._period)
        return pd.Series(values, index=df.index, name=self.name)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class MACD(BaseFeature):
    """MACD line = EMA(fast) - EMA(slow)."""

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self._fast = fast
        self._slow = slow

    @property
    def name(self) -> str:
        return f"macd_{self._fast}_{self._slow}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.macd(df["close"].tolist(), self._fast, self._slow)
        return pd.Series(values, index=df.index, name=self.name)


class MACDSignal(BaseFeature):
    """MACD signal line = EMA(signal_period) of the MACD line."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self._fast = fast
        self._slow = slow
        self._signal = signal

    @property
    def name(self) -> str:
        return f"macd_signal_{self._fast}_{self._slow}_{self._signal}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.macd_signal(
            df["close"].tolist(), self._fast, self._slow, self._signal
        )
        return pd.Series(values, index=df.index, name=self.name)


class MACDHistogram(BaseFeature):
    """MACD histogram = MACD line - signal line."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self._fast = fast
        self._slow = slow
        self._signal = signal

    @property
    def name(self) -> str:
        return f"macd_hist_{self._fast}_{self._slow}_{self._signal}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.macd_histogram(
            df["close"].tolist(), self._fast, self._slow, self._signal
        )
        return pd.Series(values, index=df.index, name=self.name)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class BollingerMid(BaseFeature):
    """Bollinger mid band = SMA(period)."""

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self._period = period
        self._num_std = num_std

    @property
    def name(self) -> str:
        return f"bb_mid_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.bb_mid(df["close"].tolist(), self._period)
        return pd.Series(values, index=df.index, name=self.name)


class BollingerUpper(BaseFeature):
    """Bollinger upper band = SMA(period) + num_std * std(period)."""

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self._period = period
        self._num_std = num_std

    @property
    def name(self) -> str:
        return f"bb_upper_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.bb_upper(df["close"].tolist(), self._period, self._num_std)
        return pd.Series(values, index=df.index, name=self.name)


class BollingerLower(BaseFeature):
    """Bollinger lower band = SMA(period) - num_std * std(period)."""

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self._period = period
        self._num_std = num_std

    @property
    def name(self) -> str:
        return f"bb_lower_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.bb_lower(df["close"].tolist(), self._period, self._num_std)
        return pd.Series(values, index=df.index, name=self.name)


class BollingerBandwidth(BaseFeature):
    """Bollinger bandwidth = 2 * num_std * std(period) / SMA(period)."""

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self._period = period
        self._num_std = num_std

    @property
    def name(self) -> str:
        return f"bb_bandwidth_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.bb_bandwidth(df["close"].tolist(), self._period, self._num_std)
        return pd.Series(values, index=df.index, name=self.name)


# ---------------------------------------------------------------------------
# Volume metrics
# ---------------------------------------------------------------------------

class VolumeSMA(BaseFeature):
    """Rolling simple moving average of volume over *period* bars."""

    def __init__(self, period: int = 20) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period

    @property
    def name(self) -> str:
        return f"volume_sma_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.volume_sma(df["volume"].tolist(), self._period)
        return pd.Series(values, index=df.index, name=self.name)


class VolumeRatio(BaseFeature):
    """Volume ratio: current volume / rolling mean volume."""

    def __init__(self, period: int = 20) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period

    @property
    def name(self) -> str:
        return f"volume_ratio_{self._period}"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        values = _qrs.features.volume_ratio(df["volume"].tolist(), self._period)
        return pd.Series(values, index=df.index, name=self.name)


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

def _build_default_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    for feat in [
        Returns(),
        LogReturns(),
        RollingMean(20),
        RollingMean(50),
        RollingStd(20),
        RSI(14),
        MACD(12, 26),
        MACDSignal(12, 26, 9),
        MACDHistogram(12, 26, 9),
        BollingerMid(20),
        BollingerUpper(20),
        BollingerLower(20),
        BollingerBandwidth(20),
        VolumeSMA(20),
        VolumeRatio(20),
    ]:
        registry.register(feat)
    return registry


DEFAULT_REGISTRY: FeatureRegistry = _build_default_registry()
