"""Feature engine for computing technical and statistical indicators from OHLCV data."""
from quant.features.base import BaseFeature
from quant.features.registry import FeatureRegistry
from quant.features.engine import FeatureEngine
from quant.features.cache import FeatureCache, InMemoryFeatureCache, RedisFeatureCache
from quant.features.built_in import (
    Returns,
    LogReturns,
    RollingMean,
    RollingStd,
    RSI,
    MACD,
    MACDSignal,
    MACDHistogram,
    BollingerUpper,
    BollingerLower,
    BollingerMid,
    BollingerBandwidth,
    VolumeSMA,
    VolumeRatio,
    DEFAULT_REGISTRY,
)

__all__ = [
    "BaseFeature",
    "FeatureRegistry",
    "FeatureEngine",
    "FeatureCache",
    "InMemoryFeatureCache",
    "RedisFeatureCache",
    "Returns",
    "LogReturns",
    "RollingMean",
    "RollingStd",
    "RSI",
    "MACD",
    "MACDSignal",
    "MACDHistogram",
    "BollingerUpper",
    "BollingerLower",
    "BollingerMid",
    "BollingerBandwidth",
    "VolumeSMA",
    "VolumeRatio",
    "DEFAULT_REGISTRY",
]
