"""Feature caching layer.

Provides an abstract FeatureCache interface with two implementations:
- InMemoryFeatureCache: simple dict-backed cache (useful for tests and single-process runs)
- RedisFeatureCache: persistent cache backed by Redis (requires ``redis`` package)

Usage
-----
cache = InMemoryFeatureCache()
key = cache.make_key("AAPL", "rsi_14", date(2024, 1, 1), date(2024, 12, 31))
cache.set(key, series)
cached = cache.get(key)   # returns Series or None
"""
from __future__ import annotations

import abc
import json
from datetime import date
from typing import Optional

import pandas as pd


def _serialize_series(series: pd.Series) -> bytes:
    """Serialize a pd.Series to JSON bytes (safe, no pickle)."""
    payload = {
        "name": str(series.name) if series.name is not None else None,
        "index": [str(i) for i in series.index],
        "values": [None if pd.isna(v) else float(v) for v in series],
    }
    return json.dumps(payload).encode("utf-8")


def _deserialize_series(data: bytes) -> pd.Series:
    """Deserialize a pd.Series from JSON bytes."""
    payload = json.loads(data.decode("utf-8"))
    values = [float("nan") if v is None else v for v in payload["values"]]
    return pd.Series(values, index=payload["index"], name=payload["name"])


class FeatureCache(abc.ABC):
    """Abstract interface for feature caching."""

    @abc.abstractmethod
    def get(self, key: str) -> Optional[pd.Series]:
        """Return cached Series or None if not found / expired."""

    @abc.abstractmethod
    def set(self, key: str, value: pd.Series, ttl_seconds: int = 3600) -> None:
        """Store *value* under *key* with optional TTL."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Evict *key* from the cache."""

    @abc.abstractmethod
    def clear(self) -> None:
        """Remove all cached entries."""

    def make_key(
        self,
        symbol: str,
        feature: str,
        start: date,
        end: date,
    ) -> str:
        """Build a cache key from the canonical feature request tuple."""
        return f"features:{symbol.upper()}:{feature}:{start.isoformat()}:{end.isoformat()}"


class InMemoryFeatureCache(FeatureCache):
    """Simple in-process dict cache.  No TTL enforcement (entries never expire).

    Suitable for single-run use, tests, and environments without Redis.
    """

    def __init__(self) -> None:
        self._store: dict[str, pd.Series] = {}

    def get(self, key: str) -> Optional[pd.Series]:
        return self._store.get(key)

    def set(self, key: str, value: pd.Series, ttl_seconds: int = 3600) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class RedisFeatureCache(FeatureCache):
    """Redis-backed feature cache.

    Serialises Series values with JSON (safe serialization).
    Requires the ``redis`` package (``pip install redis``).

    Args:
        host: Redis host. Default ``localhost``.
        port: Redis port. Default ``6379``.
        db: Redis logical database index. Default ``0``.
        key_prefix: Optional prefix added to all keys for namespace isolation.
        **kwargs: Additional kwargs forwarded to ``redis.Redis``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        key_prefix: str = "",
        **kwargs: object,
    ) -> None:
        try:
            import redis  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "RedisFeatureCache requires the 'redis' package. "
                "Install it with: pip install redis"
            ) from exc
        self._redis = redis.Redis(host=host, port=port, db=db, **kwargs)
        self._prefix = key_prefix

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}" if self._prefix else key

    def get(self, key: str) -> Optional[pd.Series]:
        raw = self._redis.get(self._full_key(key))
        if raw is None:
            return None
        return _deserialize_series(raw)

    def set(self, key: str, value: pd.Series, ttl_seconds: int = 3600) -> None:
        self._redis.setex(self._full_key(key), ttl_seconds, _serialize_series(value))

    def delete(self, key: str) -> None:
        self._redis.delete(self._full_key(key))

    def clear(self) -> None:
        pattern = f"{self._prefix}features:*" if self._prefix else "features:*"
        keys = self._redis.keys(pattern)
        if keys:
            self._redis.delete(*keys)
