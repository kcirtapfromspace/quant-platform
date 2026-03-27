"""Pluggable feature registry mapping names to BaseFeature instances."""
from __future__ import annotations

from quant.features.base import BaseFeature


class FeatureRegistry:
    """Dict-backed registry of named features.

    Usage
    -----
    registry = FeatureRegistry()
    registry.register(Returns())
    registry.register(RSI(period=14))

    feat = registry.get("rsi_14")
    series = feat.compute(df)
    """

    def __init__(self) -> None:
        self._features: dict[str, BaseFeature] = {}

    def register(self, feature: BaseFeature) -> "FeatureRegistry":
        """Register a feature. Raises ValueError if name is already taken."""
        if feature.name in self._features:
            raise ValueError(
                f"Feature '{feature.name}' is already registered. "
                "Unregister it first or use a different name."
            )
        self._features[feature.name] = feature
        return self

    def unregister(self, name: str) -> None:
        """Remove a feature by name. Raises KeyError if not found."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' is not registered.")
        del self._features[name]

    def get(self, name: str) -> BaseFeature:
        """Return feature by name. Raises KeyError if not found."""
        if name not in self._features:
            raise KeyError(
                f"Feature '{name}' is not registered. "
                f"Available: {sorted(self._features)}"
            )
        return self._features[name]

    def names(self) -> list[str]:
        """Return sorted list of all registered feature names."""
        return sorted(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return f"FeatureRegistry(features={self.names()})"
