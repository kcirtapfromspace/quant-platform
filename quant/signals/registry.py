"""Dict-backed registry of named trading signals."""
from __future__ import annotations

from quant.signals.base import BaseSignal


class SignalRegistry:
    """Maps signal names to BaseSignal instances.

    Usage
    -----
    registry = SignalRegistry()
    registry.register(MomentumSignal())
    signal = registry.get("momentum")
    """

    def __init__(self) -> None:
        self._signals: dict[str, BaseSignal] = {}

    def register(self, signal: BaseSignal) -> "SignalRegistry":
        """Register a signal. Raises ValueError if name already taken."""
        if signal.name in self._signals:
            raise ValueError(
                f"Signal '{signal.name}' is already registered."
            )
        self._signals[signal.name] = signal
        return self

    def unregister(self, name: str) -> None:
        """Remove a signal by name. Raises KeyError if not found."""
        if name not in self._signals:
            raise KeyError(f"Signal '{name}' is not registered.")
        del self._signals[name]

    def get(self, name: str) -> BaseSignal:
        """Return signal by name. Raises KeyError if not found."""
        if name not in self._signals:
            raise KeyError(
                f"Signal '{name}' is not registered. "
                f"Available: {sorted(self._signals)}"
            )
        return self._signals[name]

    def names(self) -> list[str]:
        """Return sorted list of registered signal names."""
        return sorted(self._signals)

    def all(self) -> list[BaseSignal]:
        """Return all registered signals."""
        return list(self._signals.values())

    def __contains__(self, name: str) -> bool:
        return name in self._signals

    def __len__(self) -> int:
        return len(self._signals)

    def __repr__(self) -> str:
        return f"SignalRegistry(signals={self.names()})"
