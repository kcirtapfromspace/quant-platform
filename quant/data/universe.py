"""Tradeable universe management.

Defines and maintains the set of instruments eligible for trading,
with rule-based filtering on liquidity, price, data quality, and
sector membership.

A universe is a named, versioned set of symbols that updates daily
(or on demand) by applying a chain of inclusion/exclusion filters
to a candidate pool.

Usage::

    from quant.data.universe import UniverseManager, UniverseConfig, LiquidityFilter

    mgr = UniverseManager(UniverseConfig(
        name="us_liquid_500",
        filters=[
            LiquidityFilter(min_adv=1_000_000),
            PriceFilter(min_price=5.0),
            DataQualityFilter(min_history_days=252),
        ],
    ))

    members = mgr.apply(candidate_pool)
    print(f"{len(members)} symbols passed all filters")
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

# ---------------------------------------------------------------------------
# Filter protocol
# ---------------------------------------------------------------------------


class UniverseFilter(abc.ABC):
    """Abstract filter that decides which symbols enter the universe."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short human-readable filter name."""

    @abc.abstractmethod
    def apply(self, candidates: pd.DataFrame) -> pd.Index:
        """Return the symbols that pass this filter.

        Args:
            candidates: DataFrame with at least a ``symbol`` index.
                Columns depend on the filter type (e.g. ``adv``,
                ``price``, ``history_days``).

        Returns:
            Index of symbols that pass.
        """


# ---------------------------------------------------------------------------
# Concrete filters
# ---------------------------------------------------------------------------


class LiquidityFilter(UniverseFilter):
    """Exclude symbols below minimum average daily dollar volume.

    Args:
        min_adv:  Minimum average daily dollar volume.
        column:   Column name in candidate DataFrame for ADV.
    """

    def __init__(self, min_adv: float = 1_000_000, column: str = "adv") -> None:
        self._min_adv = min_adv
        self._column = column

    @property
    def name(self) -> str:
        return f"liquidity(min_adv={self._min_adv:,.0f})"

    def apply(self, candidates: pd.DataFrame) -> pd.Index:
        if self._column not in candidates.columns:
            return candidates.index
        mask = candidates[self._column] >= self._min_adv
        return candidates.index[mask]


class PriceFilter(UniverseFilter):
    """Exclude symbols outside a price range (penny stock / extreme price filter).

    Args:
        min_price: Minimum price per share.
        max_price: Maximum price per share (None = no cap).
        column:    Column name in candidate DataFrame for price.
    """

    def __init__(
        self,
        min_price: float = 5.0,
        max_price: float | None = None,
        column: str = "price",
    ) -> None:
        self._min_price = min_price
        self._max_price = max_price
        self._column = column

    @property
    def name(self) -> str:
        parts = [f"min={self._min_price:.0f}"]
        if self._max_price is not None:
            parts.append(f"max={self._max_price:.0f}")
        return f"price({', '.join(parts)})"

    def apply(self, candidates: pd.DataFrame) -> pd.Index:
        if self._column not in candidates.columns:
            return candidates.index
        mask = candidates[self._column] >= self._min_price
        if self._max_price is not None:
            mask &= candidates[self._column] <= self._max_price
        return candidates.index[mask]


class DataQualityFilter(UniverseFilter):
    """Exclude symbols with insufficient trading history.

    Args:
        min_history_days: Minimum number of trading days of history.
        max_missing_pct:  Maximum fraction of missing data allowed.
        history_column:   Column name for history length.
        missing_column:   Column name for missing data fraction.
    """

    def __init__(
        self,
        min_history_days: int = 252,
        max_missing_pct: float = 0.05,
        history_column: str = "history_days",
        missing_column: str = "missing_pct",
    ) -> None:
        self._min_history = min_history_days
        self._max_missing = max_missing_pct
        self._history_col = history_column
        self._missing_col = missing_column

    @property
    def name(self) -> str:
        return f"data_quality(min_days={self._min_history}, max_missing={self._max_missing:.0%})"

    def apply(self, candidates: pd.DataFrame) -> pd.Index:
        mask = pd.Series(True, index=candidates.index)
        if self._history_col in candidates.columns:
            mask &= candidates[self._history_col] >= self._min_history
        if self._missing_col in candidates.columns:
            mask &= candidates[self._missing_col] <= self._max_missing
        return candidates.index[mask]


class SectorFilter(UniverseFilter):
    """Include or exclude specific sectors.

    Args:
        include:  Sectors to include (None = include all).
        exclude:  Sectors to exclude.
        column:   Column name for sector classification.
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        column: str = "sector",
    ) -> None:
        self._include = set(include) if include else None
        self._exclude = set(exclude) if exclude else set()
        self._column = column

    @property
    def name(self) -> str:
        if self._include:
            return f"sector(include={sorted(self._include)})"
        if self._exclude:
            return f"sector(exclude={sorted(self._exclude)})"
        return "sector(all)"

    def apply(self, candidates: pd.DataFrame) -> pd.Index:
        if self._column not in candidates.columns:
            return candidates.index
        mask = pd.Series(True, index=candidates.index)
        if self._include is not None:
            mask &= candidates[self._column].isin(self._include)
        if self._exclude:
            mask &= ~candidates[self._column].isin(self._exclude)
        return candidates.index[mask]


class TopNFilter(UniverseFilter):
    """Keep only the top N symbols by a given metric.

    Args:
        n:          Maximum number of symbols to keep.
        sort_by:    Column name to sort by.
        ascending:  Sort order (False = largest first).
    """

    def __init__(
        self,
        n: int = 500,
        sort_by: str = "market_cap",
        ascending: bool = False,
    ) -> None:
        self._n = n
        self._sort_by = sort_by
        self._ascending = ascending

    @property
    def name(self) -> str:
        direction = "asc" if self._ascending else "desc"
        return f"top_n(n={self._n}, by={self._sort_by}, {direction})"

    def apply(self, candidates: pd.DataFrame) -> pd.Index:
        if self._sort_by not in candidates.columns:
            return candidates.index
        sorted_df = candidates.sort_values(
            self._sort_by, ascending=self._ascending
        )
        return sorted_df.index[: self._n]


# ---------------------------------------------------------------------------
# Configuration & results
# ---------------------------------------------------------------------------


@dataclass
class UniverseConfig:
    """Configuration for a tradeable universe.

    Attributes:
        name:     Universe name (e.g. "us_liquid_500").
        filters:  Ordered list of filters to apply.  Symbols must pass
                  all filters (AND logic).
    """

    name: str = "default"
    filters: list[UniverseFilter] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Result of applying a single filter.

    Attributes:
        filter_name:    Name of the filter.
        input_count:    Number of candidates before this filter.
        output_count:   Number of candidates after this filter.
        removed_count:  Number of candidates removed.
        removed:        Symbols removed by this filter.
    """

    filter_name: str
    input_count: int
    output_count: int
    removed_count: int
    removed: list[str]


@dataclass
class UniverseSnapshot:
    """A point-in-time universe membership snapshot.

    Attributes:
        name:            Universe name.
        as_of:           Date of the snapshot.
        members:         List of symbols in the universe.
        n_members:       Number of members.
        filter_results:  Per-filter audit trail.
        candidate_count: Initial candidate count before filtering.
    """

    name: str
    as_of: date
    members: list[str]
    n_members: int
    filter_results: list[FilterResult]
    candidate_count: int

    def summary(self) -> str:
        """Human-readable universe summary."""
        lines = [
            f"Universe: {self.name} (as of {self.as_of})",
            "=" * 60,
            f"  Candidates: {self.candidate_count}",
            f"  Members:    {self.n_members}",
            "",
            f"  {'Filter':<40}{'In':>6}{'Out':>6}{'Drop':>6}",
            "-" * 60,
        ]
        for fr in self.filter_results:
            lines.append(
                f"  {fr.filter_name:<40}{fr.input_count:>6}"
                f"{fr.output_count:>6}{fr.removed_count:>6}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class UniverseManager:
    """Apply rule-based filters to construct a tradeable universe.

    Args:
        config: Universe configuration with filter chain.
    """

    def __init__(self, config: UniverseConfig | None = None) -> None:
        self._config = config or UniverseConfig()
        self._history: list[UniverseSnapshot] = []

    @property
    def config(self) -> UniverseConfig:
        return self._config

    @property
    def history(self) -> list[UniverseSnapshot]:
        """Historical universe snapshots."""
        return list(self._history)

    @property
    def latest(self) -> UniverseSnapshot | None:
        """Most recent universe snapshot, or None."""
        return self._history[-1] if self._history else None

    def apply(
        self,
        candidates: pd.DataFrame,
        as_of: date | None = None,
    ) -> UniverseSnapshot:
        """Apply all filters to the candidate pool.

        Args:
            candidates: DataFrame indexed by symbol with columns required
                by the configured filters (e.g. ``adv``, ``price``,
                ``history_days``, ``sector``, ``market_cap``).
            as_of:      Snapshot date.  Defaults to today.

        Returns:
            :class:`UniverseSnapshot` with members and audit trail.
        """
        as_of = as_of or date.today()
        current = candidates
        filter_results: list[FilterResult] = []

        for filt in self._config.filters:
            input_count = len(current)
            passing = filt.apply(current)
            removed = current.index.difference(passing)

            filter_results.append(
                FilterResult(
                    filter_name=filt.name,
                    input_count=input_count,
                    output_count=len(passing),
                    removed_count=len(removed),
                    removed=sorted(removed.tolist()),
                )
            )

            current = current.loc[current.index.intersection(passing)]

        members = sorted(current.index.tolist())

        snapshot = UniverseSnapshot(
            name=self._config.name,
            as_of=as_of,
            members=members,
            n_members=len(members),
            filter_results=filter_results,
            candidate_count=len(candidates),
        )

        self._history.append(snapshot)
        return snapshot

    def membership_changes(
        self,
    ) -> list[tuple[date, list[str], list[str]]]:
        """Compute additions and removals between consecutive snapshots.

        Returns:
            List of (date, additions, removals) tuples.
        """
        changes: list[tuple[date, list[str], list[str]]] = []
        for i in range(1, len(self._history)):
            prev = set(self._history[i - 1].members)
            curr = set(self._history[i].members)
            additions = sorted(curr - prev)
            removals = sorted(prev - curr)
            if additions or removals:
                changes.append(
                    (self._history[i].as_of, additions, removals)
                )
        return changes
