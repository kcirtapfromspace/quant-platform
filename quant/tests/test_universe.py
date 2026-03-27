"""Tests for tradeable universe management (QUA-52)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from quant.data.universe import (
    DataQualityFilter,
    FilterResult,
    LiquidityFilter,
    PriceFilter,
    SectorFilter,
    TopNFilter,
    UniverseConfig,
    UniverseManager,
    UniverseSnapshot,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_candidates(n: int = 10) -> pd.DataFrame:
    """Create a synthetic candidate pool."""
    symbols = [f"SYM{i:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "adv": [i * 1_000_000 for i in range(1, n + 1)],
            "price": [5.0 + i * 10 for i in range(n)],
            "history_days": [100 + i * 50 for i in range(n)],
            "missing_pct": [0.01 * i for i in range(n)],
            "sector": ["Tech"] * 5 + ["Finance"] * 3 + ["Energy"] * 2,
            "market_cap": [i * 1e9 for i in range(n, 0, -1)],
        },
        index=symbols,
    )


# ── Tests: Basic filtering ────────────────────────────────────────────────


class TestBasicFiltering:
    def test_no_filters_keeps_all(self):
        mgr = UniverseManager(UniverseConfig(name="all", filters=[]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        assert snapshot.n_members == len(candidates)

    def test_returns_snapshot(self):
        mgr = UniverseManager()
        snapshot = mgr.apply(_make_candidates())
        assert isinstance(snapshot, UniverseSnapshot)

    def test_snapshot_name(self):
        mgr = UniverseManager(UniverseConfig(name="test_univ"))
        snapshot = mgr.apply(_make_candidates())
        assert snapshot.name == "test_univ"

    def test_snapshot_date(self):
        mgr = UniverseManager()
        d = date(2024, 6, 15)
        snapshot = mgr.apply(_make_candidates(), as_of=d)
        assert snapshot.as_of == d

    def test_members_sorted(self):
        mgr = UniverseManager()
        snapshot = mgr.apply(_make_candidates())
        assert snapshot.members == sorted(snapshot.members)


# ── Tests: Liquidity filter ───────────────────────────────────────────────


class TestLiquidityFilter:
    def test_filters_low_adv(self):
        filt = LiquidityFilter(min_adv=5_000_000)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        # ADV: 1M, 2M, 3M, 4M, 5M, 6M, 7M, 8M, 9M, 10M → 6 pass (≥5M)
        assert snapshot.n_members == 6

    def test_all_pass_when_threshold_low(self):
        filt = LiquidityFilter(min_adv=0)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        snapshot = mgr.apply(_make_candidates())
        assert snapshot.n_members == 10

    def test_missing_column_keeps_all(self):
        filt = LiquidityFilter(min_adv=1_000_000, column="volume")
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        assert snapshot.n_members == 10

    def test_filter_name(self):
        filt = LiquidityFilter(min_adv=1_000_000)
        assert "liquidity" in filt.name


# ── Tests: Price filter ───────────────────────────────────────────────────


class TestPriceFilter:
    def test_filters_low_price(self):
        filt = PriceFilter(min_price=50.0)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        # Prices: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95 → 5 pass (≥50)
        assert snapshot.n_members == 5

    def test_max_price(self):
        filt = PriceFilter(min_price=0.0, max_price=50.0)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        # Prices ≤50: 5, 15, 25, 35, 45 → 5 pass
        assert snapshot.n_members == 5

    def test_price_range(self):
        filt = PriceFilter(min_price=20.0, max_price=60.0)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        # Prices 20–60: 25, 35, 45, 55 → 4 pass
        assert snapshot.n_members == 4


# ── Tests: Data quality filter ────────────────────────────────────────────


class TestDataQualityFilter:
    def test_filters_short_history(self):
        filt = DataQualityFilter(min_history_days=300, max_missing_pct=1.0)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        # history: 100,150,200,250,300,350,400,450,500,550 → 6 pass (≥300)
        assert snapshot.n_members == 6

    def test_filters_missing_data(self):
        filt = DataQualityFilter(min_history_days=0, max_missing_pct=0.03)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        # missing: 0,0.01,0.02,0.03,0.04,... → 4 pass (≤0.03)
        assert snapshot.n_members == 4

    def test_combined_quality_checks(self):
        filt = DataQualityFilter(min_history_days=300, max_missing_pct=0.05)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        # Must pass both: history≥300 AND missing≤0.05
        assert snapshot.n_members > 0


# ── Tests: Sector filter ─────────────────────────────────────────────────


class TestSectorFilter:
    def test_include_sectors(self):
        filt = SectorFilter(include=["Tech"])
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        assert snapshot.n_members == 5

    def test_exclude_sectors(self):
        filt = SectorFilter(exclude=["Energy"])
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        assert snapshot.n_members == 8

    def test_include_multiple_sectors(self):
        filt = SectorFilter(include=["Tech", "Finance"])
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        assert snapshot.n_members == 8

    def test_missing_column_keeps_all(self):
        filt = SectorFilter(include=["Tech"], column="industry")
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        snapshot = mgr.apply(_make_candidates())
        assert snapshot.n_members == 10


# ── Tests: Top N filter ──────────────────────────────────────────────────


class TestTopNFilter:
    def test_top_n_by_market_cap(self):
        filt = TopNFilter(n=5, sort_by="market_cap")
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        assert snapshot.n_members == 5

    def test_top_n_ascending(self):
        filt = TopNFilter(n=3, sort_by="price", ascending=True)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        assert snapshot.n_members == 3
        # Should be the 3 cheapest
        assert "SYM00" in snapshot.members

    def test_top_n_larger_than_pool(self):
        filt = TopNFilter(n=100)
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        candidates = _make_candidates(n=10)
        snapshot = mgr.apply(candidates)
        assert snapshot.n_members == 10

    def test_missing_column_keeps_all(self):
        filt = TopNFilter(n=5, sort_by="nonexistent")
        mgr = UniverseManager(UniverseConfig(filters=[filt]))
        snapshot = mgr.apply(_make_candidates())
        assert snapshot.n_members == 10


# ── Tests: Filter chaining ───────────────────────────────────────────────


class TestFilterChaining:
    def test_multiple_filters_and_logic(self):
        """Symbols must pass ALL filters."""
        filters = [
            LiquidityFilter(min_adv=3_000_000),
            PriceFilter(min_price=30.0),
        ]
        mgr = UniverseManager(UniverseConfig(filters=filters))
        candidates = _make_candidates()
        snapshot = mgr.apply(candidates)
        # ADV≥3M: SYM02..SYM09 (8 symbols)
        # Price≥30: SYM02..SYM09 (prices 25,35,45,...→ SYM03..SYM09 = 7)
        # Intersection via sequential filtering
        for sym in snapshot.members:
            idx = int(sym[3:])
            assert (idx + 1) * 1_000_000 >= 3_000_000
            assert 5.0 + idx * 10 >= 30.0

    def test_filter_audit_trail(self):
        filters = [
            LiquidityFilter(min_adv=5_000_000),
            PriceFilter(min_price=50.0),
        ]
        mgr = UniverseManager(UniverseConfig(filters=filters))
        snapshot = mgr.apply(_make_candidates())
        assert len(snapshot.filter_results) == 2
        for fr in snapshot.filter_results:
            assert isinstance(fr, FilterResult)
            assert fr.input_count >= fr.output_count

    def test_progressive_reduction(self):
        """Each filter reduces the candidate pool from the previous step."""
        filters = [
            TopNFilter(n=8),
            LiquidityFilter(min_adv=5_000_000),
        ]
        mgr = UniverseManager(UniverseConfig(filters=filters))
        snapshot = mgr.apply(_make_candidates())
        # First filter: top 8 by market_cap
        assert snapshot.filter_results[0].output_count == 8
        # Second filter: further reduces
        assert snapshot.filter_results[1].input_count == 8


# ── Tests: Universe history ──────────────────────────────────────────────


class TestUniverseHistory:
    def test_history_grows(self):
        mgr = UniverseManager()
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 1))
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 2))
        assert len(mgr.history) == 2

    def test_latest(self):
        mgr = UniverseManager()
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 1))
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 2))
        assert mgr.latest is not None
        assert mgr.latest.as_of == date(2024, 1, 2)

    def test_latest_none_when_empty(self):
        mgr = UniverseManager()
        assert mgr.latest is None


# ── Tests: Membership changes ────────────────────────────────────────────


class TestMembershipChanges:
    def test_detects_additions(self):
        mgr = UniverseManager(UniverseConfig(
            filters=[LiquidityFilter(min_adv=5_000_000)]
        ))
        # Day 1: standard pool
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 1))

        # Day 2: lower threshold → more members
        mgr._config = UniverseConfig(
            filters=[LiquidityFilter(min_adv=3_000_000)]
        )
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 2))

        changes = mgr.membership_changes()
        assert len(changes) == 1
        _, additions, removals = changes[0]
        assert len(additions) > 0
        assert len(removals) == 0

    def test_detects_removals(self):
        mgr = UniverseManager(UniverseConfig(
            filters=[LiquidityFilter(min_adv=3_000_000)]
        ))
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 1))

        # Day 2: higher threshold → fewer members
        mgr._config = UniverseConfig(
            filters=[LiquidityFilter(min_adv=5_000_000)]
        )
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 2))

        changes = mgr.membership_changes()
        assert len(changes) == 1
        _, additions, removals = changes[0]
        assert len(removals) > 0
        assert len(additions) == 0

    def test_no_changes_when_stable(self):
        mgr = UniverseManager()
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 1))
        mgr.apply(_make_candidates(), as_of=date(2024, 1, 2))
        changes = mgr.membership_changes()
        assert len(changes) == 0


# ── Tests: Summary ────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_filters(self):
        filters = [
            LiquidityFilter(min_adv=5_000_000),
            PriceFilter(min_price=50.0),
        ]
        mgr = UniverseManager(UniverseConfig(name="test", filters=filters))
        snapshot = mgr.apply(_make_candidates())
        summary = snapshot.summary()
        assert "Universe: test" in summary
        assert "Candidates" in summary
        assert "Members" in summary

    def test_summary_no_filters(self):
        mgr = UniverseManager(UniverseConfig(name="all"))
        snapshot = mgr.apply(_make_candidates())
        summary = snapshot.summary()
        assert "Universe: all" in summary


# ── Tests: Config ─────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = UniverseConfig()
        assert config.name == "default"
        assert config.filters == []

    def test_config_exposed(self):
        config = UniverseConfig(name="my_univ")
        mgr = UniverseManager(config)
        assert mgr.config.name == "my_univ"
