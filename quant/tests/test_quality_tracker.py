"""Tests for execution quality tracker (QUA-43)."""
from __future__ import annotations

import pytest

from quant.execution.quality_tracker import (
    ExecutionQualityTracker,
    QualityConfig,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_tracker(**kwargs) -> ExecutionQualityTracker:
    return ExecutionQualityTracker(QualityConfig(**kwargs))


# ── Tests: Basic recording ────────────────────────────────────────────────


class TestBasicRecording:
    def test_record_single_fill(self):
        t = _make_tracker()
        t.record("mom", slippage_bps=2.0, notional=50_000)
        assert t.strategy_stats("mom").n_fills == 1

    def test_record_multiple_fills(self):
        t = _make_tracker()
        for i in range(10):
            t.record("mom", slippage_bps=float(i), notional=10_000)
        assert t.strategy_stats("mom").n_fills == 10

    def test_rolling_window_trimming(self):
        t = _make_tracker(rolling_window=5)
        for i in range(20):
            t.record("mom", slippage_bps=float(i), notional=1_000)
        stats = t.strategy_stats("mom")
        assert stats.n_fills == 5
        # Should keep the last 5 fills (15, 16, 17, 18, 19)
        assert stats.mean_slippage_bps == pytest.approx(17.0, abs=0.01)

    def test_unknown_strategy_raises(self):
        t = _make_tracker()
        with pytest.raises(KeyError):
            t.strategy_stats("unknown")

    def test_unknown_strategy_quality_score_is_one(self):
        t = _make_tracker()
        assert t.quality_score("unknown") == 1.0


# ── Tests: Statistics ─────────────────────────────────────────────────────


class TestStatistics:
    def test_mean_slippage_weighted(self):
        t = _make_tracker()
        # Small fill with high slippage, big fill with low slippage
        t.record("s", slippage_bps=20.0, notional=10_000)
        t.record("s", slippage_bps=2.0, notional=90_000)
        stats = t.strategy_stats("s")
        # Weighted: (20*10000 + 2*90000) / 100000 = 380000/100000 = 3.8
        assert stats.mean_slippage_bps == pytest.approx(3.8, abs=0.01)

    def test_median_slippage(self):
        t = _make_tracker()
        for bps in [1.0, 3.0, 5.0, 7.0, 9.0]:
            t.record("s", slippage_bps=bps, notional=1_000)
        assert t.strategy_stats("s").median_slippage_bps == 5.0

    def test_median_even_count(self):
        t = _make_tracker()
        for bps in [1.0, 3.0, 5.0, 7.0]:
            t.record("s", slippage_bps=bps, notional=1_000)
        assert t.strategy_stats("s").median_slippage_bps == 4.0

    def test_max_slippage(self):
        t = _make_tracker()
        for bps in [1.0, 50.0, 3.0]:
            t.record("s", slippage_bps=bps, notional=1_000)
        assert t.strategy_stats("s").max_slippage_bps == 50.0

    def test_total_dollar_cost(self):
        t = _make_tracker()
        # 5 bps on $100,000 notional = $50
        t.record("s", slippage_bps=5.0, notional=100_000)
        assert t.strategy_stats("s").total_dollar_cost == pytest.approx(50.0)

    def test_total_notional(self):
        t = _make_tracker()
        t.record("s", slippage_bps=1.0, notional=50_000)
        t.record("s", slippage_bps=2.0, notional=30_000)
        assert t.strategy_stats("s").total_notional == 80_000

    def test_severe_count(self):
        t = _make_tracker(severe_slippage_bps=25.0)
        t.record("s", slippage_bps=5.0, notional=1_000)
        t.record("s", slippage_bps=30.0, notional=1_000)
        t.record("s", slippage_bps=100.0, notional=1_000)
        assert t.strategy_stats("s").severe_count == 2


# ── Tests: Quality score ──────────────────────────────────────────────────


class TestQualityScore:
    def test_perfect_execution(self):
        t = _make_tracker(cost_budget_bps=10.0)
        for _ in range(10):
            t.record("s", slippage_bps=0.0, notional=10_000)
        assert t.quality_score("s") == 1.0

    def test_negative_slippage_is_best(self):
        t = _make_tracker(cost_budget_bps=10.0)
        # Negative slippage = we got a better price
        for _ in range(10):
            t.record("s", slippage_bps=-5.0, notional=10_000)
        assert t.quality_score("s") == 1.0

    def test_score_degrades_with_slippage(self):
        t = _make_tracker(cost_budget_bps=10.0)
        for _ in range(10):
            t.record("s", slippage_bps=10.0, notional=10_000)
        score = t.quality_score("s")
        assert 0.0 < score < 1.0

    def test_score_zero_at_extreme_slippage(self):
        t = _make_tracker(cost_budget_bps=10.0)
        for _ in range(10):
            t.record("s", slippage_bps=25.0, notional=10_000)
        assert t.quality_score("s") == 0.0

    def test_severe_penalty(self):
        t = _make_tracker(cost_budget_bps=10.0, severe_slippage_bps=20.0)
        # Low mean slippage but with severe events
        for _ in range(5):
            t.record("s", slippage_bps=1.0, notional=10_000)
        for _ in range(5):
            t.record("s", slippage_bps=30.0, notional=10_000)
        stats = t.strategy_stats("s")
        assert stats.severe_count == 5
        # Score should be penalised below what mean alone would give
        assert stats.quality_score < 0.8


# ── Tests: Multi-strategy ─────────────────────────────────────────────────


class TestMultiStrategy:
    def test_independent_tracking(self):
        t = _make_tracker()
        t.record("good", slippage_bps=1.0, notional=10_000)
        t.record("bad", slippage_bps=30.0, notional=10_000)

        assert t.quality_score("good") > t.quality_score("bad")

    def test_all_stats(self):
        t = _make_tracker()
        t.record("a", slippage_bps=1.0, notional=10_000)
        t.record("b", slippage_bps=2.0, notional=10_000)
        stats = t.all_stats()
        assert len(stats) == 2
        names = {s.strategy for s in stats}
        assert names == {"a", "b"}

    def test_strategy_names(self):
        t = _make_tracker()
        t.record("z", slippage_bps=0, notional=100)
        t.record("a", slippage_bps=0, notional=100)
        assert t.strategy_names == ["a", "z"]


# ── Tests: Reset ──────────────────────────────────────────────────────────


class TestReset:
    def test_reset_single(self):
        t = _make_tracker()
        t.record("s", slippage_bps=5.0, notional=10_000)
        t.reset("s")
        assert t.quality_score("s") == 1.0

    def test_reset_all(self):
        t = _make_tracker()
        t.record("a", slippage_bps=1.0, notional=100)
        t.record("b", slippage_bps=2.0, notional=100)
        t.reset_all()
        assert t.strategy_names == []


# ── Tests: Summary ────────────────────────────────────────────────────────


class TestSummary:
    def test_empty_summary(self):
        t = _make_tracker()
        assert "no fills" in t.summary()

    def test_summary_with_data(self):
        t = _make_tracker()
        t.record("mom", slippage_bps=3.0, notional=50_000)
        summary = t.summary()
        assert "mom" in summary
        assert "bps" in summary
        assert "score" in summary
