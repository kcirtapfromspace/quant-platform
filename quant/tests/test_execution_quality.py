"""Tests for execution quality analysis (QUA-92)."""
from __future__ import annotations

import pytest

from quant.execution.quality import (
    BrokerStats,
    ExecutionAnalyzer,
    ExecutionQualityResult,
    Fill,
    FillAnalysis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _buy_fill(**overrides) -> Fill:
    defaults = {
        "symbol": "AAPL", "side": "BUY", "quantity": 1000,
        "fill_price": 150.20, "arrival_price": 150.00,
        "vwap": 150.10, "notional": 150200,
    }
    defaults.update(overrides)
    return Fill(**defaults)


def _sell_fill(**overrides) -> Fill:
    defaults = {
        "symbol": "GOOG", "side": "SELL", "quantity": 500,
        "fill_price": 99.80, "arrival_price": 100.00,
        "vwap": 99.90, "notional": 49900,
    }
    defaults.update(overrides)
    return Fill(**defaults)


def _sample_fills() -> list[Fill]:
    return [
        _buy_fill(broker="AlgoX"),
        _sell_fill(broker="AlgoX"),
        _buy_fill(symbol="MSFT", fill_price=300.50, arrival_price=300.00,
                  vwap=300.30, notional=300500, broker="AlgoY"),
    ]


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        assert isinstance(result, ExecutionQualityResult)

    def test_n_fills(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        assert result.n_fills == 3

    def test_fill_analysis_types(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        for fa in result.fill_analyses:
            assert isinstance(fa, FillAnalysis)

    def test_total_notional(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        expected = 150200 + 49900 + 300500
        assert result.total_notional == pytest.approx(expected)

    def test_empty_fills(self):
        result = ExecutionAnalyzer().analyze([])
        assert result.n_fills == 0
        assert result.total_notional == 0.0


# ---------------------------------------------------------------------------
# Slippage (vs arrival)
# ---------------------------------------------------------------------------


class TestSlippage:
    def test_buy_slippage_positive_when_fill_above_arrival(self):
        """Buying at 150.20 vs arrival 150.00 => positive cost."""
        fill = _buy_fill(fill_price=150.20, arrival_price=150.00)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].slippage_bps > 0

    def test_buy_slippage_negative_when_fill_below_arrival(self):
        """Buying at 149.80 vs arrival 150.00 => negative cost (good execution)."""
        fill = _buy_fill(fill_price=149.80, arrival_price=150.00)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].slippage_bps < 0

    def test_sell_slippage_positive_when_fill_below_arrival(self):
        """Selling at 99.80 vs arrival 100.00 => positive cost."""
        fill = _sell_fill(fill_price=99.80, arrival_price=100.00)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].slippage_bps > 0

    def test_sell_slippage_negative_when_fill_above_arrival(self):
        """Selling at 100.20 vs arrival 100.00 => negative cost (good)."""
        fill = _sell_fill(fill_price=100.20, arrival_price=100.00)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].slippage_bps < 0

    def test_slippage_magnitude(self):
        """Buy at 150.15 vs arrival 150.00 => +10 bps."""
        fill = _buy_fill(fill_price=150.15, arrival_price=150.00)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].slippage_bps == pytest.approx(10.0, rel=0.01)

    def test_zero_slippage(self):
        fill = _buy_fill(fill_price=150.00, arrival_price=150.00)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].slippage_bps == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VWAP comparison
# ---------------------------------------------------------------------------


class TestVWAP:
    def test_buy_worse_than_vwap(self):
        """Buy at 150.20 vs VWAP 150.10 => positive (worse)."""
        fill = _buy_fill(fill_price=150.20, vwap=150.10)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].vs_vwap_bps > 0

    def test_buy_better_than_vwap(self):
        fill = _buy_fill(fill_price=150.05, vwap=150.10)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].vs_vwap_bps < 0

    def test_sell_worse_than_vwap(self):
        """Sell at 99.80 vs VWAP 99.90 => positive (worse)."""
        fill = _sell_fill(fill_price=99.80, vwap=99.90)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].vs_vwap_bps > 0

    def test_vwap_beat_rate(self):
        fills = [
            _buy_fill(fill_price=149.90, vwap=150.00),  # Beat VWAP
            _buy_fill(fill_price=150.10, vwap=150.00),  # Worse than VWAP
            _buy_fill(fill_price=149.95, vwap=150.00),  # Beat VWAP
        ]
        result = ExecutionAnalyzer().analyze(fills)
        assert result.pct_fills_beat_vwap == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# TWAP comparison
# ---------------------------------------------------------------------------


class TestTWAP:
    def test_twap_used_when_provided(self):
        fill = _buy_fill(fill_price=150.20, twap=150.05)
        result = ExecutionAnalyzer().analyze([fill])
        # vs_twap should use TWAP, not VWAP
        expected = (150.20 - 150.05) / 150.05 * 10_000
        assert result.fill_analyses[0].vs_twap_bps == pytest.approx(expected, rel=0.01)

    def test_twap_defaults_to_vwap(self):
        fill = _buy_fill(fill_price=150.20, vwap=150.10, twap=None)
        result = ExecutionAnalyzer().analyze([fill])
        expected = (150.20 - 150.10) / 150.10 * 10_000
        assert result.fill_analyses[0].vs_twap_bps == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Broker aggregation
# ---------------------------------------------------------------------------


class TestBrokerStats:
    def test_broker_stats_populated(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        assert len(result.broker_stats) == 2  # AlgoX, AlgoY

    def test_broker_stats_types(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        for bs in result.broker_stats:
            assert isinstance(bs, BrokerStats)

    def test_broker_fill_counts(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        algox = next(bs for bs in result.broker_stats if bs.broker == "AlgoX")
        algoy = next(bs for bs in result.broker_stats if bs.broker == "AlgoY")
        assert algox.n_fills == 2
        assert algoy.n_fills == 1

    def test_broker_notional(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        algox = next(bs for bs in result.broker_stats if bs.broker == "AlgoX")
        assert algox.total_notional == pytest.approx(150200 + 49900)

    def test_unknown_broker_used_for_none(self):
        fills = [_buy_fill(broker=None)]
        result = ExecutionAnalyzer().analyze(fills)
        assert result.broker_stats[0].broker == "unknown"


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


class TestAggregates:
    def test_avg_slippage_notional_weighted(self):
        fills = [
            _buy_fill(fill_price=150.15, arrival_price=150.00, notional=100000),
            _buy_fill(fill_price=150.30, arrival_price=150.00, notional=200000),
        ]
        result = ExecutionAnalyzer().analyze(fills)
        # Manual: slip1 = 10bps, slip2 = 20bps
        # Weighted: (10*100k + 20*200k) / 300k = 50000/300k = 16.67 bps
        assert result.avg_slippage_bps == pytest.approx(16.67, rel=0.01)

    def test_total_cost_dollars(self):
        fills = [_buy_fill(fill_price=150.15, arrival_price=150.00, notional=1_000_000)]
        result = ExecutionAnalyzer().analyze(fills)
        # Slippage = 10 bps => cost = 10/10000 * 1M = $1000
        assert result.total_cost_dollars == pytest.approx(1000, rel=0.01)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_fill(self):
        result = ExecutionAnalyzer().analyze([_buy_fill()])
        assert result.n_fills == 1

    def test_all_same_price(self):
        fill = _buy_fill(fill_price=100.00, arrival_price=100.00, vwap=100.00)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.avg_slippage_bps == pytest.approx(0.0)
        assert result.avg_vs_vwap_bps == pytest.approx(0.0)

    def test_zero_arrival_handled(self):
        fill = _buy_fill(arrival_price=0.0)
        result = ExecutionAnalyzer().analyze([fill])
        assert result.fill_analyses[0].slippage_bps == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        summary = result.summary()
        assert "Execution Quality" in summary
        assert "slippage" in summary.lower()
        assert "VWAP" in summary
        assert "Broker" in summary

    def test_summary_shows_brokers(self):
        result = ExecutionAnalyzer().analyze(_sample_fills())
        summary = result.summary()
        assert "AlgoX" in summary
        assert "AlgoY" in summary
