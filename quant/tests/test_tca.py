"""Tests for transaction cost analysis (QUA-29)."""
from __future__ import annotations

import pytest

from quant.execution.tca import (
    ExecutionRecord,
    TCACollector,
    TCAReport,
)

# ── ExecutionRecord unit tests ───────────────────────────────────────────────


class TestExecutionRecord:
    def test_buy_implementation_shortfall_positive(self):
        """Buying at a higher price than arrival = cost."""
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="buy",
            quantity=100, fill_price=150.50, arrival_price=150.00,
        )
        assert r.implementation_shortfall == pytest.approx(0.50 / 150.00)

    def test_buy_implementation_shortfall_negative(self):
        """Buying at a lower price than arrival = improvement."""
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="buy",
            quantity=100, fill_price=149.50, arrival_price=150.00,
        )
        assert r.implementation_shortfall < 0

    def test_sell_implementation_shortfall_positive(self):
        """Selling at a lower price than arrival = cost."""
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="sell",
            quantity=100, fill_price=149.50, arrival_price=150.00,
        )
        assert r.implementation_shortfall == pytest.approx(0.50 / 150.00)

    def test_sell_implementation_shortfall_negative(self):
        """Selling at a higher price than arrival = improvement."""
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="sell",
            quantity=100, fill_price=150.50, arrival_price=150.00,
        )
        assert r.implementation_shortfall < 0

    def test_no_arrival_price_returns_none(self):
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="buy",
            quantity=100, fill_price=150.0,
        )
        assert r.implementation_shortfall is None
        assert r.dollar_cost is None

    def test_vwap_slippage(self):
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="buy",
            quantity=100, fill_price=151.00, vwap_price=150.00,
        )
        assert r.vwap_slippage == pytest.approx(1.0 / 150.0)

    def test_close_slippage(self):
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="buy",
            quantity=50, fill_price=152.00, close_price=150.00,
        )
        assert r.close_slippage == pytest.approx(2.0 / 150.0)

    def test_open_slippage(self):
        r = ExecutionRecord(
            order_id="o1", symbol="MSFT", side="sell",
            quantity=30, fill_price=300.00, open_price=302.00,
        )
        assert r.open_slippage == pytest.approx(2.0 / 302.0)

    def test_dollar_cost_buy(self):
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="buy",
            quantity=100, fill_price=150.50, arrival_price=150.00,
        )
        expected = (0.50 / 150.00) * 150.50 * 100
        assert r.dollar_cost == pytest.approx(expected)

    def test_zero_benchmark_returns_none(self):
        r = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="buy",
            quantity=100, fill_price=150.0, arrival_price=0.0,
        )
        assert r.implementation_shortfall is None


# ── TCACollector tests ───────────────────────────────────────────────────────


class TestTCACollector:
    def test_register_and_record(self):
        c = TCACollector()
        c.register_decision("o1", arrival_price=100.0, strategy_id="mom")
        rec = c.record_fill("o1", fill_price=100.50, quantity=10,
                            side="buy", symbol="AAPL")
        assert rec.arrival_price == 100.0
        assert rec.strategy_id == "mom"
        assert rec.implementation_shortfall is not None

    def test_record_without_decision(self):
        """Fills can be recorded even without a pre-registered decision."""
        c = TCACollector()
        rec = c.record_fill("o1", fill_price=100.0, quantity=10,
                            side="buy", symbol="AAPL")
        assert rec.arrival_price is None
        assert rec.implementation_shortfall is None

    def test_post_fill_benchmarks_override_decision(self):
        c = TCACollector()
        c.register_decision("o1", vwap_price=100.0)
        rec = c.record_fill("o1", fill_price=101.0, quantity=10,
                            side="buy", symbol="AAPL", vwap_price=100.5)
        assert rec.vwap_price == 100.5  # override

    def test_post_fill_benchmarks_fallback_to_decision(self):
        c = TCACollector()
        c.register_decision("o1", vwap_price=100.0, close_price=99.0)
        rec = c.record_fill("o1", fill_price=101.0, quantity=10,
                            side="buy", symbol="AAPL")
        assert rec.vwap_price == 100.0  # from decision
        assert rec.close_price == 99.0  # from decision

    def test_clear_resets_state(self):
        c = TCACollector()
        c.register_decision("o1", arrival_price=100.0)
        c.record_fill("o1", fill_price=100.5, quantity=5,
                      side="buy", symbol="AAPL")
        assert len(c.records) == 1
        c.clear()
        assert len(c.records) == 0

    def test_multiple_fills(self):
        c = TCACollector()
        c.register_decision("o1", arrival_price=100.0)
        c.register_decision("o2", arrival_price=200.0)
        c.record_fill("o1", fill_price=100.10, quantity=50,
                      side="buy", symbol="AAPL")
        c.record_fill("o2", fill_price=199.50, quantity=30,
                      side="sell", symbol="MSFT")
        assert len(c.records) == 2


# ── TCAReport tests ──────────────────────────────────────────────────────────


class TestTCAReport:
    def _make_records(self) -> list[ExecutionRecord]:
        return [
            ExecutionRecord(
                order_id="o1", symbol="AAPL", side="buy",
                quantity=100, fill_price=150.50, arrival_price=150.00,
                strategy_id="momentum",
            ),
            ExecutionRecord(
                order_id="o2", symbol="AAPL", side="buy",
                quantity=200, fill_price=151.00, arrival_price=150.00,
                strategy_id="momentum",
            ),
            ExecutionRecord(
                order_id="o3", symbol="MSFT", side="sell",
                quantity=50, fill_price=299.00, arrival_price=300.00,
                strategy_id="mean_rev",
            ),
        ]

    def test_n_fills(self):
        report = TCAReport(records=self._make_records())
        assert report.n_fills == 3

    def test_total_notional(self):
        records = self._make_records()
        report = TCAReport(records=records)
        expected = 150.50 * 100 + 151.00 * 200 + 299.00 * 50
        assert report.total_notional == pytest.approx(expected)

    def test_mean_implementation_shortfall(self):
        records = self._make_records()
        report = TCAReport(records=records)
        mis = report.mean_implementation_shortfall
        assert mis is not None
        # All three have positive shortfall so mean should be positive
        assert mis > 0

    def test_total_dollar_cost(self):
        records = self._make_records()
        report = TCAReport(records=records)
        # Each record has a dollar cost
        assert report.total_dollar_cost > 0

    def test_by_strategy(self):
        records = self._make_records()
        report = TCAReport(records=records)
        by_strat = report.by_strategy()
        assert "momentum" in by_strat
        assert "mean_rev" in by_strat
        assert by_strat["momentum"].n_fills == 2
        assert by_strat["mean_rev"].n_fills == 1

    def test_by_symbol(self):
        records = self._make_records()
        report = TCAReport(records=records)
        by_sym = report.by_symbol()
        assert "AAPL" in by_sym
        assert "MSFT" in by_sym
        assert by_sym["AAPL"].n_fills == 2
        assert by_sym["MSFT"].n_fills == 1

    def test_empty_report(self):
        report = TCAReport(records=[])
        assert report.n_fills == 0
        assert report.total_notional == 0.0
        assert report.mean_implementation_shortfall is None
        assert report.total_dollar_cost == 0.0

    def test_summary_contains_key_fields(self):
        records = self._make_records()
        report = TCAReport(records=records)
        text = report.summary()
        assert "TCA Report" in text
        assert "shortfall" in text.lower()
        assert "AAPL" in text
        assert "MSFT" in text

    def test_mean_vwap_slippage_with_vwap(self):
        records = [
            ExecutionRecord(
                order_id="o1", symbol="AAPL", side="buy",
                quantity=100, fill_price=150.50, vwap_price=150.00,
            ),
            ExecutionRecord(
                order_id="o2", symbol="AAPL", side="buy",
                quantity=100, fill_price=150.20, vwap_price=150.00,
            ),
        ]
        report = TCAReport(records=records)
        mvs = report.mean_vwap_slippage
        assert mvs is not None
        assert mvs > 0  # both fills above VWAP

    def test_mean_vwap_slippage_no_vwap(self):
        records = [
            ExecutionRecord(
                order_id="o1", symbol="AAPL", side="buy",
                quantity=100, fill_price=150.00,
            ),
        ]
        report = TCAReport(records=records)
        assert report.mean_vwap_slippage is None

    def test_weighted_mean_favors_large_trades(self):
        """Notional-weighted mean should weight large trades more heavily."""
        small_bad = ExecutionRecord(
            order_id="o1", symbol="AAPL", side="buy",
            quantity=10, fill_price=155.0, arrival_price=150.0,
        )  # 3.3% shortfall, small notional
        large_good = ExecutionRecord(
            order_id="o2", symbol="AAPL", side="buy",
            quantity=1000, fill_price=150.10, arrival_price=150.0,
        )  # 0.067% shortfall, large notional
        report = TCAReport(records=[small_bad, large_good])
        mis = report.mean_implementation_shortfall
        assert mis is not None
        # Weighted mean should be much closer to the large trade's shortfall
        assert mis < 0.005  # well under 0.5%


class TestTCAReportSingleBenchmark:
    def test_close_slippage_report(self):
        records = [
            ExecutionRecord(
                order_id="o1", symbol="AAPL", side="buy",
                quantity=100, fill_price=151.0, close_price=150.0,
            ),
        ]
        report = TCAReport(records=records)
        mcs = report.mean_close_slippage
        assert mcs is not None
        assert mcs == pytest.approx(1.0 / 150.0)
