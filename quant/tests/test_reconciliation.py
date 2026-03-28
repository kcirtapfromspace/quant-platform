"""Tests for position reconciliation (QUA-113)."""
from __future__ import annotations

import pytest

from quant.oms.reconciliation import (
    BreakType,
    CorrectionAction,
    PositionReconciler,
    PositionSnapshot,
    ReconciliationConfig,
    ReconciliationReport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap(symbol: str, qty: float, cost: float = 100.0) -> PositionSnapshot:
    return PositionSnapshot(symbol=symbol, quantity=qty, avg_cost=cost)


def _matched_positions() -> tuple[list[PositionSnapshot], list[PositionSnapshot]]:
    """OMS and broker agree on everything."""
    oms = [_snap("AAPL", 100), _snap("MSFT", 50), _snap("GOOG", -25)]
    broker = [_snap("AAPL", 100), _snap("MSFT", 50), _snap("GOOG", -25)]
    return oms, broker


def _mixed_breaks() -> tuple[list[PositionSnapshot], list[PositionSnapshot]]:
    """Mix of matched, quantity break, phantom, and missing."""
    oms = [
        _snap("AAPL", 100),       # matched
        _snap("MSFT", 50),        # quantity break — broker has 60
        _snap("TSLA", 30),        # phantom — broker has nothing
    ]
    broker = [
        _snap("AAPL", 100),       # matched
        _snap("MSFT", 60),        # quantity break
        _snap("AMZN", 40, 150.0), # missing from OMS
    ]
    return oms, broker


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_report_type(self):
        oms, broker = _matched_positions()
        report = PositionReconciler().reconcile(oms, broker)
        assert isinstance(report, ReconciliationReport)

    def test_default_config(self):
        r = PositionReconciler()
        assert r.config.quantity_tolerance == pytest.approx(1e-6)
        assert r.config.flag_price_drift is True

    def test_custom_config(self):
        cfg = ReconciliationConfig(quantity_tolerance=0.5)
        r = PositionReconciler(cfg)
        assert r.config.quantity_tolerance == 0.5

    def test_empty_positions(self):
        report = PositionReconciler().reconcile([], [])
        assert report.n_total == 0
        assert not report.has_breaks

    def test_timestamp_set(self):
        report = PositionReconciler().reconcile([], [])
        assert report.timestamp is not None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class TestMatching:
    def test_all_matched(self):
        oms, broker = _matched_positions()
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_matched == 3
        assert report.n_quantity_breaks == 0
        assert report.n_phantom == 0
        assert report.n_missing == 0
        assert not report.has_breaks

    def test_within_tolerance_matched(self):
        """Tiny difference below tolerance is treated as matched."""
        oms = [_snap("AAPL", 100.0)]
        broker = [_snap("AAPL", 100.0 + 1e-8)]
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_matched == 1

    def test_matched_items_have_correct_type(self):
        oms, broker = _matched_positions()
        report = PositionReconciler().reconcile(oms, broker)
        for item in report.items:
            assert item.break_type == BreakType.MATCHED

    def test_short_positions_matched(self):
        oms = [_snap("AAPL", -50)]
        broker = [_snap("AAPL", -50)]
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_matched == 1


# ---------------------------------------------------------------------------
# Quantity breaks
# ---------------------------------------------------------------------------


class TestQuantityBreak:
    def test_detects_quantity_break(self):
        oms = [_snap("AAPL", 100)]
        broker = [_snap("AAPL", 110)]
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_quantity_breaks == 1

    def test_quantity_diff_sign(self):
        """diff = broker - OMS, so positive means broker has more."""
        oms = [_snap("AAPL", 100)]
        broker = [_snap("AAPL", 110)]
        report = PositionReconciler().reconcile(oms, broker)
        brk = report.breaks[0]
        assert brk.quantity_diff == pytest.approx(10.0)

    def test_negative_diff(self):
        """OMS has more than broker."""
        oms = [_snap("AAPL", 100)]
        broker = [_snap("AAPL", 90)]
        report = PositionReconciler().reconcile(oms, broker)
        brk = report.breaks[0]
        assert brk.quantity_diff == pytest.approx(-10.0)

    def test_custom_tolerance(self):
        """Break only if diff exceeds custom tolerance."""
        cfg = ReconciliationConfig(quantity_tolerance=5.0)
        oms = [_snap("AAPL", 100)]
        broker = [_snap("AAPL", 103)]
        report = PositionReconciler(cfg).reconcile(oms, broker)
        assert report.n_matched == 1  # within tolerance

    def test_at_boundary(self):
        """Exactly at tolerance boundary is matched (not >)."""
        cfg = ReconciliationConfig(quantity_tolerance=5.0)
        oms = [_snap("AAPL", 100)]
        broker = [_snap("AAPL", 105)]
        report = PositionReconciler(cfg).reconcile(oms, broker)
        assert report.n_matched == 1


# ---------------------------------------------------------------------------
# Phantom positions
# ---------------------------------------------------------------------------


class TestPhantom:
    def test_detects_phantom(self):
        oms = [_snap("AAPL", 100)]
        broker: list[PositionSnapshot] = []
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_phantom == 1

    def test_phantom_item_values(self):
        oms = [_snap("AAPL", 100, 150.0)]
        report = PositionReconciler().reconcile(oms, [])
        item = report.breaks[0]
        assert item.break_type == BreakType.PHANTOM
        assert item.oms_quantity == 100.0
        assert item.broker_quantity == 0.0
        assert item.quantity_diff == pytest.approx(-100.0)

    def test_multiple_phantoms(self):
        oms = [_snap("AAPL", 100), _snap("MSFT", 50)]
        report = PositionReconciler().reconcile(oms, [])
        assert report.n_phantom == 2


# ---------------------------------------------------------------------------
# Missing positions
# ---------------------------------------------------------------------------


class TestMissing:
    def test_detects_missing(self):
        broker = [_snap("AAPL", 100)]
        report = PositionReconciler().reconcile([], broker)
        assert report.n_missing == 1

    def test_missing_item_values(self):
        broker = [_snap("AAPL", 100, 150.0)]
        report = PositionReconciler().reconcile([], broker)
        item = report.breaks[0]
        assert item.break_type == BreakType.MISSING
        assert item.oms_quantity == 0.0
        assert item.broker_quantity == 100.0
        assert item.broker_avg_cost == 150.0

    def test_multiple_missing(self):
        broker = [_snap("AAPL", 100), _snap("GOOG", -30)]
        report = PositionReconciler().reconcile([], broker)
        assert report.n_missing == 2


# ---------------------------------------------------------------------------
# Mixed scenario
# ---------------------------------------------------------------------------


class TestMixed:
    def test_mixed_break_counts(self):
        oms, broker = _mixed_breaks()
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_matched == 1       # AAPL
        assert report.n_quantity_breaks == 1  # MSFT
        assert report.n_phantom == 1       # TSLA
        assert report.n_missing == 1       # AMZN

    def test_has_breaks_true(self):
        oms, broker = _mixed_breaks()
        report = PositionReconciler().reconcile(oms, broker)
        assert report.has_breaks is True

    def test_breaks_property_excludes_matched(self):
        oms, broker = _mixed_breaks()
        report = PositionReconciler().reconcile(oms, broker)
        assert len(report.breaks) == 3
        for item in report.breaks:
            assert item.break_type != BreakType.MATCHED

    def test_total_count(self):
        oms, broker = _mixed_breaks()
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_total == 4  # AAPL, MSFT, TSLA, AMZN


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------


class TestCorrections:
    def test_quantity_break_correction(self):
        oms = [_snap("AAPL", 100)]
        broker = [_snap("AAPL", 110, 155.0)]
        report = PositionReconciler().reconcile(oms, broker)
        corrections = PositionReconciler().compute_corrections(report)
        assert len(corrections) == 1
        c = corrections[0]
        assert c.symbol == "AAPL"
        assert c.action == CorrectionAction.SET_QUANTITY
        assert c.target_qty == pytest.approx(110.0)
        assert c.target_cost == pytest.approx(155.0)

    def test_phantom_correction(self):
        oms = [_snap("TSLA", 30)]
        report = PositionReconciler().reconcile(oms, [])
        corrections = PositionReconciler().compute_corrections(report)
        assert len(corrections) == 1
        c = corrections[0]
        assert c.action == CorrectionAction.REMOVE_POSITION
        assert c.target_qty == 0.0

    def test_missing_correction(self):
        broker = [_snap("AMZN", 40, 150.0)]
        report = PositionReconciler().reconcile([], broker)
        corrections = PositionReconciler().compute_corrections(report)
        assert len(corrections) == 1
        c = corrections[0]
        assert c.action == CorrectionAction.CREATE_POSITION
        assert c.target_qty == pytest.approx(40.0)
        assert c.target_cost == pytest.approx(150.0)

    def test_no_corrections_when_matched(self):
        oms, broker = _matched_positions()
        report = PositionReconciler().reconcile(oms, broker)
        corrections = PositionReconciler().compute_corrections(report)
        assert corrections == []

    def test_mixed_corrections_count(self):
        oms, broker = _mixed_breaks()
        report = PositionReconciler().reconcile(oms, broker)
        corrections = PositionReconciler().compute_corrections(report)
        assert len(corrections) == 3  # qty break + phantom + missing

    def test_correction_has_reason(self):
        oms = [_snap("AAPL", 100)]
        broker = [_snap("AAPL", 110)]
        report = PositionReconciler().reconcile(oms, broker)
        corrections = PositionReconciler().compute_corrections(report)
        assert "mismatch" in corrections[0].reason.lower()


# ---------------------------------------------------------------------------
# Price drift
# ---------------------------------------------------------------------------


class TestPriceDrift:
    def test_flags_price_drift(self):
        """Matched quantity but divergent avg cost."""
        oms = [_snap("AAPL", 100, 150.0)]
        broker = [_snap("AAPL", 100, 160.0)]
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_matched == 1
        flags = PositionReconciler().price_drift_flags(report)
        assert len(flags) == 1
        assert flags[0].price_drift_pct > 0.01

    def test_no_drift_flag_within_tolerance(self):
        oms = [_snap("AAPL", 100, 150.0)]
        broker = [_snap("AAPL", 100, 150.5)]
        report = PositionReconciler().reconcile(oms, broker)
        flags = PositionReconciler().price_drift_flags(report)
        assert len(flags) == 0

    def test_drift_flag_disabled(self):
        cfg = ReconciliationConfig(flag_price_drift=False)
        oms = [_snap("AAPL", 100, 150.0)]
        broker = [_snap("AAPL", 100, 200.0)]
        report = PositionReconciler(cfg).reconcile(oms, broker)
        flags = PositionReconciler(cfg).price_drift_flags(report)
        assert len(flags) == 0

    def test_drift_pct_calculation(self):
        oms = [_snap("AAPL", 100, 100.0)]
        broker = [_snap("AAPL", 100, 110.0)]
        report = PositionReconciler().reconcile(oms, broker)
        item = report.items[0]
        # |100 - 110| / 110 ≈ 0.0909
        assert item.price_drift_pct == pytest.approx(10.0 / 110.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_quantity_positions(self):
        oms = [_snap("AAPL", 0.0)]
        broker = [_snap("AAPL", 0.0)]
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_matched == 1

    def test_duplicate_symbols_last_wins(self):
        """If duplicate symbols, last one in list wins (dict behavior)."""
        oms = [_snap("AAPL", 100), _snap("AAPL", 200)]
        broker = [_snap("AAPL", 200)]
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_matched == 1

    def test_very_large_positions(self):
        oms = [_snap("AAPL", 1e9)]
        broker = [_snap("AAPL", 1e9 + 1)]
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_quantity_breaks == 1

    def test_fractional_shares(self):
        oms = [_snap("BTC", 0.12345678)]
        broker = [_snap("BTC", 0.12345678)]
        report = PositionReconciler().reconcile(oms, broker)
        assert report.n_matched == 1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_clean_summary(self):
        oms, broker = _matched_positions()
        report = PositionReconciler().reconcile(oms, broker)
        summary = report.summary()
        assert "CLEAN" in summary
        assert "Reconciliation" in summary

    def test_breaks_summary(self):
        oms, broker = _mixed_breaks()
        report = PositionReconciler().reconcile(oms, broker)
        summary = report.summary()
        assert "BREAKS FOUND" in summary
        assert "TSLA" in summary  # phantom
        assert "AMZN" in summary  # missing
