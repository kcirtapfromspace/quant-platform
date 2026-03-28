"""Tests for benchmark-relative analytics (QUA-112)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.benchmark_analytics import (
    ActivePosition,
    ActiveRiskResult,
    ActiveWeightResult,
    BenchmarkAnalyzer,
    BenchmarkConfig,
    BrinsonResult,
    SectorEffect,
    TEBudgetResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DAYS = 500


def _returns_pair(
    seed: int = 42,
    alpha_bps: float = 5.0,
    te_bps: float = 50.0,
) -> tuple[pd.Series, pd.Series]:
    """Generate portfolio & benchmark return series.

    Portfolio = benchmark + alpha + tracking noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=N_DAYS)

    bench = rng.normal(0.0004, 0.01, N_DAYS)  # ~10% annual return
    noise = rng.normal(alpha_bps / 10000, te_bps / 10000, N_DAYS)
    port = bench + noise

    return (
        pd.Series(port, index=dates, name="portfolio"),
        pd.Series(bench, index=dates, name="benchmark"),
    )


def _weights() -> tuple[pd.Series, pd.Series]:
    """Portfolio and benchmark weights for a simple 5-asset universe."""
    port = pd.Series(
        {"AAPL": 0.25, "MSFT": 0.20, "GOOG": 0.20, "AMZN": 0.15, "META": 0.20},
    )
    bench = pd.Series(
        {"AAPL": 0.20, "MSFT": 0.20, "GOOG": 0.25, "AMZN": 0.20, "META": 0.15},
    )
    return port, bench


def _covariance(symbols: list[str] | None = None, seed: int = 42) -> pd.DataFrame:
    """Simple factor-model covariance for testing."""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    rng = np.random.default_rng(seed)
    n = len(symbols)
    # Factor model: 1 market factor + idiosyncratic
    beta = rng.uniform(0.8, 1.2, n)
    market_var = 0.0002
    idio_var = rng.uniform(0.0001, 0.0003, n)
    cov = np.outer(beta, beta) * market_var + np.diag(idio_var)
    return pd.DataFrame(cov, index=symbols, columns=symbols)


def _sector_map() -> dict[str, str]:
    return {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "GOOG": "Comm",
        "AMZN": "Consumer",
        "META": "Comm",
    }


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_active_risk_result(self):
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert isinstance(result, ActiveRiskResult)

    def test_config_accessible(self):
        cfg = BenchmarkConfig(trading_days=260)
        analyzer = BenchmarkAnalyzer(cfg)
        assert analyzer.config.trading_days == 260

    def test_default_config(self):
        analyzer = BenchmarkAnalyzer()
        assert analyzer.config.annualise is True
        assert analyzer.config.trading_days == 252

    def test_too_few_periods_raises(self):
        port = pd.Series([0.01], index=pd.bdate_range("2022-01-01", periods=1))
        bench = pd.Series([0.005], index=pd.bdate_range("2022-01-01", periods=1))
        with pytest.raises(ValueError, match="at least 2"):
            BenchmarkAnalyzer().active_risk(port, bench)

    def test_n_periods(self):
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert result.n_periods == N_DAYS


# ---------------------------------------------------------------------------
# Active risk metrics
# ---------------------------------------------------------------------------


class TestActiveRisk:
    def test_tracking_error_positive(self):
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert result.tracking_error > 0

    def test_information_ratio_sign(self):
        """Positive alpha should produce positive IR."""
        port, bench = _returns_pair(alpha_bps=20.0, te_bps=30.0)
        result = BenchmarkAnalyzer().active_risk(port, bench)
        # With 20bps daily alpha and 30bps TE, IR should be positive
        assert result.information_ratio > 0

    def test_zero_alpha_near_zero_ir(self):
        """Zero alpha should produce IR near zero."""
        port, bench = _returns_pair(alpha_bps=0.0, te_bps=50.0)
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert abs(result.information_ratio) < 1.0

    def test_higher_te_with_more_noise(self):
        low_te = BenchmarkAnalyzer().active_risk(
            *_returns_pair(te_bps=20.0),
        )
        high_te = BenchmarkAnalyzer().active_risk(
            *_returns_pair(te_bps=100.0),
        )
        assert high_te.tracking_error > low_te.tracking_error

    def test_hit_rate_bounded(self):
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert 0 < result.hit_rate < 1

    def test_max_relative_dd_negative(self):
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert result.max_relative_dd <= 0

    def test_active_return_series_length(self):
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert len(result.active_return_series) == N_DAYS

    def test_identical_returns_zero_te(self):
        """Identical portfolio and benchmark should have zero TE."""
        bench = pd.Series(
            np.random.default_rng(42).normal(0.0004, 0.01, 100),
            index=pd.bdate_range("2022-01-01", periods=100),
        )
        result = BenchmarkAnalyzer().active_risk(bench, bench.copy())
        assert result.tracking_error == pytest.approx(0.0, abs=1e-10)
        assert result.active_return == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Active share
# ---------------------------------------------------------------------------


class TestActiveShare:
    def test_active_share_with_weights(self):
        port, bench = _returns_pair()
        wp, wb = _weights()
        result = BenchmarkAnalyzer().active_risk(
            port, bench, portfolio_weights=wp, benchmark_weights=wb,
        )
        assert 0 < result.active_share < 1

    def test_active_share_without_weights(self):
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert result.active_share < 0  # sentinel value

    def test_identical_weights_zero_active_share(self):
        port, bench = _returns_pair()
        wp, _ = _weights()
        result = BenchmarkAnalyzer().active_risk(
            port, bench, portfolio_weights=wp, benchmark_weights=wp.copy(),
        )
        assert result.active_share == pytest.approx(0.0, abs=1e-10)

    def test_active_share_calculation(self):
        """Manual active share check."""
        wp = pd.Series({"A": 0.6, "B": 0.4})
        wb = pd.Series({"A": 0.4, "B": 0.6})
        # |0.2| + |0.2| = 0.4, / 2 = 0.2
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(
            port, bench, portfolio_weights=wp, benchmark_weights=wb,
        )
        assert result.active_share == pytest.approx(0.2, abs=1e-10)


# ---------------------------------------------------------------------------
# Active weight decomposition
# ---------------------------------------------------------------------------


class TestActiveWeights:
    def test_returns_result_type(self):
        wp, wb = _weights()
        result = BenchmarkAnalyzer().active_weights(wp, wb)
        assert isinstance(result, ActiveWeightResult)

    def test_position_types(self):
        wp, wb = _weights()
        result = BenchmarkAnalyzer().active_weights(wp, wb)
        for p in result.positions:
            assert isinstance(p, ActivePosition)

    def test_active_share_matches(self):
        wp, wb = _weights()
        result = BenchmarkAnalyzer().active_weights(wp, wb)
        expected = float(np.abs(wp.reindex(
            sorted(set(wp.index) | set(wb.index)), fill_value=0.0,
        ) - wb.reindex(
            sorted(set(wp.index) | set(wb.index)), fill_value=0.0,
        )).sum()) / 2
        assert result.active_share == pytest.approx(expected, abs=1e-10)

    def test_overweight_underweight_count(self):
        wp, wb = _weights()
        result = BenchmarkAnalyzer().active_weights(wp, wb)
        assert result.n_overweight + result.n_underweight <= len(result.positions)
        assert result.n_overweight >= 0
        assert result.n_underweight >= 0

    def test_sorted_by_abs_active_weight(self):
        wp, wb = _weights()
        result = BenchmarkAnalyzer().active_weights(wp, wb)
        abs_weights = [abs(p.active_weight) for p in result.positions]
        assert abs_weights == sorted(abs_weights, reverse=True)

    def test_with_covariance_mcte_nonzero(self):
        wp, wb = _weights()
        cov = _covariance()
        result = BenchmarkAnalyzer().active_weights(wp, wb, covariance=cov)
        mctes = [p.mcte for p in result.positions]
        assert any(abs(m) > 1e-10 for m in mctes)

    def test_without_covariance_mcte_zero(self):
        wp, wb = _weights()
        result = BenchmarkAnalyzer().active_weights(wp, wb)
        for p in result.positions:
            assert p.mcte == 0.0

    def test_disjoint_symbols(self):
        """Portfolio and benchmark with non-overlapping symbols."""
        wp = pd.Series({"A": 0.5, "B": 0.5})
        wb = pd.Series({"C": 0.5, "D": 0.5})
        result = BenchmarkAnalyzer().active_weights(wp, wb)
        assert len(result.positions) == 4
        assert result.active_share == pytest.approx(1.0, abs=1e-10)

    def test_empty_weights_raises(self):
        wp = pd.Series(dtype=float)
        wb = pd.Series(dtype=float)
        with pytest.raises(ValueError, match="empty"):
            BenchmarkAnalyzer().active_weights(wp, wb)


# ---------------------------------------------------------------------------
# Brinson-Fachler attribution
# ---------------------------------------------------------------------------


class TestBrinson:
    def test_returns_result_type(self):
        wp, wb = _weights()
        rng = np.random.default_rng(42)
        syms = list(wp.index)
        rp = pd.Series(rng.normal(0.01, 0.02, len(syms)), index=syms)
        rb = pd.Series(rng.normal(0.008, 0.02, len(syms)), index=syms)
        result = BenchmarkAnalyzer().brinson_attribution(
            wp, wb, rp, rb, _sector_map(),
        )
        assert isinstance(result, BrinsonResult)

    def test_effects_sum_to_active_return(self):
        wp, wb = _weights()
        rng = np.random.default_rng(42)
        syms = list(wp.index)
        rp = pd.Series(rng.normal(0.01, 0.02, len(syms)), index=syms)
        rb = pd.Series(rng.normal(0.008, 0.02, len(syms)), index=syms)
        result = BenchmarkAnalyzer().brinson_attribution(
            wp, wb, rp, rb, _sector_map(),
        )
        expected = result.allocation_total + result.selection_total + result.interaction_total
        assert result.active_return == pytest.approx(expected, abs=1e-12)

    def test_sector_effects_sum(self):
        wp, wb = _weights()
        rng = np.random.default_rng(42)
        syms = list(wp.index)
        rp = pd.Series(rng.normal(0.01, 0.02, len(syms)), index=syms)
        rb = pd.Series(rng.normal(0.008, 0.02, len(syms)), index=syms)
        result = BenchmarkAnalyzer().brinson_attribution(
            wp, wb, rp, rb, _sector_map(),
        )
        sector_total = sum(se.total for se in result.sector_effects)
        assert sector_total == pytest.approx(result.active_return, abs=1e-12)

    def test_sector_effect_types(self):
        wp, wb = _weights()
        rng = np.random.default_rng(42)
        syms = list(wp.index)
        rp = pd.Series(rng.normal(0.01, 0.02, len(syms)), index=syms)
        rb = pd.Series(rng.normal(0.008, 0.02, len(syms)), index=syms)
        result = BenchmarkAnalyzer().brinson_attribution(
            wp, wb, rp, rb, _sector_map(),
        )
        for se in result.sector_effects:
            assert isinstance(se, SectorEffect)

    def test_identical_weights_zero_allocation(self):
        """Same weights → zero allocation effect."""
        wp, _ = _weights()
        rng = np.random.default_rng(42)
        syms = list(wp.index)
        rp = pd.Series(rng.normal(0.01, 0.02, len(syms)), index=syms)
        rb = pd.Series(rng.normal(0.008, 0.02, len(syms)), index=syms)
        result = BenchmarkAnalyzer().brinson_attribution(
            wp, wp.copy(), rp, rb, _sector_map(),
        )
        assert result.allocation_total == pytest.approx(0.0, abs=1e-12)
        assert result.interaction_total == pytest.approx(0.0, abs=1e-12)

    def test_identical_returns_zero_selection(self):
        """Same returns across all stocks → zero selection and interaction."""
        wp, wb = _weights()
        syms = list(wp.index)
        # All stocks have the same return, so within-sector weighting
        # doesn't matter → selection = 0
        r = pd.Series(0.01, index=syms)
        result = BenchmarkAnalyzer().brinson_attribution(
            wp, wb, r, r.copy(), _sector_map(),
        )
        assert result.selection_total == pytest.approx(0.0, abs=1e-12)
        assert result.interaction_total == pytest.approx(0.0, abs=1e-12)

    def test_empty_sector_map_raises(self):
        wp, wb = _weights()
        syms = list(wp.index)
        rp = pd.Series(0.01, index=syms)
        rb = pd.Series(0.01, index=syms)
        with pytest.raises(ValueError, match="empty"):
            BenchmarkAnalyzer().brinson_attribution(wp, wb, rp, rb, {})


# ---------------------------------------------------------------------------
# Tracking error budget
# ---------------------------------------------------------------------------


class TestTEBudget:
    def test_returns_result_type(self):
        wp, wb = _weights()
        cov = _covariance()
        result = BenchmarkAnalyzer().tracking_error_budget(cov, wp, wb)
        assert isinstance(result, TEBudgetResult)

    def test_te_positive(self):
        wp, wb = _weights()
        cov = _covariance()
        result = BenchmarkAnalyzer().tracking_error_budget(cov, wp, wb)
        assert result.tracking_error > 0

    def test_risk_contrib_sums_to_one(self):
        wp, wb = _weights()
        cov = _covariance()
        result = BenchmarkAnalyzer().tracking_error_budget(cov, wp, wb)
        assert float(result.risk_contrib.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_identical_weights_zero_te(self):
        wp, _ = _weights()
        cov = _covariance()
        result = BenchmarkAnalyzer().tracking_error_budget(cov, wp, wp.copy())
        assert result.tracking_error == pytest.approx(0.0, abs=1e-10)

    def test_symbols_match_covariance(self):
        wp, wb = _weights()
        cov = _covariance()
        result = BenchmarkAnalyzer().tracking_error_budget(cov, wp, wb)
        assert result.symbols == list(cov.index)

    def test_active_weights_correct(self):
        wp, wb = _weights()
        cov = _covariance()
        result = BenchmarkAnalyzer().tracking_error_budget(cov, wp, wb)
        symbols = result.symbols
        expected = wp.reindex(symbols, fill_value=0.0).values - wb.reindex(
            symbols, fill_value=0.0,
        ).values
        np.testing.assert_array_almost_equal(result.active_weights, expected)

    def test_small_cov_raises(self):
        cov = pd.DataFrame([[0.01]], index=["A"], columns=["A"])
        wp = pd.Series({"A": 1.0})
        wb = pd.Series({"A": 1.0})
        with pytest.raises(ValueError, match="at least 2"):
            BenchmarkAnalyzer().tracking_error_budget(cov, wp, wb)

    def test_annualisation(self):
        wp, wb = _weights()
        cov = _covariance()
        ann = BenchmarkAnalyzer(BenchmarkConfig(annualise=True)).tracking_error_budget(
            cov, wp, wb,
        )
        no_ann = BenchmarkAnalyzer(BenchmarkConfig(annualise=False)).tracking_error_budget(
            cov, wp, wb,
        )
        # Annualised TE should be sqrt(252) times larger
        ratio = ann.tracking_error / no_ann.tracking_error
        assert ratio == pytest.approx(math.sqrt(252), rel=0.01)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_partial_overlap_returns(self):
        """Portfolio and benchmark with partially overlapping dates."""
        dates_a = pd.bdate_range("2022-01-01", periods=100)
        dates_b = pd.bdate_range("2022-02-01", periods=100)
        rng = np.random.default_rng(42)
        port = pd.Series(rng.normal(0.001, 0.01, 100), index=dates_a)
        bench = pd.Series(rng.normal(0.0005, 0.01, 100), index=dates_b)
        result = BenchmarkAnalyzer().active_risk(port, bench)
        assert result.n_periods > 0
        assert result.n_periods < 100

    def test_extra_symbols_in_weights(self):
        """Weights contain symbols not in covariance → they're ignored for MCTE."""
        wp = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        wb = pd.Series({"A": 0.5, "B": 0.5})
        cov = _covariance(["A", "B"])
        result = BenchmarkAnalyzer().active_weights(wp, wb, covariance=cov)
        assert len(result.positions) == 3

    def test_non_annualised(self):
        port, bench = _returns_pair()
        cfg = BenchmarkConfig(annualise=False)
        result = BenchmarkAnalyzer(cfg).active_risk(port, bench)
        # Non-annualised TE should be smaller than annualised
        ann_result = BenchmarkAnalyzer().active_risk(port, bench)
        assert result.tracking_error < ann_result.tracking_error


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_active_risk_summary(self):
        port, bench = _returns_pair()
        result = BenchmarkAnalyzer().active_risk(port, bench)
        summary = result.summary()
        assert "Active Risk" in summary
        assert "Tracking error" in summary
        assert "Information ratio" in summary

    def test_active_weight_summary(self):
        wp, wb = _weights()
        result = BenchmarkAnalyzer().active_weights(wp, wb)
        summary = result.summary()
        assert "Active Weight" in summary
        assert "Active share" in summary

    def test_brinson_summary(self):
        wp, wb = _weights()
        rng = np.random.default_rng(42)
        syms = list(wp.index)
        rp = pd.Series(rng.normal(0.01, 0.02, len(syms)), index=syms)
        rb = pd.Series(rng.normal(0.008, 0.02, len(syms)), index=syms)
        result = BenchmarkAnalyzer().brinson_attribution(
            wp, wb, rp, rb, _sector_map(),
        )
        summary = result.summary()
        assert "Brinson" in summary
        assert "Allocation" in summary

    def test_te_budget_summary(self):
        wp, wb = _weights()
        cov = _covariance()
        result = BenchmarkAnalyzer().tracking_error_budget(cov, wp, wb)
        summary = result.summary()
        assert "Tracking Error" in summary
        assert "Ex-ante TE" in summary
