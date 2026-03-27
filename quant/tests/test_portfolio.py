"""Unit tests for the portfolio construction module (QUA-22)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.alpha import AlphaCombiner, AlphaScore, CombinationMethod
from quant.portfolio.attribution import (
    AttributionReport,
    PerformanceAttributor,
    SectorAttribution,
    SignalAttribution,
)
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import ConstructionResult, PortfolioConfig, PortfolioEngine
from quant.portfolio.optimizers import (
    MaxDiversificationOptimizer,
    MeanVarianceOptimizer,
    MinimumVarianceOptimizer,
    OptimizationMethod,
    OptimizationResult,
    RiskParityOptimizer,
    get_optimizer,
)
from quant.portfolio.rebalancer import RebalanceEngine, RebalanceResult, Trade
from quant.signals.base import SignalOutput


# ── Fixtures ──────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 15, tzinfo=timezone.utc)

SYMBOLS = ["AAPL", "GOOG", "MSFT", "JPM", "XOM"]


def _make_signal(
    symbol: str, score: float, confidence: float, name: str = ""
) -> SignalOutput:
    return SignalOutput(
        symbol=symbol,
        timestamp=_NOW,
        score=score,
        confidence=confidence,
        target_position=score * confidence,
        metadata={"signal_name": name},
    )


def _make_returns(symbols: list[str], n_days: int = 500) -> pd.DataFrame:
    """Generate synthetic daily returns for testing."""
    rng = np.random.default_rng(42)
    # Correlated returns via a factor model
    n = len(symbols)
    factor = rng.normal(0.0005, 0.01, size=n_days)
    idio = rng.normal(0, 0.015, size=(n_days, n))
    betas = rng.uniform(0.5, 1.5, size=n)
    returns = factor[:, None] * betas[None, :] + idio

    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=symbols)


def _make_cov(symbols: list[str]) -> pd.DataFrame:
    returns = _make_returns(symbols)
    return returns.cov() * 252


# ── AlphaScore ────────────────────────────────────────────────────────────────

class TestAlphaScore:
    def test_valid_construction(self):
        a = AlphaScore(symbol="AAPL", timestamp=_NOW, score=0.5, confidence=0.8)
        assert a.score == 0.5
        assert a.confidence == 0.8

    def test_score_out_of_range(self):
        with pytest.raises(ValueError, match="score"):
            AlphaScore(symbol="AAPL", timestamp=_NOW, score=1.5, confidence=0.5)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            AlphaScore(symbol="AAPL", timestamp=_NOW, score=0.5, confidence=1.5)


# ── AlphaCombiner ─────────────────────────────────────────────────────────────

class TestAlphaCombinerEqualWeight:
    def test_single_signal(self):
        combiner = AlphaCombiner(method=CombinationMethod.EQUAL_WEIGHT)
        sig = _make_signal("AAPL", 0.8, 0.9, "momentum")
        alpha = combiner.combine("AAPL", _NOW, [sig])
        assert abs(alpha.score - 0.8) < 1e-9

    def test_two_signals_average(self):
        combiner = AlphaCombiner(method=CombinationMethod.EQUAL_WEIGHT)
        sigs = [
            _make_signal("AAPL", 0.6, 0.8, "momentum"),
            _make_signal("AAPL", -0.4, 0.7, "mean_reversion"),
        ]
        alpha = combiner.combine("AAPL", _NOW, sigs)
        assert abs(alpha.score - 0.1) < 1e-9

    def test_empty_signals(self):
        combiner = AlphaCombiner(method=CombinationMethod.EQUAL_WEIGHT)
        alpha = combiner.combine("AAPL", _NOW, [])
        assert alpha.score == 0.0
        assert alpha.confidence == 0.0


class TestAlphaCombinerStaticWeight:
    def test_requires_weights(self):
        with pytest.raises(ValueError):
            AlphaCombiner(method=CombinationMethod.STATIC_WEIGHT)

    def test_weighted_combination(self):
        combiner = AlphaCombiner(
            method=CombinationMethod.STATIC_WEIGHT,
            weights={"momentum": 0.7, "mean_reversion": 0.3},
        )
        sigs = [
            _make_signal("AAPL", 1.0, 1.0, "momentum"),
            _make_signal("AAPL", -1.0, 1.0, "mean_reversion"),
        ]
        alpha = combiner.combine("AAPL", _NOW, sigs)
        # 1.0*0.7 + (-1.0)*0.3 = 0.4
        assert abs(alpha.score - 0.4) < 1e-9

    def test_contributions_are_tracked(self):
        combiner = AlphaCombiner(
            method=CombinationMethod.STATIC_WEIGHT,
            weights={"momentum": 0.5, "trend": 0.5},
        )
        sigs = [
            _make_signal("AAPL", 0.6, 0.8, "momentum"),
            _make_signal("AAPL", 0.4, 0.9, "trend"),
        ]
        alpha = combiner.combine("AAPL", _NOW, sigs)
        assert "momentum" in alpha.signal_contributions
        assert "trend" in alpha.signal_contributions


class TestAlphaCombinerInverseVolatility:
    def test_higher_confidence_gets_more_weight(self):
        combiner = AlphaCombiner(method=CombinationMethod.INVERSE_VOLATILITY)
        sigs = [
            _make_signal("AAPL", 1.0, 0.9, "high_conf"),
            _make_signal("AAPL", -1.0, 0.1, "low_conf"),
        ]
        alpha = combiner.combine("AAPL", _NOW, sigs)
        # High-confidence bullish signal should dominate
        assert alpha.score > 0.0


class TestAlphaCombinerRankWeighted:
    def test_higher_magnitude_gets_more_weight(self):
        combiner = AlphaCombiner(method=CombinationMethod.RANK_WEIGHTED)
        sigs = [
            _make_signal("AAPL", 0.9, 0.5, "strong"),
            _make_signal("AAPL", 0.1, 0.5, "weak"),
        ]
        alpha = combiner.combine("AAPL", _NOW, sigs)
        assert alpha.score > 0.0
        # Strong signal should dominate due to rank
        assert alpha.signal_contributions["strong"] > alpha.signal_contributions["weak"]


class TestAlphaCombinerUniverse:
    def test_combine_universe(self):
        combiner = AlphaCombiner(method=CombinationMethod.EQUAL_WEIGHT)
        universe = {
            "AAPL": [_make_signal("AAPL", 0.5, 0.8, "mom")],
            "GOOG": [_make_signal("GOOG", -0.3, 0.6, "mom")],
        }
        result = combiner.combine_universe(_NOW, universe)
        assert set(result.keys()) == {"AAPL", "GOOG"}
        assert result["AAPL"].score > 0
        assert result["GOOG"].score < 0


# ── PortfolioConstraints ──────────────────────────────────────────────────────

class TestPortfolioConstraints:
    def test_long_only_effective_min(self):
        c = PortfolioConstraints(long_only=True)
        assert c.effective_min() == 0.0

    def test_long_short_effective_min(self):
        c = PortfolioConstraints(long_only=False, min_weight=-0.5)
        assert c.effective_min() == -0.5

    def test_valid_weights(self):
        c = PortfolioConstraints(long_only=True, max_weight=0.3, max_gross_exposure=1.0)
        weights = {"AAPL": 0.25, "GOOG": 0.25, "MSFT": 0.25}
        ok, violations = c.validate_weights(weights)
        assert ok
        assert violations == []

    def test_weight_above_max(self):
        c = PortfolioConstraints(max_weight=0.3)
        weights = {"AAPL": 0.5}
        ok, violations = c.validate_weights(weights)
        assert not ok
        assert any("above maximum" in v for v in violations)

    def test_long_only_violation(self):
        c = PortfolioConstraints(long_only=True)
        weights = {"AAPL": -0.1}
        ok, violations = c.validate_weights(weights)
        assert not ok
        assert any("below minimum" in v for v in violations)

    def test_gross_exposure_violation(self):
        c = PortfolioConstraints(max_gross_exposure=1.0)
        weights = {"AAPL": 0.6, "GOOG": 0.6}
        ok, violations = c.validate_weights(weights)
        assert not ok
        assert any("gross exposure" in v for v in violations)

    def test_sector_exposure_violation(self):
        c = PortfolioConstraints(
            max_sector_weight=0.4,
            sector_map={"AAPL": "Tech", "GOOG": "Tech"},
        )
        weights = {"AAPL": 0.3, "GOOG": 0.3}
        ok, violations = c.validate_weights(weights)
        assert not ok
        assert any("sector Tech" in v for v in violations)

    def test_turnover_violation(self):
        c = PortfolioConstraints(max_turnover=0.1)
        current = {"AAPL": 0.5}
        target = {"AAPL": 0.1, "GOOG": 0.3}
        ok, violations = c.validate_weights(target, current)
        assert not ok
        assert any("turnover" in v for v in violations)

    def test_clip_weights(self):
        c = PortfolioConstraints(long_only=True, max_weight=0.3)
        clipped = c.clip_weights({"AAPL": 0.5, "GOOG": -0.1})
        assert clipped["AAPL"] == 0.3
        assert clipped["GOOG"] == 0.0

    def test_scale_to_gross_exposure(self):
        c = PortfolioConstraints(max_gross_exposure=1.0)
        scaled = c.scale_to_gross_exposure({"AAPL": 0.8, "GOOG": 0.8})
        gross = abs(scaled["AAPL"]) + abs(scaled["GOOG"])
        assert abs(gross - 1.0) < 1e-9


# ── Optimizers ────────────────────────────────────────────────────────────────

class TestMeanVarianceOptimizer:
    def test_produces_weights(self):
        opt = MeanVarianceOptimizer(risk_aversion=1.0)
        cov = _make_cov(SYMBOLS)
        mu = pd.Series([0.10, 0.08, 0.12, 0.06, 0.05], index=SYMBOLS)
        constraints = PortfolioConstraints(max_gross_exposure=1.0)
        result = opt.optimize(SYMBOLS, cov, mu, constraints)
        assert set(result.weights.keys()) == set(SYMBOLS)
        assert result.method == OptimizationMethod.MEAN_VARIANCE
        assert result.risk > 0

    def test_long_only(self):
        opt = MeanVarianceOptimizer()
        cov = _make_cov(SYMBOLS)
        mu = pd.Series([0.10, 0.08, 0.12, 0.06, 0.05], index=SYMBOLS)
        constraints = PortfolioConstraints(long_only=True, max_gross_exposure=1.0)
        result = opt.optimize(SYMBOLS, cov, mu, constraints)
        for w in result.weights.values():
            assert w >= -1e-9

    def test_invalid_risk_aversion(self):
        with pytest.raises(ValueError):
            MeanVarianceOptimizer(risk_aversion=-1.0)


class TestMinimumVarianceOptimizer:
    def test_produces_weights(self):
        opt = MinimumVarianceOptimizer()
        cov = _make_cov(SYMBOLS)
        constraints = PortfolioConstraints(max_gross_exposure=1.0)
        result = opt.optimize(SYMBOLS, cov, None, constraints)
        assert set(result.weights.keys()) == set(SYMBOLS)
        assert result.method == OptimizationMethod.MINIMUM_VARIANCE

    def test_lower_risk_than_equal_weight(self):
        opt = MinimumVarianceOptimizer()
        cov = _make_cov(SYMBOLS)
        constraints = PortfolioConstraints(max_gross_exposure=1.0)
        result = opt.optimize(SYMBOLS, cov, None, constraints)
        # Min-variance should have lower risk than equal weight
        ew = np.ones(len(SYMBOLS)) / len(SYMBOLS)
        cov_arr = cov.loc[SYMBOLS, SYMBOLS].values
        ew_vol = np.sqrt(ew @ cov_arr @ ew)
        assert result.risk <= ew_vol + 1e-6


class TestRiskParityOptimizer:
    def test_produces_weights(self):
        opt = RiskParityOptimizer()
        cov = _make_cov(SYMBOLS)
        constraints = PortfolioConstraints(max_gross_exposure=1.0)
        result = opt.optimize(SYMBOLS, cov, None, constraints)
        assert set(result.weights.keys()) == set(SYMBOLS)
        assert result.method == OptimizationMethod.RISK_PARITY

    def test_equal_risk_contribution(self):
        opt = RiskParityOptimizer(tolerance=1e-10, max_iterations=1000)
        cov = _make_cov(SYMBOLS)
        constraints = PortfolioConstraints(max_gross_exposure=1.0)
        result = opt.optimize(SYMBOLS, cov, None, constraints)

        # Verify risk contributions are approximately equal
        w = np.array([result.weights[s] for s in SYMBOLS])
        cov_arr = cov.loc[SYMBOLS, SYMBOLS].values
        sigma_w = cov_arr @ w
        port_vol = np.sqrt(w @ sigma_w)
        rc = w * sigma_w / port_vol
        rc_pct = rc / rc.sum()

        # Each asset should contribute ~20% (1/5) to total risk
        assert np.max(np.abs(rc_pct - 0.2)) < 0.05


class TestMaxDiversificationOptimizer:
    def test_produces_weights(self):
        opt = MaxDiversificationOptimizer()
        cov = _make_cov(SYMBOLS)
        constraints = PortfolioConstraints(max_gross_exposure=1.0)
        result = opt.optimize(SYMBOLS, cov, None, constraints)
        assert set(result.weights.keys()) == set(SYMBOLS)
        assert result.method == OptimizationMethod.MAX_DIVERSIFICATION
        assert result.diversification_ratio >= 1.0 - 1e-6


class TestGetOptimizer:
    def test_factory(self):
        for method in OptimizationMethod:
            opt = get_optimizer(method)
            assert isinstance(opt, object)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            get_optimizer("nonexistent")


# ── RebalanceEngine ───────────────────────────────────────────────────────────

class TestRebalanceEngine:
    def test_basic_rebalance(self):
        engine = RebalanceEngine(min_trade_weight=0.001)
        result = engine.rebalance(
            current_weights={"AAPL": 0.5, "GOOG": 0.5},
            target_weights={"AAPL": 0.3, "GOOG": 0.3, "MSFT": 0.4},
            portfolio_value=1_000_000,
        )
        assert result.n_buys == 1  # MSFT
        assert result.n_sells == 2  # AAPL, GOOG
        assert len(result.trades) == 3

    def test_no_trades_within_dead_band(self):
        engine = RebalanceEngine(min_trade_weight=0.1)
        result = engine.rebalance(
            current_weights={"AAPL": 0.5},
            target_weights={"AAPL": 0.55},
            portfolio_value=1_000_000,
        )
        assert len(result.trades) == 0

    def test_turnover_capping(self):
        constraints = PortfolioConstraints(max_turnover=0.1)
        engine = RebalanceEngine()
        result = engine.rebalance(
            current_weights={"AAPL": 1.0},
            target_weights={"AAPL": 0.0, "GOOG": 1.0},
            portfolio_value=1_000_000,
            constraints=constraints,
        )
        assert result.turnover_capped
        assert result.turnover <= 0.1 + 1e-6

    def test_dollar_amounts(self):
        engine = RebalanceEngine()
        result = engine.rebalance(
            current_weights={"AAPL": 0.0},
            target_weights={"AAPL": 0.5},
            portfolio_value=1_000_000,
        )
        assert len(result.trades) == 1
        assert abs(result.trades[0].dollar_amount - 500_000) < 1

    def test_trade_sides(self):
        engine = RebalanceEngine()
        result = engine.rebalance(
            current_weights={"AAPL": 0.6},
            target_weights={"AAPL": 0.3},
            portfolio_value=1_000_000,
        )
        assert result.trades[0].side == "SELL"

    def test_exit_position(self):
        engine = RebalanceEngine()
        result = engine.rebalance(
            current_weights={"AAPL": 0.5, "GOOG": 0.5},
            target_weights={"AAPL": 1.0},
            portfolio_value=1_000_000,
        )
        # GOOG should be sold, AAPL bought
        sides = {t.symbol: t.side for t in result.trades}
        assert sides["GOOG"] == "SELL"
        assert sides["AAPL"] == "BUY"


# ── PortfolioEngine (integration) ────────────────────────────────────────────

class TestPortfolioEngine:
    def test_initial_construction(self):
        returns = _make_returns(SYMBOLS)
        alphas = {
            sym: AlphaScore(symbol=sym, timestamp=_NOW, score=0.3, confidence=0.7)
            for sym in SYMBOLS
        }
        config = PortfolioConfig(
            optimization_method=OptimizationMethod.RISK_PARITY,
            constraints=PortfolioConstraints(
                long_only=True, max_weight=0.4, max_gross_exposure=1.0
            ),
        )
        engine = PortfolioEngine(config)
        result = engine.construct(
            alpha_scores=alphas,
            returns_history=returns,
            portfolio_value=1_000_000,
        )
        assert result.rebalance_triggered
        assert len(result.rebalance.trades) > 0
        assert isinstance(result.optimization, OptimizationResult)

    def test_no_rebalance_below_threshold(self):
        returns = _make_returns(SYMBOLS)
        alphas = {
            sym: AlphaScore(symbol=sym, timestamp=_NOW, score=0.3, confidence=0.7)
            for sym in SYMBOLS
        }
        config = PortfolioConfig(
            optimization_method=OptimizationMethod.RISK_PARITY,
            constraints=PortfolioConstraints(long_only=True, max_gross_exposure=1.0),
            rebalance_threshold=100.0,  # impossibly high
        )
        engine = PortfolioEngine(config)
        result = engine.construct(
            alpha_scores=alphas,
            returns_history=returns,
            current_weights={sym: 0.2 for sym in SYMBOLS},
            portfolio_value=1_000_000,
        )
        assert not result.rebalance_triggered

    def test_empty_universe(self):
        returns = _make_returns(SYMBOLS)
        engine = PortfolioEngine()
        result = engine.construct(
            alpha_scores={},
            returns_history=returns,
            portfolio_value=1_000_000,
        )
        assert not result.rebalance_triggered
        assert len(result.rebalance.trades) == 0

    def test_mean_variance_method(self):
        returns = _make_returns(SYMBOLS)
        alphas = {
            sym: AlphaScore(
                symbol=sym, timestamp=_NOW,
                score=float(i) / 10, confidence=0.8
            )
            for i, sym in enumerate(SYMBOLS)
        }
        config = PortfolioConfig(
            optimization_method=OptimizationMethod.MEAN_VARIANCE,
            constraints=PortfolioConstraints(long_only=True, max_gross_exposure=1.0),
        )
        engine = PortfolioEngine(config)
        result = engine.construct(
            alpha_scores=alphas,
            returns_history=returns,
            portfolio_value=1_000_000,
        )
        assert result.optimization.method == OptimizationMethod.MEAN_VARIANCE

    def test_covariance_shrinkage(self):
        returns = _make_returns(SYMBOLS)
        alphas = {
            sym: AlphaScore(symbol=sym, timestamp=_NOW, score=0.2, confidence=0.7)
            for sym in SYMBOLS
        }
        config = PortfolioConfig(
            optimization_method=OptimizationMethod.MINIMUM_VARIANCE,
            cov_shrinkage=0.5,
            constraints=PortfolioConstraints(long_only=True, max_gross_exposure=1.0),
        )
        engine = PortfolioEngine(config)
        result = engine.construct(
            alpha_scores=alphas,
            returns_history=returns,
            portfolio_value=1_000_000,
        )
        assert result.optimization.risk > 0


# ── PerformanceAttributor ────────────────────────────────────────────────────

class TestPerformanceAttributor:
    def test_basic_attribution(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        rng = np.random.default_rng(99)
        port_ret = pd.Series(rng.normal(0.001, 0.01, 100), index=dates)
        bench_ret = pd.Series(rng.normal(0.0005, 0.008, 100), index=dates)

        attributor = PerformanceAttributor()
        report = attributor.attribute(
            portfolio_returns=port_ret,
            benchmark_returns=bench_ret,
        )
        assert isinstance(report, AttributionReport)
        assert report.tracking_error > 0
        # Active return should be portfolio - benchmark
        assert abs(
            report.active_return - (report.total_return - report.benchmark_return)
        ) < 1e-9

    def test_no_benchmark(self):
        dates = pd.bdate_range("2024-01-01", periods=50)
        port_ret = pd.Series(0.001, index=dates)
        attributor = PerformanceAttributor()
        report = attributor.attribute(portfolio_returns=port_ret)
        assert report.benchmark_return == 0.0
        assert report.total_return > 0

    def test_signal_attribution(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        rng = np.random.default_rng(77)

        port_ret = pd.Series(rng.normal(0.001, 0.01, 100), index=dates)
        asset_ret = pd.DataFrame(
            rng.normal(0.001, 0.015, (100, 3)),
            index=dates,
            columns=["AAPL", "GOOG", "MSFT"],
        )
        signal_weights = pd.DataFrame(
            rng.uniform(0, 0.5, (100, 2)),
            index=dates,
            columns=["momentum", "mean_reversion"],
        )

        attributor = PerformanceAttributor()
        report = attributor.attribute(
            portfolio_returns=port_ret,
            asset_returns=asset_ret,
            signal_weights=signal_weights,
        )
        assert len(report.signal_attributions) == 2
        for sa in report.signal_attributions:
            assert isinstance(sa, SignalAttribution)
            assert 0 <= sa.hit_rate <= 1

    def test_sector_attribution(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        rng = np.random.default_rng(55)

        port_ret = pd.Series(rng.normal(0.001, 0.01, 100), index=dates)
        bench_ret = pd.Series(rng.normal(0.0005, 0.008, 100), index=dates)
        asset_ret = pd.DataFrame(
            rng.normal(0.001, 0.015, (100, 3)),
            index=dates,
            columns=["AAPL", "JPM", "XOM"],
        )
        weights_hist = pd.DataFrame(
            np.abs(rng.normal(0.33, 0.05, (100, 3))),
            index=dates,
            columns=["AAPL", "JPM", "XOM"],
        )
        sector_map = {"AAPL": "Tech", "JPM": "Financials", "XOM": "Energy"}

        attributor = PerformanceAttributor()
        report = attributor.attribute(
            portfolio_returns=port_ret,
            benchmark_returns=bench_ret,
            weights_history=weights_hist,
            asset_returns=asset_ret,
            sector_map=sector_map,
        )
        assert len(report.sector_attributions) == 3
        for sa in report.sector_attributions:
            assert isinstance(sa, SectorAttribution)
            assert abs(sa.total - (sa.allocation + sa.selection + sa.interaction)) < 1e-12
