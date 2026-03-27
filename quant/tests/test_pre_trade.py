"""Tests for pre-trade validation and adjustment pipeline (QUA-54)."""
from __future__ import annotations

from quant.execution.cost_model import CostModelConfig, TransactionCostModel
from quant.portfolio.pre_trade import (
    PreTradeConfig,
    PreTradePipeline,
    PreTradeResult,
)
from quant.risk.limit_checker import LimitConfig, RiskLimitChecker

# ── Helpers ───────────────────────────────────────────────────────────────────


def _simple_weights() -> dict[str, float]:
    return {"AAPL": 0.15, "GOOG": -0.10, "MSFT": 0.05}


def _checker(**kwargs) -> RiskLimitChecker:
    return RiskLimitChecker(LimitConfig(**kwargs))


def _cost_model(**kwargs) -> TransactionCostModel:
    return TransactionCostModel(CostModelConfig(**kwargs))


# ── Tests: Basic pipeline ────────────────────────────────────────────────


class TestBasicPipeline:
    def test_returns_result(self):
        pipeline = PreTradePipeline()
        result = pipeline.process(target_weights=_simple_weights())
        assert isinstance(result, PreTradeResult)

    def test_passthrough_when_no_checks(self):
        """With no checker or cost model, weights pass through."""
        pipeline = PreTradePipeline(PreTradeConfig(
            min_trade_weight=0.0, min_trade_dollars=0.0
        ))
        target = _simple_weights()
        result = pipeline.process(target_weights=target)
        for sym, w in target.items():
            assert abs(result.adjusted_weights[sym] - w) < 1e-10

    def test_timestamp_set(self):
        pipeline = PreTradePipeline()
        result = pipeline.process(target_weights={"A": 0.1})
        assert result.timestamp is not None

    def test_original_preserved(self):
        pipeline = PreTradePipeline()
        target = {"A": 0.1, "B": 0.05}
        result = pipeline.process(target_weights=target)
        assert result.original_weights == target

    def test_empty_weights(self):
        pipeline = PreTradePipeline()
        result = pipeline.process(target_weights={})
        assert result.trades_remaining == 0
        assert result.trades_filtered == 0


# ── Tests: Limit enforcement ─────────────────────────────────────────────


class TestLimitEnforcement:
    def test_enforces_position_limit(self):
        checker = _checker(max_position_weight=0.10, max_concentration_hhi=None)
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            enforce_limits=True,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.20, "B": 0.05}
        result = pipeline.process(target_weights=target)
        assert abs(result.adjusted_weights["A"]) <= 0.10 + 1e-10

    def test_records_adjustment(self):
        checker = _checker(max_position_weight=0.10, max_concentration_hhi=None)
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            enforce_limits=True,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.20}
        result = pipeline.process(target_weights=target)
        limit_adjs = [a for a in result.adjustments if a.stage == "limit_enforce"]
        assert len(limit_adjs) >= 1
        assert limit_adjs[0].symbol == "A"

    def test_no_enforce_when_disabled(self):
        checker = _checker(max_position_weight=0.05, max_concentration_hhi=None)
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            enforce_limits=False,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.20}
        result = pipeline.process(target_weights=target)
        # Weights unchanged (only report, no enforcement)
        assert abs(result.adjusted_weights["A"] - 0.20) < 1e-10
        # But breach should be reported
        assert result.limit_report is not None
        assert result.limit_report.has_any_breach

    def test_compliant_unchanged(self):
        checker = _checker(max_position_weight=0.20, max_concentration_hhi=None)
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.10, "B": -0.05}
        result = pipeline.process(target_weights=target)
        for sym in target:
            assert abs(result.adjusted_weights[sym] - target[sym]) < 1e-10

    def test_sector_enforcement(self):
        checker = _checker(
            max_position_weight=1.0,
            max_sector_weight=0.15,
            max_gross_leverage=10.0,
            max_concentration_hhi=None,
        )
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.10, "B": 0.10, "C": 0.05}
        sectors = {"A": "Tech", "B": "Tech", "C": "Health"}
        result = pipeline.process(target_weights=target, sector_map=sectors)
        tech_gross = abs(result.adjusted_weights["A"]) + abs(result.adjusted_weights["B"])
        assert tech_gross <= 0.15 + 1e-10


# ── Tests: Minimum trade filter ──────────────────────────────────────────


class TestMinTradeFilter:
    def test_filters_small_weight_change(self):
        pipeline = PreTradePipeline(PreTradeConfig(
            min_trade_weight=0.01, min_trade_dollars=0.0
        ))
        target = {"A": 0.005, "B": 0.05}
        result = pipeline.process(target_weights=target)
        # A's weight change (0.005) < min (0.01) → filtered
        assert result.trades_filtered >= 1
        assert abs(result.adjusted_weights["A"]) < 1e-10

    def test_filters_small_dollar_value(self):
        pipeline = PreTradePipeline(PreTradeConfig(
            min_trade_weight=0.0, min_trade_dollars=1000.0
        ))
        target = {"A": 0.001}  # $1000 * 0.001 = $1 (default PV=$1M → $1000)
        # At $1M portfolio, 0.001 = $1000 which equals min → not filtered
        # But at $100K, 0.001 = $0.1 → filtered
        result = pipeline.process(
            target_weights=target, portfolio_value=100_000
        )
        assert result.trades_filtered >= 1

    def test_keeps_large_trades(self):
        pipeline = PreTradePipeline(PreTradeConfig(
            min_trade_weight=0.01, min_trade_dollars=100.0
        ))
        target = {"A": 0.10}
        result = pipeline.process(target_weights=target)
        assert result.trades_remaining >= 1
        assert abs(result.adjusted_weights["A"] - 0.10) < 1e-10

    def test_current_weights_considered(self):
        """Trade filter compares target vs current, not target vs zero."""
        pipeline = PreTradePipeline(PreTradeConfig(
            min_trade_weight=0.01, min_trade_dollars=0.0
        ))
        target = {"A": 0.105}
        current = {"A": 0.10}
        result = pipeline.process(target_weights=target, current_weights=current)
        # dw = 0.005 < 0.01 → filtered, reverts to current
        assert abs(result.adjusted_weights["A"] - 0.10) < 1e-10


# ── Tests: Cost-aware filtering ──────────────────────────────────────────


class TestCostFilter:
    def test_filters_expensive_trade(self):
        cost_model = _cost_model(
            default_spread_bps=50.0,  # very expensive spread
            impact_coefficient=0.0,
            commission_per_share=0.0,
        )
        pipeline = PreTradePipeline(PreTradeConfig(
            cost_model=cost_model,
            cost_alpha_ratio=0.5,  # cost can't exceed 50% of alpha
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.10}
        # Round-trip cost = 100 bps, alpha = 10 bps → ratio = 10 > 0.5
        result = pipeline.process(
            target_weights=target,
            expected_alpha_bps={"A": 10.0},
        )
        assert result.trades_filtered >= 1
        cost_adjs = [a for a in result.adjustments if a.stage == "cost_filter"]
        assert len(cost_adjs) >= 1

    def test_keeps_cheap_trade(self):
        cost_model = _cost_model(
            default_spread_bps=1.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
        )
        pipeline = PreTradePipeline(PreTradeConfig(
            cost_model=cost_model,
            cost_alpha_ratio=0.5,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.10}
        # Round-trip cost = 2 bps, alpha = 100 bps → ratio = 0.02 < 0.5
        result = pipeline.process(
            target_weights=target,
            expected_alpha_bps={"A": 100.0},
        )
        assert abs(result.adjusted_weights["A"] - 0.10) < 1e-10

    def test_no_filter_without_alpha(self):
        """Without expected alpha, cost filter is skipped."""
        cost_model = _cost_model(default_spread_bps=100.0)
        pipeline = PreTradePipeline(PreTradeConfig(
            cost_model=cost_model,
            cost_alpha_ratio=0.5,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.10}
        result = pipeline.process(target_weights=target)
        # No alpha provided → cost filter skipped
        assert abs(result.adjusted_weights["A"] - 0.10) < 1e-10

    def test_no_filter_when_ratio_none(self):
        cost_model = _cost_model(default_spread_bps=100.0)
        pipeline = PreTradePipeline(PreTradeConfig(
            cost_model=cost_model,
            cost_alpha_ratio=None,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.10}
        result = pipeline.process(
            target_weights=target,
            expected_alpha_bps={"A": 1.0},
        )
        assert abs(result.adjusted_weights["A"] - 0.10) < 1e-10


# ── Tests: Combined pipeline ─────────────────────────────────────────────


class TestCombinedPipeline:
    def test_limit_then_cost_filter(self):
        """Limit enforcement runs first, then cost filter."""
        checker = _checker(
            max_position_weight=0.10,
            max_concentration_hhi=None,
        )
        cost_model = _cost_model(
            default_spread_bps=50.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
        )
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            cost_model=cost_model,
            cost_alpha_ratio=0.5,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.20, "B": 0.05}
        result = pipeline.process(
            target_weights=target,
            expected_alpha_bps={"A": 5.0, "B": 100.0},
        )
        # A: clamped from 0.20 to 0.10, then cost-filtered (expensive)
        # B: passes limits and cost filter
        assert result.n_adjustments > 0

    def test_all_stages_produce_adjustments(self):
        checker = _checker(
            max_position_weight=0.10,
            max_concentration_hhi=None,
        )
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            min_trade_weight=0.01,
            min_trade_dollars=0.0,
        ))
        target = {"A": 0.20, "B": 0.005}
        result = pipeline.process(target_weights=target)
        stages = {a.stage for a in result.adjustments}
        assert "limit_enforce" in stages
        assert "min_weight" in stages


# ── Tests: Result properties ──────────────────────────────────────────────


class TestResultProperties:
    def test_was_modified(self):
        checker = _checker(max_position_weight=0.05, max_concentration_hhi=None)
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        result = pipeline.process(target_weights={"A": 0.20})
        assert result.was_modified

    def test_not_modified(self):
        pipeline = PreTradePipeline(PreTradeConfig(
            min_trade_weight=0.0, min_trade_dollars=0.0
        ))
        result = pipeline.process(target_weights={"A": 0.05})
        assert not result.was_modified


# ── Tests: Summary ────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_adjustments(self):
        checker = _checker(max_position_weight=0.05, max_concentration_hhi=None)
        pipeline = PreTradePipeline(PreTradeConfig(
            limit_checker=checker,
            min_trade_weight=0.0,
            min_trade_dollars=0.0,
        ))
        result = pipeline.process(target_weights={"A": 0.20})
        summary = result.summary()
        assert "Pre-Trade Pipeline" in summary
        assert "limit_enforce" in summary

    def test_summary_clean(self):
        pipeline = PreTradePipeline(PreTradeConfig(
            min_trade_weight=0.0, min_trade_dollars=0.0
        ))
        result = pipeline.process(target_weights={"A": 0.05})
        summary = result.summary()
        assert "Pre-Trade Pipeline" in summary


# ── Tests: Config ─────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = PreTradeConfig()
        assert config.limit_checker is None
        assert config.cost_model is None
        assert config.min_trade_weight == 0.005
        assert config.enforce_limits is True

    def test_config_exposed(self):
        pipeline = PreTradePipeline(PreTradeConfig(min_trade_weight=0.02))
        assert pipeline.config.min_trade_weight == 0.02
