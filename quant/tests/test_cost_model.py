"""Tests for multi-component transaction cost model (QUA-51)."""
from __future__ import annotations

from quant.execution.cost_model import (
    CostEstimate,
    CostModelConfig,
    RebalanceCostEstimate,
    TransactionCostModel,
)

# ── Tests: Single order cost ──────────────────────────────────────────────


class TestSingleOrderCost:
    def test_returns_cost_estimate(self):
        model = TransactionCostModel()
        est = model.estimate_order_cost("AAPL", notional=100_000)
        assert isinstance(est, CostEstimate)

    def test_spread_component(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=10.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
        ))
        est = model.estimate_order_cost("AAPL", notional=100_000)
        assert abs(est.spread_bps - 10.0) < 1e-6
        assert abs(est.total_bps - 10.0) < 1e-6

    def test_spread_override(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=5.0,
            spread_overrides={"AAPL": 2.0},
            impact_coefficient=0.0,
            commission_per_share=0.0,
        ))
        aapl = model.estimate_order_cost("AAPL", notional=100_000)
        goog = model.estimate_order_cost("GOOG", notional=100_000)
        assert abs(aapl.spread_bps - 2.0) < 1e-6
        assert abs(goog.spread_bps - 5.0) < 1e-6

    def test_zero_notional(self):
        model = TransactionCostModel()
        est = model.estimate_order_cost("AAPL", notional=0)
        assert est.total_bps == 0.0
        assert est.total_dollars == 0.0

    def test_total_dollars(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=10.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
        ))
        est = model.estimate_order_cost("AAPL", notional=1_000_000)
        # 10 bps of $1M = $1,000
        assert abs(est.total_dollars - 1_000.0) < 1e-6

    def test_symbol_stored(self):
        model = TransactionCostModel()
        est = model.estimate_order_cost("MSFT", notional=50_000)
        assert est.symbol == "MSFT"


# ── Tests: Market impact ─────────────────────────────────────────────────


class TestMarketImpact:
    def test_no_impact_without_adv(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0, commission_per_share=0.0
        ))
        est = model.estimate_order_cost("AAPL", notional=100_000)
        assert est.impact_bps == 0.0

    def test_impact_increases_with_participation(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            commission_per_share=0.0,
            impact_coefficient=0.1,
        ))
        small = model.estimate_order_cost(
            "AAPL", notional=100_000, adv=10_000_000, volatility=0.25
        )
        large = model.estimate_order_cost(
            "AAPL", notional=1_000_000, adv=10_000_000, volatility=0.25
        )
        assert large.impact_bps > small.impact_bps

    def test_impact_increases_with_volatility(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            commission_per_share=0.0,
            impact_coefficient=0.1,
        ))
        low_vol = model.estimate_order_cost(
            "AAPL", notional=500_000, adv=10_000_000, volatility=0.10
        )
        high_vol = model.estimate_order_cost(
            "AAPL", notional=500_000, adv=10_000_000, volatility=0.40
        )
        assert high_vol.impact_bps > low_vol.impact_bps

    def test_square_root_law(self):
        """Impact should scale as (Q/ADV)^0.5 by default."""
        config = CostModelConfig(
            default_spread_bps=0.0,
            commission_per_share=0.0,
            impact_coefficient=0.1,
            impact_exponent=0.5,
        )
        model = TransactionCostModel(config)

        est1 = model.estimate_order_cost(
            "X", notional=100_000, adv=10_000_000, volatility=0.20
        )
        est4 = model.estimate_order_cost(
            "X", notional=400_000, adv=10_000_000, volatility=0.20
        )
        # 4x notional → 2x impact (sqrt)
        ratio = est4.impact_bps / est1.impact_bps
        assert abs(ratio - 2.0) < 1e-6

    def test_no_vol_scaling(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            commission_per_share=0.0,
            impact_coefficient=0.1,
            volatility_scaling=False,
        ))
        # Without vol scaling, volatility param is ignored
        est_a = model.estimate_order_cost(
            "X", notional=100_000, adv=10_000_000, volatility=0.10
        )
        est_b = model.estimate_order_cost(
            "X", notional=100_000, adv=10_000_000, volatility=0.50
        )
        assert abs(est_a.impact_bps - est_b.impact_bps) < 1e-6


# ── Tests: Commission ────────────────────────────────────────────────────


class TestCommission:
    def test_per_share_commission(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            impact_coefficient=0.0,
            commission_per_share=0.01,
            commission_pct=0.0,
        ))
        # 1000 shares at $100 = $100K notional, commission = $10
        est = model.estimate_order_cost(
            "AAPL", notional=100_000, price=100.0, quantity=1000
        )
        # $10 / $100K = 1.0 bps
        assert abs(est.commission_bps - 1.0) < 1e-6

    def test_percentage_commission(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
            commission_pct=0.0001,  # 1 bps
        ))
        est = model.estimate_order_cost("AAPL", notional=100_000)
        assert abs(est.commission_bps - 1.0) < 1e-6

    def test_min_commission(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
            commission_pct=0.0,
            min_commission=1.0,  # $1 minimum
        ))
        est = model.estimate_order_cost("AAPL", notional=10_000)
        # $1 / $10K = 1.0 bps
        assert abs(est.commission_bps - 1.0) < 1e-6

    def test_combined_commissions(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            impact_coefficient=0.0,
            commission_per_share=0.005,
            commission_pct=0.0001,  # 1 bps
        ))
        est = model.estimate_order_cost(
            "AAPL", notional=100_000, price=100.0, quantity=1000
        )
        # Per-share: 1000 * $0.005 = $5 → 0.5 bps
        # Pct: 0.0001 * $100K = $10 → 1.0 bps
        # Total: 1.5 bps
        assert abs(est.commission_bps - 1.5) < 1e-6


# ── Tests: Cost additivity ───────────────────────────────────────────────


class TestCostAdditivity:
    def test_total_is_sum_of_components(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=5.0,
            impact_coefficient=0.1,
            commission_per_share=0.005,
        ))
        est = model.estimate_order_cost(
            "AAPL",
            notional=500_000,
            adv=20_000_000,
            volatility=0.25,
            price=150.0,
            quantity=3333,
        )
        expected_total = est.spread_bps + est.impact_bps + est.commission_bps
        assert abs(est.total_bps - expected_total) < 1e-6


# ── Tests: Rebalance cost ────────────────────────────────────────────────


class TestRebalanceCost:
    def test_returns_rebalance_estimate(self):
        model = TransactionCostModel()
        est = model.estimate_rebalance_cost(
            weight_changes={"AAPL": 0.05, "GOOG": 0.03},
            portfolio_value=1_000_000,
        )
        assert isinstance(est, RebalanceCostEstimate)

    def test_per_asset_populated(self):
        model = TransactionCostModel()
        est = model.estimate_rebalance_cost(
            weight_changes={"AAPL": 0.05, "GOOG": 0.03},
            portfolio_value=1_000_000,
        )
        assert len(est.per_asset) == 2
        symbols = {e.symbol for e in est.per_asset}
        assert symbols == {"AAPL", "GOOG"}

    def test_turnover_computed(self):
        model = TransactionCostModel()
        est = model.estimate_rebalance_cost(
            weight_changes={"AAPL": 0.05, "GOOG": 0.03},
            portfolio_value=1_000_000,
        )
        assert abs(est.turnover - 0.08) < 1e-6

    def test_total_dollars_positive(self):
        model = TransactionCostModel()
        est = model.estimate_rebalance_cost(
            weight_changes={"AAPL": 0.10},
            portfolio_value=1_000_000,
        )
        assert est.total_cost_dollars > 0

    def test_empty_rebalance(self):
        model = TransactionCostModel()
        est = model.estimate_rebalance_cost(
            weight_changes={}, portfolio_value=1_000_000
        )
        assert est.total_cost_dollars == 0.0
        assert est.turnover == 0.0

    def test_component_breakdown(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=5.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
        ))
        est = model.estimate_rebalance_cost(
            weight_changes={"AAPL": 0.10},
            portfolio_value=1_000_000,
        )
        # Only spread cost, 5 bps on 10% turnover = 0.5 bps portfolio-level
        assert abs(est.total_spread_bps - 0.5) < 1e-6
        assert est.total_impact_bps == 0.0

    def test_with_adv_and_volatility(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            impact_coefficient=0.1,
            commission_per_share=0.0,
        ))
        est = model.estimate_rebalance_cost(
            weight_changes={"AAPL": 0.10},
            portfolio_value=1_000_000,
            adv={"AAPL": 50_000_000},
            volatility={"AAPL": 0.25},
        )
        assert est.total_impact_bps > 0


# ── Tests: Break-even analysis ────────────────────────────────────────────


class TestBreakEven:
    def test_break_even_positive(self):
        model = TransactionCostModel()
        be = model.break_even_alpha("AAPL", notional=100_000)
        assert be > 0

    def test_break_even_scales_with_holding_period(self):
        model = TransactionCostModel()
        be_1d = model.break_even_alpha("AAPL", notional=100_000, holding_period_days=1)
        be_5d = model.break_even_alpha("AAPL", notional=100_000, holding_period_days=5)
        assert abs(be_1d - 5 * be_5d) < 1e-6

    def test_break_even_is_round_trip(self):
        """Break-even should be 2x one-way cost for 1-day holding."""
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=10.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
        ))
        be = model.break_even_alpha("AAPL", notional=100_000, holding_period_days=1)
        # Round-trip spread = 2 * 10 bps = 20 bps / 1 day = 20 bps
        assert abs(be - 20.0) < 1e-6


# ── Tests: Flat bps equivalent ────────────────────────────────────────────


class TestFlatBps:
    def test_flat_bps_equals_total(self):
        model = TransactionCostModel()
        flat = model.as_flat_bps("AAPL", notional=100_000)
        est = model.estimate_order_cost("AAPL", notional=100_000)
        assert abs(flat - est.total_bps) < 1e-6

    def test_flat_bps_spread_only(self):
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=8.0,
            impact_coefficient=0.0,
            commission_per_share=0.0,
        ))
        assert abs(model.as_flat_bps("X") - 8.0) < 1e-6


# ── Tests: Config ─────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = CostModelConfig()
        assert config.default_spread_bps == 5.0
        assert config.impact_coefficient == 0.10
        assert config.impact_exponent == 0.50
        assert config.commission_per_share == 0.005

    def test_config_exposed(self):
        config = CostModelConfig(default_spread_bps=3.0)
        model = TransactionCostModel(config)
        assert model.config.default_spread_bps == 3.0

    def test_custom_exponent(self):
        """Linear impact with exponent=1.0."""
        model = TransactionCostModel(CostModelConfig(
            default_spread_bps=0.0,
            commission_per_share=0.0,
            impact_coefficient=0.1,
            impact_exponent=1.0,
        ))
        est1 = model.estimate_order_cost(
            "X", notional=100_000, adv=10_000_000, volatility=0.20
        )
        est2 = model.estimate_order_cost(
            "X", notional=200_000, adv=10_000_000, volatility=0.20
        )
        # Linear: 2x notional → 2x impact
        ratio = est2.impact_bps / est1.impact_bps
        assert abs(ratio - 2.0) < 1e-6
