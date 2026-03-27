"""Unit tests for the risk engine (QUA-7)."""
from __future__ import annotations

import pytest

from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.risk.engine import Order, PortfolioState, RiskCheckResult, RiskConfig, RiskEngine
from quant.risk.limits import ExposureLimits
from quant.risk.sizing import (
    FixedFractionParams,
    KellyParams,
    PositionSizer,
    SizingMethod,
    VolatilityTargetParams,
)


# ── Position Sizing ────────────────────────────────────────────────────────────

class TestKellySizing:
    def test_positive_edge(self):
        sizer = PositionSizer()
        # 60% win rate, 1:1 win/loss → Kelly = (1*0.6 - 0.4)/1 = 0.20
        params = KellyParams(win_probability=0.6, win_loss_ratio=1.0)
        assert abs(sizer.kelly(params) - 0.20) < 1e-9

    def test_negative_edge_returns_zero(self):
        sizer = PositionSizer()
        # 40% win rate, 1:1 odds → negative Kelly → 0
        params = KellyParams(win_probability=0.4, win_loss_ratio=1.0)
        assert sizer.kelly(params) == 0.0

    def test_fractional_kelly(self):
        sizer = PositionSizer()
        params = KellyParams(win_probability=0.6, win_loss_ratio=1.0, fraction=0.5)
        assert abs(sizer.kelly(params) - 0.10) < 1e-9

    def test_kelly_clamped_to_one(self):
        sizer = PositionSizer()
        # Very high edge
        params = KellyParams(win_probability=0.99, win_loss_ratio=10.0)
        assert sizer.kelly(params) <= 1.0

    def test_invalid_win_prob(self):
        with pytest.raises(ValueError):
            KellyParams(win_probability=1.1, win_loss_ratio=1.0)

    def test_invalid_win_loss_ratio(self):
        with pytest.raises(ValueError):
            KellyParams(win_probability=0.6, win_loss_ratio=0.0)


class TestFixedFractionSizing:
    def test_basic(self):
        sizer = PositionSizer()
        params = FixedFractionParams(fraction=0.02)
        assert sizer.fixed_fraction(params) == 0.02

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            FixedFractionParams(fraction=1.5)

    def test_zero_capital_returns_zero(self):
        sizer = PositionSizer()
        result = sizer.compute(
            SizingMethod.FIXED_FRACTION,
            capital=0.0,
            fixed_fraction_params=FixedFractionParams(fraction=0.05),
        )
        assert result == 0.0


class TestVolatilityTargetSizing:
    def test_basic(self):
        sizer = PositionSizer()
        # target 10% vol, asset has 20% vol → fraction = 0.5
        params = VolatilityTargetParams(
            target_annual_volatility=0.10,
            asset_annual_volatility=0.20,
            price=100.0,
        )
        assert abs(sizer.volatility_target(params, capital=10_000.0) - 0.50) < 1e-9

    def test_clamped_to_one(self):
        sizer = PositionSizer()
        # target > asset vol → fraction > 1 → clamped
        params = VolatilityTargetParams(
            target_annual_volatility=0.50,
            asset_annual_volatility=0.10,
            price=50.0,
        )
        assert sizer.volatility_target(params, capital=10_000.0) == 1.0

    def test_zero_capital_returns_zero(self):
        sizer = PositionSizer()
        params = VolatilityTargetParams(
            target_annual_volatility=0.10,
            asset_annual_volatility=0.20,
            price=100.0,
        )
        assert sizer.volatility_target(params, capital=0.0) == 0.0


# ── Exposure Limits ────────────────────────────────────────────────────────────

class TestExposureLimits:
    def setup_method(self):
        self.limits = ExposureLimits(
            max_position_fraction=0.20,
            max_sector_fraction=0.40,
            max_gross_exposure=1.50,
            max_net_exposure=1.00,
            max_order_fraction=0.10,
        )

    def test_position_within_limit(self):
        ok, _ = self.limits.check_position("AAPL", 1_900.0, capital=10_000.0)
        assert ok

    def test_position_exceeds_limit(self):
        ok, reason = self.limits.check_position("AAPL", 2_100.0, capital=10_000.0)
        assert not ok
        assert "AAPL" in reason

    def test_order_size_within_limit(self):
        ok, _ = self.limits.check_order_size("AAPL", 900.0, capital=10_000.0)
        assert ok

    def test_order_size_exceeds_limit(self):
        ok, reason = self.limits.check_order_size("AAPL", 1_100.0, capital=10_000.0)
        assert not ok

    def test_sector_within_limit(self):
        ok, _ = self.limits.check_sector("Tech", 3_900.0, capital=10_000.0)
        assert ok

    def test_sector_exceeds_limit(self):
        ok, reason = self.limits.check_sector("Tech", 4_100.0, capital=10_000.0)
        assert not ok
        assert "Tech" in reason

    def test_gross_exposure_within_limit(self):
        ok, _ = self.limits.check_gross_exposure(14_900.0, capital=10_000.0)
        assert ok

    def test_gross_exposure_exceeds_limit(self):
        ok, _ = self.limits.check_gross_exposure(15_100.0, capital=10_000.0)
        assert not ok

    def test_net_exposure_within_limit(self):
        ok, _ = self.limits.check_net_exposure(9_900.0, capital=10_000.0)
        assert ok

    def test_net_exposure_exceeds_limit(self):
        ok, _ = self.limits.check_net_exposure(10_100.0, capital=10_000.0)
        assert not ok

    def test_zero_capital_rejected(self):
        ok, reason = self.limits.check_position("AAPL", 100.0, capital=0.0)
        assert not ok
        assert "Capital" in reason

    def test_invalid_limit_value(self):
        with pytest.raises(ValueError):
            ExposureLimits(max_position_fraction=-0.1)


# ── Drawdown Circuit Breaker ───────────────────────────────────────────────────

class TestDrawdownCircuitBreaker:
    def test_not_tripped_below_threshold(self):
        cb = DrawdownCircuitBreaker(max_drawdown_threshold=0.10)
        cb.update(10_000.0)
        cb.update(9_500.0)  # 5% drawdown
        assert not cb.is_tripped()

    def test_tripped_at_threshold(self):
        cb = DrawdownCircuitBreaker(max_drawdown_threshold=0.10)
        cb.update(10_000.0)
        cb.update(9_000.0)  # exactly 10% drawdown
        assert cb.is_tripped()

    def test_tripped_above_threshold(self):
        cb = DrawdownCircuitBreaker(max_drawdown_threshold=0.10)
        cb.update(10_000.0)
        cb.update(8_500.0)  # 15% drawdown
        assert cb.is_tripped()

    def test_resets_on_new_peak(self):
        cb = DrawdownCircuitBreaker(max_drawdown_threshold=0.10, reset_on_new_peak=True)
        cb.update(10_000.0)
        cb.update(8_500.0)
        assert cb.is_tripped()
        cb.update(11_000.0)  # new peak
        assert not cb.is_tripped()

    def test_does_not_reset_when_disabled(self):
        cb = DrawdownCircuitBreaker(max_drawdown_threshold=0.10, reset_on_new_peak=False)
        cb.update(10_000.0)
        cb.update(8_500.0)
        assert cb.is_tripped()
        cb.update(11_000.0)  # new peak, but reset disabled
        assert cb.is_tripped()

    def test_manual_reset(self):
        cb = DrawdownCircuitBreaker(max_drawdown_threshold=0.10)
        cb.update(10_000.0)
        cb.update(8_500.0)
        assert cb.is_tripped()
        cb.reset()
        assert not cb.is_tripped()

    def test_check_returns_reason(self):
        cb = DrawdownCircuitBreaker(max_drawdown_threshold=0.10)
        cb.update(10_000.0)
        ok, reason = cb.check(8_500.0)
        assert not ok
        assert "circuit breaker" in reason.lower()

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            DrawdownCircuitBreaker(max_drawdown_threshold=1.5)

    def test_initial_state_not_tripped(self):
        cb = DrawdownCircuitBreaker(max_drawdown_threshold=0.10)
        assert not cb.is_tripped()


# ── RiskEngine Integration ─────────────────────────────────────────────────────

def _make_portfolio(capital: float = 10_000.0) -> PortfolioState:
    return PortfolioState(
        capital=capital,
        positions={},
        sector_exposures={},
        peak_portfolio_value=capital,
    )


def _make_config(**kwargs) -> RiskConfig:
    return RiskConfig(
        limits=ExposureLimits(
            max_position_fraction=0.20,
            max_sector_fraction=0.40,
            max_gross_exposure=1.50,
            max_net_exposure=1.00,
            max_order_fraction=0.10,
        ),
        circuit_breaker=DrawdownCircuitBreaker(max_drawdown_threshold=0.10),
        **kwargs,
    )


class TestRiskEngineIntegration:
    def test_order_approved(self):
        engine = RiskEngine(_make_config())
        portfolio = _make_portfolio(10_000.0)
        order = Order(symbol="AAPL", quantity=5.0, price=100.0, sector="Tech")  # $500
        result = engine.validate(order, portfolio)
        assert result.approved
        assert result.adjusted_quantity == 5.0
        assert "circuit_breaker" in result.checks_passed

    def test_circuit_breaker_halts_order(self):
        config = _make_config()
        config.circuit_breaker.update(10_000.0)
        config.circuit_breaker.update(8_000.0)  # 20% drawdown, tripped
        engine = RiskEngine(config)
        portfolio = _make_portfolio(8_000.0)
        order = Order(symbol="AAPL", quantity=5.0, price=100.0)
        result = engine.validate(order, portfolio)
        assert not result.approved
        assert result.adjusted_quantity == 0.0
        assert "circuit_breaker" in result.checks_failed

    def test_order_too_large_rejected(self):
        engine = RiskEngine(_make_config())
        portfolio = _make_portfolio(10_000.0)
        # $1100 order on $10k capital = 11% > max_order_fraction 10%
        order = Order(symbol="AAPL", quantity=11.0, price=100.0)
        result = engine.validate(order, portfolio)
        assert not result.approved
        assert "max_order_size" in result.checks_failed

    def test_position_limit_blocks_order(self):
        engine = RiskEngine(_make_config())
        portfolio = _make_portfolio(10_000.0)
        # Existing position in AAPL = $1800 (18%)
        portfolio.positions["AAPL"] = 1_800.0
        # New order adds $500 → $2300 = 23% > 20% limit
        order = Order(symbol="AAPL", quantity=5.0, price=100.0)
        result = engine.validate(order, portfolio)
        assert not result.approved
        assert "position_limit" in result.checks_failed

    def test_sector_limit_blocks_order(self):
        engine = RiskEngine(_make_config())
        portfolio = _make_portfolio(10_000.0)
        portfolio.sector_exposures["Tech"] = 3_800.0  # 38%
        # New order adds $500 → $4300 = 43% > 40% limit
        order = Order(symbol="MSFT", quantity=5.0, price=100.0, sector="Tech")
        result = engine.validate(order, portfolio)
        assert not result.approved
        assert "sector_limit" in result.checks_failed

    def test_gross_exposure_limit(self):
        engine = RiskEngine(_make_config())
        portfolio = _make_portfolio(10_000.0)
        # Already at $14800 gross (148%)
        portfolio.positions = {"A": 7_400.0, "B": 7_400.0}
        # Adding $500 more would push to $15300 = 153% > 150%
        order = Order(symbol="C", quantity=5.0, price=100.0)
        result = engine.validate(order, portfolio)
        assert not result.approved
        assert "gross_exposure" in result.checks_failed

    def test_net_exposure_limit(self):
        engine = RiskEngine(_make_config())
        portfolio = _make_portfolio(10_000.0)
        portfolio.positions["AAPL"] = 9_900.0  # net = 99%
        # Adding $500 → net = $10400 = 104% > 100%
        order = Order(symbol="MSFT", quantity=5.0, price=100.0)
        result = engine.validate(order, portfolio)
        assert not result.approved
        assert "net_exposure" in result.checks_failed

    def test_zero_capital_blocks_all(self):
        engine = RiskEngine(_make_config())
        portfolio = _make_portfolio(0.0)
        order = Order(symbol="AAPL", quantity=1.0, price=100.0)
        result = engine.validate(order, portfolio)
        assert not result.approved

    def test_compute_position_size_fixed_fraction(self):
        config = _make_config(
            sizing_method=SizingMethod.FIXED_FRACTION,
            fixed_fraction_params=FixedFractionParams(fraction=0.02),
        )
        engine = RiskEngine(config)
        # 2% of $10k = $200, at price $100 = 2 units
        units = engine.compute_position_size(capital=10_000.0, price=100.0)
        assert abs(units - 2.0) < 1e-9

    def test_compute_position_size_zero_capital(self):
        engine = RiskEngine(_make_config())
        assert engine.compute_position_size(capital=0.0, price=100.0) == 0.0

    def test_compute_position_size_zero_price(self):
        engine = RiskEngine(_make_config())
        assert engine.compute_position_size(capital=10_000.0, price=0.0) == 0.0

    def test_all_checks_in_passed_list_on_approval(self):
        engine = RiskEngine(_make_config())
        portfolio = _make_portfolio(10_000.0)
        order = Order(symbol="AAPL", quantity=1.0, price=50.0, sector="Tech")
        result = engine.validate(order, portfolio)
        assert result.approved
        for check in [
            "circuit_breaker",
            "max_order_size",
            "position_limit",
            "sector_limit",
            "gross_exposure",
            "net_exposure",
        ]:
            assert check in result.checks_passed
