"""Tests for factor-based signals — volatility, quality, breakout (QUA-32)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.signals.factors import BreakoutSignal, ReturnQualitySignal, VolatilitySignal


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_returns(
    n: int = 100,
    mean: float = 0.0004,
    std: float = 0.012,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(rng.normal(mean, std, n), index=dates, name="returns")


# ── VolatilitySignal tests ───────────────────────────────────────────────────


class TestVolatilitySignal:
    def test_low_vol_stock_positive_score(self):
        """A low-volatility stock should get a positive (long) score."""
        returns = _make_returns(n=100, std=0.005)  # ~8% annual vol
        signal = VolatilitySignal(period=20, low_vol=0.12, high_vol=0.40)
        out = signal.compute("AAPL", {"returns": returns}, _utcnow())
        assert out.score > 0  # low vol → positive
        assert out.metadata["vol"] < 0.15

    def test_high_vol_stock_negative_score(self):
        """A high-volatility stock should get a negative (short/avoid) score."""
        returns = _make_returns(n=100, std=0.04)  # ~63% annual vol
        signal = VolatilitySignal(period=20, low_vol=0.12, high_vol=0.40)
        out = signal.compute("MEME", {"returns": returns}, _utcnow())
        assert out.score < 0  # high vol → negative

    def test_score_bounded(self):
        returns = _make_returns(n=100, std=0.08)  # extremely high vol
        signal = VolatilitySignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert -1.0 <= out.score <= 1.0
        assert -1.0 <= out.target_position <= 1.0

    def test_insufficient_data_returns_zero(self):
        returns = _make_returns(n=5)
        signal = VolatilitySignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert out.score == 0.0
        assert out.confidence == 0.0

    def test_confidence_scales_with_data(self):
        short_returns = _make_returns(n=20)
        long_returns = _make_returns(n=200)
        signal = VolatilitySignal(period=20)
        short_out = signal.compute("SYM", {"returns": short_returns}, _utcnow())
        long_out = signal.compute("SYM", {"returns": long_returns}, _utcnow())
        assert long_out.confidence > short_out.confidence

    def test_metadata_contains_vol(self):
        returns = _make_returns(n=100)
        signal = VolatilitySignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert "vol" in out.metadata
        assert out.metadata["vol"] > 0

    def test_name(self):
        assert VolatilitySignal().name == "volatility"

    def test_required_features(self):
        assert "returns" in VolatilitySignal().required_features


# ── ReturnQualitySignal tests ────────────────────────────────────────────────


class TestReturnQualitySignal:
    def test_positive_sharpe_positive_score(self):
        """Asset with consistently positive returns should score positively."""
        returns = _make_returns(n=200, mean=0.002, std=0.01)  # strong Sharpe
        signal = ReturnQualitySignal(period=60)
        out = signal.compute("AAPL", {"returns": returns}, _utcnow())
        assert out.score > 0
        assert out.metadata["sharpe"] > 0

    def test_negative_sharpe_negative_score(self):
        """Asset with consistently negative returns should score negatively."""
        returns = _make_returns(n=200, mean=-0.003, std=0.01)
        signal = ReturnQualitySignal(period=60)
        out = signal.compute("BAD", {"returns": returns}, _utcnow())
        assert out.score < 0
        assert out.metadata["sharpe"] < 0

    def test_score_bounded(self):
        returns = _make_returns(n=200, mean=0.01, std=0.001)  # extreme Sharpe
        signal = ReturnQualitySignal(period=60, sharpe_cap=3.0)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert -1.0 <= out.score <= 1.0

    def test_insufficient_data_returns_zero(self):
        returns = _make_returns(n=10)
        signal = ReturnQualitySignal(period=60)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert out.score == 0.0
        assert out.confidence == 0.0

    def test_zero_vol_returns_zero_sharpe(self):
        returns = pd.Series([0.001] * 100, index=pd.bdate_range("2023-01-01", periods=100))
        signal = ReturnQualitySignal(period=60)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert out.metadata["sharpe"] == 0.0

    def test_name(self):
        assert ReturnQualitySignal().name == "return_quality"

    def test_required_features(self):
        assert "returns" in ReturnQualitySignal().required_features


# ── BreakoutSignal tests ─────────────────────────────────────────────────────


class TestBreakoutSignal:
    def _trending_up_returns(self, n: int = 50) -> pd.Series:
        """Returns that produce a clear uptrend (breaking above channel)."""
        rng = np.random.default_rng(99)
        ret = rng.normal(0.005, 0.005, n)  # strong positive drift
        dates = pd.bdate_range("2023-01-01", periods=n)
        return pd.Series(ret, index=dates, name="returns")

    def _trending_down_returns(self, n: int = 50) -> pd.Series:
        """Returns that produce a clear downtrend (breaking below channel)."""
        rng = np.random.default_rng(99)
        ret = rng.normal(-0.005, 0.005, n)  # strong negative drift
        dates = pd.bdate_range("2023-01-01", periods=n)
        return pd.Series(ret, index=dates, name="returns")

    def test_uptrend_breakout_positive_score(self):
        returns = self._trending_up_returns()
        signal = BreakoutSignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert out.score > 0  # breaking above channel

    def test_downtrend_breakout_negative_score(self):
        returns = self._trending_down_returns()
        signal = BreakoutSignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert out.score < 0  # breaking below channel

    def test_score_bounded(self):
        returns = _make_returns(n=100)
        signal = BreakoutSignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert -1.0 <= out.score <= 1.0
        assert 0.0 <= out.confidence <= 1.0
        assert -1.0 <= out.target_position <= 1.0

    def test_insufficient_data_returns_zero(self):
        returns = _make_returns(n=10)
        signal = BreakoutSignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert out.score == 0.0
        assert out.confidence == 0.0

    def test_metadata_contains_channel_info(self):
        returns = _make_returns(n=100)
        signal = BreakoutSignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert "channel_pos" in out.metadata
        assert "channel_high" in out.metadata
        assert "channel_low" in out.metadata

    def test_name(self):
        assert BreakoutSignal().name == "breakout"

    def test_required_features(self):
        assert "returns" in BreakoutSignal().required_features

    def test_flat_channel_zero_score(self):
        """Constant returns → flat channel → zero score."""
        returns = pd.Series(
            [0.0] * 50,
            index=pd.bdate_range("2023-01-01", periods=50),
        )
        signal = BreakoutSignal(period=20)
        out = signal.compute("SYM", {"returns": returns}, _utcnow())
        assert out.score == 0.0


# ── Cross-signal consistency tests ───────────────────────────────────────────


class TestFactorSignalConsistency:
    def test_all_signals_produce_valid_output(self):
        """All factor signals produce outputs conforming to SignalOutput constraints."""
        returns = _make_returns(n=200)
        features = {"returns": returns}
        signals = [
            VolatilitySignal(),
            ReturnQualitySignal(),
            BreakoutSignal(),
        ]
        for signal in signals:
            out = signal.compute("TEST", features, _utcnow())
            assert -1.0 <= out.score <= 1.0, f"{signal.name}: score out of range"
            assert 0.0 <= out.confidence <= 1.0, f"{signal.name}: confidence out of range"
            assert -1.0 <= out.target_position <= 1.0, f"{signal.name}: target_position out of range"
            assert out.symbol == "TEST"

    def test_signals_have_unique_names(self):
        signals = [VolatilitySignal(), ReturnQualitySignal(), BreakoutSignal()]
        names = [s.name for s in signals]
        assert len(names) == len(set(names))

    def test_signals_importable_from_package(self):
        from quant.signals import BreakoutSignal, ReturnQualitySignal, VolatilitySignal

        assert VolatilitySignal is not None
        assert ReturnQualitySignal is not None
        assert BreakoutSignal is not None
