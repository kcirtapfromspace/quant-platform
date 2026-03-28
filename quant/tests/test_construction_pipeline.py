"""Tests for portfolio construction pipeline (QUA-100)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.construction_pipeline import (
    ConstructionPipeline,
    PipelineConfig,
    PipelineResult,
)
from quant.portfolio.cost_aware_optimizer import CostAwareConfig
from quant.research.alpha_model import AlphaModelConfig, ForecastMethod
from quant.risk.factor_model import FactorModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ASSETS = 15
N_OBS = 200
SYMBOLS = [f"S{i}" for i in range(N_ASSETS)]
DATES = pd.bdate_range("2023-01-01", periods=N_OBS)


def _returns(seed: int = 42) -> pd.DataFrame:
    """Generate correlated asset returns with factor structure."""
    rng = np.random.default_rng(seed)
    # 3 latent factors
    factors = rng.normal(0, 0.01, (N_OBS, 3))
    loadings = rng.normal(0, 1, (N_ASSETS, 3))
    systematic = factors @ loadings.T
    idio = rng.normal(0, 0.003, (N_OBS, N_ASSETS))
    return pd.DataFrame(systematic + idio, index=DATES, columns=SYMBOLS)


def _signals(seed: int = 42) -> pd.Series:
    """Generate cross-sectional signal scores."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, N_ASSETS), index=SYMBOLS, name="signal")


def _adv() -> dict[str, float]:
    """Average daily volumes."""
    return dict.fromkeys(SYMBOLS, 1_000_000_000)


def _default_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        alpha_config=AlphaModelConfig(
            method=ForecastMethod.IC_VOL,
            information_coefficient=0.05,
        ),
        risk_config=FactorModelConfig(n_factors=3, min_observations=50),
        optimizer_config=CostAwareConfig(
            risk_aversion=1.0, cost_penalty=5.0,
        ),
        constraints=PortfolioConstraints(
            long_only=True, max_weight=0.15, max_gross_exposure=1.0,
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        pipeline = ConstructionPipeline(_default_config())
        result = pipeline.construct(_signals(), _returns())
        assert isinstance(result, PipelineResult)

    def test_target_weights_populated(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        assert len(result.target_weights) == N_ASSETS

    def test_n_assets(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        assert result.n_assets == N_ASSETS

    def test_risk_positive(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        assert result.risk > 0

    def test_alpha_result_populated(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        assert result.alpha_result is not None
        assert result.alpha_result.n_assets == N_ASSETS

    def test_risk_result_populated(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        assert result.risk_result is not None
        assert result.risk_result.n_assets == N_ASSETS

    def test_optimizer_result_populated(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        assert result.optimizer_result is not None


# ---------------------------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_long_only(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        for w in result.target_weights.values():
            assert w >= -1e-9

    def test_max_weight(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        for w in result.target_weights.values():
            assert w <= 0.15 + 1e-6

    def test_gross_exposure(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        gross = sum(abs(w) for w in result.target_weights.values())
        assert gross <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Current weights (cost awareness)
# ---------------------------------------------------------------------------


class TestCurrentWeights:
    def test_from_zero(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        assert result.turnover > 0  # Must trade to build portfolio

    def test_from_existing_less_turnover(self):
        """Starting from existing weights should have less turnover than from zero."""
        cfg = _default_config()
        pipeline = ConstructionPipeline(cfg)

        # First construct from zero
        r0 = pipeline.construct(_signals(), _returns())

        # Then construct from previous target
        r1 = pipeline.construct(
            _signals(), _returns(), current_weights=r0.target_weights,
        )
        assert r1.turnover <= r0.turnover + 1e-6

    def test_cost_with_adv(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(), adv=_adv(),
        )
        assert result.total_cost > 0


# ---------------------------------------------------------------------------
# Universe intersection
# ---------------------------------------------------------------------------


class TestUniverseIntersection:
    def test_partial_overlap(self):
        """Pipeline should only use assets present in both signals and returns."""
        signals = pd.Series(
            np.random.default_rng(42).normal(0, 1, 5),
            index=SYMBOLS[:5],
        )
        returns = _returns()  # Has all 15 symbols
        result = ConstructionPipeline(_default_config()).construct(
            signals, returns,
        )
        assert result.n_assets == 5
        assert set(result.target_weights.keys()) == set(SYMBOLS[:5])

    def test_no_overlap_raises(self):
        signals = pd.Series([1.0], index=["XXX"])
        returns = _returns()
        with pytest.raises(ValueError, match="at least 2"):
            ConstructionPipeline(_default_config()).construct(signals, returns)


# ---------------------------------------------------------------------------
# Custom volatilities
# ---------------------------------------------------------------------------


class TestCustomVols:
    def test_explicit_vols_used(self):
        vols = pd.Series(0.30, index=SYMBOLS)
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(), asset_volatilities=vols,
        )
        assert result.alpha_result.cross_sectional_vol == pytest.approx(0.30, abs=0.01)

    def test_computed_from_returns(self):
        """Without explicit vols, should compute from returns history."""
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        assert result.alpha_result.cross_sectional_vol > 0


# ---------------------------------------------------------------------------
# Different alpha methods
# ---------------------------------------------------------------------------


class TestAlphaMethods:
    def test_rank_method(self):
        cfg = _default_config()
        cfg.alpha_config = AlphaModelConfig(method=ForecastMethod.RANK, target_spread=0.10)
        result = ConstructionPipeline(cfg).construct(_signals(), _returns())
        assert result.alpha_result.method == ForecastMethod.RANK

    def test_raw_method(self):
        cfg = _default_config()
        cfg.alpha_config = AlphaModelConfig(method=ForecastMethod.RAW, raw_multiplier=0.01)
        result = ConstructionPipeline(cfg).construct(_signals(), _returns())
        assert result.alpha_result.method == ForecastMethod.RAW


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = ConstructionPipeline(_default_config()).construct(
            _signals(), _returns(),
        )
        summary = result.summary()
        assert "Portfolio Construction Pipeline" in summary
        assert "Alpha Model" in summary
        assert "Risk Model" in summary
        assert "Optimizer" in summary
        assert "Target Weights" in summary
