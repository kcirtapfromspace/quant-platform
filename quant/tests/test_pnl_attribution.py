"""Tests for position-level PnL attribution (QUA-101)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.pnl_attribution import (
    PnLAttributionResult,
    PnLAttributor,
    PnLConfig,
    PositionPnL,
    SectorPnL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DAYS = 100
SYMBOLS = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
DATES = pd.bdate_range("2024-01-01", periods=N_DAYS)


def _weights(seed: int = 42) -> pd.DataFrame:
    """Equal-weight portfolio with small daily drift."""
    rng = np.random.default_rng(seed)
    base = 1.0 / len(SYMBOLS)
    w = np.full((N_DAYS, len(SYMBOLS)), base)
    w += rng.normal(0, 0.01, w.shape)
    w = np.abs(w)
    w /= w.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, index=DATES, columns=SYMBOLS)


def _returns(seed: int = 42) -> pd.DataFrame:
    """Simulated daily returns with one strong winner."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, (N_DAYS, len(SYMBOLS)))
    # Make AMZN a consistent winner
    r[:, 3] += 0.002
    # Make META a consistent loser
    r[:, 4] -= 0.001
    return pd.DataFrame(r, index=DATES, columns=SYMBOLS)


def _sector_map() -> dict[str, str]:
    return {
        "AAPL": "Tech",
        "GOOG": "Tech",
        "MSFT": "Tech",
        "AMZN": "Consumer",
        "META": "Social",
    }


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert isinstance(result, PnLAttributionResult)

    def test_n_days(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert result.n_days == N_DAYS

    def test_n_positions(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert result.n_positions == len(SYMBOLS)

    def test_daily_pnl_series(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert isinstance(result.daily_pnl, pd.Series)
        assert len(result.daily_pnl) == N_DAYS

    def test_position_pnl_shape(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert result.position_pnl.shape == (N_DAYS, len(SYMBOLS))

    def test_positions_populated(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert len(result.positions) == len(SYMBOLS)

    def test_position_types(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        for p in result.positions:
            assert isinstance(p, PositionPnL)


# ---------------------------------------------------------------------------
# PnL decomposition
# ---------------------------------------------------------------------------


class TestDecomposition:
    def test_position_pnl_sums_to_total(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        sum_positions = sum(p.total_pnl for p in result.positions)
        assert abs(sum_positions - result.total_pnl) < 1e-10

    def test_daily_pnl_sums_to_total(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert abs(float(result.daily_pnl.sum()) - result.total_pnl) < 1e-10

    def test_position_pnl_equals_weight_times_return(self):
        """Position PnL should equal w * r for each date/symbol."""
        w = _weights()
        r = _returns()
        result = PnLAttributor().attribute(w, r)
        # Align columns to match attributor's sorted order
        cols = list(result.position_pnl.columns)
        expected = w[cols] * r[cols]
        np.testing.assert_allclose(
            result.position_pnl.values, expected.values, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Winners and losers
# ---------------------------------------------------------------------------


class TestWinnersLosers:
    def test_top_contributors_ordered(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        pnls = [p.total_pnl for p in result.top_contributors]
        for i in range(len(pnls) - 1):
            assert pnls[i] >= pnls[i + 1]

    def test_amzn_is_winner(self):
        """AMZN has positive drift and should be a top contributor."""
        result = PnLAttributor().attribute(_weights(), _returns())
        top_names = [p.symbol for p in result.top_contributors]
        assert "AMZN" in top_names

    def test_meta_is_loser(self):
        """META has negative drift and should be a bottom contributor."""
        result = PnLAttributor().attribute(_weights(), _returns())
        bottom_names = [p.symbol for p in result.bottom_contributors]
        assert "META" in bottom_names

    def test_win_loss_days(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        for p in result.positions:
            assert p.win_days + p.loss_days <= p.n_days
            assert p.win_days >= 0
            assert p.loss_days >= 0


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------


class TestConcentration:
    def test_hhi_in_range(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert 0 <= result.hhi <= 1.0

    def test_top_n_share_in_range(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert 0 <= result.top_n_share <= 1.0

    def test_equal_pnl_low_hhi(self):
        """Equal PnL across all positions should give low HHI."""
        # Uniform weights and uniform returns
        w = pd.DataFrame(
            np.full((N_DAYS, 5), 0.20), index=DATES, columns=SYMBOLS,
        )
        r = pd.DataFrame(
            np.full((N_DAYS, 5), 0.01), index=DATES, columns=SYMBOLS,
        )
        result = PnLAttributor().attribute(w, r)
        assert result.hhi < 0.30  # For 5 equal positions: HHI = 0.20


# ---------------------------------------------------------------------------
# Sector attribution
# ---------------------------------------------------------------------------


class TestSectorAttribution:
    def test_sectors_populated(self):
        cfg = PnLConfig(sector_map=_sector_map())
        result = PnLAttributor(cfg).attribute(_weights(), _returns())
        assert len(result.sectors) >= 2

    def test_sector_types(self):
        cfg = PnLConfig(sector_map=_sector_map())
        result = PnLAttributor(cfg).attribute(_weights(), _returns())
        for s in result.sectors:
            assert isinstance(s, SectorPnL)

    def test_sector_pnl_sums_to_total(self):
        cfg = PnLConfig(sector_map=_sector_map())
        result = PnLAttributor(cfg).attribute(_weights(), _returns())
        sector_total = sum(s.total_pnl for s in result.sectors)
        assert abs(sector_total - result.total_pnl) < 1e-10

    def test_no_sectors_without_map(self):
        result = PnLAttributor().attribute(_weights(), _returns())
        assert len(result.sectors) == 0

    def test_tech_sector_three_positions(self):
        cfg = PnLConfig(sector_map=_sector_map())
        result = PnLAttributor(cfg).attribute(_weights(), _returns())
        tech = next(s for s in result.sectors if s.sector == "Tech")
        assert tech.n_positions == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_common_dates_raises(self):
        w = pd.DataFrame(
            [[0.5, 0.5]], index=pd.bdate_range("2024-01-01", periods=1),
            columns=["A", "B"],
        )
        r = pd.DataFrame(
            [[0.01, 0.02]], index=pd.bdate_range("2025-01-01", periods=1),
            columns=["A", "B"],
        )
        with pytest.raises(ValueError, match="at least 1"):
            PnLAttributor().attribute(w, r)

    def test_single_day(self):
        dates = pd.bdate_range("2024-01-01", periods=1)
        w = pd.DataFrame([[0.5, 0.5]], index=dates, columns=["A", "B"])
        r = pd.DataFrame([[0.01, -0.01]], index=dates, columns=["A", "B"])
        result = PnLAttributor().attribute(w, r)
        assert result.n_days == 1
        assert abs(result.total_pnl) < 0.01

    def test_single_position(self):
        w = pd.DataFrame(
            np.full((N_DAYS, 1), 1.0), index=DATES, columns=["X"],
        )
        rng = np.random.default_rng(42)
        r = pd.DataFrame(
            rng.normal(0, 0.01, (N_DAYS, 1)), index=DATES, columns=["X"],
        )
        result = PnLAttributor().attribute(w, r)
        assert result.n_positions == 1
        assert result.hhi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        cfg = PnLConfig(sector_map=_sector_map())
        result = PnLAttributor(cfg).attribute(_weights(), _returns())
        summary = result.summary()
        assert "PnL Attribution" in summary
        assert "Top contributors" in summary
        assert "Bottom contributors" in summary
        assert "Sector" in summary
        assert "HHI" in summary
