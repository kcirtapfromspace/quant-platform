"""Tests for cross-sectional ranking signals and scoring utilities (QUA-37)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from quant.signals.cross_sectional import (
    CrossSectionalMeanReversion,
    CrossSectionalMomentum,
    CrossSectionalVolatility,
    QuantileSelection,
    QuantileSelector,
    QuantileWeights,
    percentile_rank,
    scores_to_quantile_weights,
    winsorize,
    z_score_normalize,
)

_NOW = datetime(2025, 1, 15, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helper: build synthetic universe features
# ---------------------------------------------------------------------------


def _make_universe(
    symbols: list[str],
    n_days: int = 200,
    base_return: float = 0.0005,
    vol: float = 0.02,
    seed: int = 42,
) -> dict[str, dict[str, pd.Series]]:
    """Generate synthetic per-symbol returns with controlled drift and vol."""
    import random

    rng = random.Random(seed)
    universe: dict[str, dict[str, pd.Series]] = {}

    for i, sym in enumerate(symbols):
        # Vary drift per symbol so momentum ranking is deterministic
        drift = base_return * (i + 1)
        returns = [drift + rng.gauss(0, vol) for _ in range(n_days)]
        universe[sym] = {
            "returns": pd.Series(returns, name=sym),
        }

    return universe


# ---------------------------------------------------------------------------
# Percentile rank tests
# ---------------------------------------------------------------------------


class TestPercentileRank:
    def test_basic_ranking(self):
        scores = {"A": 1.0, "B": 3.0, "C": 2.0}
        ranks = percentile_rank(scores)
        assert ranks["A"] < ranks["C"] < ranks["B"]

    def test_all_ranks_in_0_1(self):
        scores = {f"S{i}": float(i) for i in range(20)}
        ranks = percentile_rank(scores)
        for v in ranks.values():
            assert 0 <= v <= 1

    def test_ties_get_average_rank(self):
        scores = {"A": 1.0, "B": 1.0, "C": 2.0}
        ranks = percentile_rank(scores)
        assert ranks["A"] == ranks["B"]

    def test_single_asset(self):
        ranks = percentile_rank({"X": 5.0})
        assert ranks["X"] == 0.5

    def test_empty(self):
        assert percentile_rank({}) == {}

    def test_symmetric_extremes(self):
        """First and last ranks should be symmetric around 0.5."""
        scores = {"A": 0.0, "B": 1.0}
        ranks = percentile_rank(scores)
        assert abs(ranks["A"] + ranks["B"] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Z-score normalisation tests
# ---------------------------------------------------------------------------


class TestZScoreNormalize:
    def test_zero_mean(self):
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        z = z_score_normalize(scores)
        assert abs(sum(z.values())) < 1e-10

    def test_unit_variance(self):
        scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
        z = z_score_normalize(scores)
        vals = list(z.values())
        var = sum(v**2 for v in vals) / len(vals)
        assert abs(var - 1.0) < 0.01

    def test_all_same_returns_zeros(self):
        scores = {"A": 5.0, "B": 5.0, "C": 5.0}
        z = z_score_normalize(scores)
        for v in z.values():
            assert v == 0.0

    def test_single_asset_returns_zero(self):
        z = z_score_normalize({"X": 10.0})
        assert z["X"] == 0.0


# ---------------------------------------------------------------------------
# Winsorize tests
# ---------------------------------------------------------------------------


class TestWinsorize:
    def test_clips_extremes(self):
        scores = {f"S{i}": float(i) for i in range(100)}
        clipped = winsorize(scores, lower=0.10, upper=0.90)
        vals = list(clipped.values())
        assert min(vals) >= 9  # approx 10th percentile
        assert max(vals) <= 90  # approx 90th percentile

    def test_moderate_values_unchanged(self):
        scores = {"A": 50.0, "B": 51.0, "C": 52.0}
        clipped = winsorize(scores, lower=0.05, upper=0.95)
        for sym in scores:
            assert clipped[sym] == scores[sym]

    def test_empty(self):
        assert winsorize({}) == {}


# ---------------------------------------------------------------------------
# QuantileSelector tests
# ---------------------------------------------------------------------------


class TestQuantileSelector:
    def test_basic_selection(self):
        scores = {f"S{i}": float(i) for i in range(10)}
        sel = QuantileSelector(long_quantile=0.2, short_quantile=0.2)
        result = sel.select(scores)

        assert isinstance(result, QuantileSelection)
        assert len(result.long_symbols) == 2
        assert len(result.short_symbols) == 2

    def test_long_is_top_short_is_bottom(self):
        scores = {"A": 10, "B": 20, "C": 30, "D": 40, "E": 50}
        sel = QuantileSelector(long_quantile=0.2, short_quantile=0.2)
        result = sel.select(scores)

        assert "E" in result.long_symbols
        assert "A" in result.short_symbols

    def test_no_overlap(self):
        scores = {f"S{i}": float(i) for i in range(20)}
        sel = QuantileSelector(long_quantile=0.3, short_quantile=0.3)
        result = sel.select(scores)

        long_set = set(result.long_symbols)
        short_set = set(result.short_symbols)
        assert long_set.isdisjoint(short_set)

    def test_empty_universe(self):
        result = QuantileSelector().select({})
        assert result.long_symbols == []
        assert result.short_symbols == []

    def test_invalid_quantiles(self):
        with pytest.raises(ValueError):
            QuantileSelector(long_quantile=0.6, short_quantile=0.6)

    def test_single_asset(self):
        result = QuantileSelector(long_quantile=0.5, short_quantile=0.5).select({"X": 1.0})
        # With one asset, it should handle gracefully
        total = len(result.long_symbols) + len(result.short_symbols)
        assert total >= 1

    def test_all_ranks_present(self):
        scores = {f"S{i}": float(i) for i in range(10)}
        result = QuantileSelector().select(scores)
        assert len(result.all_ranks) == 10


# ---------------------------------------------------------------------------
# CrossSectionalMomentum tests
# ---------------------------------------------------------------------------


class TestCrossSectionalMomentum:
    def test_higher_drift_ranks_higher(self):
        """Symbols with higher drift should get higher momentum scores."""
        universe = _make_universe(
            ["LOW", "MED", "HIGH"],
            n_days=200,
            base_return=0.002,
            vol=0.005,
            seed=77,
        )
        signal = CrossSectionalMomentum(lookback=63, skip_recent=5)
        outputs = signal.score_universe(universe, _NOW)

        assert len(outputs) == 3
        assert outputs["HIGH"].score > outputs["LOW"].score

    def test_scores_in_valid_range(self):
        universe = _make_universe(
            [f"S{i}" for i in range(10)],
            n_days=200,
        )
        signal = CrossSectionalMomentum(lookback=63)
        outputs = signal.score_universe(universe, _NOW)

        for out in outputs.values():
            assert -1.0 <= out.score <= 1.0
            assert 0.0 <= out.confidence <= 1.0
            assert -1.0 <= out.target_position <= 1.0

    def test_skip_recent(self):
        """skip_recent should exclude most recent days from return calc."""
        universe = _make_universe(["A", "B"], n_days=200)
        sig_skip = CrossSectionalMomentum(lookback=63, skip_recent=10)
        sig_no_skip = CrossSectionalMomentum(lookback=63, skip_recent=0)

        out_skip = sig_skip.score_universe(universe, _NOW)
        out_no_skip = sig_no_skip.score_universe(universe, _NOW)

        # Different skip_recent should yield different raw scores
        assert out_skip["A"].metadata["raw_score"] != out_no_skip["A"].metadata["raw_score"]

    def test_insufficient_data_excluded(self):
        """Symbols with too few returns should be excluded."""
        universe = {
            "LONG": {"returns": pd.Series([0.01] * 200)},
            "SHORT": {"returns": pd.Series([0.01] * 5)},  # too few
        }
        signal = CrossSectionalMomentum(lookback=63, skip_recent=5)
        outputs = signal.score_universe(universe, _NOW)

        assert "LONG" in outputs
        assert "SHORT" not in outputs

    def test_name(self):
        assert CrossSectionalMomentum().name == "xs_momentum"

    def test_required_features(self):
        assert "returns" in CrossSectionalMomentum().required_features

    def test_zscore_normalize(self):
        """Z-score normalisation should produce different scores than percentile."""
        universe = _make_universe([f"S{i}" for i in range(10)], n_days=200)
        signal = CrossSectionalMomentum(lookback=63)

        pct = signal.score_universe(universe, _NOW, normalize="percentile")
        zsc = signal.score_universe(universe, _NOW, normalize="zscore")

        # Scores should differ between methods
        some_sym = list(pct.keys())[0]
        assert pct[some_sym].score != zsc[some_sym].score


# ---------------------------------------------------------------------------
# CrossSectionalMeanReversion tests
# ---------------------------------------------------------------------------


class TestCrossSectionalMeanReversion:
    def test_contrarian_scoring(self):
        """Assets that dropped should get higher scores (contrarian)."""
        # Asset A: recent crash (mean reversion candidate)
        import random
        rng = random.Random(44)
        returns_a = [0.005 + rng.gauss(0, 0.01) for _ in range(180)] + [-0.05 + rng.gauss(0, 0.01) for _ in range(20)]
        # Asset B: steady with some noise (neutral)
        returns_b = [0.005 + rng.gauss(0, 0.01) for _ in range(200)]

        universe = {
            "CRASHED": {"returns": pd.Series(returns_a)},
            "STEADY": {"returns": pd.Series(returns_b)},
        }
        signal = CrossSectionalMeanReversion(lookback=21, vol_lookback=63)
        outputs = signal.score_universe(universe, _NOW)

        assert len(outputs) == 2
        # Crashed asset should have higher score (contrarian buy signal)
        assert outputs["CRASHED"].score > outputs["STEADY"].score

    def test_scores_valid(self):
        universe = _make_universe([f"S{i}" for i in range(8)], n_days=200)
        signal = CrossSectionalMeanReversion()
        outputs = signal.score_universe(universe, _NOW)

        for out in outputs.values():
            assert -1.0 <= out.score <= 1.0

    def test_name(self):
        assert CrossSectionalMeanReversion().name == "xs_mean_reversion"


# ---------------------------------------------------------------------------
# CrossSectionalVolatility tests
# ---------------------------------------------------------------------------


class TestCrossSectionalVolatility:
    def test_low_vol_ranks_higher(self):
        """Low-volatility assets should rank higher (low-vol anomaly)."""
        import random

        rng = random.Random(55)
        # Low vol asset
        returns_low = [rng.gauss(0, 0.005) for _ in range(200)]
        # High vol asset
        returns_high = [rng.gauss(0, 0.05) for _ in range(200)]

        universe = {
            "LOW_VOL": {"returns": pd.Series(returns_low)},
            "HIGH_VOL": {"returns": pd.Series(returns_high)},
        }
        signal = CrossSectionalVolatility(lookback=63)
        outputs = signal.score_universe(universe, _NOW)

        assert outputs["LOW_VOL"].score > outputs["HIGH_VOL"].score

    def test_scores_valid(self):
        universe = _make_universe([f"S{i}" for i in range(8)], n_days=200)
        signal = CrossSectionalVolatility()
        outputs = signal.score_universe(universe, _NOW)

        for out in outputs.values():
            assert -1.0 <= out.score <= 1.0

    def test_name(self):
        assert CrossSectionalVolatility().name == "xs_volatility"

    def test_required_features(self):
        assert "returns" in CrossSectionalVolatility().required_features


# ---------------------------------------------------------------------------
# scores_to_quantile_weights tests
# ---------------------------------------------------------------------------


class TestScoresToQuantileWeights:
    def test_dollar_neutral_sums_to_zero(self):
        scores = {f"S{i}": float(i) for i in range(20)}
        qw = scores_to_quantile_weights(scores, dollar_neutral=True)

        assert isinstance(qw, QuantileWeights)
        total = sum(qw.weights.values())
        assert abs(total) < 0.01

    def test_long_only_sums_to_one(self):
        scores = {f"S{i}": float(i) for i in range(20)}
        qw = scores_to_quantile_weights(scores, dollar_neutral=False)

        # Only long leg, no shorts
        assert all(w >= 0 for w in qw.weights.values())
        total = sum(qw.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_equal_weight_within_legs(self):
        scores = {f"S{i}": float(i) for i in range(10)}
        qw = scores_to_quantile_weights(
            scores, long_quantile=0.2, short_quantile=0.2, equal_weight=True, dollar_neutral=True,
        )

        long_weights = [qw.weights[s] for s in qw.long_symbols]
        short_weights = [qw.weights[s] for s in qw.short_symbols]

        # All long weights should be equal
        if len(long_weights) > 1:
            assert abs(long_weights[0] - long_weights[1]) < 1e-10

        # All short weights should be equal
        if len(short_weights) > 1:
            assert abs(short_weights[0] - short_weights[1]) < 1e-10

    def test_score_weighted(self):
        scores = {f"S{i}": float(i) for i in range(10)}
        qw = scores_to_quantile_weights(
            scores, long_quantile=0.2, short_quantile=0.2, equal_weight=False, dollar_neutral=True,
        )

        # Weights should exist for longs and shorts
        assert len(qw.long_symbols) > 0
        assert len(qw.short_symbols) > 0

    def test_metadata(self):
        scores = {f"S{i}": float(i) for i in range(10)}
        qw = scores_to_quantile_weights(scores)

        assert qw.metadata["method"] == "quantile"
        assert qw.metadata["universe_size"] == 10

    def test_empty_scores(self):
        qw = scores_to_quantile_weights({})
        assert qw.weights == {}
        assert qw.long_symbols == []
        assert qw.short_symbols == []

    def test_gross_exposure(self):
        """Gross exposure should be ~1.0 for dollar-neutral."""
        scores = {f"S{i}": float(i) for i in range(20)}
        qw = scores_to_quantile_weights(scores, dollar_neutral=True, equal_weight=True)

        gross = sum(abs(w) for w in qw.weights.values())
        assert abs(gross - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Integration: signal → quantile weights pipeline
# ---------------------------------------------------------------------------


class TestIntegrationPipeline:
    def test_momentum_to_weights(self):
        """End-to-end: momentum signal → quantile weights."""
        universe = _make_universe(
            [f"S{i}" for i in range(10)],
            n_days=200,
            base_return=0.001,
            seed=99,
        )
        signal = CrossSectionalMomentum(lookback=63)
        outputs = signal.score_universe(universe, _NOW)

        # Extract raw scores for weight construction
        raw_scores = {sym: out.metadata["raw_score"] for sym, out in outputs.items()}
        qw = scores_to_quantile_weights(raw_scores, dollar_neutral=True)

        assert len(qw.long_symbols) > 0
        assert len(qw.short_symbols) > 0
        assert abs(sum(qw.weights.values())) < 0.01  # dollar neutral

    def test_volatility_to_weights(self):
        """End-to-end: volatility signal → quantile weights."""
        universe = _make_universe(
            [f"S{i}" for i in range(10)],
            n_days=200,
            seed=88,
        )
        signal = CrossSectionalVolatility(lookback=63)
        outputs = signal.score_universe(universe, _NOW)

        raw_scores = {sym: out.metadata["raw_score"] for sym, out in outputs.items()}
        qw = scores_to_quantile_weights(raw_scores, dollar_neutral=True)

        assert len(qw.weights) > 0
