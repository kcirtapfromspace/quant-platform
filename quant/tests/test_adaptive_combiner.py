"""Tests for adaptive IC-weighted signal combination (QUA-50, QUA-68)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.signals.adaptive_combiner import (
    AdaptiveCombinerConfig,
    AdaptiveSignalCombiner,
    AdaptiveWeights,
    BayesianAdaptiveCombinerConfig,
    BayesianAdaptiveSignalCombiner,
    _NormalGammaTracker,
)
from quant.signals.base import SignalOutput

# ── Helpers ───────────────────────────────────────────────────────────────────

NOW = datetime(2024, 6, 15, tzinfo=timezone.utc)


def _make_signal_output(
    symbol: str,
    score: float,
    confidence: float = 0.8,
    signal_name: str = "sig",
) -> SignalOutput:
    return SignalOutput(
        symbol=symbol,
        timestamp=NOW,
        score=score,
        confidence=confidence,
        target_position=score * confidence,
        metadata={"signal_name": signal_name},
    )


def _make_predictive_data(
    n_obs: int = 50,
    n_assets: int = 20,
    seed: int = 42,
) -> tuple[list[pd.Series], list[pd.Series], list[datetime]]:
    """Generate n_obs cross-sectional observations where signal predicts returns.

    Returns (signal_scores_list, returns_list, timestamps).
    Each entry is a Series indexed by symbol.
    """
    rng = np.random.default_rng(seed)
    symbols = [f"S{i:02d}" for i in range(n_assets)]
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc).__class__(
            2024, 1, 1 + i, tzinfo=timezone.utc
        )
        for i in range(n_obs)
    ]

    scores_list: list[pd.Series] = []
    returns_list: list[pd.Series] = []

    for _ in range(n_obs):
        s = rng.uniform(-1, 1, n_assets)
        r = s * 0.02 + rng.normal(0, 0.005, n_assets)
        scores_list.append(pd.Series(s, index=symbols))
        returns_list.append(pd.Series(r, index=symbols))

    return scores_list, returns_list, timestamps


def _make_noise_data(
    n_obs: int = 50,
    n_assets: int = 20,
    seed: int = 99,
) -> tuple[list[pd.Series], list[pd.Series], list[datetime]]:
    """Generate data where signal has zero predictive power."""
    rng = np.random.default_rng(seed)
    symbols = [f"S{i:02d}" for i in range(n_assets)]
    timestamps = [
        datetime(2024, 1, 1 + i, tzinfo=timezone.utc) for i in range(n_obs)
    ]

    scores_list: list[pd.Series] = []
    returns_list: list[pd.Series] = []

    for _ in range(n_obs):
        scores_list.append(pd.Series(rng.uniform(-1, 1, n_assets), index=symbols))
        returns_list.append(pd.Series(rng.normal(0, 0.02, n_assets), index=symbols))

    return scores_list, returns_list, timestamps


def _feed_combiner(
    combiner: AdaptiveSignalCombiner,
    signal_name: str,
    scores_list: list[pd.Series],
    returns_list: list[pd.Series],
    timestamps: list[datetime],
) -> None:
    """Feed a series of observations into the combiner."""
    for scores, returns, ts in zip(
        scores_list, returns_list, timestamps, strict=True
    ):
        combiner.update({signal_name: scores}, returns, ts)


# ── Tests: Basic functionality ────────────────────────────────────────────


class TestBasicCombine:
    def test_empty_signals_returns_zero(self):
        combiner = AdaptiveSignalCombiner()
        alpha = combiner.combine("AAPL", NOW, [])
        assert alpha.score == 0.0
        assert alpha.confidence == 0.0

    def test_single_signal_equal_weight(self):
        """With no IC history, falls back to equal weight."""
        combiner = AdaptiveSignalCombiner()
        sig = _make_signal_output("AAPL", 0.5, signal_name="momentum")
        alpha = combiner.combine("AAPL", NOW, [sig])
        assert abs(alpha.score - 0.5) < 1e-6

    def test_two_signals_equal_weight_fallback(self):
        """Without IC data, two signals get equal weight."""
        combiner = AdaptiveSignalCombiner()
        sigs = [
            _make_signal_output("AAPL", 0.6, signal_name="momentum"),
            _make_signal_output("AAPL", -0.2, signal_name="mean_rev"),
        ]
        alpha = combiner.combine("AAPL", NOW, sigs)
        expected = (0.6 + (-0.2)) / 2.0
        assert abs(alpha.score - expected) < 1e-6

    def test_combine_universe(self):
        combiner = AdaptiveSignalCombiner()
        universe = {
            "AAPL": [_make_signal_output("AAPL", 0.5, signal_name="mom")],
            "GOOG": [_make_signal_output("GOOG", -0.3, signal_name="mom")],
        }
        alphas = combiner.combine_universe(NOW, universe)
        assert set(alphas.keys()) == {"AAPL", "GOOG"}
        assert abs(alphas["AAPL"].score - 0.5) < 1e-6


# ── Tests: IC update ─────────────────────────────────────────────────────


class TestICUpdate:
    def test_update_returns_ic(self):
        combiner = AdaptiveSignalCombiner()
        scores, returns, timestamps = _make_predictive_data(n_obs=1)
        ics = combiner.update({"sig_a": scores[0]}, returns[0], timestamps[0])
        assert "sig_a" in ics
        assert isinstance(ics["sig_a"], float)

    def test_ic_history_grows(self):
        combiner = AdaptiveSignalCombiner()
        scores, returns, timestamps = _make_predictive_data(n_obs=5)
        for i in range(5):
            combiner.update({"sig_a": scores[i]}, returns[i], timestamps[i])
        assert len(combiner._ic_history["sig_a"]) == 5

    def test_ic_history_trimmed_to_lookback(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(ic_lookback=10)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=20)
        for i in range(20):
            combiner.update({"sig_a": scores[i]}, returns[i], timestamps[i])
        assert len(combiner._ic_history["sig_a"]) == 10

    def test_multiple_signals_tracked(self):
        combiner = AdaptiveSignalCombiner()
        scores, returns, timestamps = _make_predictive_data(n_obs=5)
        for i in range(5):
            combiner.update(
                {"momentum": scores[i], "mean_rev": scores[i]},
                returns[i],
                timestamps[i],
            )
        assert set(combiner.tracked_signals) == {"momentum", "mean_rev"}

    def test_insufficient_assets_skipped(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_assets=100)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=1, n_assets=10)
        ics = combiner.update({"sig": scores[0]}, returns[0], timestamps[0])
        assert len(ics) == 0


# ── Tests: Adaptive weights ──────────────────────────────────────────────


class TestAdaptiveWeights:
    def test_equal_fallback_when_no_data(self):
        combiner = AdaptiveSignalCombiner()
        aw = combiner.get_weights(["a", "b", "c"])
        assert aw.method_used == "equal_fallback"
        assert abs(aw.weights["a"] - 1 / 3) < 1e-6

    def test_equal_fallback_when_insufficient_periods(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_ic_periods=30)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=10)
        _feed_combiner(combiner, "sig", scores, returns, timestamps)
        aw = combiner.get_weights(["sig"])
        assert aw.method_used == "equal_fallback"

    def test_ic_weighted_with_enough_data(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_ic_periods=10)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=30)
        _feed_combiner(combiner, "good_sig", scores, returns, timestamps)

        noise_s, noise_r, noise_t = _make_noise_data(n_obs=30)
        _feed_combiner(combiner, "noise_sig", noise_s, noise_r, noise_t)

        aw = combiner.get_weights(["good_sig", "noise_sig"])
        assert aw.method_used == "ic_weighted"
        # Good signal should get more weight
        assert aw.weights["good_sig"] > aw.weights["noise_sig"]

    def test_weights_sum_to_one(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_ic_periods=10)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=30)
        _feed_combiner(combiner, "sig_a", scores, returns, timestamps)
        _feed_combiner(combiner, "sig_b", scores, returns, timestamps)

        aw = combiner.get_weights(["sig_a", "sig_b"])
        total = sum(aw.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_adaptive_weights_type(self):
        combiner = AdaptiveSignalCombiner()
        aw = combiner.get_weights(["a"])
        assert isinstance(aw, AdaptiveWeights)

    def test_empty_signal_list(self):
        combiner = AdaptiveSignalCombiner()
        aw = combiner.get_weights([])
        assert aw.n_signals == 0
        assert aw.method_used == "equal_fallback"

    def test_no_signal_names_uses_tracked(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_ic_periods=5)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=10)
        _feed_combiner(combiner, "auto_sig", scores, returns, timestamps)
        aw = combiner.get_weights()
        assert "auto_sig" in aw.weights


# ── Tests: Shrinkage ─────────────────────────────────────────────────────


class TestShrinkage:
    def test_full_shrinkage_equals_equal_weight(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(shrinkage=1.0, min_ic_periods=5)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=20)
        _feed_combiner(combiner, "sig_a", scores, returns, timestamps)

        noise_s, noise_r, noise_t = _make_noise_data(n_obs=20)
        _feed_combiner(combiner, "sig_b", noise_s, noise_r, noise_t)

        aw = combiner.get_weights(["sig_a", "sig_b"])
        # Full shrinkage → equal weights
        assert abs(aw.weights["sig_a"] - 0.5) < 1e-6
        assert abs(aw.weights["sig_b"] - 0.5) < 1e-6

    def test_zero_shrinkage_pure_ic(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(shrinkage=0.0, min_ic_periods=5)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=20)
        _feed_combiner(combiner, "good", scores, returns, timestamps)

        noise_s, noise_r, noise_t = _make_noise_data(n_obs=20)
        _feed_combiner(combiner, "bad", noise_s, noise_r, noise_t)

        aw = combiner.get_weights(["good", "bad"])
        # With zero shrinkage and noise signal likely having ≤0 IC,
        # good signal should dominate
        if aw.method_used == "ic_weighted":
            assert aw.weights["good"] > aw.weights["bad"]


# ── Tests: Min IC threshold ──────────────────────────────────────────────


class TestMinICThreshold:
    def test_min_ic_filters_weak_signals(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_ic=0.05, min_ic_periods=5, shrinkage=0.0)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=20)
        _feed_combiner(combiner, "strong", scores, returns, timestamps)

        noise_s, noise_r, noise_t = _make_noise_data(n_obs=20)
        _feed_combiner(combiner, "weak", noise_s, noise_r, noise_t)

        aw = combiner.get_weights(["strong", "weak"])
        # Weak signal (noise) likely below 0.05 IC threshold
        # Strong signal should get more weight
        assert aw.weights["strong"] >= aw.weights["weak"]

    def test_zero_min_ic_includes_all(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_ic=0.0, min_ic_periods=5)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=20)
        _feed_combiner(combiner, "a", scores, returns, timestamps)
        _feed_combiner(combiner, "b", scores, returns, timestamps)

        aw = combiner.get_weights(["a", "b"])
        assert aw.weights["a"] > 0
        assert aw.weights["b"] > 0


# ── Tests: IC history accessor ────────────────────────────────────────────


class TestICHistoryAccessor:
    def test_ic_history_returns_series(self):
        combiner = AdaptiveSignalCombiner()
        scores, returns, timestamps = _make_predictive_data(n_obs=5)
        _feed_combiner(combiner, "sig", scores, returns, timestamps)
        ic_ts = combiner.ic_history("sig")
        assert isinstance(ic_ts, pd.Series)
        assert len(ic_ts) == 5

    def test_ic_history_empty_for_unknown(self):
        combiner = AdaptiveSignalCombiner()
        ic_ts = combiner.ic_history("nonexistent")
        assert ic_ts.empty

    def test_tracked_signals(self):
        combiner = AdaptiveSignalCombiner()
        scores, returns, timestamps = _make_predictive_data(n_obs=3)
        for i in range(3):
            combiner.update(
                {"alpha": scores[i], "beta": scores[i]},
                returns[i],
                timestamps[i],
            )
        assert set(combiner.tracked_signals) == {"alpha", "beta"}


# ── Tests: Reset ─────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_history(self):
        combiner = AdaptiveSignalCombiner()
        scores, returns, timestamps = _make_predictive_data(n_obs=5)
        _feed_combiner(combiner, "sig", scores, returns, timestamps)
        assert len(combiner.tracked_signals) == 1
        combiner.reset()
        assert len(combiner.tracked_signals) == 0


# ── Tests: Combine with IC weights ───────────────────────────────────────


class TestCombineWithICWeights:
    def test_combine_uses_adaptive_weights(self):
        """After feeding IC data, combine should use IC-based weights."""
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_ic_periods=5, shrinkage=0.0)
        )
        # Feed good IC for momentum, noise for mean_rev
        good_s, good_r, timestamps = _make_predictive_data(n_obs=20)
        _feed_combiner(combiner, "momentum", good_s, good_r, timestamps)

        noise_s, noise_r, noise_t = _make_noise_data(n_obs=20)
        _feed_combiner(combiner, "mean_rev", noise_s, noise_r, noise_t)

        # Now combine: momentum should dominate
        sigs = [
            _make_signal_output("AAPL", 0.8, signal_name="momentum"),
            _make_signal_output("AAPL", -0.8, signal_name="mean_rev"),
        ]
        alpha = combiner.combine("AAPL", NOW, sigs)
        # With good momentum IC and noise mean_rev, score should be positive
        # (momentum weight > mean_rev weight)
        aw = combiner.get_weights(["momentum", "mean_rev"])
        if aw.method_used == "ic_weighted":
            assert alpha.score > 0

    def test_contributions_tracked(self):
        combiner = AdaptiveSignalCombiner()
        sigs = [
            _make_signal_output("AAPL", 0.5, signal_name="mom"),
            _make_signal_output("AAPL", 0.3, signal_name="val"),
        ]
        alpha = combiner.combine("AAPL", NOW, sigs)
        assert "mom" in alpha.signal_contributions
        assert "val" in alpha.signal_contributions


# ── Tests: Summary ────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_data(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(min_ic_periods=5)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=10)
        _feed_combiner(combiner, "momentum", scores, returns, timestamps)
        summary = combiner.summary()
        assert "Adaptive Signal Combiner" in summary
        assert "momentum" in summary

    def test_summary_empty(self):
        combiner = AdaptiveSignalCombiner()
        summary = combiner.summary()
        assert "no signals tracked" in summary


# ── Tests: Config ─────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = AdaptiveCombinerConfig()
        assert config.ic_lookback == 126
        assert config.min_ic_periods == 20
        assert config.shrinkage == 0.3
        assert config.ic_halflife == 21

    def test_config_exposed(self):
        config = AdaptiveCombinerConfig(shrinkage=0.5)
        combiner = AdaptiveSignalCombiner(config)
        assert combiner.config.shrinkage == 0.5

    def test_custom_halflife(self):
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(ic_halflife=5, min_ic_periods=5)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=20)
        _feed_combiner(combiner, "sig", scores, returns, timestamps)
        # Should not crash with short half-life
        aw = combiner.get_weights(["sig"])
        assert aw.weights["sig"] > 0


# ── Tests: EWM mean ──────────────────────────────────────────────────────


class TestEWMMean:
    def test_ewm_single_value(self):
        combiner = AdaptiveSignalCombiner()
        result = combiner._ewm_mean(np.array([0.5]))
        assert abs(result - 0.5) < 1e-6

    def test_ewm_emphasises_recent(self):
        """Recent values should have more influence."""
        combiner = AdaptiveSignalCombiner(
            AdaptiveCombinerConfig(ic_halflife=5)
        )
        # Old values low, recent values high
        values = np.array([0.0] * 20 + [1.0] * 5)
        result = combiner._ewm_mean(values)
        # Should be closer to 1.0 than 0.0 due to recency bias
        assert result > 0.3

    def test_ewm_empty(self):
        combiner = AdaptiveSignalCombiner()
        result = combiner._ewm_mean(np.array([]))
        assert result == 0.0


# ── Tests: _NormalGammaTracker ────────────────────────────────────────────


class TestNormalGammaTracker:
    def test_initial_state(self):
        t = _NormalGammaTracker()
        assert t.mu_n == 0.0
        assert t.kappa_n == 1.0
        assert t.alpha_n == 2.0
        assert t.beta_n == 1.0
        assert t.n == 0

    def test_n_increments(self):
        t = _NormalGammaTracker()
        for i in range(5):
            t.update(float(i))
        assert t.n == 5

    def test_prior_shrinkage_single_update(self):
        """After one update of 0.8, posterior mean is 0.4 (prior pulls toward 0)."""
        t = _NormalGammaTracker()
        t.update(0.8)
        # kappa_prev=1, mu_prev=0 → mu_n = (1*0 + 0.8) / 2 = 0.4
        assert abs(t.posterior_mean - 0.4) < 1e-9

    def test_prior_shrinkage_three_updates(self):
        """After 3 updates of 0.8, posterior mean is 0.6 (converging to truth)."""
        t = _NormalGammaTracker()
        for _ in range(3):
            t.update(0.8)
        # Trace: 0 → 0.4 → 0.533 → 0.6
        assert abs(t.posterior_mean - 0.6) < 1e-9

    def test_posterior_mean_converges_to_sample_mean(self):
        """With many observations, posterior mean approaches the sample mean."""
        t = _NormalGammaTracker()
        values = [0.5] * 200
        for v in values:
            t.update(v)
        # With 200 updates, kappa=201, mu_n ≈ 200*0.5/201 ≈ 0.4975
        # Should be within 1% of true mean
        assert abs(t.posterior_mean - 0.5) < 0.01

    def test_zero_input_stays_at_prior(self):
        t = _NormalGammaTracker()
        for _ in range(10):
            t.update(0.0)
        assert abs(t.posterior_mean) < 1e-9


# ── Tests: BayesianAdaptiveCombinerConfig ────────────────────────────────


class TestBayesianAdaptiveCombinerConfig:
    def test_is_subclass_of_adaptive_config(self):
        cfg = BayesianAdaptiveCombinerConfig()
        assert isinstance(cfg, AdaptiveCombinerConfig)

    def test_inherits_defaults(self):
        cfg = BayesianAdaptiveCombinerConfig()
        assert cfg.ic_lookback == 126
        assert cfg.min_ic_periods == 20
        assert cfg.shrinkage == 0.3

    def test_custom_params_propagate(self):
        cfg = BayesianAdaptiveCombinerConfig(min_ic_periods=10, shrinkage=0.1)
        assert cfg.min_ic_periods == 10
        assert cfg.shrinkage == 0.1


# ── Tests: BayesianAdaptiveSignalCombiner ────────────────────────────────


class TestBayesianAdaptiveSignalCombiner:
    def test_default_construction(self):
        combiner = BayesianAdaptiveSignalCombiner()
        assert combiner._ng_trackers == {}

    def test_equal_fallback_before_warmup(self):
        combiner = BayesianAdaptiveSignalCombiner(
            BayesianAdaptiveCombinerConfig(min_ic_periods=30)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=5)
        _feed_combiner(combiner, "sig", scores, returns, timestamps)
        aw = combiner.get_weights(["sig"])
        assert aw.method_used == "equal_fallback"

    def test_bayesian_ng_method_after_warmup(self):
        combiner = BayesianAdaptiveSignalCombiner(
            BayesianAdaptiveCombinerConfig(min_ic_periods=10)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=25)
        _feed_combiner(combiner, "sig", scores, returns, timestamps)
        aw = combiner.get_weights(["sig"])
        assert aw.method_used == "bayesian_ng"

    def test_weights_sum_to_one(self):
        combiner = BayesianAdaptiveSignalCombiner(
            BayesianAdaptiveCombinerConfig(min_ic_periods=10)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=30)
        _feed_combiner(combiner, "sig_a", scores, returns, timestamps)
        _feed_combiner(combiner, "sig_b", scores, returns, timestamps)
        aw = combiner.get_weights(["sig_a", "sig_b"])
        assert abs(sum(aw.weights.values()) - 1.0) < 1e-9

    def test_good_signal_outweighs_noise(self):
        combiner = BayesianAdaptiveSignalCombiner(
            BayesianAdaptiveCombinerConfig(min_ic_periods=10, shrinkage=0.0)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=25)
        _feed_combiner(combiner, "good", scores, returns, timestamps)
        noise_s, noise_r, noise_t = _make_noise_data(n_obs=25)
        _feed_combiner(combiner, "noise", noise_s, noise_r, noise_t)
        aw = combiner.get_weights(["good", "noise"])
        if aw.method_used == "bayesian_ng":
            assert aw.weights["good"] > aw.weights["noise"]

    def test_update_creates_ng_tracker(self):
        combiner = BayesianAdaptiveSignalCombiner()
        scores, returns, timestamps = _make_predictive_data(n_obs=1)
        combiner.update({"momentum": scores[0]}, returns[0], timestamps[0])
        assert "momentum" in combiner._ng_trackers

    def test_reset_clears_ng_trackers_and_history(self):
        combiner = BayesianAdaptiveSignalCombiner()
        scores, returns, timestamps = _make_predictive_data(n_obs=5)
        _feed_combiner(combiner, "sig", scores, returns, timestamps)
        assert len(combiner._ng_trackers) == 1
        combiner.reset()
        assert len(combiner._ng_trackers) == 0
        assert len(combiner.tracked_signals) == 0

    def test_full_shrinkage_equals_equal_weight(self):
        combiner = BayesianAdaptiveSignalCombiner(
            BayesianAdaptiveCombinerConfig(shrinkage=1.0, min_ic_periods=10)
        )
        scores, returns, timestamps = _make_predictive_data(n_obs=30)
        _feed_combiner(combiner, "a", scores, returns, timestamps)
        noise_s, noise_r, noise_t = _make_noise_data(n_obs=30)
        _feed_combiner(combiner, "b", noise_s, noise_r, noise_t)
        aw = combiner.get_weights(["a", "b"])
        assert abs(aw.weights["a"] - 0.5) < 1e-6
        assert abs(aw.weights["b"] - 0.5) < 1e-6

    def test_ng_posterior_is_conservative_vs_sample_mean_early(self):
        """NG posterior mean < sample mean during warmup (prior shrinkage)."""
        t = _NormalGammaTracker()
        for _ in range(5):
            t.update(0.8)
        sample_mean = 0.8
        # After 5 updates: kappa=6, mu_n = 5*0.8/6 ≈ 0.667
        assert t.posterior_mean < sample_mean

    def test_is_subclass_of_adaptive_combiner(self):
        combiner = BayesianAdaptiveSignalCombiner()
        assert isinstance(combiner, AdaptiveSignalCombiner)
