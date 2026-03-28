"""Tests for signal correlation monitoring (QUA-93)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.signal_correlation import (
    ConvergenceEvent,
    CorrelationConfig,
    SignalCluster,
    SignalCorrelationMonitor,
    SignalCorrelationResult,
    SignalPair,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DATES = 200
DATES = pd.bdate_range("2023-01-01", periods=N_DATES)


def _make_independent_signals(n: int = 4, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create independent random signals (low cross-correlation)."""
    rng = np.random.default_rng(seed)
    symbols = [f"S{i}" for i in range(10)]
    return {
        f"sig_{i}": pd.DataFrame(
            rng.normal(0, 1, (N_DATES, len(symbols))),
            index=DATES, columns=symbols,
        )
        for i in range(n)
    }


def _make_correlated_signals(n: int = 4, corr: float = 0.90, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create signals that are highly correlated with each other."""
    rng = np.random.default_rng(seed)
    symbols = [f"S{i}" for i in range(10)]
    base = pd.DataFrame(
        rng.normal(0, 1, (N_DATES, len(symbols))),
        index=DATES, columns=symbols,
    )
    signals = {}
    for i in range(n):
        noise = pd.DataFrame(
            rng.normal(0, 1, (N_DATES, len(symbols))),
            index=DATES, columns=symbols,
        )
        signals[f"sig_{i}"] = base * corr + noise * (1 - corr)
    return signals


def _make_df_signals(seed: int = 42) -> pd.DataFrame:
    """Create a simple DataFrame of signal columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 1, (N_DATES, 4)),
        index=DATES,
        columns=["alpha", "beta", "gamma", "delta"],
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result_from_dict(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals())
        assert isinstance(result, SignalCorrelationResult)

    def test_returns_result_from_dataframe(self):
        result = SignalCorrelationMonitor().analyze(_make_df_signals())
        assert isinstance(result, SignalCorrelationResult)

    def test_n_signals(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals(4))
        assert result.n_signals == 4

    def test_n_dates(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals())
        assert result.n_dates == N_DATES

    def test_pairs_populated(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals(4))
        # 4 choose 2 = 6 pairs
        assert len(result.pairs) == 6

    def test_pair_types(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals())
        for p in result.pairs:
            assert isinstance(p, SignalPair)


# ---------------------------------------------------------------------------
# Independent signals
# ---------------------------------------------------------------------------


class TestIndependentSignals:
    def test_low_avg_correlation(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals())
        assert abs(result.current_avg_corr) < 0.30

    def test_high_effective_count(self):
        """Independent signals should have effective count near N."""
        result = SignalCorrelationMonitor().analyze(_make_independent_signals(4))
        assert result.effective_signal_count > 3.0

    def test_not_converged(self):
        result = SignalCorrelationMonitor(
            CorrelationConfig(convergence_threshold=0.50),
        ).analyze(_make_independent_signals())
        assert not result.is_converged

    def test_no_clusters(self):
        result = SignalCorrelationMonitor(
            CorrelationConfig(cluster_threshold=0.70),
        ).analyze(_make_independent_signals())
        assert len(result.clusters) == 0


# ---------------------------------------------------------------------------
# Correlated signals
# ---------------------------------------------------------------------------


class TestCorrelatedSignals:
    def test_high_avg_correlation(self):
        result = SignalCorrelationMonitor().analyze(
            _make_correlated_signals(corr=0.90),
        )
        assert result.current_avg_corr > 0.50

    def test_low_effective_count(self):
        result = SignalCorrelationMonitor().analyze(
            _make_correlated_signals(4, corr=0.95),
        )
        assert result.effective_signal_count < 3.0

    def test_is_converged(self):
        result = SignalCorrelationMonitor(
            CorrelationConfig(convergence_threshold=0.40),
        ).analyze(_make_correlated_signals(corr=0.90))
        assert result.is_converged

    def test_clusters_formed(self):
        result = SignalCorrelationMonitor(
            CorrelationConfig(cluster_threshold=0.40),
        ).analyze(_make_correlated_signals(4, corr=0.90))
        assert len(result.clusters) >= 1

    def test_cluster_types(self):
        result = SignalCorrelationMonitor(
            CorrelationConfig(cluster_threshold=0.40),
        ).analyze(_make_correlated_signals(4, corr=0.90))
        for cl in result.clusters:
            assert isinstance(cl, SignalCluster)
            assert len(cl.signals) >= 2


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------


class TestRolling:
    def test_rolling_avg_populated(self):
        result = SignalCorrelationMonitor(
            CorrelationConfig(window=30, min_periods=20),
        ).analyze(_make_independent_signals())
        assert len(result.rolling_avg_corr) > 0

    def test_rolling_avg_is_series(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals())
        assert isinstance(result.rolling_avg_corr, pd.Series)


# ---------------------------------------------------------------------------
# Convergence events
# ---------------------------------------------------------------------------


class TestConvergenceEvents:
    def test_convergence_detected(self):
        """Correlated signals should trigger convergence events."""
        result = SignalCorrelationMonitor(
            CorrelationConfig(
                window=30, min_periods=20,
                convergence_threshold=0.40,
            ),
        ).analyze(_make_correlated_signals(corr=0.90))
        assert len(result.convergence_events) >= 1

    def test_event_types(self):
        result = SignalCorrelationMonitor(
            CorrelationConfig(
                window=30, min_periods=20,
                convergence_threshold=0.40,
            ),
        ).analyze(_make_correlated_signals(corr=0.90))
        for ev in result.convergence_events:
            assert isinstance(ev, ConvergenceEvent)
            assert ev.duration_days >= 1
            assert ev.peak_avg_correlation >= 0.40

    def test_no_convergence_for_independent(self):
        result = SignalCorrelationMonitor(
            CorrelationConfig(convergence_threshold=0.80),
        ).analyze(_make_independent_signals())
        assert len(result.convergence_events) == 0


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


class TestCorrelationMatrix:
    def test_matrix_populated(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals(3))
        # 3 choose 2 = 3 pairs
        assert len(result.correlation_matrix) == 3

    def test_values_in_range(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals())
        for c in result.correlation_matrix.values():
            assert -1.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_two_signals_minimum(self):
        sigs = _make_independent_signals(2)
        result = SignalCorrelationMonitor().analyze(sigs)
        assert result.n_signals == 2
        assert len(result.pairs) == 1

    def test_single_signal_raises(self):
        sigs = _make_independent_signals(1)
        with pytest.raises(ValueError, match="at least 2"):
            SignalCorrelationMonitor().analyze(sigs)

    def test_many_signals(self):
        sigs = _make_independent_signals(8)
        result = SignalCorrelationMonitor().analyze(sigs)
        # 8 choose 2 = 28 pairs
        assert len(result.pairs) == 28


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals())
        summary = result.summary()
        assert "Signal Correlation Monitor" in summary
        assert "avg correlation" in summary.lower()
        assert "Effective signal count" in summary

    def test_summary_shows_convergence_status(self):
        result = SignalCorrelationMonitor().analyze(_make_independent_signals())
        summary = result.summary()
        assert "converged" in summary.lower()
