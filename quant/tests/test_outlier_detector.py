"""Tests for the statistical outlier detector (QUA-118)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.data.outlier_detector import (
    CleanMethod,
    OutlierConfig,
    OutlierDetector,
    OutlierReport,
    OutlierType,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_dates(n: int = 100) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-01", periods=n, freq="B")


def _clean_prices(n: int = 100, n_symbols: int = 3) -> pd.DataFrame:
    """Generate clean price data with realistic random walks."""
    rng = np.random.default_rng(42)
    dates = _make_dates(n)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    prices = {}
    for sym in symbols:
        returns = rng.normal(0.0005, 0.015, size=n)
        prices[sym] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(prices, index=dates)


def _clean_volumes(n: int = 100, n_symbols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = _make_dates(n)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    volumes = {}
    for sym in symbols:
        volumes[sym] = rng.lognormal(mean=14.0, sigma=0.5, size=n)
    return pd.DataFrame(volumes, index=dates)


# ── Clean data → no outliers ────────────────────────────────────────────────


class TestCleanData:
    def test_clean_data_no_outliers(self):
        prices = _clean_prices(100)
        detector = OutlierDetector()
        report = detector.scan(prices)
        # Clean data should have zero or very few outliers
        assert report.outlier_rate < 0.01

    def test_empty_dataframe(self):
        prices = pd.DataFrame(dtype=float)
        detector = OutlierDetector()
        report = detector.scan(prices)
        assert report.n_outliers == 0
        assert not report.has_outliers

    def test_single_column(self):
        dates = _make_dates(50)
        rng = np.random.default_rng(7)
        prices = pd.DataFrame(
            {"A": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 50)))},
            index=dates,
        )
        detector = OutlierDetector()
        report = detector.scan(prices)
        # Single column can't do cross-sectional checks (needs >=3)
        assert report.n_observations > 0

    def test_short_history_skips_checks(self):
        dates = _make_dates(10)
        prices = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=dates)
        detector = OutlierDetector(OutlierConfig(min_history=30))
        report = detector.scan(prices)
        assert report.n_outliers == 0


# ── Return Z-score detection ────────────────────────────────────────────────


class TestReturnZscore:
    def test_spike_detected(self):
        prices = _clean_prices(100, n_symbols=1)
        # Inject a 30% spike on day 70
        prices.iloc[70, 0] *= 1.30
        detector = OutlierDetector(OutlierConfig(return_z_threshold=3.0))
        report = detector.scan(prices)
        z_outliers = [o for o in report.outliers if o.outlier_type == OutlierType.RETURN_ZSCORE]
        assert len(z_outliers) >= 1
        # Should flag day 70 (the spike) or day 71 (the reversal)
        flagged_days = {o.date for o in z_outliers}
        assert prices.index[70] in flagged_days or prices.index[71] in flagged_days

    def test_crash_detected(self):
        prices = _clean_prices(100, n_symbols=1)
        # Inject a 25% crash on day 60
        prices.iloc[60, 0] *= 0.75
        detector = OutlierDetector(OutlierConfig(return_z_threshold=3.0))
        report = detector.scan(prices)
        z_outliers = [o for o in report.outliers if o.outlier_type == OutlierType.RETURN_ZSCORE]
        assert len(z_outliers) >= 1

    def test_tight_threshold_flags_more(self):
        prices = _clean_prices(200, n_symbols=1)
        prices.iloc[80, 0] *= 1.10  # 10% bump
        loose = OutlierDetector(OutlierConfig(return_z_threshold=5.0))
        tight = OutlierDetector(OutlierConfig(return_z_threshold=2.0))
        r_loose = loose.scan(prices)
        r_tight = tight.scan(prices)
        assert r_tight.n_outliers >= r_loose.n_outliers


# ── Hampel filter ────────────────────────────────────────────────────────────


class TestHampel:
    def test_price_jump_detected(self):
        prices = _clean_prices(100, n_symbols=1)
        # Inject a sudden jump that stays (not just a spike in returns)
        original = prices.iloc[50, 0]
        prices.iloc[50, 0] = original * 3.0  # 3x the price for one day
        detector = OutlierDetector(OutlierConfig(hampel_threshold=3.0))
        report = detector.scan(prices)
        hampel = [o for o in report.outliers if o.outlier_type == OutlierType.HAMPEL]
        assert len(hampel) >= 1
        assert any(o.date == prices.index[50] for o in hampel)

    def test_gradual_trend_not_flagged(self):
        dates = _make_dates(100)
        # Steady uptrend — no outliers expected
        prices = pd.DataFrame(
            {"A": np.linspace(100, 200, 100)},
            index=dates,
        )
        detector = OutlierDetector()
        report = detector.scan(prices)
        hampel = [o for o in report.outliers if o.outlier_type == OutlierType.HAMPEL]
        assert len(hampel) == 0


# ── Volume spike detection ──────────────────────────────────────────────────


class TestVolumeSpike:
    def test_volume_spike_detected(self):
        prices = _clean_prices(100, n_symbols=1)
        volumes = _clean_volumes(100, n_symbols=1)
        # Inject a 20x volume spike
        volumes.iloc[70, 0] *= 20.0
        detector = OutlierDetector(OutlierConfig(volume_spike_multiple=10.0))
        report = detector.scan(prices, volumes)
        vol_outliers = [o for o in report.outliers if o.outlier_type == OutlierType.VOLUME_SPIKE]
        assert len(vol_outliers) >= 1
        assert any(o.date == volumes.index[70] for o in vol_outliers)

    def test_normal_volume_not_flagged(self):
        prices = _clean_prices(100, n_symbols=1)
        volumes = _clean_volumes(100, n_symbols=1)
        detector = OutlierDetector(OutlierConfig(volume_spike_multiple=10.0))
        report = detector.scan(prices, volumes)
        vol_outliers = [o for o in report.outliers if o.outlier_type == OutlierType.VOLUME_SPIKE]
        assert len(vol_outliers) == 0

    def test_no_volumes_no_crash(self):
        prices = _clean_prices(100)
        detector = OutlierDetector()
        report = detector.scan(prices, volumes=None)
        vol_outliers = [o for o in report.outliers if o.outlier_type == OutlierType.VOLUME_SPIKE]
        assert len(vol_outliers) == 0


# ── Stale price detection ───────────────────────────────────────────────────


class TestStalePrice:
    def test_stale_price_detected(self):
        dates = _make_dates(50)
        prices = pd.DataFrame(
            {"A": np.concatenate([
                np.linspace(100, 110, 20),
                np.full(10, 110.0),  # 10 days unchanged
                np.linspace(110, 120, 20),
            ])},
            index=dates,
        )
        detector = OutlierDetector(OutlierConfig(stale_price_days=5))
        report = detector.scan(prices)
        stale = [o for o in report.outliers if o.outlier_type == OutlierType.STALE_PRICE]
        assert len(stale) >= 1
        assert stale[0].symbol == "A"

    def test_stale_mask_covers_run(self):
        dates = _make_dates(30)
        prices = pd.DataFrame(
            {"A": np.concatenate([
                np.linspace(100, 104, 10),
                np.full(8, 105.0),  # 8 days unchanged at different value
                np.linspace(106, 110, 12),
            ])},
            index=dates,
        )
        detector = OutlierDetector(OutlierConfig(stale_price_days=5))
        report = detector.scan(prices)
        # All 8 stale days should be flagged in the mask
        assert report.flagged_mask is not None
        flagged_count = report.flagged_mask["A"].sum()
        assert flagged_count == 8

    def test_short_run_not_flagged(self):
        dates = _make_dates(30)
        prices = pd.DataFrame(
            {"A": np.concatenate([
                np.linspace(100, 104, 15),
                np.full(3, 105.0),  # 3 days — below threshold of 5
                np.linspace(106, 110, 12),
            ])},
            index=dates,
        )
        detector = OutlierDetector(OutlierConfig(stale_price_days=5))
        report = detector.scan(prices)
        stale = [o for o in report.outliers if o.outlier_type == OutlierType.STALE_PRICE]
        assert len(stale) == 0


# ── Cross-sectional detection ───────────────────────────────────────────────


class TestCrossSectional:
    def test_single_stock_spike_detected(self):
        prices = _clean_prices(100, n_symbols=20)
        # One stock spikes 500% while others are normal — clearly extreme
        # vs the cross-section (with 20 symbols, the z-score will be high)
        prices.iloc[70, 2] *= 5.0
        detector = OutlierDetector(OutlierConfig(cross_sectional_z=3.0))
        report = detector.scan(prices)
        cs = [o for o in report.outliers if o.outlier_type == OutlierType.CROSS_SECTIONAL]
        assert len(cs) >= 1

    def test_uniform_move_not_flagged(self):
        # All stocks move together (e.g. market-wide event)
        dates = _make_dates(100)
        rng = np.random.default_rng(42)
        market_returns = rng.normal(0.0005, 0.01, size=100)
        prices = {}
        for i in range(5):
            stock_specific = rng.normal(0, 0.002, size=100)
            prices[f"S{i}"] = 100.0 * np.exp(np.cumsum(market_returns + stock_specific))
        df = pd.DataFrame(prices, index=dates)
        detector = OutlierDetector(OutlierConfig(cross_sectional_z=4.0))
        report = detector.scan(df)
        cs = [o for o in report.outliers if o.outlier_type == OutlierType.CROSS_SECTIONAL]
        assert len(cs) == 0

    def test_too_few_symbols_skips_check(self):
        prices = _clean_prices(100, n_symbols=2)
        prices.iloc[70, 0] *= 1.50
        detector = OutlierDetector()
        report = detector.scan(prices)
        cs = [o for o in report.outliers if o.outlier_type == OutlierType.CROSS_SECTIONAL]
        # With only 2 symbols, cross-sectional check is skipped
        assert len(cs) == 0


# ── Report methods ──────────────────────────────────────────────────────────


class TestReport:
    def test_by_type_grouping(self):
        prices = _clean_prices(100, n_symbols=3)
        dates = _make_dates(100)
        volumes = _clean_volumes(100, n_symbols=3)
        # Inject both a return spike and a volume spike
        prices.iloc[70, 0] *= 1.30
        volumes.iloc[80, 1] *= 25.0
        detector = OutlierDetector(OutlierConfig(return_z_threshold=3.0))
        report = detector.scan(prices, volumes)
        by_type = report.by_type()
        assert isinstance(by_type, dict)

    def test_by_symbol_grouping(self):
        prices = _clean_prices(100, n_symbols=3)
        prices.iloc[70, 0] *= 1.30
        detector = OutlierDetector(OutlierConfig(return_z_threshold=3.0))
        report = detector.scan(prices)
        by_sym = report.by_symbol()
        assert isinstance(by_sym, dict)
        if report.has_outliers:
            assert all(isinstance(k, str) for k in by_sym)

    def test_summary_string(self):
        prices = _clean_prices(100)
        prices.iloc[70, 0] *= 1.50
        detector = OutlierDetector(OutlierConfig(return_z_threshold=3.0))
        report = detector.scan(prices)
        summary = report.summary()
        assert "Outlier scan" in summary
        assert "observations" in summary

    def test_empty_report_summary(self):
        report = OutlierReport()
        assert not report.has_outliers
        assert report.outlier_rate == 0.0
        assert "0 outliers" in report.summary()


# ── Cleaned prices ──────────────────────────────────────────────────────────


class TestCleanedPrices:
    def _get_flagged_report(self) -> tuple[pd.DataFrame, OutlierReport]:
        prices = _clean_prices(100, n_symbols=1)
        prices.iloc[70, 0] *= 2.0  # huge spike
        detector = OutlierDetector(OutlierConfig(
            return_z_threshold=3.0,
            hampel_threshold=3.0,
        ))
        report = detector.scan(prices)
        return prices, report

    def test_nan_cleaning(self):
        prices, report = self._get_flagged_report()
        if not report.has_outliers:
            pytest.skip("No outliers detected for this seed")
        clean = report.cleaned_prices(prices, method="nan")
        # Flagged cells should be NaN
        assert clean.isna().any().any()
        # Non-flagged cells should be unchanged
        not_flagged = ~report.flagged_mask
        pd.testing.assert_frame_equal(
            clean[not_flagged].dropna(how="all"),
            prices[not_flagged].dropna(how="all"),
        )

    def test_ffill_cleaning(self):
        prices, report = self._get_flagged_report()
        if not report.has_outliers:
            pytest.skip("No outliers detected for this seed")
        clean = report.cleaned_prices(prices, method="ffill")
        # After ffill, there should be fewer NaNs than nan method
        nan_count = clean.isna().sum().sum()
        # ffill should fill forward, so at most the first row might be NaN
        assert nan_count <= len(prices.columns)

    def test_median_cleaning(self):
        prices, report = self._get_flagged_report()
        if not report.has_outliers:
            pytest.skip("No outliers detected for this seed")
        clean = report.cleaned_prices(prices, method="median")
        # Cleaned values should not be NaN (replaced with median)
        assert not clean.isna().any().any()

    def test_clean_report_returns_original(self):
        prices = _clean_prices(50, n_symbols=1)
        detector = OutlierDetector()
        report = detector.scan(prices)
        clean = report.cleaned_prices(prices, method="nan")
        pd.testing.assert_frame_equal(clean, prices)

    def test_string_method_accepted(self):
        prices, report = self._get_flagged_report()
        # String should work as well as enum
        clean = report.cleaned_prices(prices, method="nan")
        assert isinstance(clean, pd.DataFrame)


# ── Config validation ────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        cfg = OutlierConfig()
        assert cfg.return_z_threshold == 4.0
        assert cfg.hampel_window == 21
        assert cfg.volume_spike_multiple == 10.0
        assert cfg.stale_price_days == 5
        assert cfg.cross_sectional_z == 4.0

    def test_custom_config(self):
        cfg = OutlierConfig(
            return_z_threshold=3.0,
            stale_price_days=10,
        )
        assert cfg.return_z_threshold == 3.0
        assert cfg.stale_price_days == 10

    def test_detector_exposes_config(self):
        cfg = OutlierConfig(return_z_threshold=5.0)
        detector = OutlierDetector(cfg)
        assert detector.config.return_z_threshold == 5.0
