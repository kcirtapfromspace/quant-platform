"""Tests for data validation rules."""
from datetime import date

import pytest

from quant.data.ingest.base import OHLCVRecord
from quant.data.validation import validate


def _record(**kwargs) -> OHLCVRecord:
    defaults = dict(
        symbol="AAPL",
        date=date(2024, 1, 2),
        open=185.0,
        high=190.0,
        low=183.0,
        close=188.0,
        volume=50_000_000.0,
        adj_close=188.0,
    )
    defaults.update(kwargs)
    return OHLCVRecord(**defaults)


def test_valid_record_passes():
    result = validate([_record()])
    assert result.ok
    assert len(result.valid) == 1


def test_zero_close_fails():
    result = validate([_record(close=0.0, adj_close=0.0)])
    assert not result.ok
    rules = [i.rule for i in result.issues]
    assert "non_positive_price" in rules


def test_negative_volume_fails():
    result = validate([_record(volume=-1.0)])
    assert not result.ok
    assert any(i.rule == "negative_volume" for i in result.issues)


def test_high_below_close_fails():
    result = validate([_record(high=180.0, close=188.0)])
    assert not result.ok
    assert any(i.rule == "ohlc_high_violation" for i in result.issues)


def test_low_above_open_fails():
    result = validate([_record(low=190.0, open=185.0)])
    assert not result.ok
    assert any(i.rule == "ohlc_low_violation" for i in result.issues)


def test_duplicate_record_fails():
    rec = _record()
    result = validate([rec, rec])
    assert not result.ok
    assert any(i.rule == "duplicate" for i in result.issues)


def test_multiple_symbols_no_false_duplicate():
    r1 = _record(symbol="AAPL", date=date(2024, 1, 2))
    r2 = _record(symbol="MSFT", date=date(2024, 1, 2))
    result = validate([r1, r2])
    assert result.ok


def test_nan_field_fails():
    import math
    result = validate([_record(close=math.nan, adj_close=math.nan)])
    assert not result.ok
    assert any(i.rule == "null_field" for i in result.issues)


def test_summary_string():
    result = validate([_record()])
    assert "valid" in result.summary()
