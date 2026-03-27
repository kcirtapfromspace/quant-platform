"""Data quality checks for OHLCV records.

Validates structural integrity and business rules for market data. Returns
a ValidationResult so callers can decide how to handle failures (warn vs.
reject). Does not mutate records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from loguru import logger

from quant.data.ingest.base import OHLCVRecord


@dataclass
class ValidationIssue:
    symbol: str
    date: date
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.symbol} {self.date}] {self.rule}: {self.detail}"


@dataclass
class ValidationResult:
    valid: list[OHLCVRecord] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    def summary(self) -> str:
        return (
            f"{len(self.valid)} valid, {len(self.issues)} issues"
        )


def validate(records: Sequence[OHLCVRecord]) -> ValidationResult:
    """Run all data quality rules over *records*.

    Rules applied (in order):
    1. No null/NaN fields
    2. OHLC positivity
    3. High >= max(Open, Close) and Low <= min(Open, Close)
    4. Volume >= 0
    5. Adj_close > 0
    6. Duplicate (symbol, date) detection
    7. Extreme price change detection (>50% single-day move flagged as warning)
    """
    result = ValidationResult()
    seen: set[tuple[str, date]] = set()

    # Group by symbol for inter-record checks
    by_symbol: dict[str, list[OHLCVRecord]] = {}
    for rec in records:
        by_symbol.setdefault(rec.symbol, []).append(rec)

    for rec in records:
        issues = _check_record(rec, seen)
        if issues:
            result.issues.extend(issues)
        else:
            result.valid.append(rec)
        seen.add((rec.symbol, rec.date))

    # Inter-record: flag extreme moves per symbol
    for sym, sym_records in by_symbol.items():
        sym_records.sort(key=lambda r: r.date)
        for prev, curr in zip(sym_records, sym_records[1:]):
            if prev.close > 0:
                pct = abs(curr.close - prev.close) / prev.close
                if pct > 0.5:
                    logger.warning(
                        "{} {}: extreme price change {:.1%} (prev close={:.4f}, curr close={:.4f})",
                        sym,
                        curr.date,
                        pct,
                        prev.close,
                        curr.close,
                    )

    if result.issues:
        logger.warning(
            "Validation: {} issues in {} records", len(result.issues), len(records)
        )
        for issue in result.issues[:10]:  # log up to 10 for brevity
            logger.warning("  {}", issue)
    else:
        logger.debug("Validation passed for {} records", len(records))

    return result


def _check_record(
    rec: OHLCVRecord,
    seen: set[tuple[str, date]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(rule: str, detail: str) -> None:
        issues.append(ValidationIssue(rec.symbol, rec.date, rule, detail))

    # 1. Nullability (Python floats can be NaN from pandas)
    import math

    for field_name in ("open", "high", "low", "close", "volume", "adj_close"):
        val = getattr(rec, field_name)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            add("null_field", f"{field_name} is null/NaN")

    if issues:  # can't proceed with OHLC checks if nulls
        return issues

    # 2. Positivity
    for field_name in ("open", "high", "low", "close"):
        val = getattr(rec, field_name)
        if val <= 0:
            add("non_positive_price", f"{field_name}={val}")

    # 3. OHLC relationships
    if rec.high < rec.open or rec.high < rec.close:
        add(
            "ohlc_high_violation",
            f"high={rec.high} < max(open={rec.open}, close={rec.close})",
        )
    if rec.low > rec.open or rec.low > rec.close:
        add(
            "ohlc_low_violation",
            f"low={rec.low} > min(open={rec.open}, close={rec.close})",
        )

    # 4. Volume
    if rec.volume < 0:
        add("negative_volume", f"volume={rec.volume}")

    # 5. Adj close
    if rec.adj_close <= 0:
        add("non_positive_adj_close", f"adj_close={rec.adj_close}")

    # 6. Duplicate
    if (rec.symbol, rec.date) in seen:
        add("duplicate", f"(symbol={rec.symbol}, date={rec.date}) already seen")

    return issues
