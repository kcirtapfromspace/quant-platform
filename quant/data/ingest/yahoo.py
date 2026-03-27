"""Yahoo Finance data source via yfinance."""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Sequence

import pandas as pd
import yfinance as yf
from loguru import logger

from quant.data.ingest.base import DataSource, OHLCVRecord

# yfinance rate limit: burst up to ~2000 req/hour; we stay well under
_BATCH_SIZE = 100
_BATCH_DELAY_SEC = 0.5


class YahooFinanceSource(DataSource):
    """Fetches adjusted OHLCV data from Yahoo Finance via yfinance.

    Uses batch downloads to minimise round trips. Splits large symbol lists
    into chunks of _BATCH_SIZE to respect Yahoo's informal rate limits.
    """

    @property
    def name(self) -> str:
        return "yahoo_finance"

    def fetch(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> list[OHLCVRecord]:
        records: list[OHLCVRecord] = []
        symbols = list(symbols)

        # yfinance end date is exclusive — add one day
        end_exclusive = end + timedelta(days=1)
        start_str = start.isoformat()
        end_str = end_exclusive.isoformat()

        for i in range(0, len(symbols), _BATCH_SIZE):
            chunk = symbols[i : i + _BATCH_SIZE]
            logger.debug(
                "Fetching {} symbols from Yahoo Finance ({}/{})",
                len(chunk),
                i + len(chunk),
                len(symbols),
            )
            chunk_records = self._fetch_chunk(chunk, start_str, end_str)
            records.extend(chunk_records)

            if i + _BATCH_SIZE < len(symbols):
                time.sleep(_BATCH_DELAY_SEC)

        records.sort(key=lambda r: (r.symbol, r.date))
        return records

    def _fetch_chunk(
        self,
        symbols: list[str],
        start_str: str,
        end_str: str,
    ) -> list[OHLCVRecord]:
        try:
            raw: pd.DataFrame = yf.download(
                tickers=symbols,
                start=start_str,
                end=end_str,
                auto_adjust=True,  # prices reflect splits/dividends
                actions=False,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.error("yfinance download failed for {}: {}", symbols, exc)
            return []

        if raw.empty:
            logger.warning("No data returned for symbols: {}", symbols)
            return []

        return self._parse_raw(raw, symbols)

    def _parse_raw(
        self,
        raw: pd.DataFrame,
        symbols: list[str],
    ) -> list[OHLCVRecord]:
        records: list[OHLCVRecord] = []

        # Single symbol: flat columns; multiple symbols: MultiIndex columns
        if len(symbols) == 1:
            sym = symbols[0]
            raw = raw.copy()
            raw.columns = [c.lower() for c in raw.columns]
            for ts, row in raw.iterrows():
                rec = self._row_to_record(sym, ts, row)
                if rec is not None:
                    records.append(rec)
        else:
            for sym in symbols:
                if sym not in raw.columns.get_level_values(1):
                    logger.warning("Symbol {} not found in Yahoo response", sym)
                    continue
                sym_df = raw.xs(sym, axis=1, level=1).copy()
                sym_df.columns = [c.lower() for c in sym_df.columns]
                for ts, row in sym_df.iterrows():
                    rec = self._row_to_record(sym, ts, row)
                    if rec is not None:
                        records.append(rec)

        return records

    @staticmethod
    def _row_to_record(
        symbol: str,
        ts: pd.Timestamp,
        row: pd.Series,
    ) -> OHLCVRecord | None:
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(row.index)):
            return None
        if pd.isna(row["close"]) or pd.isna(row["open"]):
            return None

        return OHLCVRecord(
            symbol=symbol.upper(),
            date=ts.date(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0) or 0),
            adj_close=float(row["close"]),  # auto_adjust=True so close IS adj_close
        )
