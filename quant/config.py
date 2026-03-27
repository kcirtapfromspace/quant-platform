"""Configuration and universe loading for the quant platform.

Reads from environment variables (or .env file) and the universe file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Default S&P 500-like universe for development — small set to stay within free-tier limits
_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "BRK-B", "JPM", "V", "JNJ",
    "UNH", "XOM", "PG", "MA", "HD",
    "CVX", "LLY", "ABBV", "MRK", "AVGO",
    "SPY", "QQQ", "IWM",  # broad ETFs for factor proxies
]

# Path overrides (env vars take precedence)
_DB_PATH_DEFAULT = Path.home() / ".quant" / "market.duckdb"


def get_db_path() -> str:
    return os.environ.get("QUANT_DB_PATH", str(_DB_PATH_DEFAULT))


def load_universe() -> list[str]:
    """Load the symbol universe.

    Checks QUANT_UNIVERSE_FILE env var first. Falls back to _DEFAULT_UNIVERSE.
    The file should have one symbol per line, with optional # comments.
    """
    universe_file = os.environ.get("QUANT_UNIVERSE_FILE")
    if universe_file:
        path = Path(universe_file)
        if not path.exists():
            raise FileNotFoundError(f"Universe file not found: {universe_file}")
        symbols = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.upper())
        return symbols

    return list(_DEFAULT_UNIVERSE)
