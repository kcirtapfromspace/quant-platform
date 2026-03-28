"""Universe configuration: 100 S&P 500 liquid names + 10 ETF overlays.

Curated selection criteria:
  - 20-day ADTV > $50M (all names easily exceed this)
  - Float market cap > $5B
  - Sector-balanced across all 11 GICS sectors
  - Stable S&P 500 constituents with 8+ year history

ETF overlays provide broad macro/sector exposure for regime detection:
  SPY, QQQ, IWM  — broad market (large/growth/small)
  XLF, XLE, XLK, XLV, XLI — sector ETFs (Finance, Energy, Tech, Health, Industrial)
  TLT, GLD       — macro overlays (long-duration rates, gold)

Usage::

    from quant.data.config import SP500_UNIVERSE, ETF_OVERLAYS, FULL_UNIVERSE, SECTOR_MAP

    symbols = FULL_UNIVERSE          # 110 symbols
    sector = SECTOR_MAP["AAPL"]      # "Information Technology"
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# ETF overlays (10)
# ---------------------------------------------------------------------------

ETF_OVERLAYS: list[str] = [
    "SPY",  # S&P 500 broad market
    "QQQ",  # Nasdaq-100 / tech-growth
    "IWM",  # Russell 2000 small-cap
    "XLF",  # Financials sector
    "XLE",  # Energy sector
    "XLK",  # Technology sector
    "XLV",  # Health Care sector
    "XLI",  # Industrials sector
    "TLT",  # 20+ Year Treasury (rates macro)
    "GLD",  # Gold (inflation/crisis hedge)
]

# ---------------------------------------------------------------------------
# S&P 500 liquid universe: 100 names, sector-balanced across 11 GICS sectors
# ---------------------------------------------------------------------------
# Approximately 9-10 names per sector; slightly more for IT and Health Care
# which dominate S&P 500 weights.

SP500_UNIVERSE: list[str] = [
    # ── Information Technology (18) ────────────────────────────────────────
    "AAPL",   # Apple — mega-cap, consumer hardware/services
    "MSFT",   # Microsoft — cloud, enterprise software
    "NVDA",   # Nvidia — GPU/AI semiconductors
    "AVGO",   # Broadcom — networking semiconductors
    "ORCL",   # Oracle — database, cloud infrastructure
    "CRM",    # Salesforce — CRM/SaaS
    "CSCO",   # Cisco — enterprise networking
    "INTU",   # Intuit — financial software
    "IBM",    # IBM — enterprise IT/AI
    "ACN",    # Accenture — IT consulting
    "TXN",    # Texas Instruments — analog semiconductors
    "QCOM",   # Qualcomm — mobile/wireless semiconductors
    "ADI",    # Analog Devices — mixed-signal semiconductors
    "AMAT",   # Applied Materials — semiconductor equipment
    "LRCX",   # Lam Research — semiconductor equipment
    "KLAC",   # KLA Corp — semiconductor process control
    "ADP",    # ADP — HR/payroll software
    "CDNS",   # Cadence Design — EDA software
    # ── Health Care (12) ───────────────────────────────────────────────────
    "JNJ",    # Johnson & Johnson — diversified pharma/medtech
    "UNH",    # UnitedHealth — managed care/insurance
    "LLY",    # Eli Lilly — pharma (GLP-1 drugs)
    "ABBV",   # AbbVie — biopharmaceuticals
    "MRK",    # Merck — pharma
    "TMO",    # Thermo Fisher — life science tools
    "ABT",    # Abbott Labs — medical devices/diagnostics
    "DHR",    # Danaher — life science instruments
    "SYK",    # Stryker — orthopedic devices
    "ISRG",   # Intuitive Surgical — robotic surgery
    "EW",     # Edwards Lifesciences — cardiac devices
    "IDXX",   # Idexx Labs — veterinary diagnostics
    # ── Consumer Discretionary (10) ────────────────────────────────────────
    "AMZN",   # Amazon — e-commerce/cloud (consumer segment)
    "TSLA",   # Tesla — EVs/energy
    "HD",     # Home Depot — home improvement retail
    "MCD",    # McDonald's — fast food
    "NKE",    # Nike — athletic footwear/apparel
    "LOW",    # Lowe's — home improvement
    "TJX",    # TJX Companies — off-price retail
    "SBUX",   # Starbucks — coffee retail
    "BKNG",   # Booking Holdings — online travel
    "MAR",    # Marriott — lodging
    # ── Financials (10) ────────────────────────────────────────────────────
    "JPM",    # JPMorgan Chase — universal bank
    "BAC",    # Bank of America — universal bank
    "WFC",    # Wells Fargo — commercial bank
    "GS",     # Goldman Sachs — investment bank
    "MS",     # Morgan Stanley — investment bank/wealth
    "BLK",    # BlackRock — asset management
    "AXP",    # American Express — payments/card
    "V",      # Visa — payments network
    "MA",     # Mastercard — payments network
    "COF",    # Capital One — consumer credit
    # ── Communication Services (8) ─────────────────────────────────────────
    "GOOGL",  # Alphabet Class A — search/cloud/ads
    "META",   # Meta Platforms — social media/ads
    "NFLX",   # Netflix — streaming
    "DIS",    # Disney — entertainment/parks
    "CMCSA",  # Comcast — cable/broadband/NBCUniversal
    "T",      # AT&T — telecom
    "VZ",     # Verizon — telecom
    "TMUS",   # T-Mobile — wireless telecom
    # ── Industrials (9) ────────────────────────────────────────────────────
    "GE",     # GE Aerospace — aerospace engines
    "RTX",    # RTX (Raytheon) — defense/aerospace
    "HON",    # Honeywell — industrial conglomerate
    "CAT",    # Caterpillar — heavy machinery
    "UPS",    # UPS — logistics/delivery
    "FDX",    # FedEx — logistics
    "LMT",    # Lockheed Martin — defense
    "BA",     # Boeing — commercial/defense aerospace
    "DE",     # Deere — agricultural equipment
    # ── Consumer Staples (8) ───────────────────────────────────────────────
    "PG",     # Procter & Gamble — household products
    "KO",     # Coca-Cola — beverages
    "PEP",    # PepsiCo — beverages/snacks
    "WMT",    # Walmart — mass retail
    "COST",   # Costco — warehouse retail
    "PM",     # Philip Morris — tobacco
    "MO",     # Altria — tobacco
    "CL",     # Colgate-Palmolive — personal care
    # ── Energy (7) ─────────────────────────────────────────────────────────
    "XOM",    # ExxonMobil — integrated oil & gas
    "CVX",    # Chevron — integrated oil & gas
    "COP",    # ConocoPhillips — E&P
    "EOG",    # EOG Resources — shale E&P
    "SLB",    # SLB (Schlumberger) — oilfield services
    "MPC",    # Marathon Petroleum — refining
    "PSX",    # Phillips 66 — refining/chemicals
    # ── Utilities (6) ──────────────────────────────────────────────────────
    "NEE",    # NextEra Energy — renewable utilities
    "DUK",    # Duke Energy — regulated electric utility
    "SO",     # Southern Company — regulated utility
    "AEP",    # American Electric Power — regulated utility
    "D",      # Dominion Energy — regulated utility
    "EXC",    # Exelon — nuclear/regulated utility
    # ── Real Estate (6) ────────────────────────────────────────────────────
    "PLD",    # Prologis — industrial REITs
    "AMT",    # American Tower — cell tower REIT
    "EQIX",   # Equinix — data center REIT
    "PSA",    # Public Storage — self-storage REIT
    "WELL",   # Welltower — healthcare REIT
    "DLR",    # Digital Realty — data center REIT
    # ── Materials (6) ──────────────────────────────────────────────────────
    "LIN",    # Linde — industrial gases
    "APD",    # Air Products — industrial gases
    "SHW",    # Sherwin-Williams — paints/coatings
    "FCX",    # Freeport-McMoRan — copper mining
    "NEM",    # Newmont — gold mining
    "NUE",    # Nucor — steel
]

# ---------------------------------------------------------------------------
# Full 110-symbol universe
# ---------------------------------------------------------------------------

FULL_UNIVERSE: list[str] = sorted(set(SP500_UNIVERSE + ETF_OVERLAYS))

# ---------------------------------------------------------------------------
# GICS sector mapping (all 110 symbols)
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    # ETFs — classified by primary exposure
    "SPY":   "ETF_Broad",
    "QQQ":   "ETF_Broad",
    "IWM":   "ETF_Broad",
    "XLF":   "ETF_Sector",
    "XLE":   "ETF_Sector",
    "XLK":   "ETF_Sector",
    "XLV":   "ETF_Sector",
    "XLI":   "ETF_Sector",
    "TLT":   "ETF_Macro",
    "GLD":   "ETF_Macro",
    # Information Technology
    "AAPL":  "Information Technology",
    "MSFT":  "Information Technology",
    "NVDA":  "Information Technology",
    "AVGO":  "Information Technology",
    "ORCL":  "Information Technology",
    "CRM":   "Information Technology",
    "CSCO":  "Information Technology",
    "INTU":  "Information Technology",
    "IBM":   "Information Technology",
    "ACN":   "Information Technology",
    "TXN":   "Information Technology",
    "QCOM":  "Information Technology",
    "ADI":   "Information Technology",
    "AMAT":  "Information Technology",
    "LRCX":  "Information Technology",
    "KLAC":  "Information Technology",
    "ADP":   "Information Technology",
    "CDNS":  "Information Technology",
    # Health Care
    "JNJ":   "Health Care",
    "UNH":   "Health Care",
    "LLY":   "Health Care",
    "ABBV":  "Health Care",
    "MRK":   "Health Care",
    "TMO":   "Health Care",
    "ABT":   "Health Care",
    "DHR":   "Health Care",
    "SYK":   "Health Care",
    "ISRG":  "Health Care",
    "EW":    "Health Care",
    "IDXX":  "Health Care",
    # Consumer Discretionary
    "AMZN":  "Consumer Discretionary",
    "TSLA":  "Consumer Discretionary",
    "HD":    "Consumer Discretionary",
    "MCD":   "Consumer Discretionary",
    "NKE":   "Consumer Discretionary",
    "LOW":   "Consumer Discretionary",
    "TJX":   "Consumer Discretionary",
    "SBUX":  "Consumer Discretionary",
    "BKNG":  "Consumer Discretionary",
    "MAR":   "Consumer Discretionary",
    # Financials
    "JPM":   "Financials",
    "BAC":   "Financials",
    "WFC":   "Financials",
    "GS":    "Financials",
    "MS":    "Financials",
    "BLK":   "Financials",
    "AXP":   "Financials",
    "V":     "Financials",
    "MA":    "Financials",
    "COF":   "Financials",
    # Communication Services
    "GOOGL": "Communication Services",
    "META":  "Communication Services",
    "NFLX":  "Communication Services",
    "DIS":   "Communication Services",
    "CMCSA": "Communication Services",
    "T":     "Communication Services",
    "VZ":    "Communication Services",
    "TMUS":  "Communication Services",
    # Industrials
    "GE":    "Industrials",
    "RTX":   "Industrials",
    "HON":   "Industrials",
    "CAT":   "Industrials",
    "UPS":   "Industrials",
    "FDX":   "Industrials",
    "LMT":   "Industrials",
    "BA":    "Industrials",
    "DE":    "Industrials",
    # Consumer Staples
    "PG":    "Consumer Staples",
    "KO":    "Consumer Staples",
    "PEP":   "Consumer Staples",
    "WMT":   "Consumer Staples",
    "COST":  "Consumer Staples",
    "PM":    "Consumer Staples",
    "MO":    "Consumer Staples",
    "CL":    "Consumer Staples",
    # Energy
    "XOM":   "Energy",
    "CVX":   "Energy",
    "COP":   "Energy",
    "EOG":   "Energy",
    "SLB":   "Energy",
    "MPC":   "Energy",
    "PSX":   "Energy",
    # Utilities
    "NEE":   "Utilities",
    "DUK":   "Utilities",
    "SO":    "Utilities",
    "AEP":   "Utilities",
    "D":     "Utilities",
    "EXC":   "Utilities",
    # Real Estate
    "PLD":   "Real Estate",
    "AMT":   "Real Estate",
    "EQIX":  "Real Estate",
    "PSA":   "Real Estate",
    "WELL":  "Real Estate",
    "DLR":   "Real Estate",
    # Materials
    "LIN":   "Materials",
    "APD":   "Materials",
    "SHW":   "Materials",
    "FCX":   "Materials",
    "NEM":   "Materials",
    "NUE":   "Materials",
}

# ---------------------------------------------------------------------------
# Universe metadata
# ---------------------------------------------------------------------------

UNIVERSE_VERSION = "1.0.0"
UNIVERSE_AS_OF = "2026-03-28"
UNIVERSE_DESCRIPTION = (
    "110-symbol curated universe: 100 S&P 500 liquid names (sector-balanced) "
    "+ 10 ETF overlays (broad market, sector, macro)."
)

# Sector counts for validation
_EXPECTED_SECTOR_COUNTS = {
    "Information Technology": 18,
    "Health Care": 12,
    "Consumer Discretionary": 10,
    "Financials": 10,
    "Communication Services": 8,
    "Industrials": 9,
    "Consumer Staples": 8,
    "Energy": 7,
    "Utilities": 6,
    "Real Estate": 6,
    "Materials": 6,
    "ETF_Broad": 3,
    "ETF_Sector": 5,
    "ETF_Macro": 2,
}


def validate_universe() -> None:
    """Assert universe is self-consistent."""
    assert len(SP500_UNIVERSE) == 100, f"Expected 100 S&P names, got {len(SP500_UNIVERSE)}"
    assert len(ETF_OVERLAYS) == 10, f"Expected 10 ETFs, got {len(ETF_OVERLAYS)}"
    assert len(FULL_UNIVERSE) == 110, f"Expected 110 total, got {len(FULL_UNIVERSE)}"
    missing = [s for s in FULL_UNIVERSE if s not in SECTOR_MAP]
    assert not missing, f"Missing sector mapping for: {missing}"
    duplicates = [s for s in SP500_UNIVERSE if SP500_UNIVERSE.count(s) > 1]
    assert not duplicates, f"Duplicate symbols in SP500_UNIVERSE: {set(duplicates)}"
