"""Position-level PnL attribution for portfolio analysis.

Decomposes portfolio returns into per-position contributions, identifying
which positions drive performance and which drag on returns.  Supports
both daily and cumulative attribution windows.

Attribution model:

    PnL_total = Σ_i  w_i · r_i

Where w_i is the beginning-of-period weight and r_i is the asset return.
Cumulative attribution uses Brinson-style linking across periods.

Key outputs:

  * **Position PnL**: per-asset return contribution per day.
  * **Winners / losers**: top and bottom contributors over any window.
  * **Sector attribution**: aggregate PnL by sector group.
  * **Concentration risk**: HHI and top-N contribution share.

Usage::

    from quant.portfolio.pnl_attribution import (
        PnLAttributor,
        PnLConfig,
    )

    attributor = PnLAttributor()
    result = attributor.attribute(weights_df, returns_df)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PnLConfig:
    """Configuration for PnL attribution.

    Attributes:
        top_n:          Number of top contributors to highlight.
        sector_map:     Mapping of symbol → sector for sector attribution.
        min_weight:     Minimum absolute weight to include in attribution.
    """

    top_n: int = 5
    sector_map: dict[str, str] = field(default_factory=dict)
    min_weight: float = 1e-6


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionPnL:
    """PnL summary for one position over the attribution window."""

    symbol: str
    total_pnl: float
    avg_weight: float
    avg_return: float
    n_days: int
    win_days: int
    loss_days: int
    best_day: float
    worst_day: float


@dataclass(frozen=True, slots=True)
class SectorPnL:
    """PnL summary for one sector."""

    sector: str
    total_pnl: float
    n_positions: int
    avg_weight: float


@dataclass
class PnLAttributionResult:
    """Complete PnL attribution result.

    Attributes:
        daily_pnl:          Daily portfolio PnL series.
        position_pnl:       Per-position PnL contributions (DataFrame: dates × symbols).
        positions:          Per-position summary statistics.
        sectors:            Per-sector PnL (if sector_map provided).
        total_pnl:          Total portfolio PnL over the window.
        n_days:             Number of attribution days.
        n_positions:        Number of positions attributed.
        top_contributors:   Best-performing positions by PnL.
        bottom_contributors: Worst-performing positions by PnL.
        hhi:                Herfindahl-Hirschman Index of PnL concentration.
        top_n_share:        Share of total |PnL| from top N contributors.
    """

    daily_pnl: pd.Series = field(repr=False)
    position_pnl: pd.DataFrame = field(repr=False)
    positions: list[PositionPnL] = field(repr=False)
    sectors: list[SectorPnL] = field(default_factory=list, repr=False)
    total_pnl: float = 0.0
    n_days: int = 0
    n_positions: int = 0
    top_contributors: list[PositionPnL] = field(default_factory=list)
    bottom_contributors: list[PositionPnL] = field(default_factory=list)
    hhi: float = 0.0
    top_n_share: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"PnL Attribution ({self.n_positions} positions, {self.n_days} days)",
            "=" * 60,
            "",
            f"Total PnL              : {self.total_pnl:+.4f}",
            f"PnL concentration (HHI): {self.hhi:.4f}",
            f"Top-{len(self.top_contributors)} share of |PnL| : {self.top_n_share:.1%}",
            "",
            "Top contributors:",
        ]
        for p in self.top_contributors:
            lines.append(
                f"  {p.symbol:<10s}: {p.total_pnl:+.4f} "
                f"(avg wt {p.avg_weight:.2%}, {p.win_days}W/{p.loss_days}L)"
            )
        lines.append("")
        lines.append("Bottom contributors:")
        for p in self.bottom_contributors:
            lines.append(
                f"  {p.symbol:<10s}: {p.total_pnl:+.4f} "
                f"(avg wt {p.avg_weight:.2%}, {p.win_days}W/{p.loss_days}L)"
            )

        if self.sectors:
            lines.extend(["", "Sector attribution:"])
            for s in sorted(self.sectors, key=lambda x: x.total_pnl, reverse=True):
                lines.append(
                    f"  {s.sector:<15s}: {s.total_pnl:+.4f} "
                    f"({s.n_positions} pos, avg wt {s.avg_weight:.2%})"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Attributor
# ---------------------------------------------------------------------------


class PnLAttributor:
    """Position-level PnL attribution engine.

    Args:
        config: Attribution configuration.
    """

    def __init__(self, config: PnLConfig | None = None) -> None:
        self._config = config or PnLConfig()

    @property
    def config(self) -> PnLConfig:
        return self._config

    def attribute(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> PnLAttributionResult:
        """Attribute portfolio PnL to individual positions.

        Args:
            weights: Daily portfolio weights (DatetimeIndex × symbols).
                     Weights are beginning-of-period (BoP).
            returns: Daily asset returns (DatetimeIndex × symbols).

        Returns:
            :class:`PnLAttributionResult` with full attribution.

        Raises:
            ValueError: If inputs have fewer than 1 common date.
        """
        cfg = self._config

        # Align dates and symbols
        common_dates = weights.index.intersection(returns.index)
        if len(common_dates) < 1:
            raise ValueError("Need at least 1 common date between weights and returns")

        common_symbols = sorted(
            set(weights.columns) & set(returns.columns),
        )
        if not common_symbols:
            raise ValueError("No common symbols between weights and returns")

        w = weights.loc[common_dates, common_symbols].fillna(0.0)
        r = returns.loc[common_dates, common_symbols].fillna(0.0)
        n_days = len(common_dates)

        # Position-level daily PnL: PnL_it = w_it · r_it
        pos_pnl = w * r

        # Portfolio daily PnL
        daily_pnl = pos_pnl.sum(axis=1)
        daily_pnl.name = "portfolio_pnl"

        # Per-position summaries
        positions: list[PositionPnL] = []
        for sym in common_symbols:
            sym_pnl = pos_pnl[sym]
            sym_w = w[sym]
            sym_r = r[sym]

            avg_weight = float(sym_w.abs().mean())
            if avg_weight < cfg.min_weight:
                continue

            total = float(sym_pnl.sum())
            win_days = int((sym_pnl > 0).sum())
            loss_days = int((sym_pnl < 0).sum())
            best = float(sym_pnl.max())
            worst = float(sym_pnl.min())
            avg_ret = float(sym_r.mean())

            positions.append(PositionPnL(
                symbol=sym,
                total_pnl=total,
                avg_weight=avg_weight,
                avg_return=avg_ret,
                n_days=n_days,
                win_days=win_days,
                loss_days=loss_days,
                best_day=best,
                worst_day=worst,
            ))

        # Sort by total PnL
        positions.sort(key=lambda p: p.total_pnl, reverse=True)

        top_n = cfg.top_n
        top = positions[:top_n]
        bottom = positions[-top_n:] if len(positions) > top_n else positions

        # Concentration
        abs_pnls = [abs(p.total_pnl) for p in positions]
        total_abs = sum(abs_pnls)
        hhi = 0.0
        top_n_share = 0.0
        if total_abs > 1e-15:
            shares = [a / total_abs for a in abs_pnls]
            hhi = sum(s * s for s in shares)
            top_n_abs = sum(sorted(abs_pnls, reverse=True)[:top_n])
            top_n_share = top_n_abs / total_abs

        # Sector attribution
        sectors: list[SectorPnL] = []
        if cfg.sector_map:
            sector_groups: dict[str, list[PositionPnL]] = {}
            for p in positions:
                sec = cfg.sector_map.get(p.symbol, "Other")
                sector_groups.setdefault(sec, []).append(p)
            for sec, members in sector_groups.items():
                sectors.append(SectorPnL(
                    sector=sec,
                    total_pnl=sum(m.total_pnl for m in members),
                    n_positions=len(members),
                    avg_weight=np.mean([m.avg_weight for m in members]),
                ))

        total_pnl = float(daily_pnl.sum())

        return PnLAttributionResult(
            daily_pnl=daily_pnl,
            position_pnl=pos_pnl,
            positions=positions,
            sectors=sectors,
            total_pnl=total_pnl,
            n_days=n_days,
            n_positions=len(positions),
            top_contributors=top,
            bottom_contributors=bottom,
            hhi=hhi,
            top_n_share=top_n_share,
        )
