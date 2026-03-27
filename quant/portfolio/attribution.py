"""Performance attribution: decompose portfolio returns by signal, factor, and sector.

Implements Brinson-style attribution and factor-based decomposition to answer:
  - Which signals contributed to P&L?
  - Which sectors drove returns?
  - How much came from asset selection vs allocation?
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant.portfolio.factor_attribution import FactorAttributor

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class SignalAttribution:
    """Per-signal contribution to portfolio returns.

    Attributes:
        signal_name:  Name of the signal.
        contribution:  Absolute return contribution.
        weight:        Average portfolio weight driven by this signal.
        hit_rate:      Fraction of days where signal direction matched return direction.
    """

    signal_name: str
    contribution: float
    weight: float
    hit_rate: float


@dataclass(frozen=True, slots=True)
class SectorAttribution:
    """Per-sector Brinson-style attribution.

    Attributes:
        sector:     Sector label.
        allocation: Return from over/under-weighting the sector vs benchmark.
        selection:  Return from stock selection within the sector.
        interaction: Allocation × selection interaction term.
        total:      allocation + selection + interaction.
    """

    sector: str
    allocation: float
    selection: float
    interaction: float
    total: float


@dataclass
class AttributionReport:
    """Complete attribution report for a portfolio over a period.

    Attributes:
        total_return:         Portfolio total return over the period.
        benchmark_return:     Benchmark total return over the period.
        active_return:        Portfolio return − benchmark return.
        signal_attributions:  Per-signal return decomposition.
        sector_attributions:  Per-sector Brinson decomposition.
        factor_exposures:     Dict of {factor_name: average_exposure}.
        tracking_error:       Annualised tracking error vs benchmark.
        information_ratio:    Annualised active return / tracking error.
    """

    total_return: float
    benchmark_return: float
    active_return: float
    signal_attributions: list[SignalAttribution] = field(default_factory=list)
    sector_attributions: list[SectorAttribution] = field(default_factory=list)
    factor_exposures: dict[str, float] = field(default_factory=dict)
    tracking_error: float = 0.0
    information_ratio: float = 0.0


class PerformanceAttributor:
    """Compute return attribution across signals, sectors, and factors.

    Usage::

        attributor = PerformanceAttributor()
        report = attributor.attribute(
            portfolio_returns=daily_returns_series,
            benchmark_returns=benchmark_series,
            weights_history=weights_df,      # DatetimeIndex × symbols
            signal_weights=signal_weights_df, # DatetimeIndex × signals
            sector_map={"AAPL": "Tech", "JPM": "Financials"},
            asset_returns=asset_returns_df,   # DatetimeIndex × symbols
        )
    """

    def attribute(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        weights_history: pd.DataFrame | None = None,
        signal_weights: pd.DataFrame | None = None,
        sector_map: dict[str, str] | None = None,
        asset_returns: pd.DataFrame | None = None,
        benchmark_weights: pd.DataFrame | None = None,
    ) -> AttributionReport:
        """Run full attribution analysis.

        Args:
            portfolio_returns: Daily portfolio returns (net of costs).
            benchmark_returns: Daily benchmark returns. If None, uses zero.
            weights_history:   DataFrame of portfolio weights over time
                               (index=date, columns=symbols).
            signal_weights:    DataFrame of per-signal weight contributions
                               (index=date, columns=signal_names).
            sector_map:        {symbol: sector} mapping.
            asset_returns:     DataFrame of per-asset daily returns
                               (index=date, columns=symbols).
            benchmark_weights: DataFrame of benchmark weights
                               (index=date, columns=symbols).

        Returns:
            AttributionReport with full decomposition.
        """
        if benchmark_returns is None:
            benchmark_returns = pd.Series(0.0, index=portfolio_returns.index)

        # Align indices
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.reindex(common_idx).fillna(0.0)
        bench_ret = benchmark_returns.reindex(common_idx).fillna(0.0)

        total_ret = float((1 + port_ret).prod() - 1)
        bench_total = float((1 + bench_ret).prod() - 1)
        active_ret = total_ret - bench_total

        # Tracking error and information ratio
        active_daily = port_ret - bench_ret
        te = float(active_daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        ir = float(active_daily.mean() / active_daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if active_daily.std() > 1e-12 else 0.0

        # Signal attribution
        signal_attrs: list[SignalAttribution] = []
        if signal_weights is not None and asset_returns is not None:
            signal_attrs = self._signal_attribution(
                signal_weights, asset_returns, port_ret
            )

        # Sector attribution (Brinson)
        sector_attrs: list[SectorAttribution] = []
        if (
            sector_map
            and weights_history is not None
            and asset_returns is not None
        ):
            sector_attrs = self._sector_attribution(
                weights_history, asset_returns, sector_map,
                benchmark_weights, bench_ret,
            )

        # Factor attribution (populates factor_exposures)
        factor_exposures: dict[str, float] = {}
        if asset_returns is not None:
            try:
                fa = FactorAttributor(min_observations=20)
                fa_report = fa.attribute(
                    portfolio_returns=port_ret,
                    asset_returns=asset_returns,
                )
                factor_exposures = fa_report.factor_exposures
            except Exception:
                pass

        return AttributionReport(
            total_return=total_ret,
            benchmark_return=bench_total,
            active_return=active_ret,
            signal_attributions=signal_attrs,
            sector_attributions=sector_attrs,
            factor_exposures=factor_exposures,
            tracking_error=te,
            information_ratio=ir,
        )

    def _signal_attribution(
        self,
        signal_weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
        portfolio_returns: pd.Series,
    ) -> list[SignalAttribution]:
        """Decompose returns by signal contribution."""
        results: list[SignalAttribution] = []

        for sig_name in signal_weights.columns:
            sig_w = signal_weights[sig_name]

            # Signal contribution: average weight × average return attribution
            # Approximate: contribution ≈ correlation(sig_weight, port_return) × scale
            common = sig_w.index.intersection(portfolio_returns.index)
            if len(common) < 2:
                continue

            sw = sig_w.reindex(common).fillna(0.0)
            pr = portfolio_returns.reindex(common).fillna(0.0)

            # Simple contribution: sum of (signal_weight × portfolio_return) / total
            contribution = float((sw * pr).sum())
            avg_weight = float(sw.abs().mean())

            # Hit rate: fraction of days where sign(signal_weight) == sign(return)
            hits = ((sw > 0) & (pr > 0)) | ((sw < 0) & (pr < 0))
            active_days = (sw.abs() > 1e-9).sum()
            hit_rate = float(hits.sum() / active_days) if active_days > 0 else 0.0

            results.append(
                SignalAttribution(
                    signal_name=sig_name,
                    contribution=contribution,
                    weight=avg_weight,
                    hit_rate=hit_rate,
                )
            )

        return results

    def _sector_attribution(
        self,
        weights_history: pd.DataFrame,
        asset_returns: pd.DataFrame,
        sector_map: dict[str, str],
        benchmark_weights: pd.DataFrame | None,
        benchmark_returns: pd.Series,
    ) -> list[SectorAttribution]:
        """Brinson-style sector attribution: allocation + selection + interaction."""
        common = weights_history.index.intersection(asset_returns.index)
        if len(common) < 2:
            return []

        wh = weights_history.reindex(common).fillna(0.0)
        ar = asset_returns.reindex(common).fillna(0.0)

        # Build per-symbol sector mapping
        sectors = sorted(set(sector_map.values()))
        results: list[SectorAttribution] = []

        for sector in sectors:
            sector_symbols = [s for s in wh.columns if sector_map.get(s) == sector]
            if not sector_symbols:
                continue

            # Portfolio sector weight and return
            port_sector_w = wh[sector_symbols].sum(axis=1).mean()
            if abs(port_sector_w) < 1e-12:
                port_sector_ret = 0.0
            else:
                # Weighted average return in this sector
                sector_w = wh[sector_symbols]
                sector_r = ar[[s for s in sector_symbols if s in ar.columns]]
                daily_sector_ret = (sector_w * sector_r).sum(axis=1)
                port_sector_ret = float(daily_sector_ret.mean())

            # Benchmark sector weight and return
            if benchmark_weights is not None:
                bw = benchmark_weights.reindex(common).fillna(0.0)
                bench_sector_w = bw[[s for s in sector_symbols if s in bw.columns]].sum(axis=1).mean()
            else:
                # Equal weight benchmark as fallback
                n_assets = len(wh.columns)
                n_sector = len(sector_symbols)
                bench_sector_w = n_sector / n_assets if n_assets > 0 else 0.0

            bench_total_ret = float(benchmark_returns.mean())

            # Brinson decomposition
            allocation = (port_sector_w - bench_sector_w) * bench_total_ret
            selection = bench_sector_w * (port_sector_ret - bench_total_ret)
            interaction = (port_sector_w - bench_sector_w) * (port_sector_ret - bench_total_ret)

            results.append(
                SectorAttribution(
                    sector=sector,
                    allocation=allocation,
                    selection=selection,
                    interaction=interaction,
                    total=allocation + selection + interaction,
                )
            )

        return results
