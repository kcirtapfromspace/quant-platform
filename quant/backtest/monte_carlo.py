"""Monte Carlo simulation for backtest confidence intervals.

Bootstrap-resamples a strategy's daily returns to build empirical
distributions of key performance metrics (Sharpe, max drawdown, CAGR,
total return, volatility).  This quantifies the uncertainty around
point-estimate backtest results and supports tail-risk analysis.

Two resampling modes are supported:

  * **IID bootstrap** (default): samples daily returns with replacement.
    Fastest but destroys autocorrelation structure.
  * **Block bootstrap**: samples contiguous blocks of returns, preserving
    short-horizon serial dependence (momentum, mean-reversion clustering).

Key outputs:

  * Percentile confidence intervals for each metric.
  * VaR and CVaR at configurable confidence level.
  * Probability of loss (P[total_return < 0]).
  * Full empirical distributions for custom analysis.

Usage::

    from quant.backtest.monte_carlo import (
        MonteCarloAnalyzer,
        MonteCarloConfig,
    )

    analyzer = MonteCarloAnalyzer()
    result = analyzer.run(
        returns=backtest_report.returns_series,
        config=MonteCarloConfig(n_simulations=10_000),
    )
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant.backtest import metrics as m

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation.

    Attributes:
        n_simulations: Number of bootstrap paths to generate.
        confidence_level: Confidence level for intervals (e.g. 0.95).
        block_size: Block length for block bootstrap.  Set to 1 for IID.
        seed: Random seed for reproducibility (None for random).
        initial_capital: Starting capital for equity curves.
        var_confidence: Confidence level for VaR/CVaR (e.g. 0.95 → 5% tail).
    """

    n_simulations: int = 5_000
    confidence_level: float = 0.95
    block_size: int = 1
    seed: int | None = 42
    initial_capital: float = 1_000_000.0
    var_confidence: float = 0.95


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A percentile confidence interval for a metric."""

    metric: str
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo backtest simulation."""

    n_simulations: int
    n_days: int
    confidence_level: float

    # Confidence intervals
    sharpe_ci: ConfidenceInterval
    max_drawdown_ci: ConfidenceInterval
    total_return_ci: ConfidenceInterval
    cagr_ci: ConfidenceInterval
    volatility_ci: ConfidenceInterval

    # Tail risk
    var: float
    cvar: float
    var_confidence: float
    prob_loss: float

    # Full distributions (for custom analysis)
    sharpe_dist: np.ndarray = field(repr=False)
    max_drawdown_dist: np.ndarray = field(repr=False)
    total_return_dist: np.ndarray = field(repr=False)
    cagr_dist: np.ndarray = field(repr=False)
    volatility_dist: np.ndarray = field(repr=False)
    terminal_wealth_dist: np.ndarray = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary."""
        pct = self.confidence_level * 100
        lines = [
            f"Monte Carlo Simulation ({self.n_simulations:,} paths, {self.n_days} days)",
            "=" * 60,
            "",
            f"Metric               Point Est    {pct:.0f}% CI",
            "-" * 60,
            self._format_ci(self.sharpe_ci, fmt=".2f"),
            self._format_ci(self.total_return_ci, fmt="+.2%"),
            self._format_ci(self.cagr_ci, fmt="+.2%"),
            self._format_ci(self.volatility_ci, fmt=".2%"),
            self._format_ci(self.max_drawdown_ci, fmt=".2%"),
            "",
            "-" * 60,
            f"VaR ({self.var_confidence:.0%})            : {self.var:+.2%}",
            f"CVaR ({self.var_confidence:.0%})           : {self.cvar:+.2%}",
            f"P(loss)              : {self.prob_loss:.1%}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_ci(ci: ConfidenceInterval, fmt: str = ".2f") -> str:
        label = f"{ci.metric:<20s}"
        point = f"{ci.point_estimate:{fmt}}"
        lower = f"{ci.lower:{fmt}}"
        upper = f"{ci.upper:{fmt}}"
        return f"{label} {point:>10s}    [{lower}, {upper}]"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MonteCarloAnalyzer:
    """Monte Carlo bootstrap analyzer for backtest returns."""

    def run(
        self,
        returns: pd.Series,
        config: MonteCarloConfig | None = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on a daily returns series.

        Args:
            returns: Daily returns series from a backtest.
            config: Simulation configuration.

        Returns:
            :class:`MonteCarloResult` with confidence intervals and
            tail risk metrics.

        Raises:
            ValueError: If returns are empty or too short.
        """
        if config is None:
            config = MonteCarloConfig()

        returns_arr = np.asarray(returns, dtype=np.float64)
        n_days = len(returns_arr)

        if n_days < 2:
            raise ValueError(
                f"Need at least 2 days of returns, got {n_days}"
            )

        rng = np.random.default_rng(config.seed)

        # Generate bootstrap paths
        paths = self._generate_paths(returns_arr, config, rng)

        # Compute metrics for each path
        sharpe_vals = np.empty(config.n_simulations)
        mdd_vals = np.empty(config.n_simulations)
        total_ret_vals = np.empty(config.n_simulations)
        cagr_vals = np.empty(config.n_simulations)
        vol_vals = np.empty(config.n_simulations)
        terminal_vals = np.empty(config.n_simulations)

        for i in range(config.n_simulations):
            path = paths[i]
            ret_series = pd.Series(path)
            equity = (1 + ret_series).cumprod()

            sharpe_vals[i] = m.sharpe_ratio(ret_series)
            mdd_vals[i] = m.max_drawdown(equity)
            total_ret_vals[i] = float(equity.iloc[-1] - 1.0)
            n_years = n_days / 252
            if equity.iloc[-1] > 0 and n_years > 0:
                cagr_vals[i] = float(equity.iloc[-1] ** (1 / n_years) - 1)
            else:
                cagr_vals[i] = 0.0
            vol_vals[i] = (
                float(ret_series.std() * math.sqrt(252))
                if len(ret_series) > 1
                else 0.0
            )
            terminal_vals[i] = equity.iloc[-1] * config.initial_capital

        # Point estimates from original returns
        orig_series = pd.Series(returns_arr)
        orig_equity = (1 + orig_series).cumprod()
        orig_sharpe = m.sharpe_ratio(orig_series)
        orig_mdd = m.max_drawdown(orig_equity)
        orig_ret = float(orig_equity.iloc[-1] - 1.0)
        n_years = n_days / 252
        orig_cagr = (
            float(orig_equity.iloc[-1] ** (1 / n_years) - 1)
            if orig_equity.iloc[-1] > 0 and n_years > 0
            else 0.0
        )
        orig_vol = (
            float(orig_series.std() * math.sqrt(252))
            if len(orig_series) > 1
            else 0.0
        )

        # Confidence intervals
        alpha = 1 - config.confidence_level
        lo_pct = alpha / 2 * 100
        hi_pct = (1 - alpha / 2) * 100

        sharpe_ci = ConfidenceInterval(
            metric="Sharpe",
            point_estimate=orig_sharpe,
            lower=float(np.percentile(sharpe_vals, lo_pct)),
            upper=float(np.percentile(sharpe_vals, hi_pct)),
            confidence_level=config.confidence_level,
        )
        mdd_ci = ConfidenceInterval(
            metric="Max Drawdown",
            point_estimate=orig_mdd,
            lower=float(np.percentile(mdd_vals, lo_pct)),
            upper=float(np.percentile(mdd_vals, hi_pct)),
            confidence_level=config.confidence_level,
        )
        ret_ci = ConfidenceInterval(
            metric="Total Return",
            point_estimate=orig_ret,
            lower=float(np.percentile(total_ret_vals, lo_pct)),
            upper=float(np.percentile(total_ret_vals, hi_pct)),
            confidence_level=config.confidence_level,
        )
        cagr_ci = ConfidenceInterval(
            metric="CAGR",
            point_estimate=orig_cagr,
            lower=float(np.percentile(cagr_vals, lo_pct)),
            upper=float(np.percentile(cagr_vals, hi_pct)),
            confidence_level=config.confidence_level,
        )
        vol_ci = ConfidenceInterval(
            metric="Volatility",
            point_estimate=orig_vol,
            lower=float(np.percentile(vol_vals, lo_pct)),
            upper=float(np.percentile(vol_vals, hi_pct)),
            confidence_level=config.confidence_level,
        )

        # Tail risk: VaR and CVaR from terminal return distribution
        var_pctile = (1 - config.var_confidence) * 100
        var_val = float(np.percentile(total_ret_vals, var_pctile))
        tail_mask = total_ret_vals <= var_val
        cvar_val = (
            float(np.mean(total_ret_vals[tail_mask]))
            if tail_mask.any()
            else var_val
        )

        prob_loss = float(np.mean(total_ret_vals < 0))

        return MonteCarloResult(
            n_simulations=config.n_simulations,
            n_days=n_days,
            confidence_level=config.confidence_level,
            sharpe_ci=sharpe_ci,
            max_drawdown_ci=mdd_ci,
            total_return_ci=ret_ci,
            cagr_ci=cagr_ci,
            volatility_ci=vol_ci,
            var=var_val,
            cvar=cvar_val,
            var_confidence=config.var_confidence,
            prob_loss=prob_loss,
            sharpe_dist=sharpe_vals,
            max_drawdown_dist=mdd_vals,
            total_return_dist=total_ret_vals,
            cagr_dist=cagr_vals,
            volatility_dist=vol_vals,
            terminal_wealth_dist=terminal_vals,
        )

    # ── Path generation ───────────────────────────────────────────────

    @staticmethod
    def _generate_paths(
        returns: np.ndarray,
        config: MonteCarloConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate bootstrap return paths.

        Returns shape (n_simulations, n_days).
        """
        n_days = len(returns)
        n_sims = config.n_simulations
        block = config.block_size

        if block <= 1:
            # IID bootstrap: sample with replacement
            indices = rng.integers(0, n_days, size=(n_sims, n_days))
            return returns[indices]

        # Block bootstrap: sample contiguous blocks
        n_blocks = math.ceil(n_days / block)
        max_start = n_days - block
        if max_start < 0:
            max_start = 0

        paths = np.empty((n_sims, n_days))
        for i in range(n_sims):
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            segments = [returns[s : s + block] for s in starts]
            concatenated = np.concatenate(segments)
            paths[i] = concatenated[:n_days]

        return paths
