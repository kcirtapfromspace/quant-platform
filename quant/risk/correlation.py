"""Correlation monitoring and correlation-aware risk controls.

Tracks rolling pairwise correlations across portfolio holdings and
provides risk checks that tighten limits when correlations are elevated.
This closes the gap between regime detection (which classifies
correlation regimes) and the risk engine (which enforces position limits).

Key components:

  * **CorrelationMonitor** — maintains a rolling correlation matrix and
    computes aggregate portfolio-level correlation metrics.
  * **CorrelationRiskCheck** — risk check that rejects or scales orders
    when portfolio concentration (measured via effective N and
    diversification ratio) is too high.
  * **CorrelationConfig** — thresholds for alerts and risk controls.

Usage::

    from quant.risk.correlation import CorrelationMonitor, CorrelationRiskCheck

    monitor = CorrelationMonitor(window=63)
    monitor.update(returns_matrix)  # list[list[float]], [n_days][n_assets]

    state = monitor.current_state(position_weights)
    check = CorrelationRiskCheck()
    approved, reason = check.check(state)

All computations are pure Python — no scipy, no numpy dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrelationConfig:
    """Configuration for correlation monitoring and risk checks.

    Attributes:
        window: Rolling window for correlation computation (trading days).
        min_observations: Minimum data points required before correlation
            is considered valid.
        avg_corr_warn: Average pairwise correlation above which a warning
            is generated.
        avg_corr_critical: Average pairwise correlation above which
            position limits are tightened.
        min_effective_n: Minimum effective number of independent bets.
            If the portfolio's effective N drops below this, new
            concentrating orders are rejected.
        min_diversification_ratio: Minimum diversification ratio (weighted
            average vol / portfolio vol).  Below this the portfolio is
            considered too concentrated.
        position_scale_factor: When avg correlation exceeds
            ``avg_corr_critical``, position limits are scaled by this
            factor (e.g. 0.5 = halve limits).
    """

    window: int = 63
    min_observations: int = 21
    avg_corr_warn: float = 0.60
    avg_corr_critical: float = 0.75
    min_effective_n: float = 2.0
    min_diversification_ratio: float = 0.8
    position_scale_factor: float = 0.5


# ---------------------------------------------------------------------------
# Correlation state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrelationState:
    """Snapshot of portfolio-level correlation metrics.

    Attributes:
        avg_pairwise_corr: Average pairwise correlation across holdings.
        max_pairwise_corr: Maximum pairwise correlation observed.
        max_corr_pair: Tuple of symbols with the highest correlation.
        effective_n: Effective number of independent bets (higher = more
            diversified).  Computed as 1 / sum(w_i^2 * sum(w_j * rho_ij)).
        diversification_ratio: Ratio of weighted average volatility to
            portfolio volatility.  Values > 1 indicate diversification
            benefit.
        herfindahl: Herfindahl-Hirschman index of position weights
            (0 = perfectly diversified, 1 = single position).
        n_assets: Number of assets with non-zero weight.
        correlation_matrix: Full correlation matrix as nested dict
            ``{sym_i: {sym_j: rho}}``.
        volatilities: Per-asset annualised volatilities.
        level: Risk level — ``"normal"``, ``"elevated"``, or
            ``"critical"``.
    """

    avg_pairwise_corr: float
    max_pairwise_corr: float
    max_corr_pair: tuple[str, str]
    effective_n: float
    diversification_ratio: float
    herfindahl: float
    n_assets: int
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    volatilities: dict[str, float] = field(default_factory=dict)
    level: str = "normal"


# ---------------------------------------------------------------------------
# Pure-Python math helpers
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var) if var > 0 else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two series."""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx, my = _mean(xs[:n]), _mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx < 1e-12 or dy < 1e-12:
        return 0.0
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Correlation monitor
# ---------------------------------------------------------------------------


class CorrelationMonitor:
    """Track rolling correlations across portfolio holdings.

    Maintains a buffer of recent multi-asset returns and computes
    the pairwise correlation matrix and portfolio-level metrics on
    demand.

    Args:
        config: Monitoring configuration.
        symbols: Ordered list of asset symbols matching the column
            order of returns data.
    """

    def __init__(
        self,
        symbols: list[str],
        config: CorrelationConfig | None = None,
    ) -> None:
        self._config = config or CorrelationConfig()
        self._symbols = list(symbols)
        self._n = len(symbols)
        self._buffer: list[list[float]] = []  # [n_days][n_assets]

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def config(self) -> CorrelationConfig:
        return self._config

    def update(self, returns: list[list[float]]) -> None:
        """Append new return observations to the rolling buffer.

        Args:
            returns: ``[n_days][n_assets]`` — each inner list has one
                return per asset in the same order as ``symbols``.
        """
        for row in returns:
            if len(row) != self._n:
                raise ValueError(
                    f"Expected {self._n} assets per row, got {len(row)}"
                )
            self._buffer.append(list(row))

        # Trim to window size
        max_len = self._config.window
        if len(self._buffer) > max_len:
            self._buffer = self._buffer[-max_len:]

    def reset(self) -> None:
        """Clear the return buffer."""
        self._buffer.clear()

    def has_sufficient_data(self) -> bool:
        """True if enough observations exist for valid correlation."""
        return len(self._buffer) >= self._config.min_observations

    def correlation_matrix(self) -> dict[str, dict[str, float]]:
        """Compute the pairwise correlation matrix from buffered returns.

        Returns:
            Nested dict ``{sym_i: {sym_j: pearson_rho}}``.
        """
        n_days = len(self._buffer)
        if n_days < 3:
            return {
                s: {t: (1.0 if s == t else 0.0) for t in self._symbols}
                for s in self._symbols
            }

        # Extract per-asset return series
        series: list[list[float]] = [
            [self._buffer[t][a] for t in range(n_days)]
            for a in range(self._n)
        ]

        matrix: dict[str, dict[str, float]] = {}
        for i, si in enumerate(self._symbols):
            matrix[si] = {}
            for j, sj in enumerate(self._symbols):
                if i == j:
                    matrix[si][sj] = 1.0
                elif j < i:
                    matrix[si][sj] = matrix[sj][si]
                else:
                    matrix[si][sj] = _pearson(series[i], series[j])

        return matrix

    def asset_volatilities(self) -> dict[str, float]:
        """Compute annualised volatility per asset from buffered returns.

        Returns:
            ``{symbol: annualised_vol}``.
        """
        n_days = len(self._buffer)
        if n_days < 2:
            return {s: 0.0 for s in self._symbols}

        vols: dict[str, float] = {}
        for a, sym in enumerate(self._symbols):
            series = [self._buffer[t][a] for t in range(n_days)]
            vols[sym] = _std(series) * math.sqrt(252)

        return vols

    def current_state(
        self,
        weights: dict[str, float] | None = None,
    ) -> CorrelationState:
        """Compute current correlation metrics for the portfolio.

        Args:
            weights: ``{symbol: weight}`` — portfolio weights.  If None,
                equal-weight is assumed across all tracked symbols.

        Returns:
            :class:`CorrelationState` snapshot.
        """
        if weights is None:
            w_val = 1.0 / self._n if self._n > 0 else 0.0
            weights = {s: w_val for s in self._symbols}

        # Normalise weights to absolute values summing to 1
        active = {s: abs(w) for s, w in weights.items() if s in self._symbols and abs(w) > 1e-12}
        total_w = sum(active.values())
        if total_w > 0:
            active = {s: w / total_w for s, w in active.items()}

        n_assets = len(active)
        if n_assets < 2 or not self.has_sufficient_data():
            return CorrelationState(
                avg_pairwise_corr=0.0,
                max_pairwise_corr=0.0,
                max_corr_pair=("", ""),
                effective_n=float(n_assets),
                diversification_ratio=1.0,
                herfindahl=sum(w**2 for w in active.values()) if active else 0.0,
                n_assets=n_assets,
                level="normal",
            )

        corr_mat = self.correlation_matrix()
        vols = self.asset_volatilities()

        # Average and max pairwise correlation (among active positions)
        active_syms = sorted(active.keys())
        pair_corrs: list[float] = []
        max_corr = -2.0
        max_pair = (active_syms[0], active_syms[1]) if len(active_syms) >= 2 else ("", "")

        for i, si in enumerate(active_syms):
            for j in range(i + 1, len(active_syms)):
                sj = active_syms[j]
                rho = corr_mat.get(si, {}).get(sj, 0.0)
                pair_corrs.append(rho)
                if rho > max_corr:
                    max_corr = rho
                    max_pair = (si, sj)

        avg_corr = _mean(pair_corrs) if pair_corrs else 0.0

        # Herfindahl index
        herfindahl = sum(w**2 for w in active.values())

        # Effective N: 1 / (w' * C * w) where C is correlation matrix
        # This measures how many independent bets the portfolio represents
        wcw = 0.0
        for si in active_syms:
            for sj in active_syms:
                rho = corr_mat.get(si, {}).get(sj, 0.0)
                wcw += active[si] * active[sj] * rho

        effective_n = 1.0 / wcw if wcw > 1e-12 else float(n_assets)

        # Diversification ratio: weighted avg vol / portfolio vol
        w_avg_vol = sum(active.get(s, 0.0) * vols.get(s, 0.0) for s in active_syms)
        port_var = 0.0
        for si in active_syms:
            for sj in active_syms:
                rho = corr_mat.get(si, {}).get(sj, 0.0)
                port_var += (
                    active[si]
                    * active[sj]
                    * vols.get(si, 0.0)
                    * vols.get(sj, 0.0)
                    * rho
                )
        port_vol = math.sqrt(port_var) if port_var > 0 else 1e-12
        div_ratio = w_avg_vol / port_vol if port_vol > 1e-12 else 1.0

        # Determine risk level
        cfg = self._config
        if avg_corr >= cfg.avg_corr_critical:
            level = "critical"
        elif avg_corr >= cfg.avg_corr_warn:
            level = "elevated"
        else:
            level = "normal"

        return CorrelationState(
            avg_pairwise_corr=avg_corr,
            max_pairwise_corr=max_corr if max_corr > -2.0 else 0.0,
            max_corr_pair=max_pair,
            effective_n=effective_n,
            diversification_ratio=div_ratio,
            herfindahl=herfindahl,
            n_assets=n_assets,
            correlation_matrix=corr_mat,
            volatilities=vols,
            level=level,
        )


# ---------------------------------------------------------------------------
# Correlation risk check
# ---------------------------------------------------------------------------


class CorrelationRiskCheck:
    """Risk check that validates portfolio concentration and diversification.

    Designed to be called alongside the main :class:`RiskEngine`
    validation.  Returns ``(approved, reason)`` tuples following the
    same convention as :class:`ExposureLimits`.

    Args:
        config: Correlation monitoring configuration.
    """

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self._config = config or CorrelationConfig()

    def check(self, state: CorrelationState) -> tuple[bool, str]:
        """Run all correlation-based risk checks.

        Checks (in order):
          1. Average correlation is below critical threshold.
          2. Effective N is above minimum.
          3. Diversification ratio is above minimum.

        Args:
            state: Current correlation state from
                :meth:`CorrelationMonitor.current_state`.

        Returns:
            ``(approved, reason)`` — approved is True if all checks pass.
        """
        cfg = self._config

        # 1. Average correlation
        if state.avg_pairwise_corr >= cfg.avg_corr_critical:
            return (
                False,
                f"Average pairwise correlation {state.avg_pairwise_corr:.2f} "
                f"exceeds critical threshold {cfg.avg_corr_critical:.2f}",
            )

        # 2. Effective N
        if state.effective_n < cfg.min_effective_n and state.n_assets >= 2:
            return (
                False,
                f"Effective N {state.effective_n:.1f} below minimum "
                f"{cfg.min_effective_n:.1f} — portfolio is too concentrated",
            )

        # 3. Diversification ratio
        if state.diversification_ratio < cfg.min_diversification_ratio and state.n_assets >= 2:
            return (
                False,
                f"Diversification ratio {state.diversification_ratio:.2f} below "
                f"minimum {cfg.min_diversification_ratio:.2f}",
            )

        return True, ""

    def adjusted_position_limit(
        self,
        base_limit: float,
        state: CorrelationState,
    ) -> float:
        """Compute correlation-adjusted position limit.

        When correlations are elevated, position limits should be
        tighter to prevent concentration risk.

        Args:
            base_limit: Base per-position limit fraction (e.g. 0.20).
            state: Current correlation state.

        Returns:
            Adjusted limit fraction — lower when correlations are high.
        """
        cfg = self._config

        if state.avg_pairwise_corr >= cfg.avg_corr_critical:
            return base_limit * cfg.position_scale_factor

        if state.avg_pairwise_corr >= cfg.avg_corr_warn:
            # Linear interpolation between base and scaled
            warn_range = cfg.avg_corr_critical - cfg.avg_corr_warn
            if warn_range > 0:
                t = (state.avg_pairwise_corr - cfg.avg_corr_warn) / warn_range
                return base_limit * (1.0 - t * (1.0 - cfg.position_scale_factor))
            return base_limit * cfg.position_scale_factor

        return base_limit

    def position_limit_scale(self, state: CorrelationState) -> float:
        """Return the scaling factor for position limits given correlation state.

        Convenience method: ``adjusted_limit = base_limit * scale``.

        Args:
            state: Current correlation state.

        Returns:
            Scale factor in (0, 1].
        """
        return self.adjusted_position_limit(1.0, state)
