//! Pure-Rust vectorised backtesting engine.
//!
//! Phase 5: bar-replay loop, equity curve, trade log, and performance metric kernels.
//!
//! Design mirrors `quant.backtest.engine.BacktestEngine`:
//! - Signal at bar *t* applies to the close-to-close return from *t* → *t+1*
//!   (enforced by shifting signals one bar forward).
//! - Commission is proportional to |change in position size|.
//! - All metric formulas match the Python implementation to 1e-9 tolerance.

pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;

// ── Public types ─────────────────────────────────────────────────────────────

/// A completed round-trip trade identified in the position series.
#[derive(Debug, Clone)]
pub struct Trade {
    /// Bar index at which the trade was entered (position became non-zero).
    pub entry_idx: usize,
    /// Bar index at which the trade was exited (last non-zero position bar).
    pub exit_idx: usize,
    /// 1 = long, -1 = short.
    pub direction: i8,
    /// Geometric compound return over the trade bars: ∏(1+r_i) − 1.
    pub ret: f64,
}

/// Full result of a `run_backtest` call.
pub struct BacktestResult {
    /// Portfolio value at each bar (length == `adj_close.len()`).
    pub equity_curve: Vec<f64>,
    /// Rolling drawdown at each bar (non-positive fraction of peak).
    pub drawdown_curve: Vec<f64>,
    /// Net return per bar (gross return minus transaction cost).
    pub net_returns: Vec<f64>,
    /// Completed round-trip trades.
    pub trades: Vec<Trade>,
    /// Annualised Sharpe ratio (ddof=1, rf=0).
    pub sharpe_ratio: f64,
    /// Maximum peak-to-trough drawdown as a positive fraction in [0, 1].
    pub max_drawdown: f64,
    /// Compound annual growth rate over the full window.
    pub cagr: f64,
    /// Fraction of trades with positive return.
    pub win_rate: f64,
    /// Gross profit / gross loss (inf when no losing trades).
    pub profit_factor: f64,
    /// Total return over the full window as a decimal.
    pub total_return: f64,
}

// ── Core engine ───────────────────────────────────────────────────────────────

/// Run a vectorised single-asset backtest.
///
/// # Arguments
/// * `adj_close`       — adjusted close prices in ascending date order.
/// * `signals` — raw position signal per bar (same length). The engine
///   shifts these one bar forward; `signals[t]` is the position *entered*
///   at the close of bar *t* and *held* during bar *t+1*.
/// * `commission_pct` — one-way commission as a fraction of trade value
///   (e.g. `0.001` for 10 bps), applied on each position change.
/// * `initial_capital` — starting portfolio value.
///
/// # Panics
/// Panics if `adj_close` and `signals` have different lengths or are empty.
pub fn run_backtest(
    adj_close: &[f64],
    signals: &[f64],
    commission_pct: f64,
    initial_capital: f64,
) -> BacktestResult {
    let n = adj_close.len();
    assert_eq!(
        signals.len(),
        n,
        "adj_close and signals must have the same length"
    );
    assert!(n > 0, "adj_close must not be empty");

    // ── Step 1: Daily returns (pct_change; first bar = 0) ─────────────────
    let mut daily_returns = vec![0.0_f64; n];
    for i in 1..n {
        let prev = adj_close[i - 1];
        if prev != 0.0 {
            daily_returns[i] = (adj_close[i] - prev) / prev;
        }
    }

    // ── Step 2: Positions = signals shifted by 1 bar (no lookahead) ───────
    // positions[0] = 0 always — no prior signal available.
    let mut positions = vec![0.0_f64; n];
    positions[1..n].copy_from_slice(&signals[..(n - 1)]);

    // ── Step 3: Net returns = gross − transaction costs ────────────────────
    // pos_delta[0] = 0 (mirrors pandas .diff().abs().fillna(0.0)).
    let mut net_returns = vec![0.0_f64; n];
    for i in 0..n {
        let gross = positions[i] * daily_returns[i];
        let delta = if i == 0 {
            0.0
        } else {
            (positions[i] - positions[i - 1]).abs()
        };
        net_returns[i] = gross - commission_pct * delta;
    }

    // ── Step 4: Equity curve ───────────────────────────────────────────────
    let mut equity = vec![0.0_f64; n];
    equity[0] = initial_capital * (1.0 + net_returns[0]);
    for i in 1..n {
        equity[i] = equity[i - 1] * (1.0 + net_returns[i]);
    }

    // ── Step 5: Drawdown series ────────────────────────────────────────────
    let mut drawdown_curve = vec![0.0_f64; n];
    let mut running_max = equity[0];
    for i in 0..n {
        if equity[i] > running_max {
            running_max = equity[i];
        }
        drawdown_curve[i] = if running_max > 0.0 {
            (equity[i] - running_max) / running_max
        } else {
            0.0
        };
    }

    // ── Step 6: Trade log ──────────────────────────────────────────────────
    let trades = build_trade_log(&positions, &net_returns, n);

    // ── Step 7: Scalar metrics ─────────────────────────────────────────────
    let sharpe = sharpe_ratio(&net_returns);
    // max_drawdown: drawdown_curve values are ≤ 0; max_dd is the largest magnitude.
    let max_dd = drawdown_curve.iter().cloned().fold(0.0_f64, f64::min).abs();
    let total_ret = if initial_capital > 0.0 {
        equity[n - 1] / initial_capital - 1.0
    } else {
        0.0
    };
    let cagr = cagr_metric(initial_capital, equity[n - 1], n);
    let trade_rets: Vec<f64> = trades.iter().map(|t| t.ret).collect();
    let wr = win_rate(&trade_rets);
    let pf = profit_factor(&trade_rets);

    BacktestResult {
        equity_curve: equity,
        drawdown_curve,
        net_returns,
        trades,
        sharpe_ratio: sharpe,
        max_drawdown: max_dd,
        cagr,
        win_rate: wr,
        profit_factor: pf,
        total_return: total_ret,
    }
}

// ── Trade log builder ─────────────────────────────────────────────────────────

fn build_trade_log(positions: &[f64], net_returns: &[f64], n: usize) -> Vec<Trade> {
    let mut trades: Vec<Trade> = Vec::new();
    let mut in_trade = false;
    let mut entry_idx: usize = 0;
    let mut direction: i8 = 0;
    let mut trade_rets: Vec<f64> = Vec::new();

    for i in 0..n {
        let pos = positions[i];
        let ret = net_returns[i];

        if !in_trade {
            if pos != 0.0 {
                in_trade = true;
                entry_idx = i;
                direction = if pos > 0.0 { 1 } else { -1 };
                trade_rets = vec![ret];
            }
        } else if pos != 0.0 {
            trade_rets.push(ret);
            direction = if pos > 0.0 { 1 } else { -1 };
        } else {
            // Position closed — record the round-trip.
            let compound = trade_rets.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
            trades.push(Trade {
                entry_idx,
                exit_idx: i - 1,
                direction,
                ret: compound,
            });
            in_trade = false;
            trade_rets.clear();
        }
    }

    // Close any open trade at end of data.
    if in_trade && !trade_rets.is_empty() {
        let compound = trade_rets.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
        trades.push(Trade {
            entry_idx,
            exit_idx: n - 1,
            direction,
            ret: compound,
        });
    }

    trades
}

// ── Metric kernels ────────────────────────────────────────────────────────────

/// Annualised Sharpe ratio (ddof=1, rf=0).  Returns 0.0 when std is zero.
pub fn sharpe_ratio(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / n as f64;
    // ddof=1 matches pandas Series.std() default.
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = var.sqrt();
    if std == 0.0 || std.is_nan() {
        return 0.0;
    }
    (mean / std) * TRADING_DAYS_PER_YEAR.sqrt()
}

/// Maximum peak-to-trough drawdown as a positive fraction.
pub fn max_drawdown_metric(equity: &[f64]) -> f64 {
    if equity.is_empty() {
        return 0.0;
    }
    let mut running_max = equity[0];
    let mut max_dd = 0.0_f64;
    for &e in equity {
        if e > running_max {
            running_max = e;
        }
        if running_max > 0.0 {
            let dd = (running_max - e) / running_max;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }
    max_dd
}

/// CAGR over a full window of `n_bars` trading days.
///
/// Mirrors `quant.backtest.metrics.cagr(equity_curve, n_trading_days)` where
/// `equity_curve` is rebased to 1.0 at the start.
pub fn cagr_metric(initial_capital: f64, final_equity: f64, n_bars: usize) -> f64 {
    if n_bars == 0 || initial_capital <= 0.0 {
        return 0.0;
    }
    let total_return = final_equity / initial_capital;
    let years = n_bars as f64 / TRADING_DAYS_PER_YEAR;
    if total_return <= 0.0 || years <= 0.0 {
        return 0.0;
    }
    total_return.powf(1.0 / years) - 1.0
}

/// Fraction of trades with positive return.  Returns 0.0 for empty slice.
pub fn win_rate(trade_returns: &[f64]) -> f64 {
    if trade_returns.is_empty() {
        return 0.0;
    }
    let winners = trade_returns.iter().filter(|&&r| r > 0.0).count();
    winners as f64 / trade_returns.len() as f64
}

/// Gross profit / gross loss.  Returns `f64::INFINITY` when there are no
/// losing trades (with at least one winner), or 0.0 when there are no trades
/// or no winning trades.
pub fn profit_factor(trade_returns: &[f64]) -> f64 {
    if trade_returns.is_empty() {
        return 0.0;
    }
    let gross_profit: f64 = trade_returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = trade_returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
    if gross_loss == 0.0 {
        if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        gross_profit / gross_loss
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-asset portfolio backtester
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for a multi-asset portfolio backtest.
#[derive(Debug, Clone)]
pub struct PortfolioBacktestConfig {
    /// Starting portfolio value.
    pub initial_capital: f64,
    /// One-way commission as a fraction (e.g. 0.001 = 10 bps).
    pub commission_pct: f64,
    /// Rebalance every N bars. 1 = every bar, 21 = monthly, etc.
    pub rebalance_every: usize,
}

impl Default for PortfolioBacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 1_000_000.0,
            commission_pct: 0.001,
            rebalance_every: 21,
        }
    }
}

/// Per-asset contribution breakdown at each bar.
#[derive(Debug, Clone)]
pub struct AssetAttribution {
    pub symbol: String,
    /// Weight at each bar (length == n_bars).
    pub weight_series: Vec<f64>,
    /// Return contribution at each bar (weight × asset return).
    pub return_contribution: Vec<f64>,
    /// Cumulative return contribution over the backtest.
    pub total_contribution: f64,
}

/// Full result of a `run_portfolio_backtest` call.
pub struct PortfolioBacktestResult {
    /// Portfolio equity curve (length == n_bars).
    pub equity_curve: Vec<f64>,
    /// Rolling drawdown at each bar (non-positive fraction of peak).
    pub drawdown_curve: Vec<f64>,
    /// Portfolio-level net return per bar.
    pub net_returns: Vec<f64>,
    /// Annualised Sharpe ratio (ddof=1, rf=0).
    pub sharpe_ratio: f64,
    /// Maximum peak-to-trough drawdown as a positive fraction.
    pub max_drawdown: f64,
    /// Compound annual growth rate.
    pub cagr: f64,
    /// Total return over the backtest window.
    pub total_return: f64,
    /// Average one-way turnover per rebalance (fraction of portfolio).
    pub avg_turnover: f64,
    /// Number of rebalance events executed.
    pub n_rebalances: usize,
    /// Per-asset attribution breakdown.
    pub asset_attribution: Vec<AssetAttribution>,
}

/// Run a vectorised multi-asset portfolio backtest.
///
/// # Arguments
/// * `symbols`   — asset names (length `n_assets`).
/// * `returns`   — flat row-major returns matrix, shape `[n_bars × n_assets]`.
///   `returns[bar * n_assets + asset_idx]` = return for that bar and asset.
/// * `signals`   — flat row-major signal matrix, same shape as `returns`.
///   Signal at bar *t* determines the target weight for bar *t+1* (no lookahead).
///   Signals are normalised to sum to 1.0 at each rebalance point.
/// * `config`    — backtest configuration.
///
/// # Panics
/// Panics if matrix dimensions are inconsistent or empty.
pub fn run_portfolio_backtest(
    symbols: &[String],
    returns: &[f64],
    signals: &[f64],
    config: &PortfolioBacktestConfig,
) -> PortfolioBacktestResult {
    let n_assets = symbols.len();
    assert!(n_assets > 0, "symbols must not be empty");

    let n_bars = returns.len() / n_assets;
    assert_eq!(returns.len(), n_bars * n_assets, "returns matrix size mismatch");
    assert_eq!(signals.len(), n_bars * n_assets, "signals matrix size mismatch");
    assert!(n_bars > 0, "must have at least one bar");

    // ── Step 1: Build target-weight matrix (shifted by 1 bar, no lookahead) ─
    // weights[bar][asset] = target weight for bar `bar`.
    // Bar 0: no signal → equal weight or cash.
    let mut weights = vec![vec![0.0_f64; n_assets]; n_bars];

    // Bar 0 is always cash (no prior signal).
    // For subsequent bars, normalise signals from bar t-1 into weights for bar t.
    // Only update weights at rebalance points; hold between them.
    let mut last_rebalance_weights = vec![0.0_f64; n_assets];

    for bar in 1..n_bars {
        let is_rebalance = (bar - 1) % config.rebalance_every == 0;

        if is_rebalance {
            // Read signals from prior bar (no lookahead).
            let sig_start = (bar - 1) * n_assets;
            let raw_signals = &signals[sig_start..sig_start + n_assets];

            // Normalise to weights summing to 1.0 (long-only: clamp negatives to 0).
            let positive: Vec<f64> = raw_signals.iter().map(|&s| s.max(0.0)).collect();
            let total: f64 = positive.iter().sum();

            if total > 1e-12 {
                for j in 0..n_assets {
                    last_rebalance_weights[j] = positive[j] / total;
                }
            }
            // If total ≈ 0, keep previous weights (stay in current allocation).
        }

        weights[bar] = last_rebalance_weights.clone();
    }

    // ── Step 2: Compute portfolio returns with drift + transaction costs ─────
    let mut net_returns = vec![0.0_f64; n_bars];
    let mut turnover_sum = 0.0_f64;
    let mut n_rebalances = 0usize;

    // Track actual (drifted) weights for turnover calculation.
    let mut actual_weights = vec![0.0_f64; n_assets];

    // Per-asset return contribution tracking.
    let mut asset_weight_series: Vec<Vec<f64>> = vec![vec![0.0; n_bars]; n_assets];
    let mut asset_ret_contrib: Vec<Vec<f64>> = vec![vec![0.0; n_bars]; n_assets];

    for bar in 0..n_bars {
        let ret_start = bar * n_assets;
        let target = &weights[bar];

        // Transaction cost: turnover from actual → target weights.
        let mut bar_turnover = 0.0_f64;
        for j in 0..n_assets {
            bar_turnover += (target[j] - actual_weights[j]).abs();
        }
        let one_way_turnover = bar_turnover / 2.0;
        let cost = config.commission_pct * bar_turnover;

        if bar > 0 && bar_turnover > 1e-12 {
            turnover_sum += one_way_turnover;
            n_rebalances += 1;
        }

        // Portfolio return = Σ(w_j × r_j) − cost
        let mut gross_ret = 0.0_f64;
        for j in 0..n_assets {
            let asset_ret = returns[ret_start + j];
            let contribution = target[j] * asset_ret;
            gross_ret += contribution;
            asset_ret_contrib[j][bar] = contribution;
            asset_weight_series[j][bar] = target[j];
        }

        net_returns[bar] = gross_ret - cost;

        // Drift actual weights based on asset returns.
        let total_growth = 1.0 + gross_ret;
        if total_growth > 1e-12 {
            for j in 0..n_assets {
                actual_weights[j] = target[j] * (1.0 + returns[ret_start + j]) / total_growth;
            }
        } else {
            actual_weights = target.to_vec();
        }
    }

    // ── Step 3: Equity curve ────────────────────────────────────────────────
    let mut equity = vec![0.0_f64; n_bars];
    equity[0] = config.initial_capital * (1.0 + net_returns[0]);
    for i in 1..n_bars {
        equity[i] = equity[i - 1] * (1.0 + net_returns[i]);
    }

    // ── Step 4: Drawdown series ─────────────────────────────────────────────
    let mut drawdown_curve = vec![0.0_f64; n_bars];
    let mut running_max = equity[0];
    for i in 0..n_bars {
        if equity[i] > running_max {
            running_max = equity[i];
        }
        drawdown_curve[i] = if running_max > 0.0 {
            (equity[i] - running_max) / running_max
        } else {
            0.0
        };
    }

    // ── Step 5: Scalar metrics ──────────────────────────────────────────────
    let sr = sharpe_ratio(&net_returns);
    let max_dd = drawdown_curve.iter().cloned().fold(0.0_f64, f64::min).abs();
    let total_ret = if config.initial_capital > 0.0 {
        equity[n_bars - 1] / config.initial_capital - 1.0
    } else {
        0.0
    };
    let cagr_val = cagr_metric(config.initial_capital, equity[n_bars - 1], n_bars);
    let avg_turnover = if n_rebalances > 0 {
        turnover_sum / n_rebalances as f64
    } else {
        0.0
    };

    // ── Step 6: Per-asset attribution ───────────────────────────────────────
    let asset_attribution: Vec<AssetAttribution> = (0..n_assets)
        .map(|j| {
            let total_contribution: f64 = asset_ret_contrib[j].iter().sum();
            AssetAttribution {
                symbol: symbols[j].clone(),
                weight_series: asset_weight_series[j].clone(),
                return_contribution: asset_ret_contrib[j].clone(),
                total_contribution,
            }
        })
        .collect();

    PortfolioBacktestResult {
        equity_curve: equity,
        drawdown_curve,
        net_returns,
        sharpe_ratio: sr,
        max_drawdown: max_dd,
        cagr: cagr_val,
        total_return: total_ret,
        avg_turnover,
        n_rebalances,
        asset_attribution,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn prices_linear(n: usize, start: f64, step: f64) -> Vec<f64> {
        (0..n).map(|i| start + i as f64 * step).collect()
    }

    // ── Metric kernels ────────────────────────────────────────────────────

    #[test]
    fn test_sharpe_zero_returns() {
        let ret = vec![0.0_f64; 100];
        assert_eq!(sharpe_ratio(&ret), 0.0);
    }

    #[test]
    fn test_sharpe_positive_drift() {
        let ret: Vec<f64> = vec![0.001_f64; 252];
        let sr = sharpe_ratio(&ret);
        assert!(sr > 0.0, "positive drift should yield positive Sharpe");
    }

    #[test]
    fn test_max_drawdown_no_drawdown() {
        let equity = vec![1.0, 1.1, 1.2, 1.3];
        assert!(max_drawdown_metric(&equity) < 1e-12);
    }

    #[test]
    fn test_max_drawdown_simple() {
        let equity = vec![1.0, 1.2, 0.8, 1.0];
        let expected = (1.2 - 0.8) / 1.2;
        let result = max_drawdown_metric(&equity);
        assert!((result - expected).abs() < 1e-9, "got {result}");
    }

    #[test]
    fn test_cagr_flat_equity() {
        let result = cagr_metric(1.0, 1.0, 252);
        assert!(result.abs() < 1e-12);
    }

    #[test]
    fn test_cagr_doubles_in_one_year() {
        let result = cagr_metric(1.0, 2.0, 252);
        assert!((result - 1.0).abs() < 1e-9, "got {result}");
    }

    #[test]
    fn test_win_rate_empty() {
        assert_eq!(win_rate(&[]), 0.0);
    }

    #[test]
    fn test_win_rate_all_winners() {
        assert!((win_rate(&[0.01, 0.02, 0.005]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_win_rate_half() {
        assert!((win_rate(&[0.01, -0.01]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_profit_factor_no_losers() {
        assert_eq!(profit_factor(&[0.1, 0.2, 0.3]), f64::INFINITY);
    }

    #[test]
    fn test_profit_factor_no_winners() {
        assert_eq!(profit_factor(&[-0.1, -0.2]), 0.0);
    }

    #[test]
    fn test_profit_factor_mixed() {
        let pf = profit_factor(&[0.3, -0.1]);
        assert!((pf - 3.0).abs() < 1e-9, "got {pf}");
    }

    // ── Engine ────────────────────────────────────────────────────────────

    #[test]
    fn test_flat_strategy_equity_flat() {
        let prices = prices_linear(30, 100.0, 1.0);
        let signals = vec![0.0_f64; 30];
        let r = run_backtest(&prices, &signals, 0.001, 1.0);
        assert!(r.equity_curve.iter().all(|&v| (v - 1.0).abs() < 1e-12));
        assert_eq!(r.trades.len(), 0);
        assert!(r.total_return.abs() < 1e-12);
    }

    #[test]
    fn test_always_long_rising_market() {
        let prices: Vec<f64> = (0..252).map(|i| 100.0 * 1.001_f64.powi(i)).collect();
        let signals = vec![1.0_f64; 252];
        let r = run_backtest(&prices, &signals, 0.0, 1.0);
        assert!(r.total_return > 0.0);
        assert!(r.cagr > 0.0);
    }

    #[test]
    fn test_no_lookahead_bias() {
        let prices = vec![100.0, 105.0, 103.0, 108.0];
        let signals = vec![1.0_f64; 4];
        let r = run_backtest(&prices, &signals, 0.0, 1.0);
        // equity[0] = initial_capital because positions[0] = 0 (shift) and
        // daily_returns[0] = 0 (pct_change first bar).
        assert!((r.equity_curve[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_commission_reduces_return() {
        let prices: Vec<f64> = (0..252).map(|i| 100.0 * 1.001_f64.powi(i)).collect();
        let signals = vec![1.0_f64; 252];
        let zero_cost = run_backtest(&prices, &signals, 0.0, 1.0);
        let with_cost = run_backtest(&prices, &signals, 0.001, 1.0);
        assert!(with_cost.total_return < zero_cost.total_return);
    }

    #[test]
    fn test_single_long_trade_detected() {
        // Signal: long for 5 bars then flat
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 105.0];
        let signals = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let r = run_backtest(&prices, &signals, 0.0, 1.0);
        assert_eq!(r.trades.len(), 1);
        assert_eq!(r.trades[0].direction, 1);
        assert!(r.trades[0].ret > 0.0);
    }

    #[test]
    fn test_single_short_trade_detected() {
        let prices = vec![105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 100.0];
        let signals = vec![-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0];
        let r = run_backtest(&prices, &signals, 0.0, 1.0);
        assert_eq!(r.trades.len(), 1);
        assert_eq!(r.trades[0].direction, -1);
        assert!(r.trades[0].ret > 0.0);
    }

    #[test]
    fn test_equity_length_matches_input() {
        let prices = prices_linear(50, 100.0, 1.0);
        let signals = vec![1.0_f64; 50];
        let r = run_backtest(&prices, &signals, 0.001, 1.0);
        assert_eq!(r.equity_curve.len(), 50);
        assert_eq!(r.drawdown_curve.len(), 50);
        assert_eq!(r.net_returns.len(), 50);
    }

    #[test]
    fn test_max_drawdown_declining_long() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 * 0.99_f64.powi(i)).collect();
        let signals = vec![1.0_f64; 50];
        let r = run_backtest(&prices, &signals, 0.0, 1.0);
        assert!(r.max_drawdown > 0.0);
    }

    // ── Metric kernel edge cases ──────────────────────────────────────────

    #[test]
    fn test_max_drawdown_metric_empty() {
        assert_eq!(max_drawdown_metric(&[]), 0.0);
    }

    #[test]
    fn test_max_drawdown_metric_single_bar() {
        assert_eq!(max_drawdown_metric(&[1.0]), 0.0);
    }

    #[test]
    fn test_cagr_zero_bars() {
        assert_eq!(cagr_metric(1.0, 2.0, 0), 0.0);
    }

    #[test]
    fn test_cagr_negative_total_return() {
        // final_equity = 0 → total_return <= 0 → returns 0.0
        assert_eq!(cagr_metric(1.0, 0.0, 252), 0.0);
    }

    #[test]
    fn test_sharpe_single_observation() {
        // n < 2 → 0.0
        assert_eq!(sharpe_ratio(&[0.01]), 0.0);
    }

    #[test]
    fn test_win_rate_all_losers() {
        assert_eq!(win_rate(&[-0.01, -0.02, -0.005]), 0.0);
    }

    // ── Trade log edge cases ──────────────────────────────────────────────

    #[test]
    fn test_open_trade_captured_at_end_of_data() {
        // Signal never returns to zero — trade should still be recorded.
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let signals = vec![1.0_f64; 5];
        let r = run_backtest(&prices, &signals, 0.0, 1.0);
        assert_eq!(r.trades.len(), 1);
        assert_eq!(r.trades[0].exit_idx, 4); // pinned to last bar
    }

    #[test]
    fn test_multiple_trades_alternating_signal() {
        // Two distinct round-trips: long then short.
        let prices = vec![100.0, 102.0, 101.0, 103.0, 100.0, 98.0, 99.0];
        // long for 2 bars, flat, short for 2 bars, flat
        let signals = vec![1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0];
        let r = run_backtest(&prices, &signals, 0.0, 1.0);
        assert_eq!(r.trades.len(), 2);
        assert_eq!(r.trades[0].direction, 1);
        assert_eq!(r.trades[1].direction, -1);
    }

    #[test]
    fn test_total_return_consistent_with_equity_curve() {
        let prices: Vec<f64> = (0..252).map(|i| 100.0 * 1.001_f64.powi(i)).collect();
        let signals = vec![1.0_f64; 252];
        let r = run_backtest(&prices, &signals, 0.0, 10_000.0);
        let expected_tr = r.equity_curve[251] / 10_000.0 - 1.0;
        assert!((r.total_return - expected_tr).abs() < 1e-9);
    }

    // ── Portfolio backtest tests ─────────────────────────────────────────

    fn syms(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("S{}", i)).collect()
    }

    /// Build a flat returns matrix: n_bars × n_assets, each asset returns `drift`.
    fn uniform_returns(n_bars: usize, n_assets: usize, drift: f64) -> Vec<f64> {
        vec![drift; n_bars * n_assets]
    }

    /// Build a signal matrix: equal signal across all assets.
    fn equal_signals(n_bars: usize, n_assets: usize) -> Vec<f64> {
        vec![1.0; n_bars * n_assets]
    }

    #[test]
    fn test_portfolio_zero_signal_stays_cash() {
        let n = 2;
        let bars = 50;
        let returns = uniform_returns(bars, n, 0.01);
        let signals = vec![0.0_f64; bars * n];
        let config = PortfolioBacktestConfig {
            initial_capital: 1_000_000.0,
            commission_pct: 0.0,
            rebalance_every: 1,
        };
        let r = run_portfolio_backtest(&syms(n), &returns, &signals, &config);
        // Zero signals → zero weights → portfolio stays flat.
        assert!(r.total_return.abs() < 1e-12, "expected flat, got {}", r.total_return);
    }

    #[test]
    fn test_portfolio_equal_weight_positive_drift() {
        let n = 4;
        let bars = 252;
        let returns = uniform_returns(bars, n, 0.001);
        let signals = equal_signals(bars, n);
        let config = PortfolioBacktestConfig {
            initial_capital: 1_000_000.0,
            commission_pct: 0.0,
            rebalance_every: 1,
        };
        let r = run_portfolio_backtest(&syms(n), &returns, &signals, &config);
        assert!(r.total_return > 0.0, "positive drift should yield positive return");
        assert!(r.cagr > 0.0);
        assert!(r.sharpe_ratio > 0.0);
    }

    #[test]
    fn test_portfolio_equity_length_matches_bars() {
        let n = 3;
        let bars = 100;
        let returns = uniform_returns(bars, n, 0.001);
        let signals = equal_signals(bars, n);
        let config = PortfolioBacktestConfig::default();
        let r = run_portfolio_backtest(&syms(n), &returns, &signals, &config);
        assert_eq!(r.equity_curve.len(), bars);
        assert_eq!(r.drawdown_curve.len(), bars);
        assert_eq!(r.net_returns.len(), bars);
    }

    #[test]
    fn test_portfolio_commission_reduces_return() {
        let n = 4;
        let bars = 252;
        let returns = uniform_returns(bars, n, 0.001);
        let signals = equal_signals(bars, n);

        let no_cost = run_portfolio_backtest(
            &syms(n),
            &returns,
            &signals,
            &PortfolioBacktestConfig {
                commission_pct: 0.0,
                rebalance_every: 1,
                ..Default::default()
            },
        );
        let with_cost = run_portfolio_backtest(
            &syms(n),
            &returns,
            &signals,
            &PortfolioBacktestConfig {
                commission_pct: 0.01,
                rebalance_every: 1,
                ..Default::default()
            },
        );
        assert!(with_cost.total_return < no_cost.total_return);
    }

    #[test]
    fn test_portfolio_rebalance_frequency_affects_turnover() {
        let n = 4;
        let bars = 252;
        // Alternating signals to create turnover.
        let mut signals = vec![0.0_f64; bars * n];
        for bar in 0..bars {
            let asset = bar % n;
            signals[bar * n + asset] = 1.0;
        }
        let returns = uniform_returns(bars, n, 0.001);

        let frequent = run_portfolio_backtest(
            &syms(n),
            &returns,
            &signals,
            &PortfolioBacktestConfig {
                commission_pct: 0.0,
                rebalance_every: 1,
                ..Default::default()
            },
        );
        let infrequent = run_portfolio_backtest(
            &syms(n),
            &returns,
            &signals,
            &PortfolioBacktestConfig {
                commission_pct: 0.0,
                rebalance_every: 21,
                ..Default::default()
            },
        );
        assert!(frequent.n_rebalances > infrequent.n_rebalances);
    }

    #[test]
    fn test_portfolio_attribution_sums_to_gross_return() {
        let n = 3;
        let bars = 100;
        let returns = uniform_returns(bars, n, 0.002);
        let signals = equal_signals(bars, n);
        let config = PortfolioBacktestConfig {
            commission_pct: 0.0,
            rebalance_every: 1,
            ..Default::default()
        };
        let r = run_portfolio_backtest(&syms(n), &returns, &signals, &config);

        // At each bar, sum of per-asset contributions ≈ portfolio gross return.
        for bar in 1..bars {
            let contrib_sum: f64 = r.asset_attribution.iter().map(|a| a.return_contribution[bar]).sum();
            // Without commission, net_returns == gross return.
            assert!(
                (contrib_sum - r.net_returns[bar]).abs() < 1e-12,
                "bar {}: contrib={} vs net_ret={}",
                bar, contrib_sum, r.net_returns[bar]
            );
        }
    }

    #[test]
    fn test_portfolio_attribution_has_all_assets() {
        let n = 5;
        let bars = 50;
        let returns = uniform_returns(bars, n, 0.001);
        let signals = equal_signals(bars, n);
        let config = PortfolioBacktestConfig::default();
        let r = run_portfolio_backtest(&syms(n), &returns, &signals, &config);
        assert_eq!(r.asset_attribution.len(), n);
        for (j, attr) in r.asset_attribution.iter().enumerate() {
            assert_eq!(attr.symbol, format!("S{}", j));
            assert_eq!(attr.weight_series.len(), bars);
            assert_eq!(attr.return_contribution.len(), bars);
        }
    }

    #[test]
    fn test_portfolio_no_lookahead_bar_zero_is_cash() {
        let n = 2;
        let bars = 10;
        let returns = uniform_returns(bars, n, 0.05);
        let signals = equal_signals(bars, n);
        let config = PortfolioBacktestConfig {
            initial_capital: 1_000_000.0,
            commission_pct: 0.0,
            rebalance_every: 1,
        };
        let r = run_portfolio_backtest(&syms(n), &returns, &signals, &config);
        // Bar 0 has zero weights (no prior signal), so equity[0] = initial_capital.
        assert!(
            (r.equity_curve[0] - 1_000_000.0).abs() < 1e-6,
            "bar 0 equity should be initial capital, got {}",
            r.equity_curve[0]
        );
    }

    #[test]
    fn test_portfolio_total_return_consistent_with_equity() {
        let n = 3;
        let bars = 252;
        let returns = uniform_returns(bars, n, 0.001);
        let signals = equal_signals(bars, n);
        let config = PortfolioBacktestConfig {
            initial_capital: 1_000_000.0,
            commission_pct: 0.001,
            rebalance_every: 21,
        };
        let r = run_portfolio_backtest(&syms(n), &returns, &signals, &config);
        let expected_tr = r.equity_curve[bars - 1] / 1_000_000.0 - 1.0;
        assert!(
            (r.total_return - expected_tr).abs() < 1e-9,
            "total_return={} vs equity-derived={}",
            r.total_return, expected_tr
        );
    }

    #[test]
    fn test_portfolio_declining_market_has_drawdown() {
        let n = 2;
        let bars = 100;
        let returns = uniform_returns(bars, n, -0.005);
        let signals = equal_signals(bars, n);
        let config = PortfolioBacktestConfig {
            commission_pct: 0.0,
            rebalance_every: 1,
            ..Default::default()
        };
        let r = run_portfolio_backtest(&syms(n), &returns, &signals, &config);
        assert!(r.max_drawdown > 0.0, "declining market should have drawdown");
        assert!(r.total_return < 0.0, "declining market should have negative return");
    }

    #[test]
    fn test_portfolio_single_asset_matches_single_engine() {
        // With 1 asset, portfolio backtest should produce similar results
        // to the single-asset engine (modulo rebalance mechanics).
        let bars = 252;
        let drift = 0.001;
        let returns = vec![drift; bars];
        let signals = vec![1.0_f64; bars];
        let config = PortfolioBacktestConfig {
            initial_capital: 1.0,
            commission_pct: 0.0,
            rebalance_every: 1,
        };
        let r = run_portfolio_backtest(
            &["AAPL".to_string()],
            &returns,
            &signals,
            &config,
        );
        // With constant drift and no commission, both should compound equally.
        let expected_final = (1.0 + drift).powi((bars - 1) as i32);
        let actual_final = r.equity_curve[bars - 1];
        assert!(
            (actual_final - expected_final).abs() / expected_final < 1e-6,
            "expected {expected_final}, got {actual_final}"
        );
    }
}
