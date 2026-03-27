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
/// * `signals`         — raw position signal per bar (same length). The engine
///                       shifts these one bar forward; `signals[t]` is the
///                       position *entered* at the close of bar *t* and *held*
///                       during bar *t+1*.
/// * `commission_pct`  — one-way commission as a fraction of trade value
///                       (e.g. `0.001` for 10 bps), applied on each position change.
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
    assert_eq!(signals.len(), n, "adj_close and signals must have the same length");
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
    for i in 1..n {
        positions[i] = signals[i - 1];
    }

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
}
