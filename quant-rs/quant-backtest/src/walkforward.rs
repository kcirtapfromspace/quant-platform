//! Expanding walk-forward back-test engine for CRO gate validation.
//!
//! Signal-agnostic: callers pre-compute a flat row-major signal matrix
//! (same shape as the returns matrix) and pass it in.  The engine slices
//! that matrix into IS / OOS windows and evaluates portfolio metrics for
//! each fold.
//!
//! # Configuration defaults (QUA-85 / QUA-92 gate spec)
//! | param | value | meaning |
//! |-------|-------|---------|
//! | `is_days`   | 90  | initial in-sample bars |
//! | `oos_days`  | 30  | out-of-sample bars per fold |
//! | `step_days` | 30  | IS grows by this per fold (expanding) |
//! | `n_folds`   | 64  | maximum folds |
//! | `expanding` | true | anchored IS start (vs rolling) |

use crate::{run_portfolio_backtest, PortfolioBacktestConfig};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Walk-forward engine configuration.
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    /// Initial in-sample window length in bars.
    pub is_days: usize,
    /// Out-of-sample window per fold in bars.
    pub oos_days: usize,
    /// Bars added to the IS window on each fold step.
    pub step_days: usize,
    /// If `true`, IS start is anchored at bar 0 (expanding).
    /// If `false`, IS window is a fixed-length rolling window.
    pub expanding: bool,
    /// Maximum number of folds to evaluate.
    pub n_folds: usize,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            is_days: 90,
            oos_days: 30,
            step_days: 30,
            expanding: true,
            n_folds: 64,
        }
    }
}

// ── Fold index ────────────────────────────────────────────────────────────────

/// Bar-index ranges for a single walk-forward fold.
#[derive(Debug, Clone)]
pub struct WalkForwardFold {
    /// Zero-based fold index.
    pub fold_idx: usize,
    /// IS window: `[is_start, is_end)`.
    pub is_start: usize,
    /// IS window end (exclusive).
    pub is_end: usize,
    /// OOS window: `[oos_start, oos_end)`.
    pub oos_start: usize,
    /// OOS window end (exclusive).
    pub oos_end: usize,
}

// ── Per-fold results ──────────────────────────────────────────────────────────

/// Metrics for one fold.
#[derive(Debug, Clone)]
pub struct WalkForwardFoldResult {
    /// Zero-based fold index.
    pub fold_idx: usize,
    /// Annualised Sharpe on the IS window.
    pub is_sharpe: f64,
    /// Annualised Sharpe on the OOS window.
    pub oos_sharpe: f64,
    /// Profit factor (gross win / gross loss) on OOS bar returns.
    /// Capped at 5.0 before aggregation to avoid ∞ dominating the mean.
    pub oos_profit_factor: f64,
    /// Maximum peak-to-trough drawdown on the OOS equity curve.
    pub oos_max_drawdown: f64,
    /// Walk Forward Efficiency ratio = `oos_sharpe / is_sharpe`.
    /// `NaN` when `is_sharpe ≤ 0` (undefined).
    pub wfe_ratio: f64,
}

// ── Aggregate results ─────────────────────────────────────────────────────────

/// Aggregated walk-forward results across all completed folds.
pub struct WalkForwardResult {
    /// Per-fold detail.
    pub folds: Vec<WalkForwardFoldResult>,
    /// Number of folds that completed (may be < `config.n_folds` if data ran short).
    pub n_folds_completed: usize,
    /// Mean OOS annualised Sharpe across folds.
    pub oos_sharpe: f64,
    /// Mean OOS profit factor across folds (∞ values capped at 5.0).
    pub oos_profit_factor: f64,
    /// Maximum OOS drawdown across folds (worst fold).
    pub oos_max_drawdown: f64,
    /// Walk Forward Efficiency = mean(oos_sharpe / is_sharpe) for folds where is_sharpe > 0.
    pub wfe: f64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Generate the fold definitions for a given total bar count.
///
/// Folds are truncated when the OOS end would exceed `total_bars`.
pub fn generate_folds(total_bars: usize, config: &WalkForwardConfig) -> Vec<WalkForwardFold> {
    let mut folds = Vec::new();
    for fold_idx in 0..config.n_folds {
        let is_start = if config.expanding {
            0
        } else {
            fold_idx * config.step_days
        };
        let is_end = config.is_days + fold_idx * config.step_days;
        let oos_start = is_end;
        let oos_end = oos_start + config.oos_days;

        if is_end <= is_start || oos_end > total_bars {
            break;
        }
        folds.push(WalkForwardFold {
            fold_idx,
            is_start,
            is_end,
            oos_start,
            oos_end,
        });
    }
    folds
}

/// Run a walk-forward back-test.
///
/// # Arguments
/// * `symbols`  — asset names (length `n_assets`).
/// * `returns`  — flat row-major `[total_bars × n_assets]` returns matrix.
/// * `signals`  — flat row-major `[total_bars × n_assets]` signal matrix.
///   Signal at bar *t* is generated using only data available at bar *t*
///   (enforced by the caller; the engine makes no checks).
/// * `config`   — walk-forward parameters.
/// * `bt_config` — portfolio back-test parameters (commission, rebalance freq).
pub fn run_walk_forward(
    symbols: &[String],
    returns: &[f64],
    signals: &[f64],
    config: &WalkForwardConfig,
    bt_config: &PortfolioBacktestConfig,
) -> WalkForwardResult {
    let n_assets = symbols.len();
    assert!(n_assets > 0, "symbols must not be empty");
    assert_eq!(
        returns.len() % n_assets,
        0,
        "returns length must be a multiple of n_assets"
    );
    assert_eq!(
        returns.len(),
        signals.len(),
        "returns and signals must have the same length"
    );

    let total_bars = returns.len() / n_assets;
    let folds = generate_folds(total_bars, config);
    let mut fold_results = Vec::with_capacity(folds.len());

    for fold in &folds {
        // ── IS window ────────────────────────────────────────────────────────
        let is_ret = extract_window(returns, n_assets, fold.is_start, fold.is_end);
        let is_sig = extract_window(signals, n_assets, fold.is_start, fold.is_end);
        let is_result = run_portfolio_backtest(symbols, &is_ret, &is_sig, bt_config);
        let is_sharpe = is_result.sharpe_ratio;

        // ── OOS window ───────────────────────────────────────────────────────
        let oos_ret = extract_window(returns, n_assets, fold.oos_start, fold.oos_end);
        let oos_sig = extract_window(signals, n_assets, fold.oos_start, fold.oos_end);
        let oos_result = run_portfolio_backtest(symbols, &oos_ret, &oos_sig, bt_config);
        let oos_sharpe = oos_result.sharpe_ratio;
        let oos_pf = bar_returns_profit_factor(&oos_result.net_returns);
        let oos_maxdd = oos_result.max_drawdown;

        let wfe_ratio = if is_sharpe > 0.0 {
            oos_sharpe / is_sharpe
        } else {
            f64::NAN
        };

        fold_results.push(WalkForwardFoldResult {
            fold_idx: fold.fold_idx,
            is_sharpe,
            oos_sharpe,
            oos_profit_factor: oos_pf,
            oos_max_drawdown: oos_maxdd,
            wfe_ratio,
        });
    }

    aggregate(fold_results)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Extract a bar-slice from a flat row-major matrix.
fn extract_window(matrix: &[f64], n_assets: usize, start: usize, end: usize) -> Vec<f64> {
    let n_bars = end - start;
    let mut out = Vec::with_capacity(n_bars * n_assets);
    for bar in start..end {
        let row = bar * n_assets;
        out.extend_from_slice(&matrix[row..row + n_assets]);
    }
    out
}

/// Profit factor from a slice of per-bar portfolio returns.
///
/// Returns `f64::INFINITY` when all bars are winning and there are no losses,
/// or `0.0` when there are no trades / no winners.
fn bar_returns_profit_factor(returns: &[f64]) -> f64 {
    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
    if gross_loss < 1e-12 {
        if gross_profit > 1e-12 {
            f64::INFINITY
        } else {
            0.0
        }
    } else {
        gross_profit / gross_loss
    }
}

/// Compute also the annualised Sharpe from a flat concatenation of OOS bar returns.
///
/// Used to cross-check against the per-fold mean.
#[allow(dead_code)]
fn concat_oos_sharpe(folds: &[WalkForwardFoldResult]) -> f64 {
    // We don't have the raw returns here — just use the mean-of-folds approach.
    let n = folds.len();
    if n == 0 {
        return 0.0;
    }
    folds.iter().map(|f| f.oos_sharpe).sum::<f64>() / n as f64
}

/// Aggregate per-fold results into a `WalkForwardResult`.
fn aggregate(folds: Vec<WalkForwardFoldResult>) -> WalkForwardResult {
    let n = folds.len();
    if n == 0 {
        return WalkForwardResult {
            folds,
            n_folds_completed: 0,
            oos_sharpe: 0.0,
            oos_profit_factor: 0.0,
            oos_max_drawdown: 0.0,
            wfe: 0.0,
        };
    }

    let oos_sharpe = folds.iter().map(|f| f.oos_sharpe).sum::<f64>() / n as f64;

    // Cap ∞ at 5.0 so the mean is finite.
    let oos_pf = folds
        .iter()
        .map(|f| f.oos_profit_factor.min(5.0))
        .sum::<f64>()
        / n as f64;

    let oos_maxdd = folds
        .iter()
        .map(|f| f.oos_max_drawdown)
        .fold(0.0_f64, f64::max);

    let wfe_vals: Vec<f64> = folds
        .iter()
        .filter(|f| f.wfe_ratio.is_finite())
        .map(|f| f.wfe_ratio)
        .collect();
    let wfe = if wfe_vals.is_empty() {
        0.0
    } else {
        wfe_vals.iter().sum::<f64>() / wfe_vals.len() as f64
    };

    WalkForwardResult {
        n_folds_completed: n,
        folds,
        oos_sharpe,
        oos_profit_factor: oos_pf,
        oos_max_drawdown: oos_maxdd,
        wfe,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_prices(n: usize, drift: f64) -> Vec<f64> {
        let mut p = vec![100.0_f64];
        for i in 1..n {
            p.push(p[i - 1] * (1.0 + drift + (i as f64 * 0.001).sin() * 0.01));
        }
        p
    }

    fn prices_to_returns(prices: &[f64]) -> Vec<f64> {
        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    #[test]
    fn generate_folds_count() {
        // 90 IS + 30 OOS = 120; step=30; 64 folds requested.
        // total = 90 + 64*30 = 2010 bars → all 64 fit.
        // Use total = 500; folds until oos_end > 500.
        // fold 0: IS=[0,90), OOS=[90,120)  oos_end=120 ≤ 500 ✓
        // fold k: oos_end = 90 + 30k + 30 = 120+30k
        //         fits while 120+30k ≤ 500 → k ≤ 12.67 → 13 folds (k=0..12)
        let config = WalkForwardConfig {
            is_days: 90,
            oos_days: 30,
            step_days: 30,
            expanding: true,
            n_folds: 64,
        };
        let folds = generate_folds(500, &config);
        assert_eq!(folds.len(), 13);
    }

    #[test]
    fn folds_no_overlap_expanding() {
        let config = WalkForwardConfig::default();
        let folds = generate_folds(500, &config);
        for f in &folds {
            assert!(f.is_start < f.is_end);
            assert_eq!(f.oos_start, f.is_end);
            assert!(f.oos_start < f.oos_end);
        }
        // Consecutive OOS windows are contiguous.
        for w in folds.windows(2) {
            assert_eq!(w[1].oos_start, w[0].oos_end);
        }
    }

    #[test]
    fn run_walk_forward_basic() {
        // Single asset, trending up → expect positive OOS Sharpe.
        let prices = flat_prices(600, 0.001);
        let rets = prices_to_returns(&prices);
        let n = rets.len(); // 599 bars
        let symbols = vec!["ASSET".to_string()];
        // Signal = always 1.0 (long).
        let signals = vec![1.0_f64; n];
        let config = WalkForwardConfig {
            is_days: 90,
            oos_days: 30,
            step_days: 30,
            expanding: true,
            n_folds: 10,
        };
        let bt = PortfolioBacktestConfig::default();
        let result = run_walk_forward(&symbols, &rets, &signals, &config, &bt);
        assert!(result.n_folds_completed > 0);
        assert!(result.oos_sharpe > 0.0, "uptrend → positive OOS Sharpe");
        assert!(result.oos_max_drawdown >= 0.0);
        assert!(result.oos_profit_factor > 0.0);
    }

    #[test]
    fn run_walk_forward_zero_signal() {
        // Signal = 0 everywhere → no positions → returns ≈ 0 → Sharpe = 0.
        let prices = flat_prices(600, 0.001);
        let rets = prices_to_returns(&prices);
        let n = rets.len();
        let symbols = vec!["A".to_string()];
        let signals = vec![0.0_f64; n];
        let config = WalkForwardConfig::default();
        let bt = PortfolioBacktestConfig::default();
        let result = run_walk_forward(&symbols, &rets, &signals, &config, &bt);
        // Zero signal → portfolio stays in cash → all OOS returns ≈ -commission or 0.
        // Sharpe should be ≤ 0.
        assert!(result.oos_sharpe <= 0.01);
    }

    #[test]
    fn wfe_defined_when_is_sharpe_positive() {
        let prices = flat_prices(600, 0.001);
        let rets = prices_to_returns(&prices);
        let n = rets.len();
        let symbols = vec!["A".to_string()];
        let signals = vec![1.0_f64; n];
        let config = WalkForwardConfig::default();
        let bt = PortfolioBacktestConfig::default();
        let result = run_walk_forward(&symbols, &rets, &signals, &config, &bt);
        // WFE should be defined (finite) given positive IS Sharpe.
        let defined_folds = result
            .folds
            .iter()
            .filter(|f| f.wfe_ratio.is_finite())
            .count();
        assert!(defined_folds > 0);
    }
}
