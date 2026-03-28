use quant_backtest::{run_backtest, BacktestResult};
use quant_features::{bb_lower, bb_mid, bb_upper, ema, macd_histogram, returns, rsi};
use quant_signals::{
    mean_reversion_signal, momentum_signal, trend_following_signal, AdaptiveSignalCombiner,
    RegimeDetector, RegimeState, RegimeWeightAdapter,
};

const TRAIN_WINDOW: usize = 90;
const OOS_WINDOW: usize = 30;
const N_FOLDS: usize = 5;
const N_BARS: usize = 252;
const COMMISSION: f64 = 0.001;

fn synthetic_prices() -> Vec<f64> {
    let mut prices = Vec::with_capacity(N_BARS);
    let mut price = 100.0_f64;
    prices.push(price);

    for bar in 1..N_BARS {
        let regime_bucket = (bar / 21) % 4;
        let drift = match regime_bucket {
            0 => 0.0012,
            1 => -0.0009,
            2 => 0.0015,
            _ => -0.0011,
        };
        let vol = match regime_bucket {
            0 | 1 => 0.006,
            _ => 0.025,
        };
        let cyclical = (bar as f64 * 0.37).sin() + 0.6 * (bar as f64 * 0.11).cos();
        let ret = (drift + vol * cyclical).clamp(-0.095, 0.095);
        price *= 1.0 + ret;
        prices.push(price.max(1.0));
    }

    prices
}

fn quantise(signal: f64) -> f64 {
    if signal > 0.05 {
        1.0
    } else if signal < -0.05 {
        -1.0
    } else {
        0.0
    }
}

fn fold_backtest(prices: &[f64], signals: &[f64], fold: usize) -> BacktestResult {
    let oos_start = TRAIN_WINDOW + fold * OOS_WINDOW;
    let oos_end = oos_start + OOS_WINDOW;

    let oos_prices = &prices[oos_start..=oos_end];
    let mut oos_signals = signals[oos_start..oos_end].to_vec();
    oos_signals.push(0.0);

    run_backtest(oos_prices, &oos_signals, COMMISSION, 1.0)
}

#[test]
fn bayesian_signal_path_integrates_through_backtest_engine() {
    let prices = synthetic_prices();
    let rets = returns(&prices);
    let rsi_vals = rsi(&prices, 14);
    let bb_mid_vals = bb_mid(&prices, 20);
    let bb_upper_vals = bb_upper(&prices, 20, 2.0);
    let bb_lower_vals = bb_lower(&prices, 20, 2.0);
    let macd_hist = macd_histogram(&prices, 12, 26, 9);
    let fast_ma = ema(&prices, 12);
    let slow_ma = ema(&prices, 26);

    let mut combiner = AdaptiveSignalCombiner::new(3);
    let mut regime_detector = RegimeDetector::new();
    let regime_adapter = RegimeWeightAdapter::default();

    let fit_rets: Vec<f64> = rets[1..TRAIN_WINDOW]
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();
    regime_detector.fit(&fit_rets);

    let mut signals = vec![0.0_f64; prices.len()];
    let mut posterior_history: Vec<[f64; 3]> = Vec::new();
    let mut low_vol_visits = 0usize;
    let mut high_vol_visits = 0usize;

    for bar in 0..prices.len() - 1 {
        let next_ret = rets[bar + 1];
        if !next_ret.is_finite() {
            continue;
        }

        if bar < 50 {
            regime_detector.update(next_ret);
            continue;
        }

        let (mom, _, _) = momentum_signal(&rsi_vals[..=bar], &rets[..=bar], 20, 0.02);
        let (mr, _, _) = mean_reversion_signal(
            &bb_mid_vals[..=bar],
            &bb_upper_vals[..=bar],
            &bb_lower_vals[..=bar],
            &rets[..=bar],
            2.0,
        );
        let (tf, _, _) =
            trend_following_signal(&macd_hist[..=bar], &fast_ma[..=bar], &slow_ma[..=bar]);

        let regime_probs = regime_detector.regime_probs();
        let weighted_signal =
            combiner.combine(&[mom, mr, tf]) * regime_adapter.weight(regime_probs);
        signals[bar] = quantise(weighted_signal);

        combiner.update_ic(0, mom * next_ret);
        combiner.update_ic(1, mr * next_ret);
        combiner.update_ic(2, tf * next_ret);
        regime_detector.update(next_ret);

        let ic_estimates = combiner.ic_estimates();
        posterior_history.push([ic_estimates[0], ic_estimates[1], ic_estimates[2]]);

        match regime_detector.most_likely_regime() {
            RegimeState::LowVol => low_vol_visits += 1,
            RegimeState::HighVol => high_vol_visits += 1,
        }
    }

    assert!(
        posterior_history
            .iter()
            .flatten()
            .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)),
        "posterior IC estimates must stay finite and bounded"
    );
    assert!(low_vol_visits > 0, "expected LowVol regime visits");
    assert!(high_vol_visits > 0, "expected HighVol regime visits");

    for fold in 0..N_FOLDS {
        let result = fold_backtest(&prices, &signals, fold);
        let oos_start = TRAIN_WINDOW + fold * OOS_WINDOW;
        let oos_end = oos_start + OOS_WINDOW;
        let fold_signals = &signals[oos_start..oos_end];
        let rebalance_count = fold_signals
            .windows(2)
            .filter(|window| (window[1] - window[0]).abs() > 1e-12)
            .count();

        assert!(rebalance_count > 0, "fold {fold} had zero rebalances");
        assert!(
            fold_signals.iter().any(|signal| signal.abs() > 1e-12),
            "fold {fold} had all-zero signals"
        );
        assert!(
            result.net_returns.iter().all(|r| r.is_finite()),
            "fold {fold} net returns contained non-finite values"
        );
        assert!(
            result.equity_curve.iter().all(|v| v.is_finite()),
            "fold {fold} equity curve contained non-finite values"
        );
        assert!(
            result.sharpe_ratio.is_finite(),
            "fold {fold} sharpe was not finite"
        );
        assert!(
            result.max_drawdown.is_finite(),
            "fold {fold} max drawdown was not finite"
        );
        assert!(result.cagr.is_finite(), "fold {fold} cagr was not finite");
        assert!(
            !result.trades.is_empty(),
            "fold {fold} should generate at least one trade"
        );
    }
}
