//! Pure-Rust signal framework.
//!
//! Phase 4: full port of MomentumSignal, MeanReversionSignal, TrendFollowingSignal.
//! Phase 1 trait/struct definitions are preserved for API compatibility.
//! Phase 5: HMM-based regime detection via `RegimeDetector` / `RegimeWeightAdapter`.
//! Phase 6: Bayesian IC-weighted `AdaptiveSignalCombiner`.

pub use quant_bayes::RegimeState;
use quant_bayes::{BayesianModel, HmmRegimeModel, NormalGammaTracker};

// ── Phase-1 types (preserved) ─────────────────────────────────────────────

/// Signal direction emitted by a strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    Long,
    Short,
    Flat,
}

/// A signal emitted at a given bar.
#[derive(Debug, Clone)]
pub struct Signal {
    pub symbol: String,
    pub direction: SignalDirection,
    /// Strength in [0, 1]; 0.0 means no conviction.
    pub strength: f64,
}

impl Signal {
    pub fn new(symbol: impl Into<String>, direction: SignalDirection, strength: f64) -> Self {
        Self {
            symbol: symbol.into(),
            direction,
            strength: strength.clamp(0.0, 1.0),
        }
    }

    pub fn is_flat(&self) -> bool {
        self.direction == SignalDirection::Flat
    }
}

/// Trait implemented by all signal generators.
pub trait BaseSignal: Send + Sync {
    fn name(&self) -> &str;
    fn generate(&self, symbol: &str, closes: &[f64]) -> Option<Signal>;
}

// ── Internal helpers ──────────────────────────────────────────────────────

#[inline]
fn clamp11(v: f64) -> f64 {
    v.clamp(-1.0, 1.0)
}

#[inline]
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Return the last finite value in `slice`, or `None`.
fn last_valid(slice: &[f64]) -> Option<f64> {
    slice.iter().rev().copied().find(|v| v.is_finite())
}

/// Sample standard deviation (ddof=1) of the last `window` elements in `data`.
/// `data.len()` must be >= `window` and `window` >= 2.
fn rolling_sample_std(data: &[f64], window: usize) -> f64 {
    debug_assert!(window >= 2 && data.len() >= window);
    let slice = &data[data.len() - window..];
    let mean = slice.iter().sum::<f64>() / window as f64;
    let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (window - 1) as f64;
    var.sqrt()
}

// ── MomentumSignal kernel ─────────────────────────────────────────────────

/// Momentum signal: RSI-based score, return-magnitude confidence.
///
/// Returns `(score, confidence, target_position)` in \[-1, 1\] / \[0, 1\] / \[-1, 1\].
///
/// Algorithm mirrors `quant.signals.strategies.MomentumSignal.compute()`:
/// - score = clamp((rsi_last − 50) / 20)
/// - confidence = clamp(mean(|returns[-lookback:]|) / return_scale)
///   or 0.5 when fewer than `lookback` valid returns exist
/// - target_position = clamp(score × confidence)
///
/// Returns `(0, 0, 0)` when no finite RSI value is available.
pub fn momentum_signal(
    rsi_values: &[f64],
    returns: &[f64],
    lookback: usize,
    return_scale: f64,
) -> (f64, f64, f64) {
    let rsi_val = match last_valid(rsi_values) {
        Some(v) => v,
        None => return (0.0, 0.0, 0.0),
    };

    let score = clamp11((rsi_val - 50.0) / 20.0);

    let valid_rets: Vec<f64> = returns.iter().copied().filter(|v| v.is_finite()).collect();
    let confidence = if valid_rets.len() >= lookback {
        let recent_abs: f64 = valid_rets[valid_rets.len() - lookback..]
            .iter()
            .map(|v| v.abs())
            .sum::<f64>()
            / lookback as f64;
        clamp01(recent_abs / return_scale)
    } else {
        0.5
    };

    let target_position = clamp11(score * confidence);
    (score, confidence, target_position)
}

// ── MeanReversionSignal kernel ────────────────────────────────────────────

/// Mean-reversion signal: Bollinger Band z-score.
///
/// Returns `(score, confidence, target_position)`.
///
/// Algorithm mirrors `quant.signals.strategies.MeanReversionSignal.compute()`:
/// - Approximate current price as `mid × (1 + last_valid_return)`
/// - z = (price_approx − mid) / (band_width / 2)
/// - score = clamp(−z / num_std)   (price above mid → sell)
/// - confidence = clamp(|z| / num_std)
/// - target_position = clamp(score × confidence)
///
/// Returns `(0, 0, 0)` when any BB series is empty or bandwidth < 1e-12.
pub fn mean_reversion_signal(
    bb_mid: &[f64],
    bb_upper: &[f64],
    bb_lower: &[f64],
    returns: &[f64],
    num_std: f64,
) -> (f64, f64, f64) {
    let mid = match last_valid(bb_mid) {
        Some(v) => v,
        None => return (0.0, 0.0, 0.0),
    };
    let upper = match last_valid(bb_upper) {
        Some(v) => v,
        None => return (0.0, 0.0, 0.0),
    };
    let lower = match last_valid(bb_lower) {
        Some(v) => v,
        None => return (0.0, 0.0, 0.0),
    };

    let band_width = upper - lower;
    if band_width < 1e-12 {
        return (0.0, 0.0, 0.0);
    }

    let last_ret = last_valid(returns).unwrap_or(0.0);
    let price_approx = mid * (1.0 + last_ret);
    let half_band = band_width / 2.0;
    let z = if half_band > 0.0 {
        (price_approx - mid) / half_band
    } else {
        0.0
    };

    let score = clamp11(-z / num_std);
    let confidence = clamp01(z.abs() / num_std);
    let target_position = clamp11(score * confidence);

    (score, confidence, target_position)
}

// ── TrendFollowingSignal kernel ───────────────────────────────────────────

/// Trend-following signal: MACD histogram normalised by rolling std.
///
/// Returns `(score, confidence, target_position)`.
///
/// Algorithm mirrors `quant.signals.strategies.TrendFollowingSignal.compute()`:
/// - Collect finite values from `macd_hist`.
/// - If n ≥ 10: hist_std = sample_std(last min(20,n) values)
///   else:       hist_std = mean(|hist|) or 1.0
/// - score = clamp(last_hist / hist_std)
/// - confidence = clamp(|score| × 1.2 if SMA aligned else 0.6)
/// - target_position = clamp(score × confidence)
///
/// Returns `(0, 0, 0)` when no finite histogram or MA values are available.
pub fn trend_following_signal(
    macd_hist: &[f64],
    fast_ma: &[f64],
    slow_ma: &[f64],
) -> (f64, f64, f64) {
    let hist_valid: Vec<f64> = macd_hist
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();

    let fast_val = match last_valid(fast_ma) {
        Some(v) => v,
        None => return (0.0, 0.0, 0.0),
    };
    let slow_val = match last_valid(slow_ma) {
        Some(v) => v,
        None => return (0.0, 0.0, 0.0),
    };

    if hist_valid.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let last_hist = *hist_valid.last().unwrap();

    let hist_std = if hist_valid.len() >= 10 {
        let window = hist_valid.len().min(20);
        rolling_sample_std(&hist_valid, window)
    } else {
        let mean_abs: f64 =
            hist_valid.iter().map(|v| v.abs()).sum::<f64>() / hist_valid.len() as f64;
        if mean_abs == 0.0 {
            1.0
        } else {
            mean_abs
        }
    };

    let hist_std = if hist_std < 1e-12 { 1.0 } else { hist_std };

    let score = clamp11(last_hist / hist_std);

    let sma_bullish = fast_val > slow_val;
    let hist_bullish = last_hist > 0.0;
    let aligned = sma_bullish == hist_bullish;

    let base_confidence = score.abs();
    let confidence = clamp01(base_confidence * if aligned { 1.2 } else { 0.6 });
    let target_position = clamp11(score * confidence);

    (score, confidence, target_position)
}

// ── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Phase-1 scaffold tests (preserved) ───────────────────────────────

    struct AlwaysLong;
    impl BaseSignal for AlwaysLong {
        fn name(&self) -> &str {
            "always_long"
        }
        fn generate(&self, symbol: &str, _closes: &[f64]) -> Option<Signal> {
            Some(Signal::new(symbol, SignalDirection::Long, 1.0))
        }
    }

    #[test]
    fn test_signal_strength_clamped() {
        let s = Signal::new("AAPL", SignalDirection::Long, 2.5);
        assert_eq!(s.strength, 1.0);
    }

    #[test]
    fn test_always_long_signal() {
        let gen = AlwaysLong;
        let sig = gen.generate("AAPL", &[100.0, 101.0, 102.0]).unwrap();
        assert_eq!(sig.direction, SignalDirection::Long);
        assert_eq!(sig.symbol, "AAPL");
    }

    #[test]
    fn test_flat_detection() {
        let s = Signal::new("TSLA", SignalDirection::Flat, 0.0);
        assert!(s.is_flat());
    }

    // ── Phase-4: momentum_signal ──────────────────────────────────────────

    #[test]
    fn test_momentum_no_rsi_returns_zero() {
        let (s, c, t) = momentum_signal(&[f64::NAN; 20], &[0.01; 20], 5, 0.05);
        assert_eq!((s, c, t), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_momentum_rsi_50_score_is_zero() {
        let rsi = vec![50.0_f64];
        let rets = vec![0.01_f64; 10];
        let (score, _, _) = momentum_signal(&rsi, &rets, 5, 0.05);
        assert!((score).abs() < 1e-12);
    }

    #[test]
    fn test_momentum_rsi_70_score_is_one() {
        let rsi = vec![70.0_f64];
        let rets = vec![0.05_f64; 10];
        let (score, _, _) = momentum_signal(&rsi, &rets, 5, 0.05);
        assert!((score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_momentum_rsi_30_score_is_neg_one() {
        let rsi = vec![30.0_f64];
        let rets = vec![0.0_f64; 10];
        let (score, _, _) = momentum_signal(&rsi, &rets, 5, 0.05);
        assert!((score + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_momentum_few_returns_gives_half_confidence() {
        let rsi = vec![60.0_f64];
        let rets = vec![0.01_f64; 3]; // < lookback=5
        let (_, confidence, _) = momentum_signal(&rsi, &rets, 5, 0.05);
        assert!((confidence - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_momentum_score_confidence_in_range() {
        let rsi: Vec<f64> = (0..20).map(|i| 40.0 + i as f64 * 2.0).collect();
        let rets: Vec<f64> = (0..20).map(|i| 0.01 * (i as f64 % 5.0)).collect();
        let (s, c, t) = momentum_signal(&rsi, &rets, 5, 0.05);
        assert!((-1.0..=1.0).contains(&s));
        assert!((0.0..=1.0).contains(&c));
        assert!((-1.0..=1.0).contains(&t));
    }

    // ── Phase-4: mean_reversion_signal ───────────────────────────────────

    #[test]
    fn test_mean_reversion_no_bands_returns_zero() {
        let (s, c, t) = mean_reversion_signal(&[f64::NAN], &[f64::NAN], &[f64::NAN], &[0.0], 2.0);
        assert_eq!((s, c, t), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_mean_reversion_zero_bandwidth_returns_zero() {
        // upper == lower → zero bandwidth
        let (s, c, t) = mean_reversion_signal(&[100.0], &[100.0], &[100.0], &[0.0], 2.0);
        assert_eq!((s, c, t), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_mean_reversion_at_mid_returns_neutral() {
        // last_ret=0 → price_approx=mid → z=0 → score=0
        let (score, confidence, _) =
            mean_reversion_signal(&[100.0], &[102.0], &[98.0], &[0.0], 2.0);
        assert!(score.abs() < 1e-12);
        assert!(confidence.abs() < 1e-12);
    }

    #[test]
    fn test_mean_reversion_output_in_range() {
        let (s, c, t) = mean_reversion_signal(&[100.0], &[106.0], &[94.0], &[0.05], 2.0);
        assert!((-1.0..=1.0).contains(&s));
        assert!((0.0..=1.0).contains(&c));
        assert!((-1.0..=1.0).contains(&t));
    }

    // ── Phase-4: trend_following_signal ──────────────────────────────────

    #[test]
    fn test_trend_following_no_hist_returns_zero() {
        let (s, c, t) = trend_following_signal(&[f64::NAN; 5], &[100.0], &[99.0]);
        assert_eq!((s, c, t), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_trend_following_no_fast_ma_returns_zero() {
        let hist: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        let (s, c, t) = trend_following_signal(&hist, &[f64::NAN], &[100.0]);
        assert_eq!((s, c, t), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_trend_following_positive_hist_bullish_fast_above_slow() {
        // Positive histogram + fast > slow → aligned → score > 0
        let hist: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        let (score, _, _) = trend_following_signal(&hist, &[105.0], &[100.0]);
        assert!(score > 0.0);
    }

    #[test]
    fn test_trend_following_negative_hist_bearish() {
        let hist: Vec<f64> = (1..=20).map(|i| -(i as f64 * 0.1)).collect();
        let (score, _, _) = trend_following_signal(&hist, &[95.0], &[100.0]);
        assert!(score < 0.0);
    }

    #[test]
    fn test_trend_following_output_in_range() {
        let hist: Vec<f64> = (0..30).map(|i| (i as f64 - 15.0) * 0.05).collect();
        let (s, c, t) = trend_following_signal(&hist, &[102.0], &[100.0]);
        assert!((-1.0..=1.0).contains(&s));
        assert!((0.0..=1.0).contains(&c));
        assert!((-1.0..=1.0).contains(&t));
    }
}

// ── HMM-based regime detection ────────────────────────────────────────────────

/// Regime detector backed by a 2-state HMM with Gaussian emissions.
///
/// Replaces the legacy threshold rule (`vol > μ + 1.5σ`) with a smooth
/// probabilistic regime estimate updated online at O(1) per bar.
///
/// # Usage
/// ```
/// use quant_signals::{RegimeDetector, RegimeState};
///
/// let rets: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();
/// let mut detector = RegimeDetector::new();
/// detector.fit(&rets);
/// detector.update(0.02);
/// let [low_prob, high_prob] = detector.regime_probs();
/// assert!((low_prob + high_prob - 1.0).abs() < 1e-10);
/// ```
pub struct RegimeDetector {
    model: HmmRegimeModel,
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl RegimeDetector {
    /// Create a new detector with uninformative priors.
    pub fn new() -> Self {
        Self {
            model: HmmRegimeModel::new(),
        }
    }

    /// Train the HMM on a historical return series via Baum-Welch EM.
    pub fn fit(&mut self, returns: &[f64]) {
        self.model.fit(returns);
    }

    /// Online forward-pass update with a single new observation — O(1).
    pub fn update(&mut self, ret: f64) {
        self.model.update(ret);
    }

    /// Current smooth regime probabilities `[P(LowVol), P(HighVol)]`.
    pub fn regime_probs(&self) -> [f64; 2] {
        self.model.regime_probs()
    }

    /// Most likely current regime (argmax of filtered state probabilities).
    pub fn most_likely_regime(&self) -> RegimeState {
        self.model.most_likely_regime()
    }
}

/// Converts smooth HMM regime probabilities into a position-size weight.
///
/// Interpolates linearly between `low_vol_weight` (calm regime) and
/// `high_vol_weight` (turbulent regime), replacing the old binary on/off flag.
///
/// Default: full weight (1.0) in `LowVol`, half weight (0.5) in `HighVol`.
pub struct RegimeWeightAdapter {
    /// Multiplier applied when fully in the `LowVol` regime.
    pub low_vol_weight: f64,
    /// Multiplier applied when fully in the `HighVol` regime.
    pub high_vol_weight: f64,
}

impl Default for RegimeWeightAdapter {
    fn default() -> Self {
        Self::new(1.0, 0.5)
    }
}

impl RegimeWeightAdapter {
    /// Create with explicit per-regime weights.
    pub fn new(low_vol_weight: f64, high_vol_weight: f64) -> Self {
        Self {
            low_vol_weight,
            high_vol_weight,
        }
    }

    /// Smooth position weight: `low_vol_weight × P(LowVol) + high_vol_weight × P(HighVol)`.
    #[inline]
    pub fn weight(&self, regime_probs: [f64; 2]) -> f64 {
        self.low_vol_weight * regime_probs[0] + self.high_vol_weight * regime_probs[1]
    }
}

// ── Regime tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod regime_tests {
    use super::*;

    fn two_regime_returns() -> Vec<f64> {
        let mut v: Vec<f64> = (0..50)
            .map(|i| if i % 2 == 0 { 0.01_f64 } else { -0.01_f64 })
            .collect();
        v.extend((0..50).map(|i| if i % 2 == 0 { 0.05_f64 } else { -0.05_f64 }));
        v
    }

    #[test]
    fn test_regime_detector_fit_and_update() {
        let rets = two_regime_returns();
        let mut det = RegimeDetector::new();
        det.fit(&rets);

        let probs = det.regime_probs();
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-10);

        det.update(0.02);
        let probs2 = det.regime_probs();
        assert!((probs2[0] + probs2[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_regime_detector_detects_high_vol_end() {
        let rets = two_regime_returns();
        let mut det = RegimeDetector::new();
        det.fit(&rets);
        assert_eq!(det.most_likely_regime(), RegimeState::HighVol);
    }

    #[test]
    fn test_regime_weight_adapter_bounds() {
        let adapter = RegimeWeightAdapter::default();
        // Fully LowVol
        assert!((adapter.weight([1.0, 0.0]) - 1.0).abs() < 1e-12);
        // Fully HighVol
        assert!((adapter.weight([0.0, 1.0]) - 0.5).abs() < 1e-12);
        // 50/50 mix
        let w = adapter.weight([0.5, 0.5]);
        assert!((w - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_regime_weight_adapter_interpolates() {
        let adapter = RegimeWeightAdapter::new(1.0, 0.0);
        let w = adapter.weight([0.3, 0.7]);
        assert!((w - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_regime_state_reexport() {
        // Ensure RegimeState is accessible from quant_signals.
        let _ = RegimeState::LowVol;
        let _ = RegimeState::HighVol;
    }
}

// ── Bayesian IC-weighted signal combiner ──────────────────────────────────────

/// Adaptive signal combiner that weights each signal by its Bayesian IC estimate.
///
/// One `NormalGammaTracker` per signal slot tracks the information coefficient
/// (IC) online.  At each bar, `combine()` returns an IC-weighted average of the
/// provided signal values.  Negative-IC slots are down-weighted to zero.
///
/// # Replacing EMA weighting
/// Previously, `AdaptiveSignalCombiner` used an EMA decay on raw signal scores
/// to approximate IC.  Now each slot holds a `NormalGammaTracker` whose
/// `posterior_mean()` is the current IC estimate — probabilistically calibrated
/// and uncertainty-aware.
pub struct AdaptiveSignalCombiner {
    trackers: Vec<NormalGammaTracker>,
}

impl AdaptiveSignalCombiner {
    /// Create a combiner for `n_signals` slots with default hyperpriors.
    pub fn new(n_signals: usize) -> Self {
        Self {
            trackers: (0..n_signals)
                .map(|_| NormalGammaTracker::default())
                .collect(),
        }
    }

    /// Record an IC observation for slot `idx` (e.g. `sign(signal) × next_return`).
    ///
    /// Panics if `idx ≥ n_signals`.
    pub fn update_ic(&mut self, idx: usize, ic_obs: f64) {
        self.trackers[idx].update(ic_obs);
    }

    /// Posterior IC estimates `[f64; n_signals]` — `posterior_mean()` per slot.
    pub fn ic_estimates(&self) -> Vec<f64> {
        self.trackers.iter().map(|t| t.posterior_mean()).collect()
    }

    /// IC-weighted combination of `signal_values`.
    ///
    /// Uses `max(0, posterior_mean())` as the weight for each slot so that
    /// signals with negative estimated IC contribute nothing.  Falls back to
    /// a simple average when all IC estimates are ≤ 0.
    ///
    /// `signal_values.len()` must equal `n_signals`.
    pub fn combine(&self, signal_values: &[f64]) -> f64 {
        assert_eq!(
            signal_values.len(),
            self.trackers.len(),
            "signal_values length must equal n_signals"
        );
        let weights: Vec<f64> = self
            .trackers
            .iter()
            .map(|t| t.posterior_mean().max(0.0))
            .collect();
        let total: f64 = weights.iter().sum();
        if total > 1e-15 {
            weights
                .iter()
                .zip(signal_values.iter())
                .map(|(w, v)| w * v)
                .sum::<f64>()
                / total
        } else {
            // All IC ≤ 0: equal-weight average
            signal_values.iter().sum::<f64>() / signal_values.len() as f64
        }
    }

    /// Number of signal slots.
    pub fn n_signals(&self) -> usize {
        self.trackers.len()
    }
}

// ── AdaptiveSignalCombiner tests ──────────────────────────────────────────────

#[cfg(test)]
mod combiner_tests {
    use super::*;

    #[test]
    fn test_combine_equal_ic_is_average() {
        let mut combiner = AdaptiveSignalCombiner::new(2);
        // Feed equal positive IC to both slots
        for _ in 0..20 {
            combiner.update_ic(0, 0.3);
            combiner.update_ic(1, 0.3);
        }
        let result = combiner.combine(&[1.0, -1.0]);
        // Equal IC → equal weight → average = 0.0
        assert!(result.abs() < 1e-10, "expected 0.0, got {result}");
    }

    #[test]
    fn test_combine_higher_ic_slot_dominates() {
        let mut combiner = AdaptiveSignalCombiner::new(2);
        // Slot 0 gets strong positive IC, slot 1 gets ~zero IC
        for _ in 0..100 {
            combiner.update_ic(0, 0.5);
            combiner.update_ic(1, 0.0);
        }
        // slot 0 signal = 1.0, slot 1 signal = -1.0
        let result = combiner.combine(&[1.0, -1.0]);
        // slot 0 dominates → result should be close to 1.0
        assert!(result > 0.5, "high-IC slot should dominate: {result}");
    }

    #[test]
    fn test_combine_all_negative_ic_falls_back_to_average() {
        let combiner = AdaptiveSignalCombiner::new(2);
        // Default prior mean = 0.0 → weights are 0 → fallback to average
        let result = combiner.combine(&[2.0, -2.0]);
        assert!(
            result.abs() < 1e-10,
            "expected equal-weight average: {result}"
        );
    }

    #[test]
    fn test_update_ic_increments_tracker_n() {
        let mut combiner = AdaptiveSignalCombiner::new(3);
        combiner.update_ic(1, 0.1);
        combiner.update_ic(1, 0.2);
        assert_eq!(combiner.trackers[1].n, 2);
        assert_eq!(combiner.trackers[0].n, 0);
    }

    #[test]
    fn test_ic_estimates_length_matches_n_signals() {
        let combiner = AdaptiveSignalCombiner::new(4);
        assert_eq!(combiner.ic_estimates().len(), 4);
    }

    #[test]
    fn test_n_signals() {
        let c = AdaptiveSignalCombiner::new(5);
        assert_eq!(c.n_signals(), 5);
    }
}
