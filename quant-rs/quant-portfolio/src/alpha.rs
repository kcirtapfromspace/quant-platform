//! Alpha score types and combination methods.
//!
//! Mirrors `quant.portfolio.alpha`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Alpha score ───────────────────────────────────────────────────────────────

/// A composite alpha score for a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaScore {
    pub symbol: String,
    /// Combined signal score in [-1, 1].
    pub score: f64,
    /// Confidence in [0, 1].
    pub confidence: f64,
    /// Target position fraction (score × confidence), clamped to [-1, 1].
    pub target_position: f64,
}

impl AlphaScore {
    pub fn new(symbol: impl Into<String>, score: f64, confidence: f64) -> Self {
        let target_position = (score * confidence).clamp(-1.0, 1.0);
        Self {
            symbol: symbol.into(),
            score: score.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            target_position,
        }
    }
}

// ── Combination method ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombinationMethod {
    /// Simple average of all signal scores.
    EqualWeight,
    /// Weighted average using explicit per-signal weights.
    StaticWeight,
    /// Weight proportional to confidence × absolute score.
    ConvictionWeighted,
}

// ── Individual signal input ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SignalInput {
    pub signal_name: String,
    pub score: f64,
    pub confidence: f64,
    pub target_position: f64,
}

// ── AlphaCombiner ─────────────────────────────────────────────────────────────

/// Combines multiple signal outputs into a single alpha score per asset.
pub struct AlphaCombiner {
    pub method: CombinationMethod,
    /// Static signal weights (used with `StaticWeight`).
    pub weights: Option<HashMap<String, f64>>,
}

impl AlphaCombiner {
    pub fn new(method: CombinationMethod, weights: Option<HashMap<String, f64>>) -> Self {
        Self { method, weights }
    }

    /// Combine a list of signal inputs for a single symbol.
    pub fn combine(&self, symbol: &str, signals: &[SignalInput]) -> AlphaScore {
        if signals.is_empty() {
            return AlphaScore::new(symbol, 0.0, 0.0);
        }

        let (score, confidence) = match self.method {
            CombinationMethod::EqualWeight => self.combine_equal(signals),
            CombinationMethod::StaticWeight => self.combine_static(signals),
            CombinationMethod::ConvictionWeighted => self.combine_conviction(signals),
        };

        AlphaScore::new(symbol, score, confidence)
    }

    fn combine_equal(&self, signals: &[SignalInput]) -> (f64, f64) {
        let n = signals.len() as f64;
        let score = signals.iter().map(|s| s.score).sum::<f64>() / n;
        let confidence = signals.iter().map(|s| s.confidence).sum::<f64>() / n;
        (score, confidence)
    }

    fn combine_static(&self, signals: &[SignalInput]) -> (f64, f64) {
        let weights = match &self.weights {
            Some(w) => w,
            None => return self.combine_equal(signals),
        };

        let mut total_weight = 0.0_f64;
        let mut score_sum = 0.0_f64;
        let mut conf_sum = 0.0_f64;

        for s in signals {
            let w = weights.get(&s.signal_name).copied().unwrap_or(1.0);
            score_sum += s.score * w;
            conf_sum += s.confidence * w;
            total_weight += w;
        }

        if total_weight == 0.0 {
            return (0.0, 0.0);
        }
        (score_sum / total_weight, conf_sum / total_weight)
    }

    fn combine_conviction(&self, signals: &[SignalInput]) -> (f64, f64) {
        let mut total_weight = 0.0_f64;
        let mut score_sum = 0.0_f64;
        let mut conf_sum = 0.0_f64;

        for s in signals {
            let w = (s.confidence * s.score.abs()).max(1e-9);
            score_sum += s.score * w;
            conf_sum += s.confidence * w;
            total_weight += w;
        }

        if total_weight == 0.0 {
            return self.combine_equal(signals);
        }
        (score_sum / total_weight, conf_sum / total_weight)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn signals() -> Vec<SignalInput> {
        vec![
            SignalInput { signal_name: "momentum".into(), score: 0.8, confidence: 0.6, target_position: 0.48 },
            SignalInput { signal_name: "mean_reversion".into(), score: -0.4, confidence: 0.9, target_position: -0.36 },
        ]
    }

    #[test]
    fn test_equal_weight_combine() {
        let combiner = AlphaCombiner::new(CombinationMethod::EqualWeight, None);
        let alpha = combiner.combine("AAPL", &signals());
        assert!((alpha.score - 0.2).abs() < 1e-9);
        assert!((alpha.confidence - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_empty_signals_returns_zero() {
        let combiner = AlphaCombiner::new(CombinationMethod::EqualWeight, None);
        let alpha = combiner.combine("AAPL", &[]);
        assert!(alpha.score.abs() < 1e-12);
        assert!(alpha.confidence.abs() < 1e-12);
    }

    #[test]
    fn test_alpha_score_clamps() {
        let a = AlphaScore::new("X", 2.0, 1.5);
        assert_eq!(a.score, 1.0);
        assert_eq!(a.confidence, 1.0);
    }
}
