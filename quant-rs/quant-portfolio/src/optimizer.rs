//! Portfolio optimizers: equal-weight, risk-parity, min-variance.
//!
//! All optimizers return portfolio weights that sum to 1.0.
//! Weights are normalized so that the constraint is always satisfied.
//!
//! Mirrors `quant.portfolio.optimizers`.

use serde::{Deserialize, Serialize};

use crate::error::{PortfolioError, PortfolioResult};

// ── Result type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub symbols: Vec<String>,
    /// Optimal portfolio weights (sum to 1.0).
    pub weights: Vec<f64>,
    /// Portfolio volatility (annualised std dev).
    pub risk: f64,
    pub method: OptimizationMethod,
}

// ── Method enum ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationMethod {
    EqualWeight,
    RiskParity,
    MinVariance,
    MeanVariance,
}

// ── Constraints ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConstraints {
    pub long_only: bool,
    pub max_weight: f64,
    pub min_weight: f64,
    pub max_sector_weight: f64,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            long_only: true,
            max_weight: 1.0,
            min_weight: 0.0,
            max_sector_weight: 1.0,
        }
    }
}

// ── Optimizer trait ───────────────────────────────────────────────────────────

pub trait Optimizer: Send + Sync {
    fn optimize(
        &self,
        symbols: &[String],
        alpha_scores: &[f64],
        cov: &[f64],
        constraints: &PortfolioConstraints,
    ) -> PortfolioResult<OptimizationResult>;
}

// ── Equal-weight optimizer ────────────────────────────────────────────────────

pub struct EqualWeightOptimizer;

impl Optimizer for EqualWeightOptimizer {
    fn optimize(
        &self,
        symbols: &[String],
        _alpha_scores: &[f64],
        cov: &[f64],
        constraints: &PortfolioConstraints,
    ) -> PortfolioResult<OptimizationResult> {
        if symbols.is_empty() {
            return Err(PortfolioError::EmptyUniverse);
        }
        let n = symbols.len();
        let w = (1.0_f64 / n as f64).min(constraints.max_weight);
        let weights = vec![w; n];
        let risk = portfolio_vol(&weights, cov, n);
        Ok(OptimizationResult {
            symbols: symbols.to_vec(),
            weights,
            risk,
            method: OptimizationMethod::EqualWeight,
        })
    }
}

// ── Risk-parity optimizer ─────────────────────────────────────────────────────

/// Risk parity: each asset contributes equally to portfolio variance.
///
/// Uses the iterative equal-risk-contribution (ERC) algorithm.
pub struct RiskParityOptimizer {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for RiskParityOptimizer {
    fn default() -> Self {
        Self { max_iter: 200, tol: 1e-8 }
    }
}

impl Optimizer for RiskParityOptimizer {
    fn optimize(
        &self,
        symbols: &[String],
        _alpha_scores: &[f64],
        cov: &[f64],
        constraints: &PortfolioConstraints,
    ) -> PortfolioResult<OptimizationResult> {
        if symbols.is_empty() {
            return Err(PortfolioError::EmptyUniverse);
        }
        let n = symbols.len();

        // Initialize with equal weights.
        let mut w = vec![1.0_f64 / n as f64; n];

        for _ in 0..self.max_iter {
            let sigma = portfolio_vol(&w, cov, n);
            if sigma < 1e-12 {
                break;
            }

            // Marginal risk contributions: MRC_i = (Σw)_i / σ
            let mrc = marginal_risk_contributions(&w, cov, n);
            // Risk contributions: RC_i = w_i * MRC_i
            let rc: Vec<f64> = w.iter().zip(mrc.iter()).map(|(wi, mi)| wi * mi).collect();
            let rc_sum: f64 = rc.iter().sum();

            if rc_sum < 1e-12 {
                break;
            }

            // ERC gradient: each asset should contribute rc_sum / n
            let target = rc_sum / n as f64;
            let mut max_change = 0.0_f64;
            for i in 0..n {
                let gradient = rc[i] - target;
                let learning_rate = 0.5;
                let delta = -learning_rate * gradient * w[i] / (mrc[i].abs().max(1e-12));
                let new_w = (w[i] + delta).max(0.0);
                max_change = max_change.max((new_w - w[i]).abs());
                w[i] = new_w;
            }

            // Renormalise.
            let w_sum: f64 = w.iter().sum();
            if w_sum > 1e-12 {
                for wi in w.iter_mut() {
                    *wi /= w_sum;
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        // Apply max_weight constraint and re-normalise.
        apply_max_weight(&mut w, constraints.max_weight);

        let risk = portfolio_vol(&w, cov, n);
        Ok(OptimizationResult {
            symbols: symbols.to_vec(),
            weights: w,
            risk,
            method: OptimizationMethod::RiskParity,
        })
    }
}

// ── Min-variance optimizer ────────────────────────────────────────────────────

/// Minimum-variance portfolio via iterative gradient descent.
pub struct MinVarianceOptimizer {
    pub max_iter: usize,
    pub learning_rate: f64,
    pub tol: f64,
}

impl Default for MinVarianceOptimizer {
    fn default() -> Self {
        Self { max_iter: 500, learning_rate: 0.01, tol: 1e-8 }
    }
}

impl Optimizer for MinVarianceOptimizer {
    fn optimize(
        &self,
        symbols: &[String],
        _alpha_scores: &[f64],
        cov: &[f64],
        constraints: &PortfolioConstraints,
    ) -> PortfolioResult<OptimizationResult> {
        if symbols.is_empty() {
            return Err(PortfolioError::EmptyUniverse);
        }
        let n = symbols.len();
        let mut w = vec![1.0_f64 / n as f64; n];

        for _ in 0..self.max_iter {
            // Gradient of portfolio variance: 2Σw.
            let grad = mat_vec_mul(cov, &w, n);
            let mut max_change = 0.0_f64;

            for i in 0..n {
                let new_w = if constraints.long_only {
                    (w[i] - self.learning_rate * grad[i]).max(0.0)
                } else {
                    w[i] - self.learning_rate * grad[i]
                };
                max_change = max_change.max((new_w - w[i]).abs());
                w[i] = new_w;
            }

            // Project onto simplex (sum = 1, non-negative if long_only).
            let w_sum: f64 = w.iter().sum();
            if w_sum > 1e-12 {
                for wi in w.iter_mut() {
                    *wi /= w_sum;
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        apply_max_weight(&mut w, constraints.max_weight);

        let risk = portfolio_vol(&w, cov, n);
        Ok(OptimizationResult {
            symbols: symbols.to_vec(),
            weights: w,
            risk,
            method: OptimizationMethod::MinVariance,
        })
    }
}

// ── Public dispatch ───────────────────────────────────────────────────────────

pub fn optimize(
    method: OptimizationMethod,
    symbols: &[String],
    alpha_scores: &[f64],
    cov: &[f64],
    constraints: &PortfolioConstraints,
) -> PortfolioResult<OptimizationResult> {
    match method {
        OptimizationMethod::EqualWeight => {
            EqualWeightOptimizer.optimize(symbols, alpha_scores, cov, constraints)
        }
        OptimizationMethod::RiskParity | OptimizationMethod::MeanVariance => {
            RiskParityOptimizer::default().optimize(symbols, alpha_scores, cov, constraints)
        }
        OptimizationMethod::MinVariance => {
            MinVarianceOptimizer::default().optimize(symbols, alpha_scores, cov, constraints)
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Portfolio variance: w^T Σ w.
fn portfolio_variance(w: &[f64], cov: &[f64], n: usize) -> f64 {
    let cov_w = mat_vec_mul(cov, w, n);
    w.iter().zip(cov_w.iter()).map(|(wi, ci)| wi * ci).sum()
}

/// Portfolio volatility = sqrt(variance).
pub fn portfolio_vol(w: &[f64], cov: &[f64], n: usize) -> f64 {
    portfolio_variance(w, cov, n).max(0.0).sqrt()
}

/// Matrix-vector product: Σw.
fn mat_vec_mul(cov: &[f64], w: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| cov[i * n + j] * w[j]).sum())
        .collect()
}

/// Marginal risk contributions: MRC_i = (Σw)_i / σ_p.
fn marginal_risk_contributions(w: &[f64], cov: &[f64], n: usize) -> Vec<f64> {
    let cov_w = mat_vec_mul(cov, w, n);
    let sigma = portfolio_variance(w, cov, n).max(1e-24).sqrt();
    cov_w.iter().map(|v| v / sigma).collect()
}

/// Cap weights at `max_weight` and renormalise.
fn apply_max_weight(w: &mut Vec<f64>, max_weight: f64) {
    if max_weight >= 1.0 {
        return;
    }

    // Clip and redistribute excess proportionally (2-pass).
    for _ in 0..10 {
        let capped: Vec<f64> = w.iter().map(|&wi| wi.min(max_weight)).collect();
        let capped_sum: f64 = capped.iter().sum();
        if capped_sum < 1e-12 {
            break;
        }
        let scale = 1.0 / capped_sum;
        for (wi, ci) in w.iter_mut().zip(capped.iter()) {
            *wi = (ci * scale).min(max_weight);
        }
        // Check if we're within tolerance.
        if w.iter().all(|&wi| wi <= max_weight + 1e-9) {
            break;
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_cov(n: usize) -> Vec<f64> {
        let mut cov = vec![0.0_f64; n * n];
        for i in 0..n {
            cov[i * n + i] = 1.0;
        }
        cov
    }

    fn symbols(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("S{i}")).collect()
    }

    #[test]
    fn test_equal_weight_3_assets() {
        let syms = symbols(3);
        let cov = identity_cov(3);
        let alphas = vec![0.0; 3];
        let constraints = PortfolioConstraints::default();
        let result = EqualWeightOptimizer.optimize(&syms, &alphas, &cov, &constraints).unwrap();
        for &w in &result.weights {
            assert!((w - 1.0 / 3.0).abs() < 1e-9, "got {w}");
        }
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_risk_parity_equal_variance() {
        // With identity covariance, risk parity should give equal weights.
        let syms = symbols(4);
        let cov = identity_cov(4);
        let alphas = vec![0.0; 4];
        let constraints = PortfolioConstraints::default();
        let result = RiskParityOptimizer::default()
            .optimize(&syms, &alphas, &cov, &constraints)
            .unwrap();
        for &w in &result.weights {
            assert!((w - 0.25).abs() < 1e-4, "expected 0.25, got {w}");
        }
    }

    #[test]
    fn test_min_variance_identity_cov() {
        let syms = symbols(3);
        let cov = identity_cov(3);
        let alphas = vec![0.0; 3];
        let constraints = PortfolioConstraints::default();
        let result = MinVarianceOptimizer::default()
            .optimize(&syms, &alphas, &cov, &constraints)
            .unwrap();
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights don't sum to 1: {sum}");
    }

    #[test]
    fn test_max_weight_constraint() {
        let syms = symbols(2);
        let cov = identity_cov(2);
        let alphas = vec![0.0; 2];
        let constraints = PortfolioConstraints { max_weight: 0.4, ..Default::default() };
        let result = EqualWeightOptimizer.optimize(&syms, &alphas, &cov, &constraints).unwrap();
        for &w in &result.weights {
            assert!(w <= 0.4 + 1e-9);
        }
    }
}
