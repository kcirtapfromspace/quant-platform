//! Covariance matrix estimation from return histories.
//!
//! Implements sample covariance with optional Ledoit-Wolf shrinkage
//! toward a diagonal target.  Operates on column-major return matrices
//! (each column is one asset's return series).

use crate::error::{PortfolioError, PortfolioResult};

/// Estimate a covariance matrix from a returns matrix.
///
/// # Arguments
/// * `returns` — flattened row-major matrix of shape `(n_bars, n_assets)`.
/// * `n_assets` — number of assets (columns).
/// * `shrinkage` — Ledoit-Wolf intensity in [0, 1].
///   - 0.0 = pure sample covariance
///   - 1.0 = fully shrunk toward diagonal (variances only)
///   - `None` = Oracle approximating shrinkage (Ledoit-Wolf analytical estimate)
///
/// Returns a flattened row-major covariance matrix of shape `(n_assets, n_assets)`.
pub fn estimate_covariance(
    returns: &[f64],
    n_assets: usize,
    shrinkage: Option<f64>,
) -> PortfolioResult<Vec<f64>> {
    if n_assets == 0 {
        return Err(PortfolioError::EmptyUniverse);
    }

    let n_bars = returns.len() / n_assets;
    if n_bars < 2 {
        return Err(PortfolioError::InsufficientHistory { needed: 2, got: n_bars });
    }

    let mut cov = sample_covariance(returns, n_assets, n_bars);

    let alpha = shrinkage.unwrap_or_else(|| ledoit_wolf_shrinkage(returns, &cov, n_assets, n_bars));
    let alpha = alpha.clamp(0.0, 1.0);

    if alpha > 1e-12 {
        shrink_toward_diagonal(&mut cov, n_assets, alpha);
    }

    Ok(cov)
}

/// Raw sample covariance matrix (ddof=1).
fn sample_covariance(returns: &[f64], n_assets: usize, n_bars: usize) -> Vec<f64> {
    // Compute per-asset means.
    let mut means = vec![0.0_f64; n_assets];
    for bar in 0..n_bars {
        for asset in 0..n_assets {
            means[asset] += returns[bar * n_assets + asset];
        }
    }
    for m in means.iter_mut() {
        *m /= n_bars as f64;
    }

    // Compute covariance (ddof=1).
    let mut cov = vec![0.0_f64; n_assets * n_assets];
    for bar in 0..n_bars {
        for i in 0..n_assets {
            let di = returns[bar * n_assets + i] - means[i];
            for j in 0..=i {
                let dj = returns[bar * n_assets + j] - means[j];
                cov[i * n_assets + j] += di * dj;
                if i != j {
                    cov[j * n_assets + i] += di * dj;
                }
            }
        }
    }

    let denom = (n_bars - 1) as f64;
    for c in cov.iter_mut() {
        *c /= denom;
    }

    cov
}

/// Ledoit-Wolf analytical shrinkage intensity (Oracle Approximating Shrinkage).
/// Uses the simplified formula valid for Gaussian returns.
fn ledoit_wolf_shrinkage(
    _returns: &[f64],
    cov: &[f64],
    n_assets: usize,
    n_bars: usize,
) -> f64 {
    let p = n_assets as f64;
    let n = n_bars as f64;

    // Frobenius norm squared of cov.
    let cov_frob_sq: f64 = cov.iter().map(|v| v * v).sum();

    // Trace of cov.
    let trace_cov: f64 = (0..n_assets).map(|i| cov[i * n_assets + i]).sum();
    let trace_sq = trace_cov * trace_cov;

    // Simplified Oracle shrinkage formula.
    let numerator = ((n - 2.0) / n) * cov_frob_sq + trace_sq;
    let denominator = (n + 2.0) * (cov_frob_sq - trace_sq / p);

    if denominator.abs() < 1e-12 {
        return 0.0;
    }
    (numerator / denominator).clamp(0.0, 1.0)
}

/// Shrink cov = (1 - alpha) * S + alpha * diag(S) in-place.
fn shrink_toward_diagonal(cov: &mut [f64], n_assets: usize, alpha: f64) {
    for i in 0..n_assets {
        for j in 0..n_assets {
            if i == j {
                // diagonal unchanged
            } else {
                cov[i * n_assets + j] *= 1.0 - alpha;
            }
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_cov_identity_returns() {
        // Two uncorrelated assets with identical variance.
        let n_bars = 100_usize;
        let mut returns = vec![0.0_f64; n_bars * 2];
        for i in 0..n_bars {
            returns[i * 2] = if i % 2 == 0 { 0.01 } else { -0.01 };
            returns[i * 2 + 1] = if i % 2 == 0 { -0.01 } else { 0.01 };
        }
        let cov = estimate_covariance(&returns, 2, Some(0.0)).unwrap();
        // Off-diagonal should be approximately -variance (perfectly anti-correlated).
        assert!(cov[0] > 0.0);
        assert!(cov[3] > 0.0);
        assert!(cov[1] < 0.0);
    }

    #[test]
    fn test_shrinkage_reduces_off_diagonal() {
        let n_bars = 50_usize;
        let n_assets = 3;
        let returns: Vec<f64> = (0..n_bars * n_assets)
            .map(|i| ((i as f64) * 0.01 - 0.25).sin() * 0.01)
            .collect();

        let cov_no_shrink = estimate_covariance(&returns, n_assets, Some(0.0)).unwrap();
        let cov_full_shrink = estimate_covariance(&returns, n_assets, Some(1.0)).unwrap();

        // Full shrinkage should zero off-diagonal.
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i != j {
                    let off = cov_full_shrink[i * n_assets + j];
                    assert!(off.abs() < 1e-12, "off-diagonal should be zero with alpha=1.0, got {off}");
                }
            }
        }
        drop(cov_no_shrink);
    }

    #[test]
    fn test_insufficient_history_error() {
        let returns = vec![0.01_f64; 2]; // 1 bar, 2 assets
        let result = estimate_covariance(&returns, 2, Some(0.0));
        assert!(matches!(result, Err(PortfolioError::InsufficientHistory { .. })));
    }
}
