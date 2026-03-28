//! Conjugate Bayesian models for sequential IC estimation.
//!
//! Provides `NormalGammaTracker`, a Normal-Gamma conjugate model for
//! online estimation of signal information coefficients (IC).

use crate::BayesianModel;

/// Normal-Gamma conjugate Bayesian model for online mean/variance estimation.
///
/// Maintains a running posterior `(μₙ, κₙ, αₙ, βₙ)` updated in O(1) per
/// observation.  The marginal posterior for the mean μ is a Student-t:
///
/// ```text
/// μ | data  ~  t(2αₙ, μₙ, βₙ / (κₙ · αₙ))
/// ```
///
/// # Default hyperpriors
/// `μ₀ = 0, κ₀ = 1, α₀ = 2, β₀ = 1` — weakly informative, centred on zero IC.
///
/// # Math reference
/// ```text
/// κₙ = κₙ₋₁ + 1
/// μₙ = (κₙ₋₁ · μₙ₋₁ + x) / κₙ
/// αₙ = αₙ₋₁ + 0.5
/// βₙ = βₙ₋₁ + κₙ₋₁ · (x − μₙ₋₁)² / (2 · κₙ)
/// ```
#[derive(Debug, Clone)]
pub struct NormalGammaTracker {
    // ── Hyperpriors (fixed) ───────────────────────────────────────────────────
    /// Prior mean of the location parameter.
    pub mu0: f64,
    /// Prior pseudo-observation count on the mean.
    pub kappa0: f64,
    /// Prior shape of the precision (Gamma).
    pub alpha0: f64,
    /// Prior rate of the precision (Gamma).
    pub beta0: f64,

    // ── Running posterior ─────────────────────────────────────────────────────
    /// Number of observations incorporated so far.
    pub n: u64,
    /// Posterior mean of the location.
    pub mu_n: f64,
    /// Posterior pseudo-count on the mean.
    pub kappa_n: f64,
    /// Posterior shape of the precision.
    pub alpha_n: f64,
    /// Posterior rate of the precision.
    pub beta_n: f64,
}

impl Default for NormalGammaTracker {
    fn default() -> Self {
        Self::new(0.0, 1.0, 2.0, 1.0)
    }
}

impl NormalGammaTracker {
    /// Create with explicit hyperpriors `(μ₀, κ₀, α₀, β₀)`.
    ///
    /// Requires `κ₀ > 0`, `α₀ > 0`, `β₀ > 0`.
    pub fn new(mu0: f64, kappa0: f64, alpha0: f64, beta0: f64) -> Self {
        assert!(kappa0 > 0.0, "kappa0 must be > 0");
        assert!(alpha0 > 0.0, "alpha0 must be > 0");
        assert!(beta0 > 0.0, "beta0 must be > 0");
        Self {
            mu0,
            kappa0,
            alpha0,
            beta0,
            n: 0,
            mu_n: mu0,
            kappa_n: kappa0,
            alpha_n: alpha0,
            beta_n: beta0,
        }
    }

    /// Student-t predictive standard deviation for the next observation.
    ///
    /// Returns `f64::INFINITY` when `α_n ≤ 1` (variance undefined).
    ///
    /// ```text
    /// predictive_std = sqrt(βₙ · (κₙ + 1) / (κₙ · (αₙ − 1)))
    /// ```
    pub fn predictive_std(&self) -> f64 {
        if self.alpha_n <= 1.0 {
            return f64::INFINITY;
        }
        let scale_sq =
            self.beta_n * (self.kappa_n + 1.0) / (self.kappa_n * (self.alpha_n - 1.0));
        scale_sq.sqrt()
    }
}

impl BayesianModel for NormalGammaTracker {
    /// O(1) Normal-Gamma conjugate posterior update.
    fn update(&mut self, x: f64) {
        let kappa_prev = self.kappa_n;
        let mu_prev = self.mu_n;

        self.kappa_n = kappa_prev + 1.0;
        self.mu_n = (kappa_prev * mu_prev + x) / self.kappa_n;
        self.alpha_n += 0.5;
        let residual = x - mu_prev;
        self.beta_n += kappa_prev * residual * residual / (2.0 * self.kappa_n);
        self.n += 1;
    }

    /// Posterior mean of the location parameter μ.
    ///
    /// Converges to the sample mean as n → ∞ (prior washed out).
    fn posterior_mean(&self) -> f64 {
        self.mu_n
    }

    /// Marginal posterior variance of μ.
    ///
    /// Returns `f64::INFINITY` when `α_n ≤ 1` (heavy tails, variance undefined).
    ///
    /// ```text
    /// Var(μ | data) = βₙ / (κₙ · (αₙ − 1))   when αₙ > 1
    /// ```
    fn posterior_variance(&self) -> f64 {
        if self.alpha_n <= 1.0 {
            return f64::INFINITY;
        }
        self.beta_n / (self.kappa_n * (self.alpha_n - 1.0))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `n` identical observations and return the tracker.
    fn feed(tracker: &mut NormalGammaTracker, values: &[f64]) {
        for &v in values {
            tracker.update(v);
        }
    }

    #[test]
    fn test_posterior_mean_converges_to_sample_mean() {
        // True IC = 0.3; feed 200 observations centred on 0.3.
        let true_ic = 0.3_f64;
        let mut tracker = NormalGammaTracker::default();
        // Deterministic series oscillating around true_ic
        let obs: Vec<f64> = (0..200)
            .map(|i| true_ic + if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        feed(&mut tracker, &obs);
        assert!(
            (tracker.posterior_mean() - true_ic).abs() < 0.02,
            "posterior_mean {} did not converge to {}",
            tracker.posterior_mean(),
            true_ic
        );
    }

    #[test]
    fn test_posterior_variance_decreases_monotonically() {
        let mut tracker = NormalGammaTracker::default();
        let mut prev_var = f64::INFINITY;
        for i in 0..50_i32 {
            let x = if i % 2 == 0 { 0.1 } else { -0.1 };
            tracker.update(x);
            if tracker.alpha_n > 1.0 {
                let var = tracker.posterior_variance();
                assert!(
                    var <= prev_var + 1e-15,
                    "variance increased at n={}: {} > {}",
                    tracker.n,
                    var,
                    prev_var
                );
                prev_var = var;
            }
        }
        assert!(
            prev_var.is_finite(),
            "posterior variance never became finite"
        );
    }

    #[test]
    fn test_update_increments_n() {
        let mut tracker = NormalGammaTracker::default();
        assert_eq!(tracker.n, 0);
        tracker.update(0.1);
        assert_eq!(tracker.n, 1);
        tracker.update(0.2);
        assert_eq!(tracker.n, 2);
    }

    #[test]
    fn test_posterior_mean_with_zero_observations_is_prior_mean() {
        let tracker = NormalGammaTracker::new(0.5, 1.0, 2.0, 1.0);
        assert!((tracker.posterior_mean() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_kappa_n_increases_by_one_per_update() {
        let mut tracker = NormalGammaTracker::new(0.0, 2.0, 2.0, 1.0);
        tracker.update(0.1);
        assert!((tracker.kappa_n - 3.0).abs() < 1e-15);
        tracker.update(0.2);
        assert!((tracker.kappa_n - 4.0).abs() < 1e-15);
    }

    #[test]
    fn test_alpha_n_increases_by_half_per_update() {
        let mut tracker = NormalGammaTracker::new(0.0, 1.0, 2.0, 1.0);
        tracker.update(0.1);
        assert!((tracker.alpha_n - 2.5).abs() < 1e-15);
        tracker.update(0.2);
        assert!((tracker.alpha_n - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_predictive_std_finite_after_enough_obs() {
        let mut tracker = NormalGammaTracker::default();
        // alpha0 = 2.0, so alpha_n > 1 immediately → predictive_std is finite from the start
        assert!(tracker.predictive_std().is_finite());
        for i in 0..10_i32 {
            tracker.update(if i % 2 == 0 { 0.1 } else { -0.1 });
        }
        assert!(tracker.predictive_std().is_finite());
        assert!(tracker.predictive_std() > 0.0);
    }

    #[test]
    fn test_posterior_variance_infinite_when_alpha_le_one() {
        let tracker = NormalGammaTracker::new(0.0, 1.0, 0.5, 1.0);
        // alpha_0 = 0.5 ≤ 1 → variance undefined before enough updates
        assert_eq!(tracker.posterior_variance(), f64::INFINITY);
    }

    #[test]
    fn test_single_observation_updates_mu_correctly() {
        let mut tracker = NormalGammaTracker::new(0.0, 1.0, 2.0, 1.0);
        tracker.update(1.0);
        // mu_n = (1.0 * 0.0 + 1.0) / 2.0 = 0.5
        assert!((tracker.mu_n - 0.5).abs() < 1e-15);
    }
}
