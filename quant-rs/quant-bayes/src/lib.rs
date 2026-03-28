//! Bayesian inference primitives for quant-rs.
//!
//! - Phase 1: `BayesianModel` / `PosteriorSampler` traits (conjugate scaffolding)
//! - Phase 2: `HmmRegimeModel` — 2-state HMM with Gaussian emissions

pub mod hmm;

pub use hmm::{HmmRegimeModel, RegimeState};

use rand::Rng;

/// A Bayesian model that maintains a posterior distribution over parameters
/// and updates it sequentially given scalar observations.
pub trait BayesianModel {
    /// Incorporate a new scalar observation, updating the posterior in place.
    fn update(&mut self, observation: f64);

    /// Return the posterior mean of the primary parameter of interest.
    fn posterior_mean(&self) -> f64;

    /// Return the posterior variance of the primary parameter of interest.
    fn posterior_variance(&self) -> f64;
}

/// A posterior sampler that draws Monte Carlo samples from a fitted model.
pub trait PosteriorSampler: BayesianModel {
    /// Draw `n` independent samples from the current posterior predictive.
    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<f64>;
}
