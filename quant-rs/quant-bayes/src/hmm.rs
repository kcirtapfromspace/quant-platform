//! Hidden Markov Model for market regime detection.
//!
//! Implements a 2-state HMM with Gaussian emissions, Baum-Welch EM training,
//! and an O(S²) online forward pass for sequential bar-by-bar inference.

use nalgebra::{Matrix2, Vector2};

/// sqrt(2π) — used in the Gaussian PDF denominator.
const TWO_PI_SQRT: f64 = 2.506_628_274_631_001;

/// Market regime state inferred by the HMM.
///
/// Semantics match the original threshold-based regime labels:
/// `LowVol` ↔ calm/mean-reverting, `HighVol` ↔ turbulent/trending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegimeState {
    /// Low-volatility / calm market regime.
    LowVol,
    /// High-volatility / turbulent market regime.
    HighVol,
}

/// 2-state HMM with Gaussian emissions for online regime detection.
///
/// # Workflow
/// 1. Call [`fit`](HmmRegimeModel::fit) once on a historical warm-up series.
/// 2. Call [`update`](HmmRegimeModel::update) per bar for O(1) online inference.
/// 3. Read [`regime_probs`](HmmRegimeModel::regime_probs) or
///    [`most_likely_regime`](HmmRegimeModel::most_likely_regime) for downstream use.
#[derive(Debug, Clone)]
pub struct HmmRegimeModel {
    /// Gaussian emission means per state: index 0 = `LowVol`, 1 = `HighVol`.
    pub means: [f64; 2],
    /// Gaussian emission standard deviations per state (always > 0).
    pub stds: [f64; 2],
    /// Row-stochastic transition matrix: `trans[(i, j)]` = P(next = j | cur = i).
    pub trans: Matrix2<f64>,
    /// Current filtered belief over states (sums to ≈ 1.0).
    pub state_probs: Vector2<f64>,
    /// Initial state distribution (reset each `fit` call).
    pi: [f64; 2],
}

impl Default for HmmRegimeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl HmmRegimeModel {
    /// New model with uninformative (flat) priors and sticky transitions.
    pub fn new() -> Self {
        Self {
            means: [0.0, 0.0],
            stds: [1.0, 1.0],
            trans: Matrix2::new(0.95, 0.05, 0.05, 0.95),
            state_probs: Vector2::new(0.5, 0.5),
            pi: [0.5, 0.5],
        }
    }

    // ── Emission ──────────────────────────────────────────────────────────────

    /// Gaussian PDF for `obs` under state `s`.
    #[inline]
    fn emission(&self, obs: f64, s: usize) -> f64 {
        let std = self.stds[s].max(1e-10);
        let z = (obs - self.means[s]) / std;
        (-0.5 * z * z).exp() / (std * TWO_PI_SQRT)
    }

    // ── Parameter initialisation ──────────────────────────────────────────────

    /// Seed parameters from data before running Baum-Welch.
    ///
    /// Sorts observations by absolute deviation from the global mean; the
    /// lower half (small deviations) seeds `LowVol`, the upper half seeds
    /// `HighVol`.  Transition matrix is reset to a sticky prior.
    fn initialize_from_data(&mut self, obs: &[f64]) {
        let n = obs.len();
        let mean_all: f64 = obs.iter().sum::<f64>() / n as f64;
        let var_all: f64 =
            obs.iter().map(|v| (v - mean_all).powi(2)).sum::<f64>() / n.max(1) as f64;
        let std_all = var_all.sqrt().max(1e-10);

        // Sort indices by |obs - mean|: small deviations → LowVol, large → HighVol.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let da = (obs[a] - mean_all).abs();
            let db = (obs[b] - mean_all).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        let half = (n / 2).max(1);
        let low_vol_obs: Vec<f64> = indices[..half].iter().map(|&i| obs[i]).collect();
        let high_vol_obs: Vec<f64> = indices[half..].iter().map(|&i| obs[i]).collect();

        let mean0 = low_vol_obs.iter().sum::<f64>() / low_vol_obs.len() as f64;
        let mean1 = if high_vol_obs.is_empty() {
            mean_all
        } else {
            high_vol_obs.iter().sum::<f64>() / high_vol_obs.len() as f64
        };

        let std0 = if low_vol_obs.len() < 2 {
            std_all * 0.5
        } else {
            let v = low_vol_obs.iter().map(|x| (x - mean0).powi(2)).sum::<f64>()
                / (low_vol_obs.len() - 1) as f64;
            v.sqrt().max(1e-10)
        };
        let std1 = if high_vol_obs.len() < 2 {
            std_all * 1.5
        } else {
            let v = high_vol_obs
                .iter()
                .map(|x| (x - mean1).powi(2))
                .sum::<f64>()
                / (high_vol_obs.len() - 1) as f64;
            v.sqrt().max(1e-10)
        };

        self.means = [mean0, mean1];
        self.stds = [std0, std1];
        self.trans = Matrix2::new(0.95, 0.05, 0.05, 0.95);
        self.pi = [0.5, 0.5];
        self.state_probs = Vector2::new(0.5, 0.5);
    }

    // ── Forward / Backward ────────────────────────────────────────────────────

    /// Scaled forward pass.
    ///
    /// Returns `(alpha, c)` where `alpha[t][s]` is the scaled forward variable
    /// and `c[t]` is the per-step normalisation constant used for backward
    /// scaling and log-likelihood computation.
    fn forward(&self, obs: &[f64]) -> (Vec<[f64; 2]>, Vec<f64>) {
        let n = obs.len();
        let mut alpha = vec![[0.0_f64; 2]; n];
        let mut c = vec![1.0_f64; n];

        // t = 0
        alpha[0][0] = self.pi[0] * self.emission(obs[0], 0);
        alpha[0][1] = self.pi[1] * self.emission(obs[0], 1);
        c[0] = alpha[0][0] + alpha[0][1];
        if c[0] > 1e-300 {
            alpha[0][0] /= c[0];
            alpha[0][1] /= c[0];
        } else {
            alpha[0] = [0.5, 0.5];
            c[0] = 1.0;
        }

        // t > 0
        for t in 1..n {
            alpha[t][0] = (alpha[t - 1][0] * self.trans[(0, 0)]
                + alpha[t - 1][1] * self.trans[(1, 0)])
                * self.emission(obs[t], 0);
            alpha[t][1] = (alpha[t - 1][0] * self.trans[(0, 1)]
                + alpha[t - 1][1] * self.trans[(1, 1)])
                * self.emission(obs[t], 1);
            c[t] = alpha[t][0] + alpha[t][1];
            if c[t] > 1e-300 {
                alpha[t][0] /= c[t];
                alpha[t][1] /= c[t];
            } else {
                alpha[t] = [0.5, 0.5];
                c[t] = 1.0;
            }
        }

        (alpha, c)
    }

    /// Scaled backward pass.
    fn backward(&self, obs: &[f64], c: &[f64]) -> Vec<[f64; 2]> {
        let n = obs.len();
        let mut beta = vec![[1.0_f64; 2]; n];

        for t in (0..n - 1).rev() {
            let scale = c[t + 1];
            beta[t][0] = self.trans[(0, 0)] * self.emission(obs[t + 1], 0) * beta[t + 1][0]
                + self.trans[(0, 1)] * self.emission(obs[t + 1], 1) * beta[t + 1][1];
            beta[t][1] = self.trans[(1, 0)] * self.emission(obs[t + 1], 0) * beta[t + 1][0]
                + self.trans[(1, 1)] * self.emission(obs[t + 1], 1) * beta[t + 1][1];
            if scale > 1e-300 {
                beta[t][0] /= scale;
                beta[t][1] /= scale;
            }
        }

        beta
    }

    // ── EM sufficient statistics ──────────────────────────────────────────────

    /// State occupancy probabilities γ[t][s] = P(S_t = s | obs).
    fn compute_gamma(alpha: &[[f64; 2]], beta: &[[f64; 2]]) -> Vec<[f64; 2]> {
        alpha
            .iter()
            .zip(beta.iter())
            .map(|(a, b)| {
                let g0 = a[0] * b[0];
                let g1 = a[1] * b[1];
                let sum = g0 + g1;
                if sum > 1e-300 {
                    [g0 / sum, g1 / sum]
                } else {
                    [0.5, 0.5]
                }
            })
            .collect()
    }

    /// Joint transition probabilities ξ[t][i][j] = P(S_t=i, S_{t+1}=j | obs).
    fn compute_xi(&self, obs: &[f64], alpha: &[[f64; 2]], beta: &[[f64; 2]]) -> Vec<[[f64; 2]; 2]> {
        let n = obs.len();
        (0..n - 1)
            .map(|t| {
                let mut xi = [[0.0_f64; 2]; 2];
                let mut total = 0.0_f64;
                for i in 0..2 {
                    for j in 0..2 {
                        xi[i][j] = alpha[t][i]
                            * self.trans[(i, j)]
                            * self.emission(obs[t + 1], j)
                            * beta[t + 1][j];
                        total += xi[i][j];
                    }
                }
                if total > 1e-300 {
                    for row in xi.iter_mut() {
                        row[0] /= total;
                        row[1] /= total;
                    }
                }
                xi
            })
            .collect()
    }

    // ── M-step ────────────────────────────────────────────────────────────────

    fn m_step(&mut self, obs: &[f64], gamma: &[[f64; 2]], xi: &[[[f64; 2]; 2]]) {
        let n = obs.len();

        // Initial state distribution
        self.pi = gamma[0];

        // Transition matrix
        for i in 0..2 {
            let denom: f64 = (0..n - 1).map(|t| gamma[t][i]).sum();
            for j in 0..2 {
                let numer: f64 = xi.iter().map(|x| x[i][j]).sum();
                self.trans[(i, j)] = if denom > 1e-300 { numer / denom } else { 0.5 };
            }
            // Ensure row-stochastic
            let row_sum = self.trans[(i, 0)] + self.trans[(i, 1)];
            if row_sum > 1e-300 {
                self.trans[(i, 0)] /= row_sum;
                self.trans[(i, 1)] /= row_sum;
            }
        }

        // Emission parameters
        for s in [0usize, 1] {
            let denom: f64 = (0..n).map(|t| gamma[t][s]).sum();
            if denom < 1e-300 {
                continue;
            }
            let mean: f64 = (0..n).map(|t| gamma[t][s] * obs[t]).sum::<f64>() / denom;
            let var: f64 = (0..n)
                .map(|t| gamma[t][s] * (obs[t] - mean).powi(2))
                .sum::<f64>()
                / denom;
            self.means[s] = mean;
            self.stds[s] = var.sqrt().max(1e-8);
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Batch-train via Baum-Welch EM (max 50 iterations, tol = 1e-6).
    ///
    /// Initialises parameters from `observations`, iterates until log-likelihood
    /// improvement falls below tolerance, then sets `state_probs` to the final
    /// filtered belief.  Short-circuits if fewer than 2 observations are given.
    pub fn fit(&mut self, observations: &[f64]) {
        if observations.len() < 2 {
            return;
        }
        self.initialize_from_data(observations);

        let max_iter = 50;
        let tol = 1e-6_f64;
        let mut prev_ll = f64::NEG_INFINITY;

        for _ in 0..max_iter {
            let (alpha, c) = self.forward(observations);
            let beta = self.backward(observations, &c);
            let ll: f64 = c.iter().map(|v| v.max(1e-300).ln()).sum();
            let gamma = Self::compute_gamma(&alpha, &beta);
            let xi = self.compute_xi(observations, &alpha, &beta);
            self.m_step(observations, &gamma, &xi);

            if (ll - prev_ll).abs() < tol {
                break;
            }
            prev_ll = ll;
        }

        // Set filtered belief to the last step's alpha
        let (alpha, _) = self.forward(observations);
        let last = *alpha.last().unwrap();
        self.state_probs = Vector2::new(last[0], last[1]);
    }

    /// Online forward-pass update — O(S²) = O(4) per call; no retraining.
    ///
    /// Predict-then-update: propagates the current belief through the
    /// transition matrix, then weights by the Gaussian emission likelihood.
    pub fn update(&mut self, obs: f64) {
        // Predict: marginalise over previous state
        let p0 =
            self.state_probs[0] * self.trans[(0, 0)] + self.state_probs[1] * self.trans[(1, 0)];
        let p1 =
            self.state_probs[0] * self.trans[(0, 1)] + self.state_probs[1] * self.trans[(1, 1)];

        // Update: weight by emission likelihood
        let u0 = p0 * self.emission(obs, 0);
        let u1 = p1 * self.emission(obs, 1);

        let norm = u0 + u1;
        if norm > 1e-300 {
            self.state_probs = Vector2::new(u0 / norm, u1 / norm);
        }
        // Degenerate emission: keep previous state_probs unchanged.
    }

    /// Current filtered state probabilities `[P(LowVol), P(HighVol)]`.
    #[inline]
    pub fn regime_probs(&self) -> [f64; 2] {
        [self.state_probs[0], self.state_probs[1]]
    }

    /// Most likely current regime (argmax of filtered state probabilities).
    #[inline]
    pub fn most_likely_regime(&self) -> RegimeState {
        if self.state_probs[0] >= self.state_probs[1] {
            RegimeState::LowVol
        } else {
            RegimeState::HighVol
        }
    }

    /// Transition probability from state `from` to state `to`.
    ///
    /// Indices: 0 = `LowVol`, 1 = `HighVol`.
    #[inline]
    pub fn transition_probability(&self, from: usize, to: usize) -> f64 {
        self.trans[(from, to)]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic 2-regime series: 50 small-amplitude bars (±0.01)
    /// followed by 50 large-amplitude bars (±0.05).
    fn two_regime_series() -> Vec<f64> {
        let mut obs: Vec<f64> = (0..50)
            .map(|i| if i % 2 == 0 { 0.01_f64 } else { -0.01_f64 })
            .collect();
        obs.extend((0..50).map(|i| if i % 2 == 0 { 0.05_f64 } else { -0.05_f64 }));
        obs
    }

    #[test]
    fn test_baum_welch_differentiates_two_regimes() {
        let obs = two_regime_series();
        let mut model = HmmRegimeModel::new();
        model.fit(&obs);

        // States should have meaningfully different volatilities.
        let max_std = model.stds[0].max(model.stds[1]);
        let min_std = model.stds[0].min(model.stds[1]);
        assert!(
            max_std > min_std * 1.5,
            "states not differentiated: stds = {:?}",
            model.stds
        );

        // Transition matrix must remain row-stochastic.
        for i in 0..2 {
            let row = model.trans[(i, 0)] + model.trans[(i, 1)];
            assert!((row - 1.0).abs() < 1e-10, "row {i} not stochastic: {row}");
        }

        // Filtered belief must sum to 1.
        let sum = model.state_probs[0] + model.state_probs[1];
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_baum_welch_detects_high_vol_at_end() {
        // The series ends in the high-vol block, so the model should
        // converge to believing we are in the high-vol state.
        let obs = two_regime_series();
        let mut model = HmmRegimeModel::new();
        model.fit(&obs);
        assert_eq!(model.most_likely_regime(), RegimeState::HighVol);
    }

    #[test]
    fn test_online_update_preserves_probability_sum() {
        let obs = two_regime_series();
        let mut model = HmmRegimeModel::new();
        model.fit(&obs);

        for &o in &[0.02_f64, -0.01, 0.005, -0.003, 0.0, 0.10, -0.10] {
            model.update(o);
            let probs = model.regime_probs();
            assert!(
                (probs[0] + probs[1] - 1.0).abs() < 1e-10,
                "regime_probs don't sum to 1 after update({o}): {probs:?}"
            );
        }
    }

    #[test]
    fn test_online_update_shifts_toward_high_vol() {
        let obs = two_regime_series();
        let mut model = HmmRegimeModel::new();
        model.fit(&obs);

        // Drive several extreme observations; model should increase HighVol prob.
        let before = model.regime_probs()[1];
        // Reset to a more neutral starting point first
        model.state_probs = Vector2::new(0.8, 0.2);
        for _ in 0..10 {
            model.update(0.10); // large move
        }
        let after = model.regime_probs()[1];
        assert!(
            after > before || after > 0.5,
            "HighVol prob did not increase: {after}"
        );
    }

    #[test]
    fn test_edge_case_all_same_returns_no_panic() {
        let obs = vec![0.01_f64; 50];
        let mut model = HmmRegimeModel::new();
        model.fit(&obs); // zero-variance — must not panic
        model.update(0.01); // must not panic
        let probs = model.regime_probs();
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_edge_case_empty_and_single_obs_no_panic() {
        let mut model = HmmRegimeModel::new();
        model.fit(&[]); // no-op
        model.fit(&[0.01_f64]); // too short — no-op
        model.update(0.02); // must not panic
    }

    #[test]
    fn test_transition_probability_row_stochastic() {
        let model = HmmRegimeModel::new();
        for i in 0..2 {
            let row = model.transition_probability(i, 0) + model.transition_probability(i, 1);
            assert!((row - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_regime_probs_sum_to_one_on_new_model() {
        let model = HmmRegimeModel::new();
        let probs = model.regime_probs();
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_most_likely_regime_consistent_with_probs() {
        let mut model = HmmRegimeModel::new();
        model.state_probs = Vector2::new(0.3, 0.7);
        assert_eq!(model.most_likely_regime(), RegimeState::HighVol);
        model.state_probs = Vector2::new(0.7, 0.3);
        assert_eq!(model.most_likely_regime(), RegimeState::LowVol);
    }
}
