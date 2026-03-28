# CRO Model Risk Review: QUA-71b Integration (Unstaged Changes)

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**Scope:** Unstaged changes to `quant/signals/adaptive_combiner.py` and `quant/backtest/multi_strategy.py`
**Purpose:** Pre-commit model risk review before QUA-71b portfolio A/B test run

---

## Changes Under Review

### 1. `quant/signals/adaptive_combiner.py`
- New class `_NormalGammaTracker` — O(1) Normal-Gamma conjugate posterior for online IC estimation
- New dataclass `BayesianAdaptiveCombinerConfig(AdaptiveCombinerConfig)` — config marker subclass
- New class `BayesianAdaptiveSignalCombiner(AdaptiveSignalCombiner)` — replaces EWM IC mean with Normal-Gamma posterior mean

### 2. `quant/backtest/multi_strategy.py`
- Extended import of `BayesianAdaptiveCombinerConfig`, `BayesianAdaptiveSignalCombiner`
- Added isinstance dispatch: if `adaptive_combiner_config` is `BayesianAdaptiveCombinerConfig`, instantiate `BayesianAdaptiveSignalCombiner` instead of `AdaptiveSignalCombiner`

---

## Mathematical Correctness Review

### `_NormalGammaTracker.update()` — Normal-Gamma sequential update

Standard conjugate posterior update equations for Normal-Gamma (Normal likelihood, unknown mean and precision):

```
kappa_n = kappa_{n-1} + 1
mu_n    = (kappa_{n-1} * mu_{n-1} + x) / kappa_n
alpha_n = alpha_{n-1} + 0.5
beta_n  = beta_{n-1} + kappa_{n-1} * (x - mu_{n-1})^2 / (2 * kappa_n)
```

**Code implementation:**
```python
kappa_prev = self.kappa_n
mu_prev    = self.mu_n
self.kappa_n   = kappa_prev + 1.0                                   # ✅
self.mu_n      = (kappa_prev * mu_prev + x) / self.kappa_n          # ✅
self.alpha_n  += 0.5                                                 # ✅
residual = x - mu_prev
self.beta_n += kappa_prev * residual * residual / (2.0 * self.kappa_n)  # ✅
```

**VERDICT: Mathematically correct.**

`posterior_mean` returns `mu_n` — the Bayesian posterior mean for the Normal component. Correct; converges to sample mean as n→∞, and shrinks toward prior mu_0=0 with few observations. ✅

---

## Structural Correctness Review

| Check | Finding | Status |
|-------|---------|--------|
| `reset()` clears NG trackers | `super().reset()` clears EWM history; `self._ng_trackers.clear()` clears NG state — both cleared at fold boundary | ✅ |
| No lookahead bias | `update()` uses only current-cycle IC; `get_weights()` uses accumulated posterior | ✅ |
| isinstance dispatch ordering | `BayesianAdaptiveCombinerConfig` is a subclass — checked before parent class fallback; correct | ✅ |
| `has_enough` guard | Checks `_ic_history` length against `min_ic_periods`; NG tracker also guards on `tracker.n < min_ic_periods` — consistent | ✅ |
| Weight normalization | Final weights re-normalized after shrinkage blend | ✅ |
| Fallback to equal-weight | All failure paths (no signals, all IC below threshold, total_ic near zero) return equal-weight fallback | ✅ |

---

## Model Risk Flags

### FLAG 1 — No Forgetting / Unbounded Memory (MONITORING ITEM)

**Issue:** `_NormalGammaTracker` accumulates all IC observations since initialization with no forgetting. The base `AdaptiveSignalCombiner` trims `_ic_history` to `ic_lookback=126` observations, and the EWM uses a `halflife=21` decay. The NG tracker has no equivalent limit: after 300 cycles it is computing the full-history unweighted mean of IC, shrunk toward zero by the prior.

**Implication:** In stable regimes, NG is more stable and less noisy than EWM (lower variance). In regime changes, NG is slower to adapt than EWM (more past-history weight). The QUA-71b portfolio A/B test will reveal empirically which effect dominates.

**This is a known Bayesian estimation trade-off, not a bug.** The forgetting vs. shrinkage choice is the central methodological question QUA-71b is designed to answer.

**Action:** Monitor NG posterior mean drift in walk-forward results. If the Bayesian combiner lags EMA significantly in regime-change folds, consider adding a forgetting mechanism (e.g., sliding window restart or discounted NG) in a future iteration.

### FLAG 2 — Fixed Hyperpriors (MONITORING ITEM)

**Issue:** Hyperpriors `mu0=0, kappa0=1, alpha0=2, beta0=1` are hardcoded in `__init__`. Cannot be recalibrated through config without a code change. `mu0=0` (zero IC prior) is appropriate for IC estimation; `kappa0=1` corresponds to one prior pseudo-observation.

**Implication:** If a strategy persistently generates IC >0.05 for all signals, the zero-IC prior will pull down weights slightly. For the mvp-us-equity-ensemble at single-symbol IC levels of ~0.01–0.05, this is benign — the prior effect dissipates after ~20 observations.

**Action:** Consider exposing `mu0` as a config parameter in `BayesianAdaptiveCombinerConfig` if future strategies have well-characterized prior IC. Not blocking for QUA-71b.

### FLAG 3 — Dual-Tracking Overhead (MINOR)

**Issue:** `BayesianAdaptiveSignalCombiner.update()` calls `super().update()` which maintains `_ic_history` and EWM weights. These are computed but never used by `get_weights()` (which uses NG posterior instead). The base class EWM computation is wasteful but harmless.

**Action:** Acceptable for QUA-71b evaluation. If Bayesian combiner is approved for production, a refactor to skip EWM update would reduce overhead.

---

## Verdict

**CLEARED FOR COMMIT and QUA-71b portfolio-level A/B test run.**

The implementation is mathematically correct, structurally sound, and free of lookahead bias. The three flags above are monitoring items, not blockers. They should be tracked during QUA-71b result analysis.

---

## QUA-71b Run Requirements (CRO reminder)

For the results to be gate-reviewable, the portfolio A/B test must use:
- Same framework as QUA-58: `run_mvp_backtest.py`
- Expanding walk-forward: `expanding=True`, IS_min=252, OOS=63, step=63
- Same 50-symbol universe
- Same portfolio optimizer (RiskParity or MaxSharpe)
- Two runs: (A) `AdaptiveSignalCombiner` (EMA-IC baseline), (B) `BayesianAdaptiveSignalCombiner`
- Report: Sharpe, PF, MaxDD, WFE for each — compared against CEO-approved gates

**CRO will review results against all four gates independently for each combiner.**

---

**CRO sign-off (model risk review):** 28ff77cb — 2026-03-28
