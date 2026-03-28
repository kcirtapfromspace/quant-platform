# CRO Model Risk Disposition: QUA-88 — Per-Asset HMM Regime Detection

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**Ticket:** QUA-88
**Research Branch:** Per-asset Hidden Markov Model regime signal filtering
**Disposition: CLOSED — NEGATIVE RESULT. DO NOT PURSUE.**

---

## Summary

QUA-88 tested per-asset HMM regime detection as a filter/overlay on the `signal_expansion_ensemble` strategy. The hypothesis was that detecting per-asset bull/bear regimes would improve signal quality and reduce drawdown. The result is the opposite: HMM filtering degrades all four CRO gate metrics and causes two of three runs to fail the MaxDD hard gate.

---

## Backtest Results (commit e41d331)

| Run | Sharpe | PF | WFE | MaxDD | Gate Status |
|-----|--------|----|-----|-------|-------------|
| Run A — baseline (QUA-85 approved) | 1.034 | 1.24 | 0.899 | 19.83% | **PASS** |
| Run B — per-asset HMM only | 1.005 | 1.23 | 0.812 | 20.17% | **FAIL** (MaxDD) |
| Run C — combined global + per-asset HMM | 1.003 | 1.23 | 0.876 | 20.28% | **FAIL** (MaxDD) |

All four gate metrics deteriorate when HMM is applied:
- Sharpe: -0.029 to -0.031 (degradation)
- PF: -0.01 (negligible but still degradation)
- WFE: -0.023 to -0.087 (degradation, signal stability hurt)
- MaxDD: +0.34% to +0.45% (worse, breaches 20% gate)

---

## CRO Model Risk Assessment

### Why HMM Fails Here

Per-asset regime detection introduces several risks that likely explain the degradation:

1. **State estimation latency.** HMM regimes are estimated in-sample at each fold. The Viterbi/forward-backward path is noisy at regime transitions — precisely when the strategy needs the most guidance. This produces misclassification noise during the most volatile periods, increasing MaxDD.

2. **Overfitting to in-sample regime patterns.** 50 assets each with their own HMM increases the parameter space substantially. Regime sequences that appear robust in IS may not generalize — WFE declining from 0.899 → 0.812 (Run B) confirms this.

3. **Regime fragmentation.** Per-asset regimes can contradict each other, reducing the coherence of ensemble signals. When some assets are in "bear" regime and others in "bull," the filter attenuates otherwise valid diversified signals.

4. **Insufficient diversification benefit.** The global regime filter (already in baseline) captures the primary source of regime risk. Per-asset HMM adds granularity without adding genuine diversification — it layers complexity without improving the risk/return profile.

### Model Risk Conclusion

The HMM per-asset approach increases model complexity while decreasing out-of-sample performance. It is rejected under the CRO model risk framework. The approach should not be revisited without a fundamentally different methodology (e.g., reducing universe, unsupervised clustering with validation).

---

## PF Gap — Next Steps

The primary motivation for QUA-88 was closing the PF gap (1.24 actual vs 1.30 aspiration). HMM did not help. Alternative approaches that remain viable:

| Approach | Owner | Priority | Rationale |
|----------|-------|----------|-----------|
| **Volatility regime sleeve** | CIO | HIGH | Originally scoped in QUA-85 approval. Low-vol / high-vol regime filter operates at portfolio level, not per-asset. Less overfitting risk than per-asset HMM. |
| **QUA-71b Bayesian adaptive combiner** | CTO | MEDIUM | BayesianAdaptiveSignalCombiner dynamically re-weights sleeves. If it up-weights mean_reversion (PF 1.28) during its strongest regime, portfolio-level PF could improve. |
| **Universe refinement** | CIO | LOW | Restricting to higher-quality signals (e.g., dropping weakest PF assets from universe) may improve aggregate PF. Worth exploring after vol sleeve. |

**CRO guidance:** Prioritize vol sleeve implementation (CIO) as next PF gap attempt. QUA-71b proceeds in parallel for its own merits (already model-risk approved). Do not pursue additional regime detection variants until vol sleeve results are reviewed.

---

## Implications for Live Paper Trading

The approved paper trading strategy remains unchanged: `signal_expansion_ensemble` at 90% Kelly (QUA-85 approval). QUA-88 results confirm the approved configuration is correct — do not apply HMM filtering to the live paper trading run.

**No changes to paper trading configuration required.**

---

## Actions

| Action | Owner | Priority | Deadline |
|--------|-------|----------|----------|
| Implement vol regime sleeve for PF gap attempt | CIO | HIGH | Next sprint |
| Continue QUA-71b Bayesian combiner (independent of HMM) | CTO | MEDIUM | Awaiting duckdb confirm |
| Investigate baseline MaxDD regression (14% → 20%) | CTO/CIO | HIGH | Before next gate review |
| Weekly paper trading P&L review | CRO | ONGOING | Every Monday |

---

**CRO sign-off:** 28ff77cb — 2026-03-28
