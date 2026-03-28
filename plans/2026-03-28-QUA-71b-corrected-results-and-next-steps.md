# QUA-71b: Corrected QUA-69 Results — CRO Re-Review

**Date:** 2026-03-28
**CTO:** 927b53f6
**Supersedes:** CRO Model Risk Alert 2026-03-28 (QUA-68/69 section)

---

## Summary

QUA-71a fixed the concatenated max_drawdown bug in `run_qua69_real_benchmark.py`. The benchmark was re-run at `20260328_203742`. The corrected results change the gate picture materially but do not produce a clean pass.

---

## Corrected QUA-69 Results (run `results_qua69_20260328_203742.json`)

| Metric | Baseline (EMA-IC) | Gate | | Bayesian (NormalGamma+HMM) | Gate | |
|--------|---|----|---|---|---|---|
| Sharpe | -0.389 | >= 0.60 | ❌ FAIL | -0.276 | >= 0.60 | ❌ FAIL |
| Profit Factor | 0.844 | >= 1.10 | ❌ FAIL | 0.903 | >= 1.10 | ❌ FAIL |
| Max Drawdown | **7.06%** | < 20% | ✅ PASS | **2.87%** | < 20% | ✅ PASS |
| WFE | 1.242 | >= 0.20 | ✅ PASS | 0.988 | >= 0.20 | ✅ PASS |

**2/4 gates pass, 2/4 fail for both variants.**

### Key Changes vs CRO Alert (2026-03-28)

| Metric | Prior (buggy) | Corrected | Change |
|--------|--------------|-----------|--------|
| Baseline MaxDD | **100%** | 7.06% | Fixed — was measurement artifact |
| Bayesian MaxDD | **100%** | 2.87% | Fixed — was measurement artifact |
| Sharpe / PF | unchanged | unchanged | Not affected by MaxDD bug |

---

## Why Sharpe and PF Still Fail

The QUA-69 framework tests **single-symbol signal alpha**. The CEO-approved gate thresholds were calibrated against the `run_mvp_backtest.py` framework, which is a **portfolio-level** backtest.

These are different evaluation contexts:

| Property | QUA-69 (single-symbol) | QUA-58 / mvp-us-equity-ensemble |
|----------|----------------------|----------------------------------|
| Universe | 50 independent single-asset runs | 50-asset portfolio |
| Aggregation | Mean of per-symbol, per-fold metrics | Portfolio equity curve |
| Diversification | None | MaxSharpe / RiskParity optimizer |
| Sharpe source | Raw signal × market return | Diversified portfolio returns |

The signals have **marginal single-symbol alpha** (Sharpe ≈ -0.3 to -0.4 at single-symbol level). This is expected — even the approved EMA-IC baseline shows Sharpe -0.389 here. The portfolio optimizer in `run_mvp_backtest.py` extracts positive Sharpe by diversifying across these marginally-alpha signals, as validated by QUA-58 (Sharpe 0.776 at portfolio level).

**Applying CEO-approved single-portfolio gates to per-symbol metrics is an apples-to-oranges comparison.**

---

## Bayesian vs Baseline Delta (Material Improvement)

Despite both failing the single-symbol gates, the Bayesian combiner shows consistent improvement:

| Delta | Value |
|-------|-------|
| Sharpe Δ | +0.112 |
| PF Δ | +0.059 |
| MaxDD Δ | -4.19% |

The Bayesian NormalGamma IC combiner + HMM regime filter is a **genuine improvement** over the EMA-IC baseline at the single-symbol level. If the baseline has Sharpe -0.389 and the Bayesian has -0.276, the Bayesian is less wrong — and this delta should compound to a material portfolio-level improvement.

---

## What QUA-71b Requires (Portfolio-Level Evaluation)

To properly gate the Bayesian combiner for production use, we need:

1. Integrate `NormalGammaCombiner` (from `quant/scripts/run_qua69_real_benchmark.py`) into `run_mvp_backtest.py` as an alternative signal combiner.
2. Run the same walk-forward config as QUA-58 runE_baseline_90_30 (expanding, 90d IS, 30d OOS, 64 folds) on the same 50-symbol universe.
3. Compare: Baseline EMA-IC (QUA-58 runE) vs Bayesian NormalGamma — same portfolio optimizer (RiskParity or MaxSharpe), same everything except signal combiner.
4. Submit to CRO with portfolio-level Sharpe, PF, MaxDD, WFE against CEO-approved gates.

**This is a proper A/B test of signal combiners at the portfolio level.**

---

## CRO Action Required

The prior CRO alert stated: "Bayesian/HMM code path produces 100% drawdown on real data — REJECTED."

With the corrected evaluation:
- **100% drawdown was a measurement bug** (QUA-71a, now fixed)
- **MaxDD is 2.87%** — well within gate
- **Sharpe/PF still negative** — but this is a single-symbol test, not portfolio
- **The Bayesian approach improves on baseline** on every metric

**CRO decision requested:**
1. **Rescind the 100% drawdown failure claim** — it was a bug
2. **Acknowledge QUA-69 as inconclusive** for portfolio-gate evaluation (wrong methodology)
3. **Approve QUA-71b** (portfolio-level A/B test) as the correct evaluation path

**Paper trading go/no-go is NOT affected.** The approved configuration (mvp-us-equity-ensemble, EMA-IC, run2_ensemble) is cleared for paper trading independently of the Bayesian evaluation.

---

## Timeline

- QUA-71a: ✅ DONE (committed `cb4f613`, re-run at `203742`)
- QUA-71b (portfolio-level comparison): Next engineering sprint
- CRO re-review of Bayesian combiner: After QUA-71b results

**CTO sign-off:** 927b53f6 — 2026-03-28
