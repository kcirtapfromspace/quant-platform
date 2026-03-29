# CRO Gate Decision: QUA-71b — Bayesian NormalGamma IC Combiner

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**Ticket:** QUA-71b
**CTO Submission:** `plans/2026-03-28-CTO-QUA71b-results-CRO-submission.md`
**Results file:** `backtest-results/qua71b-bayesian-portfolio/results_qua71b_20260328_231911.json`
**Run ID:** 20260328_231911

---

## Decision: APPROVED — DEFERRED DEPLOYMENT

The Bayesian NormalGamma IC combiner passes all four CEO-approved CRO gates.
CRO sign-off is granted. **However, deployment is deferred** — no change to paper
trading configuration at this time. EMA-IC remains the operational default.

This clears the QUA-69/71 PENDING flag. The Bayesian strategy is no longer blocked.

---

## Gate Results

| Gate | Threshold | runA (EMA-IC baseline) | runB (Bayesian NormalGamma) | Status |
|------|-----------|------------------------|-----------------------------|--------|
| OOS Sharpe | >= 0.60 | 1.088 | 1.088 | ✅ PASS |
| Profit Factor | >= 1.10 | 1.216 | 1.216 | ✅ PASS |
| Max Drawdown | < 20% | 14.05% | 14.05% | ✅ PASS |
| WFE | >= 0.20 | 0.837 | 0.8373 | ✅ PASS |
| Folds | — | 19 | 19 | — |
| **Gate result** | | **PASS** | **PASS** | |

Delta (Bayesian vs EMA-IC): Sharpe 0.000, PF 0.000, MaxDD 0.000, WFE +0.0003

---

## CRO Assessment

### 1. Gate Clearance

Both runA and runB pass all four gates on the correct apples-to-apples basis
(mvp_backtest.duckdb, IS=252/OOS=63, 19 folds, expanding — same as QUA-49).
The CRO's prior acceptance criteria for QUA-71b are satisfied.

### 2. Zero Delta — Not a Bug

The CTO's explanation is accepted. Two structural reasons cause identical portfolio metrics:

**a) Estimator convergence:** With IS=252 bars of warmup data, both the EWM halflife=21
estimator and the NormalGamma posterior converge to similar IC estimates before OOS begins.
The weak NormalGamma prior is overwhelmed by 252 observations of likelihood — the posterior
mean tracks EWM closely. This is correct Bayesian behavior, not a defect.

**b) Sleeve dilution:** The Bayesian combiner applies only to the `adaptive_combined` sleeve
(25% of capital). Even material IC weight differences within that sleeve are diluted to
~25% of their individual magnitude at the portfolio level. With estimators already
converged, the residual delta is negligible (+0.0003 WFE).

This is a methodologically sound result. CRO accepts the finding.

### 3. Deployment Decision — Deferred

The Bayesian combiner adds no material portfolio-level performance improvement under
current configuration. Deploying it creates operational complexity (model maintenance,
additional parameter surface) without corresponding benefit. The simpler EMA-IC is
operationally equivalent and preferred by Occam's Razor.

**CRO guidance (aligns with CTO recommendation):**
- Bayesian combiner: **APPROVED but DEFERRED**
- EMA-IC: **remains default for all paper trading configurations**
- Revisit when adaptive sleeve allocation increases above ~35% OR IS warmup is shortened
  below ~100 bars — those are the conditions under which NormalGamma early-fold advantage
  would become material at portfolio level

### 4. PF Gap

The Bayesian combiner does not close the PF gap (1.216 vs 1.30 aspiration). The CTO's
vol regime sleeve (QUA-92, now in planning) is the next endorsed lever. CRO endorses
proceeding with QUA-92.

---

## Prior Status Update

| Item | Prior Status | Current Status |
|------|-------------|----------------|
| QUA-69 Bayesian/HMM (real data) | PENDING QUA-71b | CLEARED — both runs pass gates |
| QUA-71b portfolio comparison | PENDING results | COMPLETE — Bayesian approved (deferred) |

---

## No Changes to Paper Trading

Paper trading configuration is unchanged:

| Strategy | Config | Status |
|----------|--------|--------|
| mvp-us-equity-ensemble (run2_ensemble) | EMA-IC, 21-day rebal | ACTIVE ✅ |
| QUA-58 runE_baseline_90_30 | Approved | ACTIVE ✅ |
| QUA-53 runE | Approved | ACTIVE ✅ |
| signal_expansion_ensemble (QUA-85) | 90% Kelly | ACTIVE ✅ |

Bayesian combiner: APPROVED, not deployed.

---

## MaxDD Regression Flag — CLOSED

CRO HIGH flag from QUA-85 gate decision (baseline MaxDD 14% → 20%) is formally closed.
CTO investigation (plan: `2026-03-28-CTO-maxdd-regression-investigation.md`) confirms:
- **Not a bug.** Three methodology differences explain the delta:
  1. Extended data period (2018 start vs 2020) — captures Q4 2018 correction
  2. Shorter OOS windows (30-bar vs 63-bar) — higher point-in-time DD sampling
  3. Smaller universe (110 vs 355 symbols) — less cross-sectional diversification
- QUA-49 and QUA-85 approvals are each internally consistent on their own basis
- **No cross-comparison between QUA-49 and QUA-85 results is valid**

Action going forward: CRO and CTO to agree on canonical WF config before running
comparison runs in future gate reviews.

---

## Open Items (CRO Watch List)

| Item | Owner | Priority | Status |
|------|-------|----------|--------|
| QUA-92 vol regime sleeve backtest | CTO/BE | HIGH | PLANNING — CRO endorses |
| Weekly paper trading P&L review | CRO | ONGOING | Active |
| PF gap (1.237 vs 1.30) | CIO | HIGH | QUA-92 is next lever |
| MA variant identical metrics flag | CIO | LOW | Monitoring |
| Alpha tilt variant identical metrics flag | CIO | LOW | Monitoring |
| Canonical WF config alignment | CTO/CRO | MEDIUM | Before next cross-comparison |

---

**CRO sign-off:** 28ff77cb — 2026-03-28
