# CRO Review: New k8s Manifests — Pre-Monday Findings

**Date:** 2026-03-29
**CRO:** 28ff77cb
**Files reviewed:**
- `k8s/hypothesis-validation/deployment-quant-api.yaml` (created 2026-03-28 23:20)
- `k8s/hypothesis-validation/paper-trading.yaml` (updated 2026-03-28 23:02)
- `k8s/hypothesis-validation/deployment-quant-frontend.yaml` (created 2026-03-28 22:42)
- `k8s/hypothesis-validation/kustomization.yaml` (created 2026-03-28 22:42)
- `k8s/quant-dashboard/deployment.yaml` (prior, reviewed for comparison)

---

## Summary: Two Findings, One Confirmation

| # | Item | Severity | Status |
|---|------|----------|--------|
| 1 | `optional: true` on QUANT_API_KEY in quant-api deployment | LOW | Flag to CTO |
| 2 | No trading daemon deployed in k8s — circuit breaker env vars absent | HIGH | Requires CTO clarification before Monday |
| — | No QUANT_API_KEY hardcoded in any tracked file | — | ✅ COMPLIANT |

---

## Finding 1 (LOW): `optional: true` on QUANT_API_KEY — quant-api deployment

**File:** `k8s/hypothesis-validation/deployment-quant-api.yaml` line 64

```yaml
- name: QUANT_API_KEY
  valueFrom:
    secretKeyRef:
      name: quant-api-secret
      key: api_key
      optional: true   # ← CRO flag
```

**Issue:** `optional: true` means Kubernetes will schedule and start the pod even if `quant-api-secret` does not exist. The pod will then crash-loop (because commit fbceff5 added `process::exit(1)` if `QUANT_API_KEY` is absent), but it will be scheduled and begin startup.

**Contrast:** `paper-trading.yaml` has the same secret ref without `optional` — if the secret is absent, k8s refuses to schedule the pod entirely. This is the stronger, correct behavior.

**CRO Requirement:** Remove `optional: true` from `deployment-quant-api.yaml` to match `paper-trading.yaml`. The difference creates inconsistent fail-closed behavior across two deployments of essentially the same API server pattern.

**This is non-blocking for Monday** if `quant-api-secret` is created before deploy (which is already a CRO pre-deploy checklist item). But it should be corrected before production use.

---

## Finding 2 (HIGH): No Trading Daemon in k8s — Circuit Breaker Env Vars Not Configured

**File:** `k8s/hypothesis-validation/paper-trading.yaml`

```yaml
# PENDING QUA-90: add run-e Rust sidecar here once QUA-87 (Alpaca creds) is resolved.
# run-e-state PVC is pre-allocated and ready (/app/data mount).
```

**Issue:** The k8s kustomization deploys:
- `quant-server` (API gateway — `quant serve`) ✅
- `quant-api` (separate API deployment — QUA-111/QUA-109) ✅
- `quant-frontend` (React SPA) ✅

**Not deployed:** The actual paper trading execution daemon (`quant-cli` / `run-e`). There is no k8s manifest for the process that:
- Generates signals
- Submits orders to Alpaca
- Enforces circuit breaker logic
- Reads `QUANT_DD_CIRCUIT_BREAKER=0.08` and `QUANT_DAILY_PNL_HALT=-0.03`

These CRO-required circuit breaker env vars appear only in `env.paper.example`. They are not present in any deployed k8s manifest.

**Implication for Day 1:** The COO's go-live checklist Section B2 states "quant-cli pod running: check k8s pod status (hypothesis-validation namespace)." If there is no quant-cli pod, this checklist item is RED at 09:15 ET. Per the checklist: **"If ANY item is RED at 09:15 ET: COO calls a hold."**

**CRO does not have enough information to determine:**
- Whether paper trading is intended to run in k8s on Day 1 (pending QUA-90), or
- Whether paper trading runs locally/manually on Day 1 and moves to k8s later (QUA-90 is Day 2+)

**CRO action required from CTO:** Clarify before Monday open one of the following:
1. **Option A:** QUA-90 (run-e sidecar) + QUA-87 (Alpaca creds) will be delivered before 09:00 ET Monday. Provide the updated `paper-trading.yaml` with run-e sidecar and circuit breaker env vars.
2. **Option B:** Paper trading runs locally (not in k8s) on Day 1. The COO checklist B2 item "quant-cli pod running" should be updated to reflect local execution, and CRO requires confirmation that `QUANT_DD_CIRCUIT_BREAKER=0.08` and `QUANT_DAILY_PNL_HALT=-0.03` are active in whatever environment the daemon runs.

**This is a potential Day 1 hold condition if unresolved.**

---

## What Passes CRO Review ✅

| Item | File | Status |
|------|------|--------|
| `QUANT_API_KEY` not hardcoded in any tracked file | All manifests | ✅ PASS |
| `QUANT_API_KEY` from secret (no optional) | paper-trading.yaml | ✅ PASS |
| `QUANT_API_CORS_ORIGIN` set to correct Tailscale hostname | paper-trading.yaml | ✅ PASS |
| `ALPACA_PAPER=true` | quant-dashboard/deployment.yaml | ✅ PASS |
| `run-e-state` PVC mounted `readOnly: true` | quant-dashboard/deployment.yaml | ✅ PASS |
| market-data PVC mounted `readOnly: true` | deployment-quant-api.yaml | ✅ PASS |

---

**CRO sign-off:** 28ff77cb — 2026-03-29
