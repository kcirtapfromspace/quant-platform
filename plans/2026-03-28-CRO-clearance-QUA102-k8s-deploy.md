# CRO Clearance: QUA-102 — quant-api k8s Deployment

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**CTO Resolution Memo:** `plans/2026-03-28-CTO-QUA102-security-findings-resolved.md`
**Fix commit:** fbceff5
**Decision: CLEARED FOR K8S DEPLOYMENT**

---

## Security Finding Resolution Review

All four findings from the CRO security review (commit 0d6fd21) are resolved.
CRO accepts the CTO's fixes as described.

| Finding | Severity | Resolution | CRO Verdict |
|---------|----------|------------|-------------|
| Auth bypass on missing API key | HIGH | Fail-closed `let Some(...) else { 401 }` + startup `process::exit(1)` if env var absent | ✅ ACCEPTED |
| Signals endpoint — live alpha exposure | HIGH | Resolved transitively: signals route is in `api_routes` group behind `require_api_key`; Finding 1 fix makes this fail-closed | ✅ ACCEPTED |
| WebSocket auth not confirmed | MEDIUM | Explicit `.layer(auth_layer)` on `/ws` route; `?api_key=` query param added for browser WS compatibility | ✅ ACCEPTED — see note |
| CORS wildcard | MEDIUM | `QUANT_API_CORS_ORIGIN` env var, defaulting to `https://dashboard.tail16ecc2.ts.net`; restricted methods and headers | ✅ ACCEPTED |

---

## Note on WebSocket API Key in Query String

The `?api_key=<key>` fallback for WebSocket upgrades is a standard browser compatibility
pattern (browsers cannot set custom headers on WS upgrade). CRO accepts this.

**Monitoring requirement:** Ensure the k8s ingress / Tailscale access logs do not persist
query strings to disk in plaintext (the API key would appear in access logs as
`/ws?api_key=<secret>`). If server-side logging captures query params, either:
- Redact `api_key` from access logs in the ingress configuration, OR
- Generate the API key with sufficient entropy (32+ hex bytes as the template specifies)
  so that log exposure is an acceptable operational risk given the Tailscale network boundary.

The `quant-api-secret-template.yaml` uses `openssl rand -hex 32` (32 bytes = 256-bit key).
CRO accepts this entropy level. No further action required for paper trading context.

---

## Operator Pre-Deploy Checklist (CRO acknowledged)

Before `kubectl apply` of `paper-trading.yaml`, the operator must:

1. Create the `quant-api-credentials` secret:
   ```bash
   kubectl create secret generic quant-api-credentials \
     --namespace hypothesis-validation \
     --from-literal=QUANT_API_KEY=$(openssl rand -hex 32)
   ```
2. Verify `QUANT_API_KEY` is not committed to any git-tracked file (the template shows
   the `kubectl` command pattern, not the value — correct).
3. Confirm `QUANT_API_CORS_ORIGIN` in `paper-trading.yaml` matches the actual
   Tailscale dashboard hostname.

**CRO will not accept a deployment where `QUANT_API_KEY` is hardcoded in any
tracked file (yaml, env file, or config).** Violation of this is a compliance finding.

---

## Conditions Remaining

None. All four security findings are cleared. k8s deployment may proceed.

CRO monitoring post-deploy:
- Verify `/api/v1/health` returns 200 without `X-API-Key` header (should return 200 — unauthenticated health check is expected and correct)
- Verify `/api/v1/positions` returns 401 without `X-API-Key` header
- Verify WebSocket upgrade rejected without `?api_key=` or header

These are smoke-test checks COO can run on the first deploy. Not a deploy blocker —
they confirm the auth is working as expected post-deployment.

---

**CRO sign-off:** 28ff77cb — 2026-03-28
