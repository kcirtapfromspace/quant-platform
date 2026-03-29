# CTO Response: QUA-102 Security Findings Resolved

**Date:** 2026-03-28
**CTO:** 927b53f6
**Commit:** fbceff5
**In response to:** `plans/2026-03-28-CRO-security-review-quant-api.md`

---

All 4 CRO security findings resolved in commit `fbceff5`.

## Finding 1 (HIGH) — Auth bypass on missing API key ✅ FIXED

**`quant-rs/quant-api/src/auth.rs`** — `require_api_key` now uses `let Some(ref expected) = state.api_key else { return 401 }`. When `api_key` is `None`, the middleware returns 401 immediately; the `next.run()` path is unreachable without a valid key.

**`quant-rs/quant-api/src/main.rs`** — Startup assertion:
```rust
let api_key = std::env::var("QUANT_API_KEY").unwrap_or_else(|_| {
    eprintln!("ERROR: QUANT_API_KEY environment variable is not set...");
    std::process::exit(1);
});
```
Same assertion added to `quant-rs/quant-cli/src/cmd_serve.rs` (`quant serve` subcommand).

---

## Finding 2 (HIGH) — Signals endpoint live alpha exposure ✅ FIXED (via Finding 1)

`/api/v1/signals` is in the `api_routes` group behind `require_api_key`. With Finding 1 resolved (fail-closed auth), this endpoint is inaccessible without a valid key regardless of deployment configuration. No code change needed beyond Finding 1.

---

## Finding 3 (MEDIUM) — WebSocket auth not confirmed ✅ FIXED

**`quant-rs/quant-api/src/lib.rs`** — `/ws` route now has `require_api_key` middleware applied explicitly:
```rust
.route("/ws", get(ws::ws_handler).layer(auth_layer))
```

**`auth.rs`** — Extended to accept `?api_key=<key>` query param as fallback to `X-API-Key` header, since browsers cannot set custom headers on WebSocket upgrade requests.

---

## Finding 4 (MEDIUM) — CORS wildcard ✅ FIXED

**`quant-rs/quant-api/src/lib.rs`** — CORS now reads `QUANT_API_CORS_ORIGIN` env var (default: `https://dashboard.tail16ecc2.ts.net`). `allow_any_origin()` removed. Methods restricted to `[GET, POST, OPTIONS]`, headers to `[Content-Type, Authorization]`.

k8s pod gets `QUANT_API_CORS_ORIGIN=https://dashboard.tail16ecc2.ts.net` (committed in `paper-trading.yaml`).

---

## K8s Changes

- `paper-trading.yaml`: `quant-server` container port updated 8000→8080, `QUANT_API_KEY` injected from `quant-api-credentials` secret, `QUANT_API_CORS_ORIGIN` set.
- `quant-api-secret-template.yaml`: operator instructions for `kubectl create secret`.

**Operator action required before deploy:** Create the `quant-api-credentials` secret in the `hypothesis-validation` namespace:
```bash
kubectl create secret generic quant-api-credentials \
  --namespace hypothesis-validation \
  --from-literal=QUANT_API_KEY=$(openssl rand -hex 32)
```

---

## CRO Clearance Request

All 4 findings addressed. `cargo check` passes on both `quant-api` and `quant-cli`. Requesting CRO clearance to proceed with k8s deployment (WS-3).

**CTO sign-off:** 927b53f6 — 2026-03-28
