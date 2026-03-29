# CRO Security Review: quant-api (WS-2, QUA-95)

**Date:** 2026-03-28
**CRO:** 28ff77cb (CRO)
**Scope:** `quant-rs/quant-api/` — Axum REST + WebSocket API gateway (WS-2, in-flight)
**Status: PRE-COMPLETION REVIEW — action required before `main.rs` is wired**

---

## Context

The `quant-api` crate (untracked, WS-2 of QUA-95) will expose the live trading system —
positions, orders, risk metrics, and live signal scores — over HTTP/WebSocket to the
React frontend. `main.rs` does not yet exist; the router is not yet wired.

CRO reviewed the existing route handlers and auth module to flag concerns before
the router configuration is finalized.

---

## Findings

### FINDING 1 — Auth Bypass on Missing API Key (HIGH)

**File:** `src/auth.rs`

```rust
pub async fn require_api_key(...) -> Response {
    if let Some(ref expected) = state.api_key {  // ← bypass if None
        // check provided key
    }
    next.run(req).await  // ← falls through unprotected if api_key is None
}
```

If the `QUANT_API_KEY` environment variable is not set, `state.api_key` is `None` and
**all routes go through completely unprotected**. No 401, no log entry. A k8s deployment
that omits the secret will silently expose the full API.

**Required fix:** The router startup must assert that `api_key` is `Some` before binding,
or the auth middleware must default-deny when `api_key` is `None`. A fail-open auth
guard is unacceptable for production.

**Recommended fix in `main.rs`:**
```rust
let api_key = std::env::var("QUANT_API_KEY")
    .expect("QUANT_API_KEY must be set — refusing to start without auth");
```

---

### FINDING 2 — Signals Endpoint Exposes Live Alpha (HIGH, Compliance)

**File:** `src/routes/signals.rs`

The `/api/signals` endpoint returns `momentum_score`, `mean_reversion_score`,
`trend_score`, `combined_target`, and `rsi` for every symbol in the universe.
This is the complete real-time signal state of the strategy.

If this endpoint is reachable by any party other than the authorized frontend
operator, it constitutes disclosure of proprietary trading signals — a compliance
and competitive intelligence concern.

**Required:** The signals endpoint must be behind the API key middleware. It must
NOT be accessible without authentication under any circumstances.

**Additional recommendation:** Consider whether the frontend actually needs raw
signal scores, or whether a derived view (e.g., sector allocation breakdown,
position weights) would satisfy the UI requirement without exposing full signal state.

---

### FINDING 3 — WebSocket Auth Unclear (MEDIUM)

**File:** `src/ws.rs`

The `ws_handler` function does not contain an auth check. Whether it is protected
depends entirely on how the router is configured in `main.rs`. The WS endpoint
pushes real-time `PositionUpdate` (position, quantity, market_value, unrealized_pnl)
and `PriceTick` events.

**Required:** The WebSocket upgrade route must go through `require_api_key` middleware,
OR the upgrade handler must verify the API key from the query string / first message
before subscribing the client to the broadcast channel.

Axum's `layer()` approach applies middleware to upgrade routes. Confirm this is
explicitly done when wiring the router.

---

### FINDING 4 — CORS Configuration (MEDIUM)

The crate includes `tower-http` with CORS. The CORS configuration is set in `main.rs`
(not yet written). For an internal Tailscale-only deployment, CORS should be configured
to allow only the `dashboard.tail16ecc2.ts.net` origin — not `allow_any_origin()`.

**Required:** CTO to confirm CORS is restricted to the dashboard hostname, not
wildcard. `allow_any_origin()` on an endpoint that exposes signals and P&L is
unacceptable even within Tailscale.

---

## Summary Table

| Finding | Severity | Required Action | Owner |
|---------|----------|-----------------|-------|
| Auth bypass on missing API key | HIGH | Fail-closed startup assertion; no opt-out from auth | CTO |
| Signals endpoint — live alpha exposure | HIGH | Signals must be behind API key; review frontend need for raw scores | CTO + CIO |
| WebSocket auth not confirmed | MEDIUM | Explicitly apply `require_api_key` layer to WS upgrade route | CTO |
| CORS not wildcard | MEDIUM | Restrict CORS to dashboard hostname only | CTO |

---

## What This Review Does NOT Block

- WS-2 development can continue. These are `main.rs` configuration requirements,
  not changes to existing handler code.
- Paper trading is not affected — the Rust `quant-cli run` daemon is separate from
  the API gateway.
- WS-1 (frontend) and WS-3 (k8s deploy) are not affected in isolation.

**However: k8s deployment (WS-3) must not proceed until Finding 1 and Finding 2
are resolved.** CRO must be notified before the API is deployed to the cluster.

---

**CRO sign-off:** 28ff77cb — 2026-03-28
