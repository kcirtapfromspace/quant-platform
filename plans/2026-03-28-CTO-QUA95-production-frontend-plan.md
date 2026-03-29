# QUA-95: Production Frontend Deployment

**Date:** 2026-03-28
**CTO:** 927b53f6
**Priority:** HIGH (Board-requested, QUA-94 escalation)
**Status:** IN PROGRESS

---

## Objective

Deploy the React+Express demo POC (`demo/`) to the k8s cluster as a production
service accessible at `https://dashboard.tail16ecc2.ts.net`.

The existing `quant.tail16ecc2.ts.net` (Rust `quant serve` status page) stays
unchanged. The new `dashboard` hostname will serve the full 7-page React frontend.

---

## Architecture

```
Tailscale (dashboard.tail16ecc2.ts.net)
    └─ Ingress (quant-dashboard)
        └─ Service (quant-dashboard, ClusterIP, port 3001)
            └─ Deployment (quant-dashboard)
                └─ Container: quant-dashboard
                   ├─ Express API server (port 3001)
                   └─ Serves built React SPA from /dist
```

The demo Express server is the API layer. In production:
- Static React files served from `dist/` at `/`
- API calls from the React app hit `/api/...` on the same origin (no CORS needed)
- `/ws` WebSocket for live quote streaming

---

## 4 Workstreams

### WS-1: Frontend production build (FrontendEngineer)

**Deliverables:**
1. `demo/server/index.ts` — add static file serving in production:
   - Serve `dist/` as Express static files
   - SPA fallback: `GET *` → `dist/index.html`
   - Guard with `NODE_ENV === 'production'` so dev mode still uses Vite proxy
2. `demo/Dockerfile` — multi-stage ARM64 build:
   - Stage 1: `node:20-alpine` — `npm ci && npm run build` (React assets to `dist/`)
   - Stage 2: `node:20-alpine` — `npm ci --omit=dev`, copy `server/`, copy `dist/`, `EXPOSE 3001`
   - Entrypoint: `node --import=tsx/esm server/index.ts`
   - Platform: `linux/arm64` (Turing Pi cluster)
3. `demo/tsconfig.server.json` (if needed) for server-only compile

**No changes to React pages or components needed for WS-1.**

### WS-2: Backend data bridge (BackendEngineer — after QUA-92)

**Deliverables:**
1. Connect `/api/portfolio` to Alpaca paper account via env-injected credentials:
   - Read `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` from env
   - Fallback to paper trading engine if creds absent (dev mode)
2. Replace `/api/risk/snapshot` stub with real computations from positions data
3. Replace `/api/strategies` static store with real data from `run_e_state.json`:
   - Read `RUN_E_STATE_PATH` env var (default `/app/data/run_e_state.json`)
4. Add env var: `PORT` (default 3001), `NODE_ENV`

### WS-3: K8s deployment (CTO)

**Deliverables:**
1. `k8s/quant-dashboard/deployment.yaml` — Deployment + imagePullSecrets
2. `k8s/quant-dashboard/service.yaml` — ClusterIP, port 3001
3. `k8s/quant-dashboard/ingress.yaml` — Tailscale ingress, hostname `dashboard`
4. `k8s/quant-dashboard/secret.yaml` — template for Alpaca creds (not committed)
5. `k8s/quant-dashboard/kustomization.yaml`
6. `.github/workflows/build-dashboard.yml` — build+push `ghcr.io/.../quant-dashboard:latest`

### WS-4: QA smoke tests (QAEngineer)

**Deliverables:**
1. `demo/tests/smoke.test.ts` — API endpoint smoke tests (supertest)
2. Verify all 7 pages render without errors
3. WebSocket connect/message test
4. Add `npm test` to CI

---

## K8s Manifest Plan

**Namespace:** `hypothesis-validation` (existing)
**Image:** `ghcr.io/kcirtapfromspace/quant-dashboard:latest`
**Node affinity:** CPU worker (talos-lwn-dba or any CPU node)
**Resources:** 200m CPU / 256Mi memory request; 500m / 512Mi limit
**Secret:** `alpaca-credentials` (already exists in namespace for run-e)
**Env:** `NODE_ENV=production`, `ALPACA_PAPER=true`, `RUN_E_STATE_PATH=/app/data/run_e_state.json`
**Volume mount:** `run-e-state` PVC at `/app/data` (read-only)

---

## CRO/Risk Notes

- No trading logic in the frontend; read-only connection to Alpaca
- Alpaca credentials reused from existing `alpaca-credentials` secret (no new secrets)
- Paper trading only — `ALPACA_PAPER=true` enforced at pod level

---

## Sequence

1. WS-1 (FrontendEngineer) → Dockerfile + static serving [IN PROGRESS]
2. WS-3 (CTO) → k8s manifests [IN PROGRESS]
3. WS-2 (BackendEngineer) → data bridge [after QUA-92]
4. CI: build-dashboard.yml builds and pushes on merge to main
5. ArgoCD auto-sync deploys to cluster
6. WS-4 (QAEngineer) → smoke tests [after WS-1 + WS-3]

---

**CTO sign-off:** 927b53f6 — 2026-03-28
