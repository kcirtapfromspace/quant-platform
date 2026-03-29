# syntax=docker/dockerfile:1
# Target: linux/arm64 (Turing Pi cluster)

# ── Stage 1: Build React SPA ──────────────────────────────────────────────────
FROM --platform=linux/arm64 node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── Stage 2: Production server ────────────────────────────────────────────────
FROM --platform=linux/arm64 node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3001

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY server/ server/
COPY --from=builder /app/dist dist/

EXPOSE 3001

CMD ["node", "--import=tsx/esm", "server/index.ts"]
