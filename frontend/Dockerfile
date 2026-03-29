# syntax=docker/dockerfile:1
FROM --platform=linux/arm64 node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM --platform=linux/arm64 nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# nginx.conf: listens on 8080, proxies /api/ and /ws to quant-api (cluster-internal)
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
