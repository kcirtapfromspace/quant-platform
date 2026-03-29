# syntax=docker/dockerfile:1
FROM --platform=linux/arm64 node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM --platform=linux/arm64 nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# Default nginx config serves index.html for all routes (SPA fallback)
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 80
