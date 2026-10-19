# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies with clean cache
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Copy source and build static bundle
COPY . .
RUN npm run build

# Production serve stage using high-performance lightweight Nginx
FROM nginx:alpine

# Copy built assets to Nginx html directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Configure Nginx for SPA routing and API proxying on port 5173
RUN printf 'server {\n\
    listen 5173;\n\
    server_name localhost;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
\n\
    location ~ ^/(api|v1|health|patients|predict|metrics|cohort|explain|graph|audit)/ {\n\
        proxy_pass http://api:8000;\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        proxy_set_header X-Forwarded-Proto $scheme;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]
