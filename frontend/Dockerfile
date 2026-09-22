# ==========================================
# Stage 1: Build the React Application
# ==========================================
FROM node:20-alpine AS build

WORKDIR /app

# Copy package descriptors and lockfiles
COPY package*.json ./

# Install npm dependencies
RUN npm ci

# Copy all source files
COPY . .

# Build the production bundle
RUN npm run build

# ==========================================
# Stage 2: Serve Build using Nginx
# ==========================================
FROM nginx:alpine

# Copy built assets from Stage 1 to Nginx default html directory
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
