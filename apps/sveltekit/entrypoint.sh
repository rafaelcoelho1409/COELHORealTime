#!/bin/bash
set -e

if [ "$NODE_ENV" = "production" ]; then
    echo "Starting SvelteKit in production mode..."
    pnpm build
    node build
else
    echo "Starting SvelteKit in development mode..."
    # Skip svelte-kit sync (already done in Docker build)
    # Use pre-bundled deps from Docker build cache
    exec pnpm dev
fi
