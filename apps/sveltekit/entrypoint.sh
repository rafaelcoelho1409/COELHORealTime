#!/bin/bash
set -e

# Ensure SvelteKit types are synced
pnpm svelte-kit sync 2>/dev/null || true

if [ "$NODE_ENV" = "production" ]; then
    echo "Starting SvelteKit in production mode..."
    pnpm build
    node build
else
    echo "Starting SvelteKit in development mode..."
    exec pnpm dev
fi
