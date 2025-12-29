#!/bin/bash
# Reflex Frontend Entrypoint
# Dependencies are pre-installed in Docker image via multi-stage build

set -e

# Check if Redis is external (Kubernetes) or needs to be started locally (Docker standalone)
# In K8s, REFLEX_REDIS_URL will point to redis service (not localhost)
if [[ "${REFLEX_REDIS_URL:-redis://localhost}" == "redis://localhost"* ]]; then
    # Standalone Docker mode - start Redis locally
    echo "Starting local Redis server..."
    redis-server --daemonize yes
    # Wait for Redis to be ready
    echo "Waiting for Redis..."
    timeout 10 bash -c 'until redis-cli ping > /dev/null 2>&1; do sleep 0.5; done' || echo "Warning: Redis may not be ready"
else
    # Kubernetes mode - Redis is external
    echo "Using external Redis at: ${REFLEX_REDIS_URL}"
fi

# Determine environment mode from REFLEX_ENV (default to DEV for hot-reload)
ENV_MODE="${REFLEX_ENV:-DEV}"

echo "Starting Reflex app in ${ENV_MODE} mode..."
echo "  Frontend port: ${REFLEX_FRONTEND_PORT:-3000}"
echo "  Backend port: ${REFLEX_BACKEND_PORT:-8000}"
echo "  API URL: ${REFLEX_API_URL:-http://localhost:8000}"
echo "  Redis URL: ${REFLEX_REDIS_URL:-redis://localhost}"

if [ "$ENV_MODE" = "PROD" ]; then
    # Production mode (no hot-reload, optimized)
    exec reflex run --env prod --loglevel info
else
    # Development mode with hot-reload
    exec reflex run --env dev --loglevel debug
fi
