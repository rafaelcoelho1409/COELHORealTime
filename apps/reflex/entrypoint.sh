#!/bin/bash
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

# Determine environment mode from REFLEX_ENV (default to PROD if not set)
ENV_MODE="${REFLEX_ENV:-PROD}"

if [ "$ENV_MODE" = "PROD" ]; then
    echo "Starting Reflex app in PRODUCTION mode..."
    echo "  Frontend port: ${REFLEX_FRONTEND_PORT:-3000}"
    echo "  Backend port: ${REFLEX_BACKEND_PORT:-8000}"
    echo "  API URL: ${REFLEX_API_URL:-http://localhost:8000}"
    echo "  Redis URL: ${REFLEX_REDIS_URL:-redis://localhost}"
    reflex run --env prod --loglevel info
else
    echo "Starting Reflex app in DEVELOPMENT mode with hot-reload..."
    # For development, create venv if needed
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        uv venv --python 3.13
    fi
    source .venv/bin/activate
    # Install/update dependencies
    echo "Installing dependencies..."
    uv pip install -r requirements.txt
    # Initialize Reflex
    reflex init --loglevel error || true
    reflex run --env dev --loglevel debug
fi