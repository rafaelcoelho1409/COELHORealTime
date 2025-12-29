#!/bin/bash
# River ML Training Service Entrypoint
# Dependencies are pre-installed in Docker image via multi-stage build

set -e

# Ensure logs directory exists
mkdir -p /app/logs

# Start River ML Training API with hot-reload
# Training scripts are spawned on-demand via /switch_model endpoint
exec uvicorn app:app --host 0.0.0.0 --port 8002 --reload
