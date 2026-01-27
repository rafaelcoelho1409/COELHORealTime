#!/bin/bash
# Unified ML Service Entrypoint
# Combines Incremental ML (River) and Batch ML (Scikit-Learn) APIs
# Dependencies are pre-installed in Docker image via multi-stage build

set -e

# Ensure logs directory exists
mkdir -p /app/logs

# Start Unified ML Service API with hot-reload
# Endpoints:
#   /api/v1/incremental - River (streaming ML)
#   /api/v1/batch       - Scikit-Learn/CatBoost (batch ML)
#   /api/v1/sql         - SQL queries against Delta Lake
exec uvicorn app:app --host 0.0.0.0 --port 8000 --reload
