#!/bin/bash
# Scikit-Learn Batch ML Service Entrypoint
# Dependencies are pre-installed in Docker image via multi-stage build

set -e

# Start Sklearn Batch ML service with hot-reload
exec uvicorn app:app --host 0.0.0.0 --port 8003 --reload
