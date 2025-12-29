#!/bin/sh

# Create and activate the virtual environment
rm -rf .venv
uv venv --python 3.13
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

mkdir -p /app/logs

# Start River ML Training API
# Training scripts are spawned on-demand via /switch_model endpoint
uvicorn app:app --host 0.0.0.0 --port 8002 --reload