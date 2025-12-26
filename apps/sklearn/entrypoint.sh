#!/bin/sh

# Create and activate the virtual environment
rm -rf .venv
uv venv --python 3.13
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start Sklearn Batch ML service
uvicorn app:app --host 0.0.0.0 --port 8003 --reload &

wait
