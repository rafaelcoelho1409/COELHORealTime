#!/bin/sh

# Create and activate the virtual environment
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install --system -r requirements.txt

uv run mlflow server --host 0.0.0.0 --port 5000