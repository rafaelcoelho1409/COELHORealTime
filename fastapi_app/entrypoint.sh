#!/bin/sh

# Create and activate the virtual environment
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start both applications in the background *within the activated environment*
(uvicorn app:app --host 0.0.0.0 --port 8000 --reload) &
(python3 consumer.py) &

wait