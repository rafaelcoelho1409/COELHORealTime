#!/bin/sh

export GIT_PYTHON_REFRESH=quiet

# Create and activate the virtual environment
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start applications in the background *within the activated environment*
(python3 transaction_fraud_detection_consumer.py > transaction_fraud_detection_consumer.log 2>&1) &
(uvicorn app:app --host 0.0.0.0 --port 8000 --reload) &

wait