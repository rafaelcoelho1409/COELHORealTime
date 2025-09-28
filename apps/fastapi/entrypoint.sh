#!/bin/sh

export GIT_PYTHON_REFRESH=quiet

# Create and activate the virtual environment
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

mkdir -p /app/logs

# Start both applications in the background *within the activated environment*
#python3 transaction_fraud_detection.py > transaction_fraud_detection.log 2>&1 &
#python3 estimated_time_of_arrival.py > estimated_time_of_arrival.log 2>&1 &
#python3 e_commerce_customer_interactions.py > e_commerce_customer_interactions.log 2>&1 &
#python3 sales_forecasting.py > sales_forecasting.log 2>&1 &
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &

wait