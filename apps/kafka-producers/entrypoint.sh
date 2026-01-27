#!/bin/bash
# Kafka Producers Entrypoint
# Dependencies are pre-installed in Docker image via multi-stage build

set -e

# Set up Prometheus multiprocess directory
export PROMETHEUS_MULTIPROC_DIR=$(mktemp -d)
echo "Prometheus multiprocess dir: ${PROMETHEUS_MULTIPROC_DIR}"

# Get Kafka host from environment variable (set by Helm ConfigMap)
KAFKA_BOOTSTRAP="${KAFKA_HOST:-coelho-realtime-kafka}:9092"

echo "Waiting for Kafka broker at ${KAFKA_BOOTSTRAP}..."
while ! nc -z ${KAFKA_HOST:-coelho-realtime-kafka} 9092; do
  echo "Kafka not ready, retrying in 5 seconds..."
  sleep 5
done

echo "Kafka is ready. Starting metrics server and Python producers..."

# Start Prometheus metrics server (multiprocess collector on port 8000)
PYTHONUNBUFFERED=1 python3 metrics_server.py 2>&1 &

# Start Python producers with unbuffered output
# Using PYTHONUNBUFFERED=1 ensures print statements appear immediately
PYTHONUNBUFFERED=1 python3 transaction_fraud_detection.py \
  --fraud-probability 0.01 \
  --account-age-days-limit 5 \
  --merchant-id-limit 200  \
  --cvv-provided-missing-probability 0.05 \
  --billing-address-match-missing-probability 0.1 \
  --high-value-fraud-probability 0.8 2>&1 &

PYTHONUNBUFFERED=1 python3 estimated_time_of_arrival.py \
  --heavy-traffic-multiplier 2.0 \
  --bad-weather-probability 0.1 \
  --incident-probability 0.02 2>&1 &

PYTHONUNBUFFERED=1 python3 e_commerce_customer_interactions.py \
  --event-rate-per-minute 600 2>&1 &

echo "All producers started."

wait
