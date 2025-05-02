#!/bin/bash

# Start Kafka in background using original entrypoint
/opt/bitnami/scripts/kafka/entrypoint.sh /opt/bitnami/scripts/kafka/run.sh &

# Wait for Kafka to be ready
echo "Waiting for Kafka to start..."
while ! nc -z localhost 9092; do
  sleep 0.1
done


# Start Python producers
python3 transaction_fraud_detection.py \
  --fraud-probability 0.01 \
  --account-age-days-limit 5 \
  --merchant-id-limit 200  \
  --cvv-provided-missing-probability 0.05 \
  --billing-address-match-missing-probability 0.1 \
  --high-value-fraud-probability 0.8 &

python3 estimated_time_of_arrival.py \
  --heavy-traffic-multiplier 2.0 \
  --bad-weather-probability 0.1 \
  --incident-probability 0.02 &


wait