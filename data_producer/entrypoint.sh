#!/bin/bash

# Wait for ZooKeeper to be ready
echo "Waiting for ZooKeeper to start..."
while ! nc -z zookeeper 2181; do
  sleep 0.1
done

# Start Kafka in background using original entrypoint
/opt/bitnami/scripts/kafka/entrypoint.sh /opt/bitnami/scripts/kafka/run.sh &

# Wait for Kafka to be ready
echo "Waiting for Kafka to start..."
while ! nc -z localhost 9092; do
  sleep 0.1
done

# Start Python producer
python3 producer.py