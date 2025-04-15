import json
import time
import random
import uuid
from kafka import KafkaProducer
from faker import Faker
import datetime

fake = Faker()

KAFKA_TOPIC = 'transactions'
KAFKA_BROKERS = 'kafka-producer:9092' # Use the internal Docker DNS name
FRAUD_PROBABILITY = 0.02 # 2% of transactions will be marked as fraud

def create_producer():
    # Retry connection logic
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers = KAFKA_BROKERS,
                value_serializer = lambda v: json.dumps(v).encode('utf-8'),
                client_id = 'fraud-producer'
            )
            print("Kafka Producer connected!")
            return producer
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

def generate_transaction():
    is_fraud = 1 if random.random() < FRAUD_PROBABILITY else 0
    amount = round(random.uniform(5.0, 1000.0), 2)
    if is_fraud:
        # Fraudulent transactions often have higher amounts or specific patterns
        amount = round(random.uniform(500.0, 5000.0), 2) # Example: Higher amount
    # Add more realistic fields as needed
    return {
        'transaction_id': str(uuid.uuid4()),
        'user_id': fake.uuid4(),
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'amount': amount,
        'merchant_id': f'merchant_{random.randint(1, 50)}',
        'location': { # Example location data
            'lat': float(fake.latitude()),
            'lon': float(fake.longitude())
        },
        'ip_address': fake.ipv4(),
        # CRUCIAL: Include the ground truth label for training later
        'is_fraud': is_fraud
    }

if __name__ == "__main__":
    producer = create_producer()
    print("Starting to send transaction events...")
    try:
        while True:
            transaction = generate_transaction()
            producer.send(KAFKA_TOPIC, value=transaction)
            # Limit console output frequency for readability
            if random.random() < 0.1:
                 print(f"Sent: {transaction}")
            # Simulate varying transaction rate
            time.sleep(random.uniform(0.05, 0.5))
    except KeyboardInterrupt:
        print("Stopping producer.")
    finally:
        if producer:
            producer.flush() # Ensure all messages are sent
            producer.close()
            print("Producer closed.")