import json
from kafka import KafkaConsumer
from river import (
    compose, 
    linear_model, 
    preprocessing, 
    metrics, 
    anomaly
)
import datetime
import pickle
import os
import pandas as pd


# Configuration
KAFKA_TOPIC = 'transactions'
KAFKA_BROKERS = 'kafka-producer:9092'  # Adjust as needed
MODEL_PATH = 'river_model.pkl'
DATA_PATH = 'river_data.pkl'


def load_or_create_model():
    """Load existing model or create a new one"""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        # Create a new model pipeline
        model = compose.Pipeline(
            #compose.Select("amount"),
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )
        # Alternatively for anomaly detection:
        # model = anomaly.HalfSpaceTrees(seed=42)
        return model

def process_transaction(model, transaction, metric=None):
    """Process a single transaction and update the model"""
    # Prepare features and label
    x = {
        #'transaction_id': transaction['transaction_id'],
        #'user_id': transaction['user_id'],
        #'timestamp': transaction['timestamp'],
        'amount': transaction['amount'],
        #'merchant_id': transaction['merchant_id'],
        #'location': transaction['location'],
        #'ip_address': transaction['ip_address'],
    }
    y = transaction['is_fraud']
    # Update the model
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    # Update metrics if provided
    if metric:
        metric.update(y, y_pred)
    return y_pred

def create_consumer():
    """Create and return Kafka consumer"""
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers = KAFKA_BROKERS,
        auto_offset_reset = 'earliest',
        value_deserializer = lambda v: json.loads(v.decode('utf-8')),
        group_id = 'river_trainer'
    )

def main():
    # Initialize model and metrics
    model = load_or_create_model()
    metric = metrics.Accuracy()  # Or other relevant metric
    # Create consumer
    consumer = create_consumer()
    print("Consumer started. Waiting for transactions...")
    data_df = pd.DataFrame(
        columns = [
            'transaction_id',
            'user_id',
            'timestamp',
            'amount',
            'merchant_id',
            'location',
            'ip_address',
            'is_fraud'
        ]
    )
    try:
        for message in consumer:
            transaction = message.value
            # Create a new DataFrame from the received data
            new_row = pd.DataFrame([transaction])
            # Append the new row to the existing DataFrame
            data_df = pd.concat([data_df, new_row], ignore_index = True)
            # Process the transaction
            prediction = process_transaction(model, transaction, metric)
            # Periodically log progress
            if message.offset % 1000 == 0:
                print(f"Processed {message.offset} messages")
                print(f"Current accuracy: {metric.get():.2%}")
                print(f"Last prediction: {'Fraud' if prediction == 1 else 'Legit'}")
                print(f"Metric: {metric}")
                # Save model periodically
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(model, f)
                data_df.to_pickle("river_data.pkl")
    except:
        print("Stopping consumer...")
    finally:
        # Save final model state
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        consumer.close()
        print("Consumer closed.")

if __name__ == "__main__":
    main()