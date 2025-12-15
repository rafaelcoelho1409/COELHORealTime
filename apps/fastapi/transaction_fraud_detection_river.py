from river import (
    metrics
)
import pickle
import os
import signal
import sys
import pandas as pd
import mlflow
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_data,
    load_or_create_encoders,
)


MLFLOW_HOST = os.environ["MLFLOW_HOST"]


DATA_PATH = "data/transaction_fraud_detection.parquet"
MODEL_FOLDER = "models/transaction_fraud_detection"
ENCODERS_PATH = "encoders/river/transaction_fraud_detection.pkl"
PROJECT_NAME = "Transaction Fraud Detection"

os.makedirs(MODEL_FOLDER, exist_ok = True)
os.makedirs("encoders/river", exist_ok = True)
os.makedirs("data", exist_ok = True)

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    print(f"\nReceived {signal_name}, initiating graceful shutdown...")
    _shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def main():
    # Initialize model and metrics
    print(f"Connecting to MLflow at http://{MLFLOW_HOST}:5000")
    mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
    mlflow.set_experiment(PROJECT_NAME)
    print(f"MLflow experiment '{PROJECT_NAME}' set successfully")
    encoders = load_or_create_encoders(
        PROJECT_NAME,
        "river"
    )
    print("Encoders loaded")
    model = load_or_create_model(
        PROJECT_NAME,
        "ARFClassifier",
        MODEL_FOLDER
    )
    print(f"Model loaded: {model.__class__.__name__}")
    # Create consumer
    consumer = create_consumer(PROJECT_NAME)
    print("Consumer started. Waiting for transactions...")
    data_df = load_or_create_data(
        consumer,
        PROJECT_NAME)
    print(f"Initial data loaded with {len(data_df)} rows")
    binary_classification_metrics = [
        'Accuracy',
        'Precision',          # Typically for the positive class (Fraud)
        'Recall',             # Typically for the positive class (Fraud)
        'F1',                 # Typically for the positive class (Fraud)
        'GeometricMean',
        'ROCAUC',             # Requires probabilities
    ]
    binary_classification_metrics_dict = {
        x: getattr(metrics, x)() for x in binary_classification_metrics
    }
    BATCH_SIZE_OFFSET = 100
    print(f"Starting MLflow run with model: {model.__class__.__name__}")
    with mlflow.start_run(run_name = model.__class__.__name__):
        print("MLflow run started, entering consumer loop...")
        try:
            # Use while loop to handle consumer timeout and check for shutdown
            while not _shutdown_requested:
                for message in consumer:
                    # Check for graceful shutdown request
                    if _shutdown_requested:
                        print("Shutdown requested, breaking out of consumer loop...")
                        break
                    transaction = message.value
                    # Create a new DataFrame from the received data
                    new_row = pd.DataFrame([transaction])
                    # Append the new row to the existing DataFrame
                    data_df = pd.concat([data_df, new_row], ignore_index = True)
                    # Process the transaction
                    x = {
                        'transaction_id':        transaction['transaction_id'],
                        'user_id':               transaction['user_id'],
                        'timestamp':             transaction['timestamp'],
                        'amount':                transaction['amount'],
                        'currency':              transaction['currency'],
                        'merchant_id':           transaction['merchant_id'],
                        'product_category':      transaction['product_category'],
                        'transaction_type':      transaction['transaction_type'],
                        'payment_method':        transaction['payment_method'],
                        'location':              transaction['location'],
                        'ip_address':            transaction['ip_address'],
                        'device_info':           transaction['device_info'], # Nested structure for device details
                        'user_agent':            transaction['user_agent'],
                        'account_age_days':      transaction['account_age_days'],
                        'cvv_provided':          transaction['cvv_provided'], # Boolean flag
                        'billing_address_match': transaction['billing_address_match'], # Boolean flag
                    }
                    x, encoders = process_sample(
                        x,
                        encoders,
                        PROJECT_NAME)
                    y = transaction['is_fraud']
                    # Update the model
                    model.learn_one(x, y)
                    prediction = model.predict_one(x)
                    # Update metrics if provided
                    try:
                        for metric in binary_classification_metrics:
                            binary_classification_metrics_dict[metric].update(y, prediction)
                    except Exception as e:
                        print(f"Error updating metric {metric}: {str(e)}")
                    # Periodically log progress
                    if message.offset % BATCH_SIZE_OFFSET == 0:
                        print(f"Processed {message.offset} messages")
                        for metric in binary_classification_metrics:
                            try:
                                binary_classification_metrics_dict[metric].update(y, prediction)
                            except Exception as e:
                                print(f"Error updating metric {metric}: {str(e)}")
                            mlflow.log_metric(
                                metric,
                                binary_classification_metrics_dict[metric].get(),
                                step = message.offset)
                            #print(f"{metric}: {binary_classification_metrics_dict[metric].get():.2%}")
                        with open(ENCODERS_PATH, 'wb') as f:
                            pickle.dump(encoders, f)
                        mlflow.log_artifact(ENCODERS_PATH)
                    if message.offset % (BATCH_SIZE_OFFSET) == 0:
                        MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                        with open(MODEL_VERSION, 'wb') as f:
                            pickle.dump(model, f)
                        mlflow.log_artifact(MODEL_VERSION)
                        data_df.to_parquet(DATA_PATH)
                # Consumer timeout reached, loop continues if not shutdown
            print("Graceful shutdown completed.")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
            with open(MODEL_VERSION, 'wb') as f:
                pickle.dump(model, f)
            with open(ENCODERS_PATH, 'wb') as f:
                pickle.dump(encoders, f)
            data_df.to_parquet(DATA_PATH)
            consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()