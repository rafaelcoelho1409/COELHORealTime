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
    # Batch sizes for different operations (tuned for performance)
    METRICS_LOG_INTERVAL = 100      # Log metrics to MLflow every N messages
    ARTIFACT_SAVE_INTERVAL = 1000   # Save model/encoders to S3 every N messages
    DATA_SAVE_INTERVAL = 5000       # Save parquet data every N messages

    # Buffer for efficient DataFrame building (avoid pd.concat on every message)
    pending_rows = []
    BUFFER_FLUSH_SIZE = 500  # Flush buffer to DataFrame every N rows

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
                    # Buffer rows instead of concat on every message (major performance fix)
                    pending_rows.append(transaction)
                    # Flush buffer periodically to avoid memory buildup
                    if len(pending_rows) >= BUFFER_FLUSH_SIZE:
                        data_df = pd.concat([data_df, pd.DataFrame(pending_rows)], ignore_index=True)
                        pending_rows = []
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
                        'device_info':           transaction['device_info'],
                        'user_agent':            transaction['user_agent'],
                        'account_age_days':      transaction['account_age_days'],
                        'cvv_provided':          transaction['cvv_provided'],
                        'billing_address_match': transaction['billing_address_match'],
                    }
                    x, encoders = process_sample(
                        x,
                        encoders,
                        PROJECT_NAME)
                    y = transaction['is_fraud']
                    # Update the model
                    model.learn_one(x, y)
                    prediction = model.predict_one(x)
                    # Update metrics (once per message, not twice)
                    for metric in binary_classification_metrics:
                        try:
                            binary_classification_metrics_dict[metric].update(y, prediction)
                        except Exception as e:
                            print(f"Error updating metric {metric}: {str(e)}")
                    # Log metrics to MLflow periodically (batched for efficiency)
                    if message.offset % METRICS_LOG_INTERVAL == 0:
                        print(f"Processed {message.offset} messages")
                        # Batch log all metrics in one call (reduces HTTP overhead)
                        metrics_to_log = {
                            metric: binary_classification_metrics_dict[metric].get()
                            for metric in binary_classification_metrics
                        }
                        mlflow.log_metrics(metrics_to_log, step=message.offset)
                    # Save artifacts less frequently (S3 uploads are expensive)
                    if message.offset % ARTIFACT_SAVE_INTERVAL == 0 and message.offset > 0:
                        MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                        with open(MODEL_VERSION, 'wb') as f:
                            pickle.dump(model, f)
                        with open(ENCODERS_PATH, 'wb') as f:
                            pickle.dump(encoders, f)
                        mlflow.log_artifact(MODEL_VERSION)
                        mlflow.log_artifact(ENCODERS_PATH)
                        print(f"Artifacts saved at offset {message.offset}")
                    # Save data even less frequently (parquet write is heavy)
                    if message.offset % DATA_SAVE_INTERVAL == 0 and message.offset > 0:
                        # Flush pending rows before saving
                        if pending_rows:
                            data_df = pd.concat([data_df, pd.DataFrame(pending_rows)], ignore_index=True)
                            pending_rows = []
                        data_df.to_parquet(DATA_PATH)
                        print(f"Data saved at offset {message.offset}")
                # Consumer timeout reached, loop continues if not shutdown
            print("Graceful shutdown completed.")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            # Flush any remaining buffered rows
            if pending_rows:
                data_df = pd.concat([data_df, pd.DataFrame(pending_rows)], ignore_index=True)
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