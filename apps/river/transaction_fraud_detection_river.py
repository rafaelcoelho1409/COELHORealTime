from river import (
    metrics
)
import pickle
import json
import os
import signal
import sys
import tempfile
import mlflow
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_encoders,
    load_kafka_offset_from_mlflow,
    get_best_mlflow_run,
    MLFLOW_MODEL_NAMES,
    ENCODER_ARTIFACT_NAMES,
    KAFKA_OFFSET_ARTIFACT,
)


MLFLOW_HOST = os.environ["MLFLOW_HOST"]
PROJECT_NAME = "Transaction Fraud Detection"
MODEL_NAME = MLFLOW_MODEL_NAMES[PROJECT_NAME]
ENCODER_ARTIFACT_NAME = ENCODER_ARTIFACT_NAMES[PROJECT_NAME]

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
    encoders = load_or_create_encoders(PROJECT_NAME, "river")
    print("Encoders loaded")
    model = load_or_create_model(PROJECT_NAME, MODEL_NAME)
    print(f"Model loaded: {model.__class__.__name__}")
    # Load last processed Kafka offset from MLflow
    last_offset = load_kafka_offset_from_mlflow(PROJECT_NAME)
    # Create consumer with starting offset
    consumer = create_consumer(PROJECT_NAME, start_offset=last_offset)
    print("Consumer started. Waiting for transactions...")
    # Track current offset for persistence
    current_offset = last_offset if last_offset is not None else -1
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
    # Get traceability info from best run (if continuing from existing model)
    best_run_id = get_best_mlflow_run(PROJECT_NAME, MODEL_NAME)
    baseline_metrics = {}
    if best_run_id:
        try:
            best_run = mlflow.get_run(best_run_id)
            baseline_metrics = best_run.data.metrics
            print(f"Loaded baseline metrics from best run: {best_run_id}")
        except Exception as e:
            print(f"Could not get baseline metrics from best run: {e}")
    print(f"Starting MLflow run with model: {model.__class__.__name__}")
    with mlflow.start_run(run_name = model.__class__.__name__):
        # Log traceability tags for model lineage
        if best_run_id:
            mlflow.set_tag("continued_from_run", best_run_id)
            # Log all baseline metrics as tags
            for metric_name in binary_classification_metrics:
                if metric_name in baseline_metrics:
                    mlflow.set_tag(f"baseline_{metric_name}", f"{baseline_metrics[metric_name]:.4f}")
            print(f"Traceability tags set: continued_from_run={best_run_id}")
        else:
            mlflow.set_tag("training_mode", "from_scratch")
            print("Starting fresh training (no previous model)")
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
                    # Track current offset for persistence
                    current_offset = message.offset
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
                    # Save artifacts to MLflow (using temp files)
                    if message.offset % ARTIFACT_SAVE_INTERVAL == 0 and message.offset > 0:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                            encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                            offset_path = os.path.join(tmpdir, KAFKA_OFFSET_ARTIFACT)
                            with open(model_path, 'wb') as f:
                                pickle.dump(model, f)
                            with open(encoder_path, 'wb') as f:
                                pickle.dump(encoders, f)
                            with open(offset_path, 'w') as f:
                                json.dump({"last_offset": current_offset}, f)
                            mlflow.log_artifact(model_path)
                            mlflow.log_artifact(encoder_path)
                            mlflow.log_artifact(offset_path)
                        print(f"Artifacts saved to MLflow at offset {message.offset}")
                # Consumer timeout reached, loop continues if not shutdown
            print("Graceful shutdown completed.")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            # Save final model, encoders, and offset to MLflow on shutdown
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                offset_path = os.path.join(tmpdir, KAFKA_OFFSET_ARTIFACT)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoders, f)
                with open(offset_path, 'w') as f:
                    json.dump({"last_offset": current_offset}, f)
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(encoder_path)
                mlflow.log_artifact(offset_path)
            print(f"Final artifacts saved to MLflow (offset={current_offset})")
            if consumer is not None:
                consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()