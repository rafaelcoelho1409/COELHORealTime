"""
Transaction Fraud Detection - Batch ML Training Script

This script trains an XGBClassifier model for fraud detection using batch learning.
It consumes data from Kafka, trains the model, logs to MLflow, and saves artifacts.
Supports graceful shutdown via SIGTERM/SIGINT signals.
"""
import pickle
import os
import signal
import sys
import time
from datetime import datetime
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.metrics import geometric_mean_score
from functions import (
    create_consumer,
    load_or_create_data,
    process_batch_data,
    create_batch_model,
)


MLFLOW_HOST = os.environ.get("MLFLOW_HOST", "localhost")

PROJECT_NAME = "Transaction Fraud Detection"
MODEL_NAME = "XGBClassifier"
DATA_PATH = "data/transaction_fraud_detection.parquet"
MODEL_FOLDER = "models/transaction_fraud_detection"
ENCODERS_PATH = "encoders/sklearn/transaction_fraud_detection.pkl"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs("encoders/sklearn", exist_ok=True)
os.makedirs("data", exist_ok=True)

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
    """Main training function."""
    global _shutdown_requested

    start_time = time.time()
    exit_code = 0
    consumer = None

    print(f"Starting batch ML training for {PROJECT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"MLflow host: {MLFLOW_HOST}")

    try:
        # Configure MLflow
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
        mlflow.set_experiment(PROJECT_NAME)

        # Check for shutdown before loading data
        if _shutdown_requested:
            print("Shutdown requested before data loading.")
            return 1

        # Load data
        print("Loading data...")
        consumer = create_consumer(PROJECT_NAME)
        data_df = load_or_create_data(consumer, PROJECT_NAME)

        if consumer is not None:
            consumer.close()
            consumer = None
            print("Kafka consumer closed.")

        if data_df.empty:
            print("ERROR: No data available for training.", file=sys.stderr)
            return 1

        print(f"Loaded {len(data_df)} samples")

        # Check for shutdown before processing
        if _shutdown_requested:
            print("Shutdown requested before data processing.")
            return 1

        # Process data
        print("Processing data...")
        X_train, X_test, y_train, y_test = process_batch_data(data_df, PROJECT_NAME)
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Check for shutdown before training
        if _shutdown_requested:
            print("Shutdown requested before model training.")
            return 1

        # Create model
        print("Creating model...")
        model = create_batch_model(PROJECT_NAME, y_train=y_train)

        # Train model
        print("Training model...")
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Check for shutdown before evaluation
        if _shutdown_requested:
            print("Shutdown requested before evaluation. Saving model anyway...")

        # Make predictions
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1": float(f1_score(y_test, y_pred, zero_division=0)),
            "ROCAUC": float(roc_auc_score(y_test, y_pred_proba)),
            "GeometricMean": float(geometric_mean_score(y_test, y_pred)),
        }

        print("Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # Log to MLflow
        print("Logging to MLflow...")
        with mlflow.start_run(run_name=MODEL_NAME):
            # Log parameters
            mlflow.log_param("model_type", MODEL_NAME)
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run ID: {run_id}")

        # Save model locally
        model_path = f"{MODEL_FOLDER}/{MODEL_NAME}.pkl"
        print(f"Saving model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        training_time = time.time() - start_time
        print(f"\nTraining completed successfully in {training_time:.2f} seconds")

    except Exception as e:
        print(f"ERROR during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        if consumer is not None:
            consumer.close()
            print("Kafka consumer closed.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
