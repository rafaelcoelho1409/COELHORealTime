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
from utils.incremental import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_encoders,
    load_kafka_offset_from_mlflow,
    get_best_mlflow_run,
    MLFLOW_MODEL_NAMES,
    ENCODER_ARTIFACT_NAMES,
    KAFKA_OFFSET_ARTIFACT,
    # Redis live model cache (real-time predictions during training)
    save_live_model_to_redis,
    clear_live_model_from_redis,
    REDIS_CACHE_INTERVAL,
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
    """River ML Training for Transaction Fraud Detection.

    Continues training from the best model in MLflow.
    """
    # Initialize model and metrics
    print(f"Connecting to MLflow at http://{MLFLOW_HOST}:5000")
    mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
    mlflow.set_experiment(PROJECT_NAME)
    print(f"MLflow experiment '{PROJECT_NAME}' set successfully")
    # Load or create model and encoders from MLflow
    print("Loading model from MLflow (best historical model)")
    encoders = load_or_create_encoders(PROJECT_NAME, "river")
    model = load_or_create_model(PROJECT_NAME, MODEL_NAME)
    print("Encoders loaded")
    print(f"Model loaded: {model.__class__.__name__}")
    # Load last processed Kafka offset from MLflow
    last_offset = load_kafka_offset_from_mlflow(PROJECT_NAME)
    # Create consumer with starting offset
    consumer = create_consumer(PROJECT_NAME, start_offset=last_offset)
    print("Consumer started. Waiting for transactions...")
    # Track current offset for persistence
    current_offset = last_offset if last_offset is not None else -1
    # =========================================================================
    # METRICS CONFIGURATION (Research-based optimal args for TFD)
    # =========================================================================
    # Sources:
    #   - River ML Documentation: https://riverml.xyz/dev/api/metrics/
    #   - Fraud Detection Best Practices: https://www.cesarsotovalero.net/blog/
    #     evaluation-metrics-for-real-time-financial-fraud-detection-ml-models.html
    #   - F-Beta Score Guide: https://machinelearningmastery.com/fbeta-measure-for-ml
    # =========================================================================
    # Shared confusion matrix for efficiency (metrics share TP/TN/FP/FN counts)
    shared_cm = metrics.ConfusionMatrix()
    # -----------------------------------------------------------------------------
    # CLASS-BASED METRICS (use predict_one - class labels)
    # -----------------------------------------------------------------------------
    class_metric_classes = {
        # PRIMARY METRICS (most important for fraud detection)
        "Recall": metrics.Recall,       # TP / (TP + FN) - catch rate of fraud
        "Precision": metrics.Precision, # TP / (TP + FP) - accuracy of fraud alerts
        "F1": metrics.F1,               # Harmonic mean (beta=1, balanced)
        "FBeta": metrics.FBeta,         # Weighted harmonic mean (configurable beta)
        # SECONDARY METRICS (additional insights)
        "Accuracy": metrics.Accuracy,           # Overall correct predictions
        "BalancedAccuracy": metrics.BalancedAccuracy,  # Mean recall per class
        "MCC": metrics.MCC,             # Matthews Correlation Coefficient
        "GeometricMean": metrics.GeometricMean, # sqrt(TPR * TNR) - imbalance robust
        "CohenKappa": metrics.CohenKappa,       # Agreement beyond chance
        "Jaccard": metrics.Jaccard,     # TP / (TP + FP + FN) - IoU for positive class
    }
    class_metric_args = {
        # PRIMARY METRICS
        "Recall": {"cm": shared_cm, "pos_val": 1},
        "Precision": {"cm": shared_cm, "pos_val": 1},
        "F1": {"cm": shared_cm, "pos_val": 1},
        # FBeta with beta=2.0: Industry standard for fraud detection
        # beta=2.0 weights Recall 2x more than Precision (prioritize catching fraud)
        # Alternative: beta=0.5 if customer experience (fewer false positives) is priority
        # Reference: https://www.analyticsvidhya.com/blog/2024/12/f-beta-score/
        "FBeta": {"beta": 2.0, "cm": shared_cm, "pos_val": 1},
        # SECONDARY METRICS
        "Accuracy": {"cm": shared_cm},
        "BalancedAccuracy": {"cm": shared_cm},
        # MCC: One of best metrics for imbalanced data per research
        # Reference: https://link.springer.com/article/10.1186/s12864-019-6413-7
        "MCC": {"cm": shared_cm, "pos_val": 1},
        "GeometricMean": {"cm": shared_cm},
        "CohenKappa": {"cm": shared_cm},
        "Jaccard": {"cm": shared_cm, "pos_val": 1},
    }
    # -----------------------------------------------------------------------------
    # PROBABILITY-BASED METRICS (use predict_proba_one - probabilities)
    # -----------------------------------------------------------------------------
    proba_metric_classes = {
        "ROCAUC": metrics.ROCAUC,
        "RollingROCAUC": metrics.RollingROCAUC,
        "LogLoss": metrics.LogLoss,
    }
    proba_metric_args = {
        # ROCAUC: Approximation of true ROC AUC for streaming data
        # n_thresholds: Higher = more accurate but more memory/CPU
        # Default=10, River example uses 20, production recommended=50
        # Reference: https://riverml.xyz/dev/api/metrics/ROCAUC/
        "ROCAUC": {"n_thresholds": 50, "pos_val": 1},
        # RollingROCAUC: Windowed ROCAUC for concept drift detection
        # window_size: Number of recent samples to consider
        # Default=1000, but for rare fraud (~1-5%), need larger window
        # With window=5000 and 2% fraud rate, expect ~100 fraud samples
        # This provides stable AUC while detecting recent drift
        "RollingROCAUC": {"window_size": 5000, "pos_val": 1},
        # LogLoss: Measures probability calibration (lower = better)
        # 0.693 is random baseline, target < 0.1 for good calibration
        # No configurable parameters
        "LogLoss": {},
    }
    # -----------------------------------------------------------------------------
    # MATRIX/REPORT METRICS (no .get(), display separately)
    # -----------------------------------------------------------------------------
    report_metric_classes = {
        "ConfusionMatrix": metrics.ConfusionMatrix,
        "ClassificationReport": metrics.ClassificationReport,
    }
    report_metric_args = {
        # Separate ConfusionMatrix for report (not shared_cm)
        "ConfusionMatrix": {},
        # decimals=4: More precision for monitoring subtle changes
        # Uses shared_cm for efficiency with class_metrics
        "ClassificationReport": {"decimals": 4, "cm": shared_cm},
    }
    # =============================================================================
    # INSTANTIATE ALL METRICS
    # =============================================================================
    class_metrics = {
        name: class_metric_classes[name](**class_metric_args[name])
        for name in class_metric_classes
    }
    proba_metrics = {
        name: proba_metric_classes[name](**proba_metric_args[name])
        for name in proba_metric_classes
    }
    report_metrics = {
        name: report_metric_classes[name](**report_metric_args[name])
        for name in report_metric_classes
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
    # Use MODEL_NAME for run name (not model.__class__.__name__) to maintain
    # compatibility with MLflow queries that filter by run name.
    # The actual model is RandomUnderSampler wrapping ARFClassifier.
    print(f"Starting MLflow run with model: {MODEL_NAME} (wrapper: {model.__class__.__name__})")
    with mlflow.start_run(run_name = MODEL_NAME):
        # Log traceability tags for model lineage
        if best_run_id:
            mlflow.set_tag("training_mode", "continued")
            mlflow.set_tag("continued_from_run", best_run_id)
            # Log all baseline metrics as tags (class + proba metrics)
            all_metric_names = list(class_metrics.keys()) + list(proba_metrics.keys())
            for metric_name in all_metric_names:
                if metric_name in baseline_metrics:
                    mlflow.set_tag(f"baseline_{metric_name}", f"{baseline_metrics[metric_name]:.4f}")
            print(f"Traceability tags set: continued_from_run={best_run_id}")
        else:
            mlflow.set_tag("training_mode", "from_scratch")
            print("Starting training from scratch (no previous model in MLflow)")
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
                    prediction_proba = model.predict_proba_one(x)
                    proba_positive = prediction_proba.get(1, 0.0) if prediction_proba else 0.0
                    # Track current offset for persistence
                    current_offset = message.offset
                    # Update class-based metrics (use class prediction: 0 or 1)
                    for metric in class_metrics.values():
                        try:
                            metric.update(y, prediction)
                        except Exception as e:
                            print(f"Error updating metric {metric}: {str(e)}")
                    # Update probability-based metrics (use probability: 0.0 to 1.0)
                    for metric in proba_metrics.values():
                        try:
                            metric.update(y, proba_positive)
                        except Exception as e:
                            print(f"Error updating metric {metric}: {str(e)}")
                    # Update report metrics (use class prediction)
                    for metric in report_metrics.values():
                        try:
                            metric.update(y, prediction)
                        except Exception as e:
                            print(f"Error updating metric {metric}: {str(e)}")
                    # Log metrics to MLflow periodically (batched for efficiency)
                    if message.offset % METRICS_LOG_INTERVAL == 0:
                        print(f"Processed {message.offset} messages")
                        # =============================================================================
                        # COLLECT SCALAR METRICS FOR LOGGING
                        # =============================================================================
                        # Combine class + proba metrics (all have .get() method)
                        metrics_to_log = {}
                        for name, metric in class_metrics.items():
                            metrics_to_log[name] = metric.get()
                        for name, metric in proba_metrics.items():
                            metrics_to_log[name] = metric.get()
                        mlflow.log_metrics(metrics_to_log, step = message.offset)
                    # Save live model to Redis for real-time predictions
                    if message.offset % REDIS_CACHE_INTERVAL == 0:
                        save_live_model_to_redis(PROJECT_NAME, MODEL_NAME, model, encoders)
                    # Save artifacts to MLflow (using temp files)
                    if message.offset % ARTIFACT_SAVE_INTERVAL == 0 and message.offset > 0:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                            encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                            offset_path = os.path.join(tmpdir, KAFKA_OFFSET_ARTIFACT)
                            report_path = os.path.join(tmpdir, "report_metrics.pkl")
                            with open(model_path, 'wb') as f:
                                pickle.dump(model, f)
                            with open(encoder_path, 'wb') as f:
                                pickle.dump(encoders, f)
                            with open(offset_path, 'w') as f:
                                json.dump({"last_offset": current_offset}, f)
                            with open(report_path, 'wb') as f:
                                pickle.dump(report_metrics, f)
                            mlflow.log_artifact(model_path)
                            mlflow.log_artifact(encoder_path)
                            mlflow.log_artifact(offset_path)
                            mlflow.log_artifact(report_path)
                        print(f"Artifacts saved to MLflow at offset {message.offset}")
                # Consumer timeout reached, loop continues if not shutdown
            print("Graceful shutdown completed.")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            # Save final model, encoders, offset, and report metrics to MLflow on shutdown
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                offset_path = os.path.join(tmpdir, KAFKA_OFFSET_ARTIFACT)
                report_path = os.path.join(tmpdir, "report_metrics.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoders, f)
                with open(offset_path, 'w') as f:
                    json.dump({"last_offset": current_offset}, f)
                with open(report_path, 'wb') as f:
                    pickle.dump(report_metrics, f)
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(encoder_path)
                mlflow.log_artifact(offset_path)
                mlflow.log_artifact(report_path)
            print(f"Final artifacts saved to MLflow (offset={current_offset})")
            # Clear live model from Redis (training stopped)
            clear_live_model_from_redis(PROJECT_NAME, MODEL_NAME)
            if consumer is not None:
                consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()