"""
Transaction Fraud Detection - Batch ML Training Script

Trains a CatBoostClassifier model for fraud detection using batch learning.
Reads data from Delta Lake on MinIO via DuckDB, trains the model,
logs to MLflow, and saves artifacts.

Uses DuckDB SQL for all preprocessing (JSON extraction, timestamp parsing).
No StandardScaler needed - CatBoost handles raw features efficiently.

Usage:
    python transaction_fraud_detection_sklearn.py
    python transaction_fraud_detection_sklearn.py --sample-frac 0.3
    python transaction_fraud_detection_sklearn.py --max-rows 100000

Environment variables:
    MLFLOW_HOST: MLflow server hostname (required)
"""
import pickle
import os
import signal
import sys
import time
import tempfile
import click
import mlflow
import mlflow.catboost
import requests
import pandas as pd
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from catboost import CatBoostClassifier
from utils.sklearn import (
    process_batch_data_duckdb,
    create_batch_model,
    TFD_CAT_FEATURE_INDICES,
    TFD_ALL_FEATURES,
)


MLFLOW_HOST = os.environ["MLFLOW_HOST"]
FASTAPI_STATUS_URL = "http://localhost:8001/api/v1/batch/training-status"

# =============================================================================
# GRACEFUL SHUTDOWN (Signal Handling - matches River pattern)
# =============================================================================
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


class ShutdownCallback:
    """CatBoost callback to check for shutdown requests during training."""
    def after_iteration(self, info):
        """Called after each iteration. Return True to stop training."""
        if _shutdown_requested:
            print("Shutdown requested, stopping CatBoost training early...")
            return True  # Stop training
        return False  # Continue training


def check_shutdown(stage: str = "unknown"):
    """Check if shutdown was requested and exit gracefully if so."""
    if _shutdown_requested:
        print(f"Shutdown requested during {stage}, exiting gracefully...")
        update_status(f"Training stopped during {stage}", progress=0, stage="stopped")
        sys.exit(0)


def update_status(message: str, progress: int = None, stage: str = None, metrics: dict = None, total_rows: int = None):
    """Post training status update to sklearn service."""
    try:
        payload = {"message": message}
        if progress is not None:
            payload["progress"] = progress
        if stage is not None:
            payload["stage"] = stage
        if metrics is not None:
            payload["metrics"] = metrics
        if total_rows is not None:
            payload["total_rows"] = total_rows
        requests.post(FASTAPI_STATUS_URL, json=payload, timeout=2)
    except Exception:
        pass  # Don't fail training if status update fails


PROJECT_NAME = "Transaction Fraud Detection"
MODEL_NAME = "CatBoostClassifier"
ENCODER_ARTIFACT_NAME = "sklearn_encoders.pkl"


@click.command()
@click.option(
    "--sample-frac",
    type=float,
    default=None,
    help="Fraction of data to sample (0.0-1.0). E.g., 0.3 = 30% of data.",
)
@click.option(
    "--max-rows",
    type=int,
    default=None,
    help="Maximum number of rows to load from Delta Lake.",
)
def main(sample_frac: float | None, max_rows: int | None):
    """Batch ML Training for Transaction Fraud Detection.

    Trains CatBoostClassifier on Delta Lake data and logs to MLflow.
    Uses DuckDB SQL for all preprocessing (no StandardScaler needed).
    """
    start_time = time.time()
    print(f"Starting batch ML training for {PROJECT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Preprocessing: DuckDB SQL")
    print(f"MLflow host: {MLFLOW_HOST}")
    if sample_frac:
        print(f"Data sampling: {sample_frac * 100}%")
    if max_rows:
        print(f"Max rows: {max_rows}")
    update_status("Initializing training...", progress=5, stage="init")
    try:
        # Configure MLflow
        print(f"Connecting to MLflow at http://{MLFLOW_HOST}:5000")
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
        mlflow.set_experiment(PROJECT_NAME)
        print(f"MLflow experiment '{PROJECT_NAME}' set successfully")
        update_status("Loading data from Delta Lake...", progress=10, stage="loading_data")
        # Load and process data using DuckDB SQL
        load_start = time.time()
        print("\n=== Loading and Preprocessing Data ===")
        print("All transformations done in single DuckDB SQL query...")
        X_train, X_test, y_train, y_test, preprocessor_dict = process_batch_data_duckdb(
            PROJECT_NAME,
            sample_frac=sample_frac,
            max_rows=max_rows,
        )
        cat_feature_indices = TFD_CAT_FEATURE_INDICES
        load_time = time.time() - load_start
        print(f"\nData loading/preprocessing completed in {load_time:.2f} seconds")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        total_rows = len(X_train) + len(X_test)
        update_status(
            f"Data loaded: {len(X_train):,} train, {len(X_test):,} test samples",
            progress=25,
            stage="data_loaded",
            total_rows=total_rows
        )
        # Check for shutdown after data loading
        check_shutdown("data_loading")
        # Create model
        print("\nCreating model...")
        model = create_batch_model(PROJECT_NAME, y_train = y_train)
        print(f"Categorical feature indices: {cat_feature_indices}")
        # Check for shutdown after model creation
        check_shutdown("model_creation")
        update_status("Training CatBoost model...", progress=30, stage="training", total_rows=total_rows)
        # Train model with early stopping, native categorical handling, and shutdown callback
        print("Training model...")
        model.fit(
            X_train, y_train,
            eval_set = (X_test, y_test),
            cat_features = cat_feature_indices,  # CatBoost handles categoricals natively
            early_stopping_rounds = 50,
            use_best_model = True,
            verbose = True,  # Show all iterations
            callbacks = [ShutdownCallback()],  # Check for shutdown during training
        )
        print("Model training complete.")
        # Check for shutdown after training (in case callback didn't trigger exit)
        check_shutdown("training")
        update_status("Evaluating model performance...", progress=70, stage="evaluating", total_rows=total_rows)
        # Evaluate model
        print("\nEvaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # =============================================================================
        # METRICS - SCIKIT-LEARN
        # =============================================================================
        # -----------------------------------------------------------------------------
        # PRIMARY METRICS - Class-based metrics for fraud detection
        # These use y_pred (predicted labels), not probabilities
        # -----------------------------------------------------------------------------
        primary_metric_functions = {
            "recall_score": metrics.recall_score,
            "precision_score": metrics.precision_score,
            "f1_score": metrics.f1_score,
            "fbeta_score": metrics.fbeta_score,
        }
        primary_metric_args = {
            "recall_score": {
                "pos_label": 1,
                "average": "binary",
                "zero_division": 0.0,
            },
            "precision_score": {
                "pos_label": 1,
                "average": "binary",
                "zero_division": 0.0,
            },
            "f1_score": {
                "pos_label": 1,
                "average": "binary",
                "zero_division": 0.0,
            },
            "fbeta_score": {
                "beta": 2.0,
                "pos_label": 1,
                "average": "binary",
                "zero_division": 0.0,
            },
        }
        # -----------------------------------------------------------------------------
        # SECONDARY METRICS - Good for monitoring and additional insights
        # These provide complementary information but shouldn't drive model selection
        # -----------------------------------------------------------------------------
        secondary_metric_functions = {
            "accuracy_score": metrics.accuracy_score,
            "balanced_accuracy_score": metrics.balanced_accuracy_score,
            "matthews_corrcoef": metrics.matthews_corrcoef,
            "cohen_kappa_score": metrics.cohen_kappa_score,
            "jaccard_score": metrics.jaccard_score,
        }
        secondary_metric_args = {
            "accuracy_score": {
                "normalize": True,
            },
            "balanced_accuracy_score": {
                "adjusted": False,
            },
            "matthews_corrcoef": {},
            "cohen_kappa_score": {
                "weights": None,
            },
            "jaccard_score": {
                "pos_label": 1,
                "average": "binary",
                "zero_division": 0.0,
            },
        }
        # -----------------------------------------------------------------------------
        # PROBABILISTIC METRICS - Use y_pred_proba (probability scores)
        # These measure ranking ability and probability calibration
        # -----------------------------------------------------------------------------
        probabilistic_metric_functions = {
            "roc_auc_score": metrics.roc_auc_score,
            "average_precision_score": metrics.average_precision_score,
            "log_loss": metrics.log_loss,
            "brier_score_loss": metrics.brier_score_loss,
            "d2_log_loss_score": metrics.d2_log_loss_score,
            "d2_brier_score": metrics.d2_brier_score,
        }
        probabilistic_metric_args = {
            "roc_auc_score": {},
            "average_precision_score": {
                "pos_label": 1,
            },
            "log_loss": {
                "normalize": True,
            },
            "brier_score_loss": {
                "pos_label": 1,
            },
            "d2_log_loss_score": {},
            "d2_brier_score": {
                "pos_label": 1,
            },
        }
        # -----------------------------------------------------------------------------
        # ANALYSIS/REPORTING METRICS - Handled by YellowBrick visualizers on-demand
        # Available via /yellowbrick_metric endpoint:
        #   - ConfusionMatrix, ClassificationReport, ROCAUC, PrecisionRecallCurve
        #   - DiscriminationThreshold (threshold tuning)
        # No need to store as artifacts - generated dynamically when requested
        # -----------------------------------------------------------------------------
        # COMPUTE ALL METRICS
        metrics_to_log = {}
        for name, func in primary_metric_functions.items():
            metrics_to_log[name] = float(func(y_test, y_pred, **primary_metric_args[name]))
        for name, func in secondary_metric_functions.items():
            metrics_to_log[name] = float(func(y_test, y_pred, **secondary_metric_args[name]))
        for name, func in probabilistic_metric_functions.items():
            metrics_to_log[name] = float(func(y_test, y_pred_proba, **probabilistic_metric_args[name]))
        # Add GeometricMean from imblearn (not in sklearn.metrics)
        metrics_to_log["geometric_mean_score"] = float(geometric_mean_score(y_test, y_pred))
        print("\nMetrics:")
        for name, value in metrics_to_log.items():
            print(f"  {name}: {value:.4f}")
        # Check for shutdown after evaluation (before MLflow logging)
        check_shutdown("evaluation")
        # Send metrics preview to status
        update_status(
            "Logging to MLflow...",
            progress=85,
            stage="logging_mlflow",
            metrics={
                "recall": metrics_to_log["recall_score"],
                "precision": metrics_to_log["precision_score"],
                "f1": metrics_to_log["f1_score"],
                "roc_auc": metrics_to_log["roc_auc_score"],
            },
            total_rows=total_rows
        )
        # Log to MLflow
        print("\nLogging to MLflow...")
        with mlflow.start_run(run_name=MODEL_NAME):
            # Log tags
            mlflow.set_tag("training_mode", "batch")
            mlflow.set_tag("preprocessing", "DuckDB SQL")
            # Log data parameters
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("preprocessing_method", "DuckDB SQL")
            if sample_frac:
                mlflow.log_param("sample_frac", sample_frac)
            if max_rows:
                mlflow.log_param("max_rows", max_rows)
            # Log all CatBoost model parameters
            all_params = model.get_all_params()
            print(f"Logging {len(all_params)} CatBoost parameters...")
            for param_name, param_value in all_params.items():
                # Skip complex objects that MLflow can't serialize
                if isinstance(param_value, (str, int, float, bool)):
                    mlflow.log_param(param_name, param_value)
                elif isinstance(param_value, list) and len(param_value) <= 10:
                    # Log short lists as string
                    mlflow.log_param(param_name, str(param_value))
                elif isinstance(param_value, dict) and len(param_value) <= 5:
                    # Log small dicts as string
                    mlflow.log_param(param_name, str(param_value))
            # Log timing metrics
            mlflow.log_metric("preprocessing_time_seconds", load_time)
            # Log metrics
            for metric_name, metric_value in metrics_to_log.items():
                mlflow.log_metric(metric_name, metric_value)
            # Log model, encoders, and training data as artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save model and encoders
                model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(preprocessor_dict, f)
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(encoder_path)
                # Save training data for YellowBrick visualization reproducibility
                # Uses snappy compression (fast decompression, industry default)
                print("Saving training data artifacts...")
                X_train.to_parquet(os.path.join(tmpdir, "X_train.parquet"), compression="snappy")
                X_test.to_parquet(os.path.join(tmpdir, "X_test.parquet"), compression="snappy")
                y_train.to_frame(name="target").to_parquet(os.path.join(tmpdir, "y_train.parquet"), compression="snappy")
                y_test.to_frame(name="target").to_parquet(os.path.join(tmpdir, "y_test.parquet"), compression="snappy")
                mlflow.log_artifact(os.path.join(tmpdir, "X_train.parquet"), artifact_path="training_data")
                mlflow.log_artifact(os.path.join(tmpdir, "X_test.parquet"), artifact_path="training_data")
                mlflow.log_artifact(os.path.join(tmpdir, "y_train.parquet"), artifact_path="training_data")
                mlflow.log_artifact(os.path.join(tmpdir, "y_test.parquet"), artifact_path="training_data")
                print(f"  Saved: X_train={X_train.shape}, X_test={X_test.shape}")
            # Log model using MLflow's catboost flavor
            mlflow.catboost.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run ID: {run_id}")
        training_time = time.time() - start_time
        print(f"\nTraining completed successfully in {training_time:.2f} seconds")
        update_status(
            f"Training complete! Time: {training_time:.1f}s",
            progress=100,
            stage="complete",
            metrics={
                "recall": metrics_to_log["recall_score"],
                "precision": metrics_to_log["precision_score"],
                "f1": metrics_to_log["f1_score"],
                "roc_auc": metrics_to_log["roc_auc_score"],
                "fbeta": metrics_to_log["fbeta_score"],
            },
            total_rows=total_rows
        )
    except Exception as e:
        update_status(f"Training failed: {str(e)}", progress=0, stage="error")
        print(f"Error during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
