"""
Transaction Fraud Detection - Batch ML Training Script

Trains a CatBoostClassifier model for fraud detection using batch learning.
Reads data from Delta Lake on MinIO via DuckDB, trains the model,
logs to MLflow, and saves artifacts.

Two preprocessing approaches available:
- DuckDB SQL (default): All transformations in SQL, no StandardScaler (faster, less memory)
- Pandas/Sklearn: Traditional approach with pandas transforms and StandardScaler

Usage:
    # DuckDB SQL approach (recommended)
    python transaction_fraud_detection_sklearn.py

    # With sampling
    python transaction_fraud_detection_sklearn.py --sample-frac 0.3

    # Traditional pandas approach (for comparison)
    python transaction_fraud_detection_sklearn.py --use-pandas

Environment variables:
    MLFLOW_HOST: MLflow server hostname (required)
"""
import pickle
import os
import sys
import time
import tempfile
import click
import mlflow
import mlflow.catboost
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.metrics import geometric_mean_score
from functions import (
    load_or_create_data,
    process_batch_data,
    process_batch_data_duckdb,
    create_batch_model,
    TFD_CAT_FEATURE_INDICES,
)


MLFLOW_HOST = os.environ["MLFLOW_HOST"]
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
@click.option(
    "--use-pandas",
    is_flag=True,
    default=False,
    help="Use traditional pandas/sklearn preprocessing instead of DuckDB SQL.",
)
def main(sample_frac: float | None, max_rows: int | None, use_pandas: bool):
    """Batch ML Training for Transaction Fraud Detection.

    Trains CatBoostClassifier on Delta Lake data and logs to MLflow.
    """
    start_time = time.time()
    preprocessing_method = "pandas/sklearn" if use_pandas else "DuckDB SQL"
    print(f"Starting batch ML training for {PROJECT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Preprocessing: {preprocessing_method}")
    print(f"MLflow host: {MLFLOW_HOST}")
    if sample_frac:
        print(f"Data sampling: {sample_frac * 100}%")
    if max_rows:
        print(f"Max rows: {max_rows}")
    try:
        # Configure MLflow
        print(f"Connecting to MLflow at http://{MLFLOW_HOST}:5000")
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
        mlflow.set_experiment(PROJECT_NAME)
        print(f"MLflow experiment '{PROJECT_NAME}' set successfully")

        # Load and process data
        load_start = time.time()

        if use_pandas:
            # Traditional pandas/sklearn approach
            print("\n=== Using Pandas/Sklearn Preprocessing ===")
            print("Loading data...")
            data_df = load_or_create_data(
                PROJECT_NAME,
                sample_frac=sample_frac,
                max_rows=max_rows,
            )
            if data_df.empty:
                print("ERROR: No data available for training.", file=sys.stderr)
                sys.exit(1)
            print(f"Loaded {len(data_df)} samples")
            print("Processing data (JSON extract, timestamp extract, StandardScaler)...")
            X_train, X_test, y_train, y_test, preprocessor_dict = process_batch_data(
                data_df, PROJECT_NAME
            )
            cat_feature_indices = preprocessor_dict.get("cat_feature_indices", [])
        else:
            # DuckDB SQL approach (recommended)
            print("\n=== Using DuckDB SQL Preprocessing ===")
            print("Loading and preprocessing in single SQL query...")
            X_train, X_test, y_train, y_test, preprocessor_dict = process_batch_data_duckdb(
                PROJECT_NAME,
                sample_frac=sample_frac,
                max_rows=max_rows,
            )
            # Use predefined indices (no StandardScaler needed for CatBoost)
            cat_feature_indices = TFD_CAT_FEATURE_INDICES

        load_time = time.time() - load_start
        print(f"\nData loading/preprocessing completed in {load_time:.2f} seconds")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Create model
        print("\nCreating model...")
        model = create_batch_model(PROJECT_NAME, y_train=y_train)
        print(f"Categorical feature indices: {cat_feature_indices}")
        # Train model with early stopping and native categorical handling
        print("Training model...")
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            cat_features=cat_feature_indices,  # CatBoost handles categoricals natively
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=True,  # Show all iterations
        )
        print("Model training complete.")
        # Evaluate model
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
        print("\nLogging to MLflow...")
        with mlflow.start_run(run_name=MODEL_NAME):
            # Log tags
            mlflow.set_tag("training_mode", "batch")
            mlflow.set_tag("preprocessing", preprocessing_method)
            # Log parameters
            mlflow.log_param("model_type", MODEL_NAME)
            mlflow.log_param("preprocessing_method", preprocessing_method)
            mlflow.log_param("iterations", model.get_param("iterations"))
            mlflow.log_param("learning_rate", model.get_param("learning_rate"))
            mlflow.log_param("depth", model.get_param("depth"))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            if sample_frac:
                mlflow.log_param("sample_frac", sample_frac)
            if max_rows:
                mlflow.log_param("max_rows", max_rows)
            # Log timing metrics
            mlflow.log_metric("preprocessing_time_seconds", load_time)
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            # Log model and encoders as artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(preprocessor_dict, f)
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(encoder_path)
            # Log model using MLflow's catboost flavor
            mlflow.catboost.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run ID: {run_id}")
        training_time = time.time() - start_time
        print(f"\nTraining completed successfully in {training_time:.2f} seconds")
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
