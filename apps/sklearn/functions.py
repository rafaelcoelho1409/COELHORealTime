"""
Scikit-Learn Batch ML Helper Functions

Functions for batch machine learning training and YellowBrick visualizations.
Reads data from Delta Lake on MinIO (S3-compatible storage) via DuckDB.
Models and encoders are loaded from MLflow artifacts (with local fallback).
"""
import pickle
import os
import io
import base64
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)
from catboost import CatBoostClassifier
from yellowbrick import (
    classifier,
    features,
    target,
    model_selection,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import duckdb
import orjson


# MinIO (S3-compatible) configuration for Delta Lake
MINIO_HOST = os.environ.get("MINIO_HOST", "localhost")
MINIO_ENDPOINT = f"http://{MINIO_HOST}:9000"
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")

# MLflow configuration
MLFLOW_HOST = os.environ.get("MLFLOW_HOST", "localhost")
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:5000"

# MLflow model names for each project (batch ML)
MLFLOW_MODEL_NAMES = {
    "Transaction Fraud Detection": "CatBoostClassifier",
    "Estimated Time of Arrival": "CatBoostRegressor",
    "E-Commerce Customer Interactions": "KMeans",
}

# Encoder artifact names (saved to MLflow during training)
ENCODER_ARTIFACT_NAMES = {
    "Transaction Fraud Detection": "sklearn_encoders.pkl",
    "Estimated Time of Arrival": "sklearn_encoders.pkl",
    "E-Commerce Customer Interactions": "sklearn_encoders.pkl",
}

# Best metric criteria for each project (used to select best model from MLflow)
# Similar to River's BEST_METRIC_CRITERIA
BEST_METRIC_CRITERIA = {
    # TFD: Use fbeta_score (beta=2.0) - prioritizes recall for fraud detection
    "Transaction Fraud Detection": {"metric_name": "fbeta_score", "maximize": True},
    # ETA: Use MAE (lower is better) for regression
    "Estimated Time of Arrival": {"metric_name": "mae", "maximize": False},
    # ECCI: Use Silhouette score (higher is better) for clustering
    "E-Commerce Customer Interactions": {"metric_name": "silhouette", "maximize": True},
}

# Delta Lake paths (S3 paths for DuckDB delta_scan)
DELTA_PATHS = {
    "Transaction Fraud Detection": "s3://lakehouse/delta/transaction_fraud_detection",
    "Estimated Time of Arrival": "s3://lakehouse/delta/estimated_time_of_arrival",
    "E-Commerce Customer Interactions": "s3://lakehouse/delta/e_commerce_customer_interactions",
}

# Feature definitions for Transaction Fraud Detection (DuckDB SQL approach)
TFD_NUMERICAL_FEATURES = ["amount", "account_age_days", "cvv_provided", "billing_address_match"]
TFD_CATEGORICAL_FEATURES = [
    "currency", "merchant_id", "payment_method", "product_category",
    "transaction_type", "browser", "os",
    "year", "month", "day", "hour", "minute", "second",
]
TFD_ALL_FEATURES = TFD_NUMERICAL_FEATURES + TFD_CATEGORICAL_FEATURES
TFD_CAT_FEATURE_INDICES = list(range(len(TFD_NUMERICAL_FEATURES), len(TFD_ALL_FEATURES)))


# Persistent DuckDB connection for Delta Lake queries
# Configured once at module load, reused for all queries (fast)
_duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None

def _get_duckdb_connection(force_reconnect: bool = False) -> duckdb.DuckDBPyConnection:
    """Get or create a persistent DuckDB connection configured for MinIO.

    Args:
        force_reconnect: If True, recreate connection even if one exists.
                        Used for recovery after connection errors.

    Connection is kept alive and reused. If a query fails due to stale connection,
    caller should retry with force_reconnect=True.

    Optimizations:
    - Uses DuckDB's extension auto-loading (extensions cached to ~/.duckdb/extensions)
    - Configures settings for faster Delta Lake scans
    - Reuses connection across queries
    """
    global _duckdb_conn
    if _duckdb_conn is not None and not force_reconnect:
        return _duckdb_conn
    # Create new connection and configure it
    if force_reconnect:
        print("Recreating DuckDB connection...")
    else:
        print("Creating new DuckDB connection with Delta extension...")
    # Set environment variables for delta-rs/object_store
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
    os.environ["AWS_ENDPOINT_URL"] = f"http://{MINIO_HOST}:9000"
    os.environ["AWS_ALLOW_HTTP"] = "true"
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    # Create connection with optimized settings
    conn = duckdb.connect(config={
        'threads': os.cpu_count(),  # Use all available CPU cores
        'memory_limit': '4GB',       # Limit memory to avoid OOM
        'temp_directory': '/tmp/duckdb',  # Use temp dir for spilling
    })
    # Enable autoinstall/autoload for faster subsequent loads
    conn.execute("SET autoinstall_known_extensions = true;")
    conn.execute("SET autoload_known_extensions = true;")
    # Install delta extension (skips if already installed, cached to disk)
    conn.execute("INSTALL delta;")
    conn.execute("LOAD delta;")
    # Configure S3/MinIO credentials
    minio_host_port = f"{MINIO_HOST}:9000"
    conn.execute("DROP SECRET IF EXISTS minio_secret;")
    conn.execute(f"""
        CREATE SECRET minio_secret (
            TYPE S3,
            KEY_ID '{MINIO_ACCESS_KEY}',
            SECRET '{MINIO_SECRET_KEY}',
            ENDPOINT '{minio_host_port}',
            URL_STYLE 'path',
            USE_SSL false,
            REGION 'us-east-1'
        );
    """)
    # Optimize for large data scans
    conn.execute("SET enable_progress_bar = false;")  # Disable progress bar overhead
    conn.execute("SET preserve_insertion_order = false;")  # Faster inserts
    _duckdb_conn = conn
    print("DuckDB connection ready with Delta extension loaded")
    return _duckdb_conn


# Initialize connection at module load (optional - will be created on first query if not)
try:
    _get_duckdb_connection()
except Exception as e:
    print(f"Warning: Could not initialize DuckDB connection: {e}")


# =============================================================================
# MLflow Experiment Caching (like River)
# =============================================================================
_experiment_cache: Dict[str, tuple] = {}
_EXPERIMENT_CACHE_TTL = 300  # 5 minutes


def get_cached_experiment(project_name: str) -> Optional[Any]:
    """Get MLflow experiment with caching (5 minute TTL).

    Experiment names/IDs rarely change, so caching avoids repeated API calls.
    Returns the experiment object or None if not found.
    """
    import time as _time
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cache_entry = _experiment_cache.get(project_name)
    if cache_entry:
        timestamp, experiment = cache_entry
        if _time.time() - timestamp < _EXPERIMENT_CACHE_TTL:
            return experiment
    # Cache miss or expired - fetch from MLflow
    try:
        experiment = mlflow.get_experiment_by_name(project_name)
        _experiment_cache[project_name] = (_time.time(), experiment)
        return experiment
    except Exception as e:
        print(f"Error getting MLflow experiment for {project_name}: {e}")
        return None


# =============================================================================
# MLflow Model Functions
# =============================================================================
def _run_has_model_artifact(run_id: str, model_name: str) -> bool:
    """Check if a run has the model artifact."""
    try:
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        return f"{model_name}.pkl" in artifact_names or "model" in artifact_names
    except Exception:
        return False


def get_best_mlflow_run(project_name: str, model_name: str) -> Optional[str]:
    """Get the best MLflow run ID based on metrics for a project.

    Uses BEST_METRIC_CRITERIA to determine which metric to optimize:
    - For fraud detection (TFD): maximize fbeta_score (beta=2.0, prioritizes recall)
    - For regression (ETA): minimize MAE
    - For clustering (ECCI): maximize Silhouette score

    Only selects runs that have the model artifact saved.
    Returns None if no runs exist.
    """
    try:
        experiment = get_cached_experiment(project_name)
        if experiment is None:
            print(f"No MLflow experiment found for {project_name}")
            return None

        # Search for completed runs
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=50,
        )

        if runs_df.empty:
            print(f"No FINISHED runs found in MLflow for {project_name}")
            return None

        # Filter by model name
        filtered_runs = runs_df[runs_df["tags.mlflow.runName"] == model_name]
        if filtered_runs.empty:
            print(f"No runs found for model {model_name} in {project_name}")
            return None

        # Get metric criteria for this project
        criteria = BEST_METRIC_CRITERIA.get(project_name)
        if criteria is None:
            # No metric criteria - use latest run with artifacts
            print(f"No metric criteria for {project_name}, using latest run with artifacts")
            for _, row in filtered_runs.iterrows():
                if _run_has_model_artifact(row["run_id"], model_name):
                    return row["run_id"]
            return None

        metric_name = criteria["metric_name"]
        maximize = criteria["maximize"]
        metric_column = f"metrics.{metric_name}"

        # Check if metric column exists
        if metric_column not in filtered_runs.columns:
            print(f"Metric {metric_name} not found, using latest run with artifacts")
            for _, row in filtered_runs.iterrows():
                if _run_has_model_artifact(row["run_id"], model_name):
                    return row["run_id"]
            return None

        # Filter runs with the metric and sort
        runs_with_metric = filtered_runs[filtered_runs[metric_column].notna()]
        if runs_with_metric.empty:
            print(f"No runs with metric {metric_name}, using latest run with artifacts")
            for _, row in filtered_runs.iterrows():
                if _run_has_model_artifact(row["run_id"], model_name):
                    return row["run_id"]
            return None

        # Sort by metric (best first)
        ascending = not maximize
        sorted_runs = runs_with_metric.sort_values(by=metric_column, ascending=ascending)

        # Find the best run with model artifacts
        for _, row in sorted_runs.iterrows():
            run_id = row["run_id"]
            metric_value = row[metric_column]
            if _run_has_model_artifact(run_id, model_name):
                print(f"Best run for {project_name}/{model_name}: {run_id} "
                      f"({metric_name} = {metric_value:.4f}, maximize = {maximize})")
                return run_id

        print(f"No runs with model artifact found for {project_name}/{model_name}")
        return None

    except Exception as e:
        print(f"Error finding best MLflow run: {e}")
        return None


def get_all_mlflow_runs(project_name: str, model_name: str) -> list[dict]:
    """Get all MLflow runs for a project, ordered by metric criteria (best first).

    Each project (TFD, ETA, ECCI) has its own MLflow experiment.
    Returns runs ordered by BEST_METRIC_CRITERIA:
    - TFD: fbeta_score DESC (maximize)
    - ETA: MAE ASC (minimize)
    - ECCI: silhouette_score DESC (maximize)

    Returns:
        List of run info dicts with: run_id, run_name, start_time,
        metrics, params, total_rows, is_best (True for first/best run)
    """
    try:
        experiment = get_cached_experiment(project_name)
        if experiment is None:
            print(f"No MLflow experiment found for {project_name}")
            return []

        # Search for completed runs
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=100,
        )

        if runs_df.empty:
            return []

        # Filter by model name if the column exists
        if "tags.mlflow.runName" in runs_df.columns:
            filtered_runs = runs_df[runs_df["tags.mlflow.runName"] == model_name]
        else:
            filtered_runs = runs_df
        if filtered_runs.empty:
            return []

        # Get metric criteria for sorting (each project has different criteria)
        criteria = BEST_METRIC_CRITERIA.get(project_name)
        if criteria:
            metric_name = criteria["metric_name"]
            maximize = criteria["maximize"]
            metric_column = f"metrics.{metric_name}"

            if metric_column in filtered_runs.columns:
                # Sort by metric (best first based on maximize/minimize)
                ascending = not maximize
                filtered_runs = filtered_runs.sort_values(
                    by=metric_column,
                    ascending=ascending,
                    na_position='last'
                )

        # Filter only runs with model artifacts
        valid_runs = []
        for idx, row in filtered_runs.iterrows():
            if not _run_has_model_artifact(row["run_id"], model_name):
                continue

            # Extract metrics
            metrics = {}
            for col in row.index:
                if col.startswith("metrics."):
                    metric_key = col.replace("metrics.", "")
                    val = row[col]
                    if pd.notna(val):
                        metrics[metric_key] = round(float(val), 4)

            # Extract params
            params = {}
            for col in row.index:
                if col.startswith("params."):
                    param_key = col.replace("params.", "")
                    val = row[col]
                    if pd.notna(val):
                        params[param_key] = val

            # Calculate total rows
            train_samples = int(params.get("train_samples", 0) or 0)
            test_samples = int(params.get("test_samples", 0) or 0)
            total_rows = train_samples + test_samples

            valid_runs.append({
                "run_id": row["run_id"],
                "run_name": row.get("tags.mlflow.runName", model_name),
                "start_time": row["start_time"].isoformat() if pd.notna(row["start_time"]) else None,
                "end_time": row["end_time"].isoformat() if pd.notna(row.get("end_time")) else None,
                "metrics": metrics,
                "params": params,
                "total_rows": total_rows,
                "is_best": len(valid_runs) == 0,  # First valid run is best
            })

        print(f"Found {len(valid_runs)} valid runs for {project_name}/{model_name}")
        return valid_runs

    except Exception as e:
        print(f"Error getting all MLflow runs: {e}")
        return []


def load_model_from_mlflow(project_name: str, model_name: str, run_id: str = None):
    """Load model from MLflow run.

    Args:
        project_name: MLflow experiment name (e.g., "Transaction Fraud Detection")
        model_name: Model name tag (e.g., "CatBoostClassifier")
        run_id: Optional specific run ID. If None, uses get_best_mlflow_run().

    Returns:
        Loaded model, or None if not found.
    """
    try:
        # Use provided run_id or find the best one
        if run_id is None:
            run_id = get_best_mlflow_run(project_name, model_name)
        if run_id is None:
            return None

        # Try loading via MLflow's catboost flavor first (preferred)
        try:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.catboost.load_model(model_uri)
            print(f"Model loaded from MLflow (catboost flavor): {project_name}/{model_name}")
            return model
        except Exception:
            pass

        # Fallback: load from pickle artifact
        artifact_path = f"{model_name}.pkl"
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
        with open(local_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from MLflow (pickle): {project_name}/{model_name} (run_id={run_id})")
        return model

    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return None


def load_encoders_from_mlflow(project_name: str, run_id: str = None) -> Optional[Dict[str, Any]]:
    """Load encoders from MLflow run.

    Args:
        project_name: MLflow experiment name
        run_id: Optional specific run ID. If None, uses get_best_mlflow_run().

    Returns:
        Dict with preprocessor/encoders, or None if not found.
    """
    try:
        model_name = MLFLOW_MODEL_NAMES.get(project_name)
        if not model_name:
            print(f"Unknown project for encoder loading: {project_name}")
            return None

        # Use provided run_id or find the best one
        if run_id is None:
            run_id = get_best_mlflow_run(project_name, model_name)
        if run_id is None:
            return None

        # Download encoder artifact
        artifact_name = ENCODER_ARTIFACT_NAMES.get(project_name)
        if not artifact_name:
            print(f"No encoder artifact name for {project_name}")
            return None

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_name,
        )

        with open(local_path, 'rb') as f:
            encoders = pickle.load(f)

        print(f"Sklearn encoders loaded from best MLflow run: {project_name} (run_id={run_id})")
        return encoders

    except Exception as e:
        print(f"Error loading sklearn encoders from MLflow: {e}")
        return None


def load_or_create_sklearn_encoders(project_name: str) -> Dict[str, Any]:
    """Load encoders from MLflow artifacts.

    Tries to load from MLflow (best model's encoders).
    Returns default metadata if not available (no encoders needed for DuckDB approach).

    Returns:
        Dict with feature info for CatBoost.
    """
    # Try loading from MLflow first
    encoders = load_encoders_from_mlflow(project_name)
    if encoders is not None:
        return encoders

    # No encoders in MLflow - return default metadata (DuckDB approach)
    print(f"No sklearn encoders in MLflow for {project_name}, using DuckDB defaults.")
    return {
        "numerical_features": TFD_NUMERICAL_FEATURES,
        "categorical_features": TFD_CATEGORICAL_FEATURES,
        "cat_feature_indices": TFD_CAT_FEATURE_INDICES,
        "feature_names": TFD_ALL_FEATURES,
    }


def load_training_data_from_mlflow(
    project_name: str,
    run_id: str = None,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]]:
    """Load training data from MLflow run's artifacts.

    This ensures YellowBrick visualizations use the EXACT same data
    that was used to train the selected model, guaranteeing 100% reproducibility.

    Args:
        project_name: MLflow experiment name
        run_id: Optional specific run ID. If None, uses get_best_mlflow_run().

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names) or None if not found.
    """
    try:
        model_name = MLFLOW_MODEL_NAMES.get(project_name)
        if not model_name:
            print(f"Unknown project for training data loading: {project_name}")
            return None

        # Use provided run_id or find the best one
        if run_id is None:
            run_id = get_best_mlflow_run(project_name, model_name)
        if run_id is None:
            print(f"No run found for {project_name}")
            return None

        print(f"Loading training data from MLflow run: {run_id}")

        # Download training data artifacts
        X_train_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="training_data/X_train.parquet",
        )
        X_test_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="training_data/X_test.parquet",
        )
        y_train_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="training_data/y_train.parquet",
        )
        y_test_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="training_data/y_test.parquet",
        )

        # Load parquet files
        X_train = pd.read_parquet(X_train_path)
        X_test = pd.read_parquet(X_test_path)
        y_train = pd.read_parquet(y_train_path)["target"]
        y_test = pd.read_parquet(y_test_path)["target"]

        feature_names = X_train.columns.tolist()

        print(f"  Loaded: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_names

    except Exception as e:
        print(f"Error loading training data from MLflow: {e}")
        return None


# =============================================================================
# DuckDB SQL-Based Data Loading and Preprocessing
# =============================================================================
def load_data_duckdb(
    project_name: str,
    sample_frac: float | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load and preprocess data using pure DuckDB SQL.

    All transformations (JSON extraction, timestamp parsing) done in SQL.
    No pandas transformations needed - data is ready for CatBoost.

    Key insight: CatBoost is tree-based and doesn't require feature scaling.
    StandardScaler is unnecessary overhead for gradient boosting models.

    Args:
        project_name: Project name (e.g., "Transaction Fraud Detection")
        sample_frac: Optional fraction of data to sample (0.0-1.0)
        max_rows: Optional maximum number of rows to load

    Returns:
        X: Features DataFrame ready for CatBoost
        y: Target Series
        metadata: Dict with feature info for CatBoost (cat_feature_indices, etc.)
    """
    if project_name == "Transaction Fraud Detection":
        return _load_tfd_data_duckdb(sample_frac, max_rows)
    else:
        raise ValueError(f"Unsupported project for DuckDB loading: {project_name}")


def _load_tfd_data_duckdb(
    sample_frac: float | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load Transaction Fraud Detection data with pure DuckDB SQL.

    Single SQL query does:
    - JSON extraction (device_info → browser, os)
    - Timestamp extraction (→ year, month, day, hour, minute, second)
    - Label encoding via DENSE_RANK() - 1 (all-numeric output)
    - Column selection and ordering (matches TFD_ALL_FEATURES order)
    - Sampling (if requested)

    All categorical features are label-encoded as 0-indexed integers.
    This produces all-numeric data compatible with:
    - CatBoost (pass cat_features indices for native handling)
    - YellowBrick (requires numeric data for all visualizers)
    - sklearn tools
    """
    delta_path = DELTA_PATHS["Transaction Fraud Detection"]

    # Build the SQL query - all preprocessing in one pass
    # Column order MUST match TFD_ALL_FEATURES for correct cat_feature_indices
    # All categorical features are label-encoded using DENSE_RANK() - 1
    query = f"""
    SELECT
        -- Numerical features (no scaling needed for CatBoost)
        amount,
        account_age_days,
        CAST(cvv_provided AS INTEGER) AS cvv_provided,
        CAST(billing_address_match AS INTEGER) AS billing_address_match,

        -- Categorical features: Label encoded with DENSE_RANK() - 1
        -- Produces 0-indexed integers compatible with all ML tools
        DENSE_RANK() OVER (ORDER BY currency) - 1 AS currency,
        DENSE_RANK() OVER (ORDER BY merchant_id) - 1 AS merchant_id,
        DENSE_RANK() OVER (ORDER BY payment_method) - 1 AS payment_method,
        DENSE_RANK() OVER (ORDER BY product_category) - 1 AS product_category,
        DENSE_RANK() OVER (ORDER BY transaction_type) - 1 AS transaction_type,

        -- Categorical features from JSON: extracted + label encoded
        DENSE_RANK() OVER (ORDER BY json_extract_string(device_info, '$.browser')) - 1 AS browser,
        DENSE_RANK() OVER (ORDER BY json_extract_string(device_info, '$.os')) - 1 AS os,

        -- Timestamp components (already integers)
        CAST(date_part('year', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS year,
        CAST(date_part('month', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS month,
        CAST(date_part('day', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS day,
        CAST(date_part('hour', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS hour,
        CAST(date_part('minute', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS minute,
        CAST(date_part('second', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS second,

        -- Target
        is_fraud

    FROM delta_scan('{delta_path}')
    """

    # Add sampling clause (DuckDB's efficient sampling at scan time)
    if sample_frac is not None and 0 < sample_frac < 1:
        query += f" USING SAMPLE {sample_frac * 100}%"

    # Add limit clause
    if max_rows is not None:
        query += f" LIMIT {max_rows}"

    # Execute query
    try:
        print(f"Loading TFD data via DuckDB SQL (single-pass preprocessing)...")
        if sample_frac:
            print(f"  Sampling: {sample_frac * 100}%")
        if max_rows:
            print(f"  Max rows: {max_rows}")

        conn = _get_duckdb_connection()
        df = conn.execute(query).df()

        print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")

    except Exception as e:
        print(f"DuckDB query failed, attempting reconnect: {e}")
        conn = _get_duckdb_connection(force_reconnect=True)
        df = conn.execute(query).df()
        print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")

    # Split features and target
    y = df["is_fraud"]
    X = df.drop("is_fraud", axis=1)

    # All columns are now numeric (integers from DENSE_RANK or float64)
    # No dtype conversion needed - works with CatBoost and YellowBrick
    print(f"  All features numeric: {X.select_dtypes(include=['number']).shape[1]}/{X.shape[1]} columns")

    # Metadata for CatBoost
    metadata = {
        "numerical_features": TFD_NUMERICAL_FEATURES,
        "categorical_features": TFD_CATEGORICAL_FEATURES,
        "cat_feature_indices": TFD_CAT_FEATURE_INDICES,
        "feature_names": TFD_ALL_FEATURES,
    }

    print(f"  Features: {len(TFD_NUMERICAL_FEATURES)} numerical, {len(TFD_CATEGORICAL_FEATURES)} label-encoded")
    print(f"  Categorical indices for CatBoost: {TFD_CAT_FEATURE_INDICES}")

    return X, y, metadata


def process_batch_data_duckdb(
    project_name: str,
    sample_frac: float | None = None,
    max_rows: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Process batch data using DuckDB SQL (drop-in replacement for process_batch_data).

    Combines DuckDB SQL preprocessing with sklearn's stratified train/test split.
    No StandardScaler - CatBoost doesn't need feature scaling.

    Args:
        project_name: Project name
        sample_frac: Optional fraction of data to sample (0.0-1.0)
        max_rows: Optional maximum rows to load
        test_size: Fraction for test set (default 0.2)
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test, metadata
    """
    # Load preprocessed data from DuckDB
    X, y, metadata = load_data_duckdb(project_name, sample_frac, max_rows)

    # Stratified train/test split (keeps class balance)
    print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Free memory
    del X, y

    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Calculate class balance
    fraud_rate = y_train.sum() / len(y_train) * 100
    print(f"  Fraud rate in training set: {fraud_rate:.2f}%")

    return X_train, X_test, y_train, y_test, metadata


def process_sklearn_sample(x: dict, project_name: str) -> pd.DataFrame:
    """Process a single sample for CatBoost prediction.

    For real-time predictions, we process individual samples in Python
    using the same feature order as DuckDB batch training.

    Note: No StandardScaler needed since CatBoost (tree-based) doesn't require it.
    Feature order MUST match TFD_ALL_FEATURES for correct predictions.
    """
    if project_name == "Transaction Fraud Detection":
        # Extract device_info JSON
        device_info = x.get("device_info", "{}")
        if isinstance(device_info, str):
            device_info = orjson.loads(device_info)

        # Extract timestamp components
        timestamp = pd.to_datetime(x.get("timestamp"))

        # Build feature dict in correct order (matches TFD_ALL_FEATURES)
        features = {
            # Numerical features
            "amount": x.get("amount"),
            "account_age_days": x.get("account_age_days"),
            "cvv_provided": int(x.get("cvv_provided", 0)),
            "billing_address_match": int(x.get("billing_address_match", 0)),
            # Categorical direct (order matches TFD_CATEGORICAL_FEATURES)
            "currency": x.get("currency"),
            "merchant_id": x.get("merchant_id"),
            "payment_method": x.get("payment_method"),
            "product_category": x.get("product_category"),
            "transaction_type": x.get("transaction_type"),
            # Categorical from JSON
            "browser": device_info.get("browser"),
            "os": device_info.get("os"),
            # Timestamp components
            "year": timestamp.year,
            "month": timestamp.month,
            "day": timestamp.day,
            "hour": timestamp.hour,
            "minute": timestamp.minute,
            "second": timestamp.second,
        }

        df = pd.DataFrame([features])

        # Convert categoricals to category dtype
        for col in TFD_CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df
    else:
        raise ValueError(f"Unsupported project: {project_name}")


# =============================================================================
# Model Creation
# =============================================================================
def create_batch_model(project_name: str, **kwargs):
    if project_name == "Transaction Fraud Detection":
        # Calculate class imbalance ratio for reference
        y_train = kwargs.get("y_train")
        if y_train is not None:
            neg_samples = sum(y_train == 0)
            pos_samples = sum(y_train == 1)
            imbalance_ratio = neg_samples / pos_samples if pos_samples > 0 else 1
            print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1 (negative:positive)")
            print(f"Fraud rate: {pos_samples / len(y_train) * 100:.2f}%")
            print("Using auto_class_weights='Balanced' for imbalanced data")

        # Optimized CatBoost parameters for fraud detection (1M+ rows, ~1% fraud)
        # Based on: https://catboost.ai/docs/en/references/training-parameters/common
        # Research: CatBoost achieves F1=0.92, AUC=0.99 on fraud detection benchmarks
        model = CatBoostClassifier(
            # Core parameters
            iterations=1000,                # Max trees; early stopping finds optimal
            learning_rate=0.05,             # Good balance for 1M+ rows
            depth=6,                        # CatBoost default, good for most cases

            # Imbalanced data handling (critical for fraud detection)
            auto_class_weights='Balanced',  # Weights positive class by neg/pos ratio

            # Loss function & evaluation
            loss_function='Logloss',        # Binary cross-entropy
            eval_metric='AUC',              # Best for imbalanced binary classification

            # Regularization
            l2_leaf_reg=3,                  # L2 regularization (default=3)

            # Boosting type: 'Plain' for large datasets (1M+), 'Ordered' for <100K
            boosting_type='Plain',

            # Performance
            task_type='CPU',
            thread_count=-1,                # Use all CPU cores
            random_seed=42,

            # Output
            verbose=100,                    # Print every 100 iterations
            allow_writing_files=False,
        )
        return model
    raise ValueError(f"Unknown project: {project_name}")


# =============================================================================
# YellowBrick Classification Visualizers
# Reference: https://www.scikit-yb.org/en/latest/api/classifier/index.html
# =============================================================================

# Sklearn-compatible wrappers for CatBoost (YellowBrick compatibility)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from yellowbrick.classifier.class_prediction_error import ClassPredictionError as _ClassPredictionErrorBase
from yellowbrick.exceptions import YellowbrickValueError, ModelError

try:
    from sklearn.metrics._classification import _check_targets
except ImportError:
    from sklearn.metrics.classification import _check_targets


class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    """Wraps pre-fitted CatBoost model for YellowBrick sklearn compatibility.

    Exposes feature_importances_ for FeatureImportances visualizer.
    Uses get_feature_importance() for reliable access (works with loaded models).
    """
    _estimator_type = 'classifier'

    def __init__(self, model):
        self.model = model
        self.classes_ = np.array(model.classes_)
        # Expose feature_importances_ for FeatureImportances visualizer
        # Use get_feature_importance() which is more reliable for loaded models
        try:
            fi = model.get_feature_importance()
            self.feature_importances_ = np.array(fi) if fi is not None else None
        except Exception:
            # Fallback to property
            self.feature_importances_ = model.feature_importances_

    def fit(self, X, y):
        return self  # Already fitted

    def predict(self, X):
        return self.model.predict(X).flatten()

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CatBoostWrapperCV(BaseEstimator, ClassifierMixin):
    """CatBoost wrapper for CV-based visualizers (can be cloned and re-fitted).

    Exposes feature_importances_ after fitting for RFECV and FeatureImportances.
    """
    _estimator_type = 'classifier'

    def __init__(self, iterations=100, depth=6, learning_rate=0.1,
                 auto_class_weights='Balanced', random_state=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.auto_class_weights = auto_class_weights
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            auto_class_weights=self.auto_class_weights,
            random_seed=self.random_state,
            verbose=0
        )
        self.model_.fit(X, y)
        self.classes_ = np.array(self.model_.classes_)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, X):
        return self.model_.predict(X).flatten()

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class ClassPredictionErrorFixed(_ClassPredictionErrorBase):
    """ClassPredictionError with sklearn 1.8+ compatibility fix.

    sklearn 1.8 changed _check_targets to return 4 values instead of 3.
    """

    def score(self, X, y):
        y_pred = self.predict(X)

        # FIX: Handle sklearn 1.8+ which returns 4 values
        result = _check_targets(y, y_pred)
        if len(result) == 4:
            y_type, y_true, y_pred, _ = result  # Ignore sample_weight
        else:
            y_type, y_true, y_pred = result

        if y_type not in ("binary", "multiclass"):
            raise YellowbrickValueError("{} is not supported".format(y_type))

        indices = unique_labels(y_true, y_pred)
        labels = self._labels()

        try:
            super(_ClassPredictionErrorBase, self).score(X, y)
        except ModelError as e:
            if labels is not None and len(labels) < len(indices):
                raise NotImplementedError("filtering classes is currently not supported")
            else:
                raise e

        if labels is not None and len(labels) > len(indices):
            raise ModelError("y and y_pred contain zero values for one of the specified classes")

        self.predictions_ = np.array([
            [(y_pred[y_true == label_t] == label_p).sum() for label_p in indices]
            for label_t in indices
        ])

        self.draw()
        return self.score_


def yellowbrick_classification_kwargs(
    project_name: str,
    metric_name: str,
    y_train: pd.Series,
    binary_classes: list
) -> dict:
    """Get kwargs for YellowBrick classification visualizers.

    Reference: https://www.scikit-yb.org/en/latest/api/classifier/index.html

    All visualizers use CatBoostWrapper with is_fitted=True and force_model=True
    for CatBoost compatibility, except DiscriminationThreshold which uses
    CatBoostWrapperCV for CV-based training.
    """
    # Human-readable class names for fraud detection
    class_names = ["Non-Fraud", "Fraud"] if "Fraud" in project_name else binary_classes

    kwargs = {
        # ConfusionMatrix: Essential for fraud detection
        "ConfusionMatrix": {
            "classes": class_names,
            "cmap": "Blues",
            "percent": True,
            "is_fitted": True,
            "force_model": True,
        },
        # ClassificationReport: Per-class precision, recall, F1
        "ClassificationReport": {
            "classes": class_names,
            "cmap": "YlOrRd",
            "support": "percent",
            "colorbar": True,
            "is_fitted": True,
            "force_model": True,
        },
        # ROCAUC: ROC curve for binary classification
        "ROCAUC": {
            "classes": class_names,
            "binary": True,
            "is_fitted": True,
            "force_model": True,
        },
        # PrecisionRecallCurve: BEST for imbalanced data!
        "PrecisionRecallCurve": {
            "classes": class_names,
            "fill_area": True,
            "ap_score": True,
            "iso_f1_curves": True,
            "is_fitted": True,
            "force_model": True,
        },
        # ClassPredictionError: Bar chart showing prediction errors
        "ClassPredictionError": {
            "classes": class_names,
            "is_fitted": True,
            "force_model": True,
        },
        # DiscriminationThreshold: Shows optimal threshold for binary classification
        # NOTE: Uses CatBoostWrapperCV (unfitted) for internal CV
        "DiscriminationThreshold": {
            "n_trials": 10,
            "cv": 0.2,
            "argmax": "fscore",
            "random_state": 42,
            # No is_fitted - needs to re-fit during CV
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_classification_visualizers(
    yb_kwargs: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model = None,
):
    """Create and fit YellowBrick classification visualizer.

    Reference: https://www.scikit-yb.org/en/latest/api/classifier/index.html

    Uses CatBoostWrapper for pre-fitted model visualizers, and
    CatBoostWrapperCV for CV-based visualizers (DiscriminationThreshold).
    """
    from yellowbrick.classifier import (
        ConfusionMatrix,
        ClassificationReport,
        ROCAUC,
        PrecisionRecallCurve,
        DiscriminationThreshold,
    )

    # Map visualizer names to classes
    visualizer_map = {
        "ConfusionMatrix": ConfusionMatrix,
        "ClassificationReport": ClassificationReport,
        "ROCAUC": ROCAUC,
        "PrecisionRecallCurve": PrecisionRecallCurve,
        "ClassPredictionError": ClassPredictionErrorFixed,  # sklearn 1.8+ fix
        "DiscriminationThreshold": DiscriminationThreshold,
    }

    for visualizer_name, params in yb_kwargs.items():
        vis_class = visualizer_map.get(visualizer_name)
        if vis_class is None:
            raise ValueError(f"Unknown visualizer: {visualizer_name}")

        if visualizer_name == "DiscriminationThreshold":
            # DiscriminationThreshold needs CV-compatible unfitted wrapper
            cv_estimator = CatBoostWrapperCV(iterations=100, depth=6, learning_rate=0.1)
            visualizer = vis_class(cv_estimator, **params)
            # Fit on full data (CV happens internally)
            X_full = pd.concat([X_train, X_test], ignore_index=True)
            y_full = pd.concat([y_train, y_test], ignore_index=True)
            visualizer.fit(X_full, y_full)
        else:
            # Other visualizers use pre-fitted wrapped model
            if model is None:
                raise ValueError("Model required for classification visualizers")
            wrapped_model = CatBoostWrapper(model) if 'CatBoost' in type(model).__name__ else model
            visualizer = vis_class(wrapped_model, **params)
            visualizer.fit(X_train, y_train)
            visualizer.score(X_test, y_test)

        return visualizer
    return None


# =============================================================================
# YellowBrick Feature Analysis Visualizers
# Reference: https://www.scikit-yb.org/en/latest/api/features/index.html
# =============================================================================

# Categorize visualizers by fit method (per YellowBrick docs)
FEATURE_FIT_TRANSFORM_VISUALIZERS = ["PCA", "Manifold", "ParallelCoordinates"]
FEATURE_FIT_ONLY_VISUALIZERS = ["FeatureImportances", "RFECV", "JointPlot"]
FEATURE_FIT_THEN_TRANSFORM_VISUALIZERS = ["Rank1D", "Rank2D", "RadViz"]


def yellowbrick_feature_analysis_kwargs(
    project_name: str,
    metric_name: str,
    classes: list,
    feature_names: list = None,
) -> dict:
    """
    Get kwargs for YellowBrick feature analysis visualizers.

    Reference: https://www.scikit-yb.org/en/latest/api/features/index.html

    Available visualizers:
    | Visualizer          | Fit Method      | Speed   |
    |---------------------|-----------------|---------|
    | Rank1D              | fit + transform | Fast    |
    | Rank2D              | fit + transform | Fast    |
    | PCA                 | fit_transform   | Fast    |
    | Manifold            | fit_transform   | SLOW    |
    | ParallelCoordinates | fit_transform   | Medium  |
    | RadViz              | fit + transform | Fast    |
    | JointPlot           | fit             | Fast    |

    Note: All features are numeric (DENSE_RANK encoding in DuckDB SQL),
    so all visualizers work with all features.
    """
    kwargs = {
        # =================================================================
        # RANK FEATURES - Detect covariance between features
        # Docs: https://www.scikit-yb.org/en/latest/api/features/rankd.html
        # All features are numeric (DENSE_RANK encoded) - works with all
        # =================================================================

        # Rank1D: Single feature ranking using Shapiro-Wilk normality test
        "Rank1D": {
            "algorithm": "shapiro",
            "features": feature_names,
            "orient": "h",
            "show_feature_names": True,
        },

        # Rank2D: Pairwise feature ranking (correlation matrix)
        # Algorithms: 'pearson', 'covariance', 'spearman', 'kendalltau'
        "Rank2D": {
            "algorithm": "pearson",
            "features": feature_names,
            "colormap": "RdBu_r",
            "show_feature_names": True,
        },

        # =================================================================
        # PROJECTION - Reduce dimensionality for visualization
        # =================================================================

        # PCA: Principal Component Analysis projection
        # Docs: https://www.scikit-yb.org/en/latest/api/features/pca.html
        "PCA": {
            "scale": True,
            "projection": 2,
            "proj_features": False,
            "classes": classes,
            "alpha": 0.75,
            "heatmap": False,
        },

        # Manifold: Non-linear dimensionality reduction (SLOW - 30-120s)
        # Docs: https://www.scikit-yb.org/en/latest/api/features/manifold.html
        # Algorithms: 'lle', 'ltsa', 'hessian', 'modified', 'isomap', 'mds', 'spectral', 'tsne'
        "Manifold": {
            "manifold": "tsne",
            "n_neighbors": 10,
            "classes": classes,
            "projection": 2,
            "alpha": 0.75,
            "random_state": 42,
            "target_type": "discrete",
        },

        # =================================================================
        # MULTI-DIMENSIONAL VISUALIZATION
        # =================================================================

        # ParallelCoordinates: Each instance as a line across feature axes
        # Docs: https://www.scikit-yb.org/en/latest/api/features/pcoords.html
        "ParallelCoordinates": {
            "classes": classes,
            "features": feature_names,
            "normalize": "minmax",
            "sample": 0.05,
            "shuffle": True,
            "alpha": 0.3,
            "fast": True,
        },

        # RadViz: Radial visualization (features as points on circle)
        # Docs: https://www.scikit-yb.org/en/latest/api/features/radviz.html
        "RadViz": {
            "classes": classes,
            "features": feature_names,
            "alpha": 0.5,
        },

        # =================================================================
        # DIRECT DATA VISUALIZATION
        # =================================================================

        # JointPlot: 2D correlation between two features
        # Docs: https://www.scikit-yb.org/en/latest/api/features/jointplot.html
        "JointPlot": {
            "columns": feature_names[:2] if feature_names and len(feature_names) >= 2 else None,
            "correlation": "pearson",
            "kind": "scatter",
            "hist": True,
            "alpha": 0.65,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_feature_analysis_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Create and fit YellowBrick feature analysis visualizer.

    Uses correct fit method per YellowBrick documentation:
    - fit_transform(): PCA, Manifold, ParallelCoordinates
    - fit() only: JointPlot
    - fit() + transform(): Rank1D, Rank2D, RadViz
    """
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(features, visualizer_name)(**params)

        # Use correct fit method per YellowBrick documentation
        if visualizer_name in FEATURE_FIT_TRANSFORM_VISUALIZERS:
            visualizer.fit_transform(X, y)
        elif visualizer_name in FEATURE_FIT_ONLY_VISUALIZERS:
            visualizer.fit(X, y)
        elif visualizer_name in FEATURE_FIT_THEN_TRANSFORM_VISUALIZERS:
            visualizer.fit(X, y)
            visualizer.transform(X)
        else:
            # Default: fit + transform
            visualizer.fit(X, y)
            visualizer.transform(X)

        return visualizer
    return None


# =============================================================================
# YellowBrick Target Visualizers
# =============================================================================
def yellowbrick_target_kwargs(
    project_name: str,
    metric_name: str,
    labels: list = None,
    feature_names: list = None
) -> dict:
    """Get kwargs for YellowBrick target visualizers.

    Reference: https://www.scikit-yb.org/en/latest/api/target/index.html

    Available visualizers:
    - ClassBalance: Class distribution (ESSENTIAL for fraud detection)
    - FeatureCorrelation: Feature-target correlation with mutual information
    - FeatureCorrelation_Pearson: Feature-target correlation with Pearson
    - BalancedBinningReference: Optimal bin boundaries (for regression)
    """
    kwargs = {
        # ClassBalance: CRITICAL for fraud detection - shows class imbalance
        # Docs: https://www.scikit-yb.org/en/latest/api/target/class_balance.html
        "ClassBalance": {
            "labels": labels if labels else ["Non-Fraud", "Fraud"],
            "colors": ["#2ecc71", "#e74c3c"],  # Green=non-fraud, Red=fraud
        },
        # FeatureCorrelation with Mutual Information (best for classification)
        # Captures non-linear relationships between features and target
        # Docs: https://www.scikit-yb.org/en/latest/api/target/feature_correlation.html
        "FeatureCorrelation": {
            "method": "mutual_info-classification",  # Best for binary classification
            "labels": feature_names,
            "sort": True,  # Sort by correlation (most important first)
            "color": "#3498db",  # Blue bars
        },
        # FeatureCorrelation with Pearson (faster, linear only)
        "FeatureCorrelation_Pearson": {
            "method": "pearson",
            "labels": feature_names,
            "sort": True,
            "color": "#9b59b6",  # Purple bars
        },
        # BalancedBinningReference: NOT useful for binary classification
        # Use for regression (ETA) to discretize continuous targets
        "BalancedBinningReference": {
            "target": "is_fraud",
            "bins": 4,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_target_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create and fit YellowBrick target visualizer.

    Reference: https://www.scikit-yb.org/en/latest/api/target/index.html

    Fit Methods:
    - ClassBalance: fit(y)
    - BalancedBinningReference: fit(y)
    - FeatureCorrelation: fit(X, y)
    - FeatureCorrelation_Pearson: fit(X, y)
    """
    # Map _Pearson suffix to actual class name
    visualizer_name_map = {
        "FeatureCorrelation_Pearson": "FeatureCorrelation",
    }

    for visualizer_name, params in yb_kwargs.items():
        # Get actual class name (handle _Pearson suffix)
        actual_class_name = visualizer_name_map.get(visualizer_name, visualizer_name)
        visualizer = getattr(target, actual_class_name)(**params)

        # Fit based on visualizer type
        if visualizer_name in ["BalancedBinningReference", "ClassBalance"]:
            # Target-only visualizers
            visualizer.fit(y)
        else:
            # Feature-target correlation visualizers need X
            visualizer.fit(X, y)
        return visualizer
    return None


# =============================================================================
# YellowBrick Model Selection Visualizers
# Reference: https://www.scikit-yb.org/en/latest/api/model_selection/index.html
# =============================================================================
def yellowbrick_model_selection_kwargs(
    project_name: str,
    metric_name: str,
    feature_names: list = None,
) -> dict:
    """Get kwargs for YellowBrick model selection visualizers.

    Reference: https://www.scikit-yb.org/en/latest/api/model_selection/index.html

    All 6 visualizers:
    - FeatureImportances: Feature ranking by importance (FAST)
    - CVScores: Cross-validation scores bar chart (MODERATE)
    - ValidationCurve: Hyperparameter tuning visualization (SLOW)
    - LearningCurve: Training size vs performance (SLOW)
    - RFECV: Recursive feature elimination with CV (VERY SLOW)
    - DroppingCurve: Feature subset random selection (SLOW)

    CatBoost Compatibility:
    - FeatureImportances: Uses CatBoostWrapper with is_fitted=True
    - All others: Use CatBoostWrapperCV for CV-based training
    """
    kwargs = {
        # =================================================================
        # PRIMARY: FeatureImportances (FAST, works with CatBoostWrapper)
        # =================================================================
        "FeatureImportances": {
            "labels": feature_names,
            "relative": True,              # Show as % of max importance
            "absolute": False,
            "is_fitted": True,             # CatBoostWrapper compatibility
        },

        # =================================================================
        # SECONDARY: CVScores (moderate speed with CatBoostWrapperCV)
        # =================================================================
        "CVScores": {
            "cv": 5,                       # 5-fold stratified CV
            "scoring": "f1",               # F1 score for imbalanced data
        },

        # =================================================================
        # ValidationCurve: Hyperparameter tuning (SLOW)
        # Shows how a single hyperparameter affects train/test scores
        # =================================================================
        "ValidationCurve": {
            "param_name": "iterations",    # CatBoost iterations param
            "param_range": np.array([50, 100, 150, 200]),
            "cv": 3,
            "scoring": "f1",
        },

        # =================================================================
        # LearningCurve: Training size analysis (SLOW)
        # =================================================================
        "LearningCurve": {
            "train_sizes": np.linspace(0.1, 1.0, 5),
            "cv": 3,                       # Reduce folds for speed
            "scoring": "f1",
            "random_state": 42,
        },

        # =================================================================
        # RFECV: Recursive Feature Elimination with CV (VERY SLOW)
        # Finds optimal number of features
        # =================================================================
        "RFECV": {
            "cv": 3,
            "scoring": "f1",
            "step": 1,                     # Remove 1 feature at a time
        },

        # =================================================================
        # DroppingCurve: Feature subset analysis (SLOW)
        # =================================================================
        "DroppingCurve": {
            "feature_sizes": np.linspace(0.1, 1.0, 5),
            "cv": 3,
            "scoring": "f1",
            "random_state": 42,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_model_selection_visualizers(
    yb_kwargs: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model = None,
):
    """Create and fit YellowBrick model selection visualizer.

    Reference: https://www.scikit-yb.org/en/latest/api/model_selection/index.html

    CatBoost Handling:
    - FeatureImportances: Uses CatBoostWrapper (pre-fitted, sklearn-compatible)
    - All others: Creates CatBoostWrapperCV for CV-based training
    """
    from yellowbrick.model_selection import (
        FeatureImportances,
        CVScores,
        ValidationCurve,
        LearningCurve,
        RFECV,
        DroppingCurve,
    )

    # Map visualizer names to classes
    visualizer_map = {
        "FeatureImportances": FeatureImportances,
        "CVScores": CVScores,
        "ValidationCurve": ValidationCurve,
        "LearningCurve": LearningCurve,
        "RFECV": RFECV,
        "DroppingCurve": DroppingCurve,
    }

    # Combine data for CV-based visualizers
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    for visualizer_name, params in yb_kwargs.items():
        vis_class = visualizer_map.get(visualizer_name)
        if vis_class is None:
            raise ValueError(f"Unknown visualizer: {visualizer_name}")

        if visualizer_name == "FeatureImportances":
            # FeatureImportances uses CatBoostWrapper (pre-fitted, sklearn-compatible)
            if model is None:
                raise ValueError("Model required for FeatureImportances")
            wrapped_estimator = CatBoostWrapper(model) if 'CatBoost' in type(model).__name__ else model
            visualizer = vis_class(wrapped_estimator, **params)
            visualizer.fit(X_train, y_train)
        else:
            # All other visualizers need CatBoostWrapperCV
            cv_wrapper = CatBoostWrapperCV(
                iterations=100,
                depth=6,
                learning_rate=0.1,
            )
            visualizer = vis_class(cv_wrapper, **params)
            visualizer.fit(X_full, y_full)

        return visualizer
    return None


# =============================================================================
# Data Manager Class
# =============================================================================
class ModelDataManager:
    """Manages loaded data for batch ML models."""
    def __init__(self):
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.preprocessor_dict: dict | None = None
        self.project_name: str | None = None

    def load_data(self, project_name: str):
        """Load and process data for a project using DuckDB SQL."""
        if self.project_name == project_name and self.y_train is not None:
            print(f"Data for {project_name} is already loaded.")
            return

        print(f"Loading data for project: {project_name}")
        self.X_train, self.X_test, self.y_train, self.y_test, self.preprocessor_dict = process_batch_data_duckdb(
            project_name
        )
        self.X = pd.concat([self.X_train, self.X_test])
        self.y = pd.concat([self.y_train, self.y_test])
        self.project_name = project_name
        print("Data loaded successfully.")


# =============================================================================
# YellowBrick Image Generation
# =============================================================================
def generate_yellowbrick_image(visualizer) -> str:
    """Generate base64 encoded PNG image from YellowBrick visualizer.

    The sequence is critical for correct rendering:
    1. visualizer.show() - Finalizes the visualization (required!)
    2. fig.savefig() - Save to buffer
    3. plt.close() + plt.clf() - Cleanup to prevent overlap
    """
    # Finalize the visualization (required for proper rendering)
    visualizer.show()

    # Save to buffer
    buf = io.BytesIO()
    visualizer.fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Cleanup - prevents overlapping plots and memory leaks
    plt.close(visualizer.fig)  # Close specific figure
    plt.clf()                   # Clear current figure state
    return image_base64
