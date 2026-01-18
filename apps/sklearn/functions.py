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
from resource_pool import resource_pool, SessionResources
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
    conn = duckdb.connect()
    # Install and load delta extension (install is cached to disk)
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


def load_model_from_mlflow(project_name: str, model_name: str):
    """Load model from the best MLflow run based on metrics.

    Uses get_best_mlflow_run to find the run with optimal metrics.

    Returns:
        Loaded model, or None if not found.
    """
    try:
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


def load_encoders_from_mlflow(project_name: str) -> Optional[Dict[str, Any]]:
    """Load encoders from the best MLflow run based on metrics.

    Uses get_best_mlflow_run to find the run with optimal metrics,
    ensuring encoders are loaded from the same run as the best model.

    Returns:
        Dict with preprocessor/encoders, or None if not found.
    """
    try:
        model_name = MLFLOW_MODEL_NAMES.get(project_name)
        if not model_name:
            print(f"Unknown project for encoder loading: {project_name}")
            return None

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
    - Column selection and ordering (matches TFD_ALL_FEATURES order)
    - Sampling (if requested)

    No pandas transformations, no StandardScaler (tree models don't need it).
    CatBoost handles categorical features natively - no label encoding needed.
    """
    delta_path = DELTA_PATHS["Transaction Fraud Detection"]

    # Build the SQL query - all preprocessing in one pass
    # Column order MUST match TFD_ALL_FEATURES for correct cat_feature_indices
    query = f"""
    SELECT
        -- Numerical features (no scaling needed for CatBoost)
        amount,
        account_age_days,
        CAST(cvv_provided AS INTEGER) AS cvv_provided,
        CAST(billing_address_match AS INTEGER) AS billing_address_match,

        -- Categorical features (order matches TFD_CATEGORICAL_FEATURES)
        currency,
        merchant_id,
        payment_method,
        product_category,
        transaction_type,

        -- Categorical features from JSON (extracted in SQL)
        json_extract_string(device_info, '$.browser') AS browser,
        json_extract_string(device_info, '$.os') AS os,

        -- Timestamp components (extracted in SQL, cast VARCHAR to TIMESTAMP first)
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

    # Convert categorical columns to category dtype (memory efficient + CatBoost compatible)
    for col in TFD_CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Metadata for CatBoost
    metadata = {
        "numerical_features": TFD_NUMERICAL_FEATURES,
        "categorical_features": TFD_CATEGORICAL_FEATURES,
        "cat_feature_indices": TFD_CAT_FEATURE_INDICES,
        "feature_names": TFD_ALL_FEATURES,
    }

    print(f"  Features: {len(TFD_NUMERICAL_FEATURES)} numerical, {len(TFD_CATEGORICAL_FEATURES)} categorical")
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

    # Store in resource pool for YellowBrick visualizations
    session = SessionResources(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=metadata["feature_names"],
        cat_feature_indices=metadata["cat_feature_indices"],
        project_name=project_name,
    )
    resource_pool.store(session)
    print("  Data stored in resource pool for YellowBrick visualizations")

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
# =============================================================================
def yellowbrick_classification_kwargs(
    project_name: str,
    metric_name: str,
    y_train: pd.Series,
    binary_classes: list
) -> dict:
    """Get kwargs for YellowBrick classification visualizers."""
    kwargs = {
        "ClassificationReport": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "classes": binary_classes,
            "support": True,
        },
        "ConfusionMatrix": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "classes": binary_classes,
        },
        "ROCAUC": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "classes": binary_classes,
        },
        "PrecisionRecallCurve": {
            "estimator": create_batch_model(project_name, y_train=y_train),
        },
        "ClassPredictionError": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "classes": binary_classes,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_classification_visualizers(
    yb_kwargs: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """Create and fit YellowBrick classification visualizer."""
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(classifier, visualizer_name)(**params)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        return visualizer
    return None


# =============================================================================
# YellowBrick Feature Analysis Visualizers
# =============================================================================
def yellowbrick_feature_analysis_kwargs(
    project_name: str,
    metric_name: str,
    classes: list,
    feature_names: list = None
) -> dict:
    """Get kwargs for YellowBrick feature analysis visualizers."""
    kwargs = {
        "ParallelCoordinates": {
            "classes": classes,
            "features": feature_names,
            "sample": 0.05,
            "shuffle": True,
            "n_jobs": 1,
        },
        "PCA": {
            "classes": classes,
            "scale": True,
            "n_jobs": 1,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_feature_analysis_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create and fit YellowBrick feature analysis visualizer."""
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(features, visualizer_name)(**params)
        if visualizer_name in ["ParallelCoordinates", "PCA", "Manifold"]:
            visualizer.fit_transform(X, y)
        else:
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
    """Get kwargs for YellowBrick target visualizers."""
    kwargs = {
        "BalancedBinningReference": {},
        "ClassBalance": {
            "labels": labels,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_target_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create and fit YellowBrick target visualizer."""
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(target, visualizer_name)(**params)
        if visualizer_name in ["BalancedBinningReference", "ClassBalance"]:
            visualizer.fit(y)
        else:
            visualizer.fit(X, y)
        return visualizer
    return None


# =============================================================================
# YellowBrick Model Selection Visualizers
# =============================================================================
def yellowbrick_model_selection_kwargs(
    project_name: str,
    metric_name: str,
    y_train: pd.Series
) -> dict:
    """Get kwargs for YellowBrick model selection visualizers."""
    kwargs = {
        "ValidationCurve": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "param_name": "gamma",
            "param_range": np.logspace(-6, -1, 10),
            "logx": True,
            "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "scoring": "average_precision",
            "n_jobs": 1,
        },
        "LearningCurve": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "scoring": "average_precision",
            "train_sizes": np.linspace(0.3, 1.0, 8),
            "n_jobs": 1,
        },
        "CVScores": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "scoring": "average_precision",
            "n_jobs": 1,
        },
        "FeatureImportances": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "n_jobs": 1,
        },
        "DroppingCurve": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "n_jobs": 1,
        }
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_model_selection_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create and fit YellowBrick model selection visualizer."""
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(model_selection, visualizer_name)(**params)
        if visualizer_name in ["ValidationCurve", "RFECV"]:
            X_stratified, _, y_stratified, _ = train_test_split(
                X, y,
                train_size=min(50000, len(X)),
                shuffle=True,
                stratify=y,
                random_state=42
            )
            visualizer.fit(X_stratified, y_stratified)
        else:
            visualizer.fit(X, y)
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
    """Generate base64 encoded PNG image from YellowBrick visualizer."""
    buf = io.BytesIO()
    visualizer.fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(visualizer.fig)
    return image_base64
