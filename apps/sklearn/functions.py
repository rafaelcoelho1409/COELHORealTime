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
from sklearn.preprocessing import StandardScaler
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

# Delta Lake paths (S3 paths for DuckDB delta_scan)
DELTA_PATHS = {
    "Transaction Fraud Detection": "s3://lakehouse/delta/transaction_fraud_detection",
    "Estimated Time of Arrival": "s3://lakehouse/delta/estimated_time_of_arrival",
    "E-Commerce Customer Interactions": "s3://lakehouse/delta/e_commerce_customer_interactions",
}


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
# MLflow Encoder Functions
# =============================================================================
def get_best_mlflow_run(project_name: str, model_name: str) -> Optional[str]:
    """Get the best MLflow run ID based on metrics for a project.

    For classification (TFD): Uses F1 score (higher is better)
    For regression (ETA): Uses RMSE (lower is better)
    For clustering (ECCI): Uses Silhouette score (higher is better)

    Returns:
        run_id of the best run, or None if no runs found.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(project_name)
        if experiment is None:
            print(f"No MLflow experiment found for {project_name}")
            return None

        # Define metric and order based on project type
        if project_name == "Transaction Fraud Detection":
            metric_key = "metrics.F1"
            ascending = False  # Higher F1 is better
        elif project_name == "Estimated Time of Arrival":
            metric_key = "metrics.RMSE"
            ascending = True  # Lower RMSE is better
        else:
            metric_key = "metrics.F1"
            ascending = False

        # Search for completed runs with the model name
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_name}' and status = 'FINISHED'",
            order_by=[f"{metric_key} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if runs.empty:
            print(f"No completed runs found for {project_name}/{model_name}")
            return None

        run_id = runs.iloc[0]["run_id"]
        print(f"Best MLflow run for {project_name}: {run_id}")
        return run_id

    except Exception as e:
        print(f"Error finding best MLflow run: {e}")
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
    """Load encoders from MLflow artifacts or create new ones.

    Tries to load from MLflow first (best model's encoders).
    Falls back to creating default encoders if not available.

    Returns:
        Dict with preprocessor/encoders.
    """
    # Try loading from MLflow first
    encoders = load_encoders_from_mlflow(project_name)
    if encoders is not None:
        return encoders

    # No encoders in MLflow, create new ones
    print(f"No sklearn encoders in MLflow for {project_name}, creating new ones.")
    return _create_default_encoders(project_name)


# =============================================================================
# Data Processing Functions (Sklearn)
# =============================================================================
def extract_device_info_sklearn(data):
    data = data.copy()
    # Parse JSON strings to dictionaries
    # Using orjson for speed (already imported)
    device_dicts = data["device_info"].apply(
        lambda x: orjson.loads(x) if isinstance(x, str) else x
    )
    data_to_join = pd.json_normalize(device_dicts)
    data = data.drop("device_info", axis = 1)
    data = data.join(data_to_join)
    return data


def extract_timestamp_info_sklearn(data):
    """Extract timestamp components to separate columns."""
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], format = 'ISO8601')
    data["year"] = data["timestamp"].dt.year
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["second"] = data["timestamp"].dt.second
    data = data.drop("timestamp", axis=1)
    return data


def extract_coordinates_sklearn(data):
    data = data.copy()
    location_dicts = data["location"].apply(
        lambda x: orjson.loads(x) if isinstance(x, str) else x
    )
    data_to_join = pd.json_normalize(location_dicts)
    data = data.drop("location", axis = 1)
    data = data.join(data_to_join)
    return data


def _create_default_encoders(project_name: str) -> dict:
    """Create default encoders for CatBoost (no OneHotEncoder needed).

    Used as fallback for fresh starts when no pre-trained encoders exist.
    CatBoost handles categorical features natively, so we only need StandardScaler.

    Returns:
        Dict with scaler, feature lists, and categorical indices.
    """
    if project_name == "Transaction Fraud Detection":
        numerical_features = [
            "amount",
            "account_age_days",
            "cvv_provided",
            "billing_address_match",
        ]
        categorical_features = [
            "currency",
            "merchant_id",
            "payment_method",
            "product_category",
            "transaction_type",
            "browser",
            "os",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
        ]
        # Full feature order: numerical first, then categorical
        all_features = numerical_features + categorical_features

        # Categorical feature indices (relative to all_features)
        cat_feature_indices = list(range(
            len(numerical_features),
            len(all_features)
        ))

        return {
            "scaler": StandardScaler(),  # Unfitted - will be fitted on first use
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "cat_feature_indices": cat_feature_indices,
            "feature_names": all_features,
        }
    else:
        raise ValueError(f"Unknown project: {project_name}")


def load_or_create_data(
    project_name: str,
    sample_frac: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load data from Delta Lake on MinIO via DuckDB.

    Uses persistent DuckDB connection with Delta extension.
    Retries with reconnection on failure.

    Args:
        project_name: Project name to load data for.
        sample_frac: Optional fraction of data to sample (0.0-1.0).
                     E.g., 0.3 = 30% of data.
        max_rows: Optional maximum number of rows to load.
                  Applied after sampling if both specified.

    Returns:
        DataFrame with loaded data.
    """
    delta_path = DELTA_PATHS.get(project_name, "")
    if not delta_path:
        raise ValueError(f"Unknown project: {project_name}")

    # Build query with optional sampling
    query = f"SELECT * FROM delta_scan('{delta_path}')"
    if sample_frac is not None and 0 < sample_frac < 1:
        query += f" USING SAMPLE {sample_frac * 100}%"
    if max_rows is not None:
        query += f" LIMIT {max_rows}"

    try:
        print(f"Loading data from Delta Lake via DuckDB: {delta_path}")
        if sample_frac:
            print(f"  Sampling: {sample_frac * 100}%")
        if max_rows:
            print(f"  Max rows: {max_rows}")
        conn = _get_duckdb_connection()
        result = conn.execute(query).df()
        print(f"Data loaded from Delta Lake for {project_name}: {len(result)} rows")
        return result
    except Exception as e:
        # Retry with reconnection on failure
        print(f"DuckDB query failed, attempting reconnect: {e}")
        conn = _get_duckdb_connection(force_reconnect=True)
        result = conn.execute(query).df()
        print(f"Data loaded from Delta Lake for {project_name}: {len(result)} rows")
        return result


def process_batch_data(data: pd.DataFrame, project_name: str):
    """Process batch data for CatBoost training.

    Memory-optimized version that leverages CatBoost's native categorical handling.
    No OneHotEncoder needed - CatBoost handles categoricals internally.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor_dict
        preprocessor_dict contains:
          - 'scaler': fitted StandardScaler for numerical features
          - 'numerical_features': list of numerical feature names
          - 'categorical_features': list of categorical feature names
          - 'cat_feature_indices': indices of categorical features in final X
    """
    if project_name == "Transaction Fraud Detection":
        # Extract nested JSON fields (modifies in place where possible)
        data = extract_device_info_sklearn(data)
        data = extract_timestamp_info_sklearn(data)

        # Define feature groups
        numerical_features = [
            "amount",
            "account_age_days",
        ]
        binary_features = [
            "cvv_provided",
            "billing_address_match",
        ]
        categorical_features = [
            "currency",
            "merchant_id",
            "payment_method",
            "product_category",
            "transaction_type",
            "browser",
            "os",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
        ]

        # Memory optimization: convert categoricals to category dtype
        # This reduces memory significantly for high-cardinality columns
        print("Optimizing memory with categorical dtypes...")
        for col in categorical_features:
            if col in data.columns:
                data[col] = data[col].astype('category')

        # Downcast numerical columns
        for col in numerical_features + binary_features:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], downcast='float')

        # Prepare features and target
        all_features = numerical_features + binary_features + categorical_features
        available_features = [f for f in all_features if f in data.columns]

        X = data[available_features]
        y = data['is_fraud']

        # Free memory from original data
        del data

        print(f"Features: {len(available_features)} ({len(numerical_features)} numerical, "
              f"{len(binary_features)} binary, {len(categorical_features)} categorical)")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        # Free memory
        del X, y

        # Scale only numerical features (CatBoost handles categoricals natively)
        scaler = StandardScaler()
        numerical_cols_present = [c for c in numerical_features + binary_features if c in X_train.columns]

        if numerical_cols_present:
            X_train[numerical_cols_present] = scaler.fit_transform(X_train[numerical_cols_present])
            X_test[numerical_cols_present] = scaler.transform(X_test[numerical_cols_present])

        # Get categorical feature indices for CatBoost
        cat_feature_indices = [
            X_train.columns.get_loc(col)
            for col in categorical_features
            if col in X_train.columns
        ]

        # Preprocessor dict for saving to MLflow
        preprocessor_dict = {
            "scaler": scaler,
            "numerical_features": numerical_cols_present,
            "categorical_features": categorical_features,
            "cat_feature_indices": cat_feature_indices,
            "feature_names": list(X_train.columns),
        }

        print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        print(f"Categorical feature indices for CatBoost: {cat_feature_indices}")

        return X_train, X_test, y_train, y_test, preprocessor_dict
    else:
        raise ValueError(f"Unsupported project for batch processing: {project_name}")


# =============================================================================
# DuckDB SQL-Based Preprocessing (Optimized - No Sklearn Transformations)
# =============================================================================
# Feature definitions for Transaction Fraud Detection
TFD_NUMERICAL_FEATURES = ["amount", "account_age_days", "cvv_provided", "billing_address_match"]
TFD_CATEGORICAL_FEATURES = [
    "currency", "merchant_id", "payment_method", "product_category",
    "transaction_type", "browser", "os",
    "year", "month", "day", "hour", "minute", "second",
]
TFD_ALL_FEATURES = TFD_NUMERICAL_FEATURES + TFD_CATEGORICAL_FEATURES
TFD_CAT_FEATURE_INDICES = list(range(len(TFD_NUMERICAL_FEATURES), len(TFD_ALL_FEATURES)))


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
    - Column selection and ordering
    - Sampling (if requested)

    No pandas transformations, no StandardScaler (tree models don't need it).
    """
    delta_path = DELTA_PATHS["Transaction Fraud Detection"]

    # Build the SQL query - all preprocessing in one pass
    query = f"""
    SELECT
        -- Numerical features (no scaling needed for CatBoost)
        amount,
        account_age_days,
        CAST(cvv_provided AS INTEGER) AS cvv_provided,
        CAST(billing_address_match AS INTEGER) AS billing_address_match,

        -- Categorical features from JSON (extracted in SQL)
        json_extract_string(device_info, '$.browser') AS browser,
        json_extract_string(device_info, '$.os') AS os,

        -- Categorical features (direct columns)
        currency,
        merchant_id,
        payment_method,
        product_category,
        transaction_type,

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

    return X_train, X_test, y_train, y_test, metadata


def process_sklearn_sample_duckdb(x: dict, project_name: str) -> pd.DataFrame:
    """Process a single sample using DuckDB SQL approach.

    For real-time predictions, we still need to process individual samples.
    This uses the same feature order as DuckDB training but processes in Python.

    Note: No scaling needed since CatBoost doesn't require it.
    """
    if project_name == "Transaction Fraud Detection":
        # Extract device_info JSON
        device_info = x.get("device_info", "{}")
        if isinstance(device_info, str):
            device_info = orjson.loads(device_info)

        # Extract timestamp components
        timestamp = pd.to_datetime(x.get("timestamp"))

        # Build feature dict in correct order
        features = {
            # Numerical features
            "amount": x.get("amount"),
            "account_age_days": x.get("account_age_days"),
            "cvv_provided": int(x.get("cvv_provided", 0)),
            "billing_address_match": int(x.get("billing_address_match", 0)),
            # Categorical from JSON
            "browser": device_info.get("browser"),
            "os": device_info.get("os"),
            # Categorical direct
            "currency": x.get("currency"),
            "merchant_id": x.get("merchant_id"),
            "payment_method": x.get("payment_method"),
            "product_category": x.get("product_category"),
            "transaction_type": x.get("transaction_type"),
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


def process_sklearn_sample(x: dict, project_name: str) -> pd.DataFrame:
    """Process a single sample for CatBoost prediction.

    Loads preprocessor from MLflow (best model's encoders) or creates new if not available.
    CatBoost handles categorical features natively, so we only scale numerical features.

    Returns:
        DataFrame with features in correct order for the model.
    """
    if project_name == "Transaction Fraud Detection":
        df = pd.DataFrame([x])
        df = extract_device_info_sklearn(df)
        df = extract_timestamp_info_sklearn(df)

        # Load preprocessor from MLflow or create new
        preprocessor_dict = load_or_create_sklearn_encoders(project_name)

        scaler = preprocessor_dict["scaler"]
        numerical_features = preprocessor_dict["numerical_features"]
        feature_names = preprocessor_dict["feature_names"]

        # Scale numerical features (if scaler is fitted)
        numerical_cols_present = [c for c in numerical_features if c in df.columns]
        if numerical_cols_present:
            # Check if scaler is fitted (has mean_ attribute)
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                df[numerical_cols_present] = scaler.transform(df[numerical_cols_present])
            else:
                # Scaler not fitted - this shouldn't happen in production
                # (training saves fitted scaler to MLflow)
                print("Warning: Scaler not fitted, using raw numerical values")

        # Ensure columns are in correct order (match training feature order)
        available_features = [f for f in feature_names if f in df.columns]
        df = df[available_features]

        # Convert categorical columns to category dtype for CatBoost
        categorical_features = preprocessor_dict.get("categorical_features", [])
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df
    else:
        raise ValueError(f"Unsupported project for sample processing: {project_name}")


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
        model = CatBoostClassifier(
            # =================================================================
            # CORE PARAMETERS
            # =================================================================
            iterations = 1000,              # Number of boosting rounds
                                          # High value + early stopping finds optimal
            learning_rate = 0.03,           # Lower = better generalization
                                          # Range: 0.01-0.3, lower for more data
            depth = 6,                      # Tree depth (recommended: 6-10)
                                          # CatBoost default=6, good for most cases
            # =================================================================
            # IMBALANCED DATA HANDLING (KEY FOR FRAUD DETECTION)
            # =================================================================
            auto_class_weights = 'Balanced', # Automatically balance class weights
                                           # Options: None, 'Balanced', 'SqrtBalanced'
                                           # 'Balanced' = weight inversely proportional to frequency
            # =================================================================
            # LOSS FUNCTION & EVALUATION
            # =================================================================
            loss_function = 'Logloss',      # Binary cross-entropy for classification
                                          # Options: 'Logloss', 'CrossEntropy'
            eval_metric = 'AUC',            # Area Under ROC Curve
                                          # Best metric for imbalanced binary classification
                                          # Other options: 'F1', 'Precision', 'Recall', 'PRAUC'
            # =================================================================
            # REGULARIZATION (PREVENT OVERFITTING)
            # =================================================================
            l2_leaf_reg = 3.0,              # L2 regularization coefficient
                                          # Higher = more regularization
                                          # Range: 1-10, default=3
            random_strength = 1.0,          # Randomness for scoring splits
                                          # Higher = more regularization
                                          # Range: 0-10, default=1
            bagging_temperature = 1.0,      # Bayesian bootstrap intensity
                                          # Higher = more randomness
                                          # Range: 0-10, default=1
            # =================================================================
            # EARLY STOPPING (requires eval_set in fit())
            # =================================================================
            # NOTE: early_stopping_rounds and use_best_model require eval_set
            # Pass eval_set=(X_test, y_test) when calling model.fit()
            # =================================================================
            # =================================================================
            # PERFORMANCE & REPRODUCIBILITY
            # =================================================================
            task_type = 'CPU',              # 'CPU' or 'GPU'
                                          # GPU requires CUDA, much faster for large data
            thread_count = -1,              # Use all CPU cores
            random_seed = 42,               # Reproducibility
            # =================================================================
            # MEMORY OPTIMIZATION (for containerized environments)
            # =================================================================
            max_ctr_complexity = 1,         # Limit categorical feature combinations
                                          # Default=4, reduces memory significantly
            used_ram_limit = '6gb',         # Explicit RAM limit for CatBoost
                                          # Leave 2GB headroom for other processes (8GB limit)
            boosting_type = 'Plain',        # Plain boosting uses less memory
                                          # Options: 'Ordered' (default), 'Plain'
            # =================================================================
            # OUTPUT
            # =================================================================
            verbose = 100,                  # Print every 100 iterations
                                          # Set to False for silent training
            allow_writing_files = False,    # Don't write temp files
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
        """Load and process data for a project."""
        if self.project_name == project_name and self.y_train is not None:
            print(f"Data for {project_name} is already loaded.")
            return

        print(f"Loading data for project: {project_name}")
        data_df = load_or_create_data(project_name)
        self.X_train, self.X_test, self.y_train, self.y_test, self.preprocessor_dict = process_batch_data(
            data_df, project_name
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
