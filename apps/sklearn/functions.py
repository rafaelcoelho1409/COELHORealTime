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
from catboost import CatBoostClassifier, CatBoostRegressor
from yellowbrick import (
    classifier,
    features,
    target,
    model_selection,
    regressor,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import duckdb
import orjson


# =============================================================================
# NUMPY 1.22+ COMPATIBILITY FIX FOR YELLOWBRICK INTERCLUSTERDISTANCE
# Issue: YellowBrick uses np.percentile(interpolation=...) but NumPy 1.22+
#        renamed this parameter to 'method'
# Fix: Monkey-patch the percentile_index function in yellowbrick.cluster.icdm
# =============================================================================
try:
    import yellowbrick.cluster.icdm as icdm_module

    def _patched_percentile_index(a, q):
        """
        Returns the index of the value at the Qth percentile in array a.
        NumPy 1.22+ compatible version using 'method' instead of 'interpolation'.
        """
        idx = int(np.percentile(np.arange(len(a)), q, method='nearest'))
        return idx

    icdm_module.percentile_index = _patched_percentile_index
except Exception:
    pass  # Ignore if module not available

# =============================================================================
# NUMPY 2.0+ COMPATIBILITY FIX FOR YELLOWBRICK DISPERSIONPLOT
# Issue: YellowBrick uses np.stack(generator) but NumPy 2.0+ requires sequences
# Fix: Monkey-patch DispersionPlot.fit to convert generators to lists
# From notebook 018_sklearn_duckdb_sql_clustering.ipynb
# =============================================================================
try:
    from yellowbrick.text.dispersion import DispersionPlot as _DispersionPlotClass
    from yellowbrick.exceptions import YellowbrickValueError as _YBValueError

    def _patched_dispersion_fit(self, X, y=None, **kwargs):
        """
        Patched fit method for NumPy 2.0+ compatibility.
        Converts generators to lists before calling np.stack.
        """
        if y is not None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.array([self.NULL_CLASS])

        # Create an index for the target words
        self.indexed_words_ = np.flip(self.search_terms, axis=0)
        if self.ignore_case:
            self.indexed_words_ = np.array([w.lower() for w in self.indexed_words_])

        # FIX: Convert generator to list before np.stack
        try:
            dispersion_data = list(self._compute_dispersion(X, y))
            if len(dispersion_data) == 0:
                raise ValueError('Empty')
            offsets_positions_categories = np.stack(dispersion_data)
        except ValueError:
            raise _YBValueError('No search terms were found in the corpus')

        # FIX: Convert zip to list before np.stack
        word_positions = np.stack(
            list(zip(
                offsets_positions_categories[:, 0].astype(int),
                offsets_positions_categories[:, 1].astype(int),
            ))
        )

        self.word_categories_ = offsets_positions_categories[:, 2]
        self._check_missing_words(word_positions)
        self.draw(word_positions, **kwargs)
        return self

    _DispersionPlotClass.fit = _patched_dispersion_fit
except Exception:
    pass  # Ignore if module not available


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
    "E-Commerce Customer Interactions": {"metric_name": "silhouette_score", "maximize": True},
}

# Delta Lake paths (S3 paths for DuckDB delta_scan)
DELTA_PATHS = {
    "Transaction Fraud Detection": "s3://lakehouse/delta/transaction_fraud_detection",
    "Estimated Time of Arrival": "s3://lakehouse/delta/estimated_time_of_arrival",
    "E-Commerce Customer Interactions": "s3://lakehouse/delta/e_commerce_customer_interactions",
}

# =============================================================================
# Task Type Configuration (determines split strategy, metrics, etc.)
# =============================================================================
PROJECT_TASK_TYPES = {
    "Transaction Fraud Detection": "classification",
    "Estimated Time of Arrival": "regression",
    "E-Commerce Customer Interactions": "clustering",
}

# Target column names per project
PROJECT_TARGET_COLUMNS = {
    "Transaction Fraud Detection": "is_fraud",
    "Estimated Time of Arrival": "simulated_actual_travel_time_seconds",
    "E-Commerce Customer Interactions": None,  # Clustering has no target
}

# =============================================================================
# Feature definitions for Transaction Fraud Detection (DuckDB SQL approach)
# =============================================================================
TFD_NUMERICAL_FEATURES = ["amount", "account_age_days", "cvv_provided", "billing_address_match"]
TFD_CATEGORICAL_FEATURES = [
    "currency", "merchant_id", "payment_method", "product_category",
    "transaction_type", "browser", "os",
    "year", "month", "day", "hour", "minute", "second",
]
TFD_ALL_FEATURES = TFD_NUMERICAL_FEATURES + TFD_CATEGORICAL_FEATURES
TFD_CAT_FEATURE_INDICES = list(range(len(TFD_NUMERICAL_FEATURES), len(TFD_ALL_FEATURES)))

# =============================================================================
# Feature definitions for Estimated Time of Arrival (ETA) - Regression
# =============================================================================
# From notebook 017_sklearn_duckdb_sql_regression.ipynb
ETA_NUMERICAL_FEATURES = [
    "estimated_distance_km",
    "temperature_celsius",
    "driver_rating",
    "hour_of_day",
    "initial_estimated_travel_time_seconds",
    "debug_traffic_factor",
    "debug_weather_factor",
    "debug_incident_delay_seconds",
    "debug_driver_factor",
]
ETA_CATEGORICAL_FEATURES = [
    # IDs (high cardinality - label encoded)
    "trip_id",
    "driver_id",
    "vehicle_id",
    # Locations
    "origin",
    "destination",
    # Context
    "weather",
    "day_of_week",
    "vehicle_type",
    # Temporal (extracted from timestamp)
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
]
ETA_ALL_FEATURES = ETA_NUMERICAL_FEATURES + ETA_CATEGORICAL_FEATURES
ETA_CAT_FEATURE_INDICES = list(range(len(ETA_NUMERICAL_FEATURES), len(ETA_ALL_FEATURES)))

# =============================================================================
# Feature definitions for E-Commerce Customer Interactions (ECCI) - Clustering
# =============================================================================
# Based on River's process_sample for ECCI
ECCI_NUMERICAL_FEATURES = [
    "price",
    "quantity",
    "session_event_sequence",
    "time_on_page_seconds",
]
ECCI_CATEGORICAL_FEATURES = [
    "event_type", "product_category", "product_id", "referrer_url",
    "browser", "os",
    "year", "month", "day", "hour", "minute", "second",
]
ECCI_ALL_FEATURES = ECCI_NUMERICAL_FEATURES + ECCI_CATEGORICAL_FEATURES
ECCI_CAT_FEATURE_INDICES = list(range(len(ECCI_NUMERICAL_FEATURES), len(ECCI_ALL_FEATURES)))

# Customer-level aggregated features for clustering (from notebook 018)
# These are computed via DuckDB SQL aggregation at customer level
ECCI_CUSTOMER_FEATURES = [
    # Engagement metrics
    "total_sessions",
    "total_events",
    "avg_time_on_page",
    "total_time_on_site",
    "avg_events_per_session",
    # Purchase behavior (RFM-like)
    "total_purchases",
    "total_revenue",
    "avg_order_value",
    # Behavioral patterns
    "page_views",
    "product_views",
    "add_to_carts",
    "searches",
    "unique_products_viewed",
    "unique_categories_viewed",
    # Temporal patterns
    "preferred_hour",
    "days_active",
    # Geographic
    "geo_diversity",
    # Conversion rates
    "view_to_cart_rate",
    "cart_to_purchase_rate",
]

# =============================================================================
# Unified Feature Lookup Dictionaries
# =============================================================================
PROJECT_NUMERICAL_FEATURES = {
    "Transaction Fraud Detection": TFD_NUMERICAL_FEATURES,
    "Estimated Time of Arrival": ETA_NUMERICAL_FEATURES,
    "E-Commerce Customer Interactions": ECCI_NUMERICAL_FEATURES,
}
PROJECT_CATEGORICAL_FEATURES = {
    "Transaction Fraud Detection": TFD_CATEGORICAL_FEATURES,
    "Estimated Time of Arrival": ETA_CATEGORICAL_FEATURES,
    "E-Commerce Customer Interactions": ECCI_CATEGORICAL_FEATURES,
}
PROJECT_ALL_FEATURES = {
    "Transaction Fraud Detection": TFD_ALL_FEATURES,
    "Estimated Time of Arrival": ETA_ALL_FEATURES,
    "E-Commerce Customer Interactions": ECCI_ALL_FEATURES,
}
PROJECT_CAT_FEATURE_INDICES = {
    "Transaction Fraud Detection": TFD_CAT_FEATURE_INDICES,
    "Estimated Time of Arrival": ETA_CAT_FEATURE_INDICES,
    "E-Commerce Customer Interactions": ECCI_CAT_FEATURE_INDICES,
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

    Handles different data formats:
    - Classification/Regression (TFD, ETA): X_train, X_test, y_train, y_test parquet files
    - Clustering (ECCI): X_scaled.parquet with cluster_label column

    Args:
        project_name: MLflow experiment name
        run_id: Optional specific run ID. If None, uses get_best_mlflow_run().

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names) or None if not found.
    """
    from sklearn.model_selection import train_test_split

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

        # Check if this is a clustering project (ECCI)
        task_type = PROJECT_TASK_TYPES.get(project_name)
        if task_type == "clustering":
            # ECCI clustering format: X_scaled.parquet with cluster_label column
            try:
                X_scaled_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="training_data/X_scaled.parquet",
                )
                df = pd.read_parquet(X_scaled_path)
                # Extract cluster labels and features
                if 'cluster_label' in df.columns:
                    y = df['cluster_label']
                    X = df.drop(columns=['cluster_label'])
                else:
                    # Fallback: use all columns as features, no labels
                    X = df
                    y = pd.Series([0] * len(df))
                feature_names = X.columns.tolist()
                # Split for YellowBrick visualizers that need train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                print(f"  Loaded ECCI clustering data: X_train={X_train.shape}, X_test={X_test.shape}")
                return X_train, X_test, y_train, y_test, feature_names
            except Exception as e:
                print(f"Error loading ECCI clustering data: {e}")
                return None

        # Standard format for classification/regression (TFD, ETA)
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
    elif project_name == "Estimated Time of Arrival":
        return _load_eta_data_duckdb(sample_frac, max_rows)
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


def _load_eta_data_duckdb(
    sample_frac: float | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load Estimated Time of Arrival data with pure DuckDB SQL.

    From notebook 017_sklearn_duckdb_sql_regression.ipynb

    Single SQL query does:
    - Timestamp extraction (→ year, month, day, hour, minute, second)
    - Label encoding via DENSE_RANK() - 1 for categorical features
    - Column selection and ordering (matches ETA_ALL_FEATURES order)
    - Sampling (if requested)

    Target: simulated_actual_travel_time_seconds (regression)
    """
    delta_path = DELTA_PATHS["Estimated Time of Arrival"]
    # Build the SQL query - all preprocessing in one pass
    # Column order MUST match ETA_ALL_FEATURES for correct cat_feature_indices
    query = f"""
    SELECT
        -- Numerical features (unchanged)
        estimated_distance_km,
        temperature_celsius,
        driver_rating,
        hour_of_day,
        initial_estimated_travel_time_seconds,
        debug_traffic_factor,
        debug_weather_factor,
        debug_incident_delay_seconds,
        debug_driver_factor,

        -- Categorical features: Label encoded with DENSE_RANK() - 1
        -- This produces 0-indexed integers compatible with all ML tools
        DENSE_RANK() OVER (ORDER BY trip_id) - 1 AS trip_id,
        DENSE_RANK() OVER (ORDER BY driver_id) - 1 AS driver_id,
        DENSE_RANK() OVER (ORDER BY vehicle_id) - 1 AS vehicle_id,
        DENSE_RANK() OVER (ORDER BY origin) - 1 AS origin,
        DENSE_RANK() OVER (ORDER BY destination) - 1 AS destination,
        DENSE_RANK() OVER (ORDER BY weather) - 1 AS weather,
        DENSE_RANK() OVER (ORDER BY day_of_week) - 1 AS day_of_week,
        DENSE_RANK() OVER (ORDER BY vehicle_type) - 1 AS vehicle_type,

        -- Timestamp components (already integers)
        CAST(date_part('year', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS year,
        CAST(date_part('month', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS month,
        CAST(date_part('day', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS day,
        CAST(date_part('hour', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS hour,
        CAST(date_part('minute', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS minute,
        CAST(date_part('second', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS second,

        -- Target (continuous - travel time in seconds)
        simulated_actual_travel_time_seconds

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
        print(f"Loading ETA data via DuckDB SQL (single-pass preprocessing)...")
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
    y = df["simulated_actual_travel_time_seconds"]
    X = df.drop("simulated_actual_travel_time_seconds", axis=1)
    # All columns are now numeric (integers from DENSE_RANK or float64)
    print(f"  All features numeric: {X.select_dtypes(include=['number']).shape[1]}/{X.shape[1]} columns")
    # Metadata for CatBoost
    metadata = {
        "numerical_features": ETA_NUMERICAL_FEATURES,
        "categorical_features": ETA_CATEGORICAL_FEATURES,
        "cat_feature_indices": ETA_CAT_FEATURE_INDICES,
        "feature_names": ETA_ALL_FEATURES,
    }
    print(f"  Features: {len(ETA_NUMERICAL_FEATURES)} numerical, {len(ETA_CATEGORICAL_FEATURES)} label-encoded")
    print(f"  Categorical indices for CatBoost: {ETA_CAT_FEATURE_INDICES}")
    return X, y, metadata


def load_ecci_event_data_duckdb(
    sample_frac: float | None = None,
    max_rows: int | None = None,
    include_search_queries: bool = False,
) -> tuple[pd.DataFrame, list[str]] | tuple[pd.DataFrame, list[str], list[str]]:
    """Load E-Commerce Customer Interactions data with event-level approach.

    From notebook 018_sklearn_duckdb_sql_clustering.ipynb (event-level version)

    Uses the same DENSE_RANK encoding pattern as TFD/ETA for consistency.
    Each row is an individual event (not aggregated by customer).

    Single SQL query does:
    - DENSE_RANK label encoding for categorical features
    - Timestamp extraction (year, month, day, hour, minute, second)
    - JSON extraction (device_info → browser, os)
    - NULL handling with COALESCE
    - Sampling (if requested)

    Args:
        sample_frac: Fraction of data to sample (0.0-1.0)
        max_rows: Maximum number of rows to load
        include_search_queries: If True, also returns search queries for text analysis

    Returns:
        If include_search_queries=False: (DataFrame, feature_names)
        If include_search_queries=True: (DataFrame, feature_names, search_queries)
    """
    delta_path = DELTA_PATHS["E-Commerce Customer Interactions"]

    # Build the SQL query - event-level with DENSE_RANK encoding (like TFD/ETA)
    # Column order MUST match ECCI_ALL_FEATURES for consistency
    query = f"""
    SELECT
        -- Numerical features
        COALESCE(price, 0) AS price,
        COALESCE(quantity, 0) AS quantity,
        COALESCE(session_event_sequence, 0) AS session_event_sequence,
        COALESCE(time_on_page_seconds, 0) AS time_on_page_seconds,

        -- Categorical features: Label encoded with DENSE_RANK() - 1
        -- Produces 0-indexed integers compatible with all ML tools
        DENSE_RANK() OVER (ORDER BY event_type) - 1 AS event_type,
        DENSE_RANK() OVER (ORDER BY COALESCE(product_category, 'unknown')) - 1 AS product_category,
        DENSE_RANK() OVER (ORDER BY COALESCE(product_id, 'unknown')) - 1 AS product_id,
        DENSE_RANK() OVER (ORDER BY COALESCE(referrer_url, 'unknown')) - 1 AS referrer_url,
        DENSE_RANK() OVER (ORDER BY COALESCE(device_info->>'browser', 'unknown')) - 1 AS browser,
        DENSE_RANK() OVER (ORDER BY COALESCE(device_info->>'os', 'unknown')) - 1 AS os,

        -- Timestamp components (already integers)
        CAST(date_part('year', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS year,
        CAST(date_part('month', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS month,
        CAST(date_part('day', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS day,
        CAST(date_part('hour', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS hour,
        CAST(date_part('minute', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS minute,
        CAST(date_part('second', CAST(timestamp AS TIMESTAMP)) AS INTEGER) AS second

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
        print(f"Loading ECCI data via DuckDB SQL (event-level, DENSE_RANK encoding)...")
        if sample_frac:
            print(f"  Sampling: {sample_frac * 100}%")
        if max_rows:
            print(f"  Max rows: {max_rows}")
        conn = _get_duckdb_connection()
        df = conn.execute(query).df()
        print(f"  Loaded {len(df)} events with {len(df.columns)} columns")
    except Exception as e:
        print(f"DuckDB query failed, attempting reconnect: {e}")
        conn = _get_duckdb_connection(force_reconnect=True)
        df = conn.execute(query).df()
        print(f"  Loaded {len(df)} events with {len(df.columns)} columns")

    # Feature names match ECCI_ALL_FEATURES order
    feature_names = ECCI_ALL_FEATURES

    # All columns are now numeric (integers from DENSE_RANK or float64)
    print(f"  All features numeric: {df.select_dtypes(include=['number']).shape[1]}/{df.shape[1]} columns")
    print(f"  Features: {len(ECCI_NUMERICAL_FEATURES)} numerical, {len(ECCI_CATEGORICAL_FEATURES)} label-encoded")

    if not include_search_queries:
        return df, feature_names

    # Load ALL search queries for text analysis (NO LIMIT - full dataset)
    # Using same DuckDB connection that's already connected to Delta Lake
    search_queries_df = None
    try:
        search_query = f"""
        SELECT DISTINCT search_query
        FROM delta_scan('{delta_path}')
        WHERE search_query IS NOT NULL
          AND search_query != ''
          AND LENGTH(search_query) > 2
        """
        search_queries_df = conn.execute(search_query).df()
        print(f"  Loaded {len(search_queries_df)} unique search queries for text analysis (full dataset)")
    except Exception as e:
        print(f"  Warning: Could not load search queries: {e}")
        search_queries_df = pd.DataFrame(columns=['search_query'])

    return df, feature_names, search_queries_df


def get_ecci_label_encodings() -> dict[str, dict[str, int]]:
    """Get DENSE_RANK label encodings for ECCI categorical features.

    Queries the Delta table to get the value -> integer mappings that match
    the DENSE_RANK() - 1 encoding used in load_ecci_event_data_duckdb().

    Returns:
        Dict mapping feature_name -> {value: encoded_int}
        Example: {"event_type": {"page_view": 0, "add_to_cart": 1, ...}}
    """
    delta_path = DELTA_PATHS["E-Commerce Customer Interactions"]

    # Features that need encoding (exclude timestamp components which are already integers)
    features_to_encode = ["event_type", "product_category", "product_id", "referrer_url"]
    device_features = ["browser", "os"]

    encodings = {}
    conn = _get_duckdb_connection()

    # Get encodings for direct categorical features
    for feature in features_to_encode:
        query = f"""
        SELECT DISTINCT
            COALESCE({feature}, 'unknown') AS value,
            DENSE_RANK() OVER (ORDER BY COALESCE({feature}, 'unknown')) - 1 AS encoded
        FROM delta_scan('{delta_path}')
        ORDER BY encoded
        """
        try:
            result = conn.execute(query).df()
            encodings[feature] = dict(zip(result['value'].astype(str), result['encoded'].astype(int)))
            print(f"  {feature}: {len(encodings[feature])} unique values")
        except Exception as e:
            print(f"  Warning: Could not encode {feature}: {e}")
            encodings[feature] = {}

    # Get encodings for device_info JSON fields
    for feature in device_features:
        query = f"""
        SELECT DISTINCT
            COALESCE(device_info->>'{feature}', 'unknown') AS value,
            DENSE_RANK() OVER (ORDER BY COALESCE(device_info->>'{feature}', 'unknown')) - 1 AS encoded
        FROM delta_scan('{delta_path}')
        ORDER BY encoded
        """
        try:
            result = conn.execute(query).df()
            encodings[feature] = dict(zip(result['value'].astype(str), result['encoded'].astype(int)))
            print(f"  {feature}: {len(encodings[feature])} unique values")
        except Exception as e:
            print(f"  Warning: Could not encode {feature}: {e}")
            encodings[feature] = {}

    return encodings


def process_batch_data_duckdb(
    project_name: str,
    sample_frac: float | None = None,
    max_rows: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Process batch data using DuckDB SQL (unified for all projects).

    Combines DuckDB SQL preprocessing with sklearn's train/test split.
    - Classification (TFD): Stratified split to maintain class balance
    - Regression (ETA): Random split (stratification not applicable)
    - Clustering (ECCI): Random split (no target variable)

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

    # Determine split strategy based on task type
    task_type = PROJECT_TASK_TYPES.get(project_name, "classification")

    if task_type == "classification":
        # Stratified split for classification (keeps class balance)
        print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test (stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
        # Calculate class balance
        positive_rate = y_train.sum() / len(y_train) * 100
        print(f"  Positive class rate in training set: {positive_rate:.2f}%")
    else:
        # Random split for regression/clustering (stratification not applicable)
        print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test (random)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )
        if task_type == "regression":
            # Show target statistics for regression
            print(f"  Target mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

    # Free memory
    del X, y
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test, metadata


def process_sklearn_sample(x: dict, project_name: str) -> pd.DataFrame:
    """Process a single sample for CatBoost prediction.

    For real-time predictions, we process individual samples in Python
    using the same feature order as DuckDB batch training.

    Note: No StandardScaler needed since CatBoost (tree-based) doesn't require it.
    Feature order MUST match *_ALL_FEATURES for correct predictions.
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
    elif project_name == "Estimated Time of Arrival":
        # Extract timestamp components
        timestamp = pd.to_datetime(x.get("timestamp"))
        # Build feature dict in correct order (matches ETA_ALL_FEATURES)
        features = {
            # Numerical features
            "estimated_distance_km": x.get("estimated_distance_km"),
            "temperature_celsius": x.get("temperature_celsius"),
            "driver_rating": x.get("driver_rating"),
            "hour_of_day": x.get("hour_of_day"),
            "initial_estimated_travel_time_seconds": x.get("initial_estimated_travel_time_seconds"),
            "debug_traffic_factor": x.get("debug_traffic_factor"),
            "debug_weather_factor": x.get("debug_weather_factor"),
            "debug_incident_delay_seconds": x.get("debug_incident_delay_seconds"),
            "debug_driver_factor": x.get("debug_driver_factor"),
            # Categorical features (will be label encoded by model)
            "trip_id": x.get("trip_id"),
            "driver_id": x.get("driver_id"),
            "vehicle_id": x.get("vehicle_id"),
            "origin": x.get("origin"),
            "destination": x.get("destination"),
            "weather": x.get("weather"),
            "day_of_week": x.get("day_of_week"),
            "vehicle_type": x.get("vehicle_type"),
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
        for col in ETA_CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df
    elif project_name == "E-Commerce Customer Interactions":
        # Extract device_info JSON
        device_info = x.get("device_info", "{}")
        if isinstance(device_info, str):
            device_info = orjson.loads(device_info)
        # Extract timestamp components
        timestamp = pd.to_datetime(x.get("timestamp"))
        # Build feature dict in correct order (matches ECCI_ALL_FEATURES)
        # Categorical features are kept as strings here - apply_ecci_label_encodings()
        # should be called after this to convert them to integers
        features = {
            # Numerical features
            "price": float(x.get("price", 0) or 0),
            "quantity": int(x.get("quantity", 0) or 0),
            "session_event_sequence": int(x.get("session_event_sequence", 0) or 0),
            "time_on_page_seconds": int(x.get("time_on_page_seconds", 0) or 0),
            # Categorical features (will be encoded by apply_ecci_label_encodings)
            "event_type": str(x.get("event_type") or "unknown"),
            "product_category": str(x.get("product_category") or "unknown"),
            "product_id": str(x.get("product_id") or "unknown"),
            "referrer_url": str(x.get("referrer_url") or "unknown"),
            "browser": str(device_info.get("browser") or "unknown"),
            "os": str(device_info.get("os") or "unknown"),
            # Timestamp components (already integers)
            "year": timestamp.year,
            "month": timestamp.month,
            "day": timestamp.day,
            "hour": timestamp.hour,
            "minute": timestamp.minute,
            "second": timestamp.second,
        }
        df = pd.DataFrame([features])
        return df
    else:
        raise ValueError(f"Unsupported project: {project_name}")


def apply_ecci_label_encodings(df: pd.DataFrame, label_encodings: dict) -> pd.DataFrame:
    """Apply DENSE_RANK label encodings to ECCI categorical features.

    Converts string categorical values to integers matching the DENSE_RANK
    encoding used during training. Unknown values default to 0.

    Args:
        df: DataFrame with string categorical columns
        label_encodings: Dict of {feature_name: {value: encoded_int}}

    Returns:
        DataFrame with categorical columns converted to integers
    """
    df = df.copy()

    # Features that need encoding (exclude timestamp components)
    features_to_encode = ["event_type", "product_category", "product_id", "referrer_url", "browser", "os"]

    for feature in features_to_encode:
        if feature in df.columns and feature in label_encodings:
            encoding_map = label_encodings[feature]
            # Convert string to encoded integer, default to 0 for unknown
            df[feature] = df[feature].apply(
                lambda v: encoding_map.get(str(v), 0)
            )

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


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
    elif project_name == "Estimated Time of Arrival":
        # CatBoostRegressor for ETA prediction
        # From notebook 017_sklearn_duckdb_sql_regression.ipynb
        y_train = kwargs.get("y_train")
        if y_train is not None:
            print(f"Target statistics: mean={y_train.mean():.2f}s, std={y_train.std():.2f}s")
            print(f"Target range: [{y_train.min():.0f}s, {y_train.max():.0f}s]")
        model = CatBoostRegressor(
            # Core parameters
            iterations=1000,                # Max trees; early stopping finds optimal
            learning_rate=0.05,             # Good balance for large datasets
            depth=6,                        # CatBoost default
            # Loss function & evaluation for regression
            loss_function='RMSE',           # Root Mean Squared Error
            eval_metric='MAE',              # Mean Absolute Error (interpretable)
            # Regularization
            l2_leaf_reg=3,                  # L2 regularization (default=3)
            # Boosting type: 'Plain' for large datasets (1M+)
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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
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


# =============================================================================
# Sklearn-compatible Wrappers for CatBoostRegressor (YellowBrick compatibility)
# From notebook 017_sklearn_duckdb_sql_regression.ipynb
# =============================================================================

class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wraps CatBoost regressor to make it sklearn-compatible for YellowBrick.

    Use for visualizers that need a pre-fitted model:
    - ResidualsPlot
    - PredictionError
    - FeatureImportances
    """
    _estimator_type = 'regressor'

    def __init__(self, model):
        self.model = model
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

    def score(self, X, y):
        """Return R2 score (default sklearn regression metric)."""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


class CatBoostRegressorWrapperCV(BaseEstimator, RegressorMixin):
    """CatBoost regressor wrapper that supports cross-validation.

    Use for CV-based visualizers that need to re-fit the model:
    - CVScores
    - LearningCurve
    - ValidationCurve
    """
    _estimator_type = 'regressor'

    def __init__(self, iterations=500, depth=6, learning_rate=0.05,
                 l2_leaf_reg=3, random_state=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function='RMSE',
            random_seed=self.random_state,
            verbose=0  # Suppress output during CV
        )
        self.model_.fit(X, y)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model_.predict(X).flatten()

    def score(self, X, y):
        """Return R2 score (default sklearn regression metric)."""
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'l2_leaf_reg': self.l2_leaf_reg,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


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
# YellowBrick Regression Visualizers
# Reference: https://www.scikit-yb.org/en/latest/api/regressor/index.html
# From notebook 017_sklearn_duckdb_sql_regression.ipynb
# =============================================================================

def yellowbrick_regression_kwargs(
    project_name: str,
    metric_name: str,
) -> dict:
    """Get kwargs for YellowBrick regression visualizers.

    Reference: https://www.scikit-yb.org/en/latest/api/regressor/index.html

    Available Regression Visualizers:
    - ResidualsPlot: Residuals vs predicted values (detect heteroscedasticity)
    - PredictionError: Actual vs predicted scatter (identity line check)

    NOT Included:
    - CooksDistance: Matplotlib compatibility issue (use_line_collection removed)
    - AlphaSelection: For Lasso/Ridge only (not applicable to CatBoost)
    """
    kwargs = {
        "ResidualsPlot": {
            "hist": True,
            "qqplot": False,
            "train_color": "#2196F3",
            "test_color": "#FF5722",
            "train_alpha": 0.75,
            "test_alpha": 0.75,
            "is_fitted": True,
        },
        "PredictionError": {
            "shared_limits": True,
            "bestfit": True,
            "identity": True,
            "alpha": 0.75,
            "is_fitted": True,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_regression_visualizers(
    yb_kwargs: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model=None,
):
    """Create and fit YellowBrick regression visualizer."""
    from yellowbrick.regressor import ResidualsPlot, PredictionError
    visualizer_map = {
        "ResidualsPlot": ResidualsPlot,
        "PredictionError": PredictionError,
    }
    for visualizer_name, params in yb_kwargs.items():
        vis_class = visualizer_map.get(visualizer_name)
        if vis_class is None:
            raise ValueError(f"Unknown visualizer: {visualizer_name}")
        if model is None:
            raise ValueError("Model required for regression visualizers")
        wrapped_model = CatBoostRegressorWrapper(model) if 'CatBoost' in type(model).__name__ else model
        clean_params = {k: v for k, v in params.items() if k != "is_fitted"}
        visualizer = vis_class(wrapped_model, **clean_params)
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
    classes: list = None,
    feature_names: list = None,
) -> dict:
    """
    Get kwargs for YellowBrick feature analysis visualizers.

    Reference: https://www.scikit-yb.org/en/latest/api/features/index.html

    Classification (TFD): All visualizers available
    Regression (ETA): ParallelCoordinates & RadViz excluded (require discrete classes)

    Available visualizers:
    | Visualizer          | Fit Method      | Classification | Regression |
    |---------------------|-----------------|----------------|------------|
    | Rank1D              | fit + transform | YES            | YES        |
    | Rank2D              | fit + transform | YES            | YES        |
    | PCA                 | fit_transform   | YES            | YES        |
    | Manifold            | fit_transform   | YES            | YES        |
    | ParallelCoordinates | fit_transform   | YES            | NO         |
    | RadViz              | fit + transform | YES            | NO         |
    | JointPlot           | fit             | YES            | YES        |
    """
    task_type = PROJECT_TASK_TYPES.get(project_name, "classification")
    is_regression = task_type == "regression"

    kwargs = {
        # Rank1D: Single feature ranking using Shapiro-Wilk normality test
        "Rank1D": {
            "algorithm": "shapiro",
            "features": feature_names,
            "orient": "h",
            "show_feature_names": True,
        },
        # Rank2D: Pairwise feature ranking (correlation matrix)
        "Rank2D": {
            "algorithm": "pearson",
            "features": feature_names,
            "colormap": "RdBu_r",
            "show_feature_names": True,
        },
        # PCA: Principal Component Analysis projection
        "PCA": {
            "scale": True,
            "projection": 2,
            "proj_features": False,
            "alpha": 0.75,
            "heatmap": False,
        } if is_regression else {
            "scale": True,
            "projection": 2,
            "proj_features": False,
            "classes": classes,
            "alpha": 0.75,
            "heatmap": False,
        },
        # Manifold: Non-linear dimensionality reduction (SLOW)
        "Manifold": {
            "manifold": "tsne",
            "n_neighbors": 10,
            "projection": 2,
            "alpha": 0.75,
            "random_state": 42,
        } if is_regression else {
            "manifold": "tsne",
            "n_neighbors": 10,
            "classes": classes,
            "projection": 2,
            "alpha": 0.75,
            "random_state": 42,
            "target_type": "discrete",
        },
        # JointPlot: 2D correlation between two features
        "JointPlot": {
            "columns": feature_names[:2] if feature_names and len(feature_names) >= 2 else None,
            "correlation": "pearson",
            "kind": "scatter",
            "hist": True,
            "alpha": 0.65,
        },
    }

    # Classification-only visualizers (require discrete class labels)
    if not is_regression:
        kwargs["ParallelCoordinates"] = {
            "classes": classes,
            "features": feature_names,
            "normalize": "minmax",
            "sample": 0.05,
            "shuffle": True,
            "alpha": 0.3,
            "fast": True,
        }
        kwargs["RadViz"] = {
            "classes": classes,
            "features": feature_names,
            "alpha": 0.5,
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

    Classification (TFD):
    - ClassBalance: Class distribution (shows imbalance)
    - FeatureCorrelation: mutual_info-classification
    - FeatureCorrelation_Pearson: Pearson correlation

    Regression (ETA):
    - FeatureCorrelation: mutual_info-regression
    - FeatureCorrelation_Pearson: Pearson correlation
    - BalancedBinningReference: Target distribution binning
    """
    task_type = PROJECT_TASK_TYPES.get(project_name, "classification")
    is_regression = task_type == "regression"
    is_clustering = task_type == "clustering"

    kwargs = {}

    # ClassBalance: Classification and Clustering only (not regression)
    if not is_regression:
        if is_clustering:
            # Clustering: dynamic labels for clusters
            # Generate colors for n clusters using a color palette
            cluster_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
                              "#1abc9c", "#e67e22", "#34495e", "#7f8c8d", "#16a085"]
            n_clusters = len(labels) if labels else 3
            kwargs["ClassBalance"] = {
                "labels": labels if labels else [f"Cluster {i}" for i in range(n_clusters)],
                "colors": cluster_colors[:n_clusters],
            }
        else:
            # Classification: binary fraud detection labels (TFD)
            kwargs["ClassBalance"] = {
                "labels": labels if labels else ["Non-Fraud", "Fraud"],
                "colors": ["#2ecc71", "#e74c3c"],
            }

    # FeatureCorrelation with Mutual Information (method depends on task type)
    kwargs["FeatureCorrelation"] = {
        "method": "mutual_info-regression" if is_regression else "mutual_info-classification",
        "labels": feature_names,
        "sort": True,
        "color": "#3498db",
    }

    # FeatureCorrelation with Pearson (works for both)
    kwargs["FeatureCorrelation_Pearson"] = {
        "method": "pearson",
        "labels": feature_names,
        "sort": True,
        "color": "#9b59b6",
    }

    # BalancedBinningReference (useful for regression)
    target_col = PROJECT_TARGET_COLUMNS.get(project_name, "target")
    kwargs["BalancedBinningReference"] = {
        "target": target_col,
        "bins": 10 if is_regression else 4,
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

    Classification (TFD): Uses 'f1' scoring metric
    Regression (ETA): Uses 'r2' scoring metric

    All 6 visualizers:
    - FeatureImportances: Feature ranking by importance (FAST)
    - CVScores: Cross-validation scores bar chart (MODERATE)
    - ValidationCurve: Hyperparameter tuning visualization (SLOW)
    - LearningCurve: Training size vs performance (SLOW)
    - RFECV: Recursive feature elimination with CV (VERY SLOW)
    - DroppingCurve: Feature subset random selection (SLOW)
    """
    task_type = PROJECT_TASK_TYPES.get(project_name, "classification")
    is_regression = task_type == "regression"
    is_clustering = task_type == "clustering"

    # Scoring metric based on task type
    # - Regression: r2 (coefficient of determination)
    # - Clustering: accuracy (predicting cluster membership with RandomForest)
    # - Classification: f1 (harmonic mean of precision and recall)
    if is_regression:
        scoring = "r2"
    elif is_clustering:
        scoring = "accuracy"
    else:
        scoring = "f1"

    kwargs = {
        "FeatureImportances": {
            "labels": feature_names,
            "relative": True,
            "absolute": False,
            "is_fitted": True,
        },
        "CVScores": {
            "cv": 5,
            "scoring": scoring,
        },
        "ValidationCurve": {
            "param_name": "iterations",
            "param_range": np.array([50, 100, 150, 200]),
            "cv": 3,
            "scoring": scoring,
        },
        "LearningCurve": {
            "train_sizes": np.linspace(0.1, 1.0, 5),
            "cv": 3,
            "scoring": scoring,
            "random_state": 42,
        },
        "RFECV": {
            "cv": 3,
            "scoring": scoring,
            "step": 1,
        },
        "DroppingCurve": {
            "feature_sizes": np.linspace(0.1, 1.0, 5),
            "cv": 3,
            "scoring": scoring,
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
    model=None,
    project_name: str = "Transaction Fraud Detection",
):
    """Create and fit YellowBrick model selection visualizer.

    Reference: https://www.scikit-yb.org/en/latest/api/model_selection/index.html

    CatBoost Handling (selects wrapper based on task type):
    - Classification: CatBoostWrapper/CatBoostWrapperCV
    - Regression: CatBoostRegressorWrapper/CatBoostRegressorWrapperCV

    Clustering Handling (ECCI):
    - Uses RandomForestClassifier trained on cluster labels as pseudo-targets
    - This reveals which features are most predictive of cluster membership
    """
    from yellowbrick.model_selection import (
        FeatureImportances,
        CVScores,
        ValidationCurve,
        LearningCurve,
        RFECV,
        DroppingCurve,
    )
    from sklearn.ensemble import RandomForestClassifier

    visualizer_map = {
        "FeatureImportances": FeatureImportances,
        "CVScores": CVScores,
        "ValidationCurve": ValidationCurve,
        "LearningCurve": LearningCurve,
        "RFECV": RFECV,
        "DroppingCurve": DroppingCurve,
    }

    # Determine wrapper type based on task
    task_type = PROJECT_TASK_TYPES.get(project_name, "classification")
    is_regression = task_type == "regression"
    is_clustering = task_type == "clustering"

    # Combine data for CV-based visualizers
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    for visualizer_name, params in yb_kwargs.items():
        vis_class = visualizer_map.get(visualizer_name)
        if vis_class is None:
            raise ValueError(f"Unknown visualizer: {visualizer_name}")

        if is_clustering:
            # For clustering: use RandomForestClassifier on cluster labels
            # This reveals which features are most predictive of cluster membership
            if visualizer_name == "FeatureImportances":
                # Train RandomForest on cluster labels
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                visualizer = vis_class(rf_model, **params)
                visualizer.fit(X_train, y_train)
            else:
                # CV-based visualizers use fresh RandomForest
                rf_cv = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
                # Adjust params for RandomForest (no 'iterations' param like CatBoost)
                cv_params = {k: v for k, v in params.items() if k not in ("param_name", "param_range")}
                # Override scoring to 'accuracy' for clustering
                cv_params["scoring"] = "accuracy"
                if visualizer_name == "ValidationCurve":
                    # Use max_depth for RandomForest (tests tree complexity)
                    cv_params["param_name"] = "max_depth"
                    cv_params["param_range"] = np.array([2, 4, 6, 8, 10])
                visualizer = vis_class(rf_cv, **cv_params)
                visualizer.fit(X_full, y_full)
            return visualizer

        elif visualizer_name == "FeatureImportances":
            if model is None:
                raise ValueError("Model required for FeatureImportances")
            # Select wrapper based on task type
            if 'CatBoost' in type(model).__name__:
                wrapped_estimator = CatBoostRegressorWrapper(model) if is_regression else CatBoostWrapper(model)
            else:
                wrapped_estimator = model
            visualizer = vis_class(wrapped_estimator, **params)
            visualizer.fit(X_train, y_train)
        else:
            # CV-based visualizers need unfitted wrapper
            if is_regression:
                cv_wrapper = CatBoostRegressorWrapperCV(iterations=100, depth=6, learning_rate=0.1)
            else:
                cv_wrapper = CatBoostWrapperCV(iterations=100, depth=6, learning_rate=0.1)
            visualizer = vis_class(cv_wrapper, **params)
            visualizer.fit(X_full, y_full)
        return visualizer
    return None


# =============================================================================
# YellowBrick Clustering Visualizers (ECCI)
# Reference: https://www.scikit-yb.org/en/latest/api/cluster/index.html
# =============================================================================
def yellowbrick_clustering_kwargs(
    project_name: str,
    metric_name: str,
    n_clusters: int = 5,
) -> dict:
    """Get kwargs for YellowBrick clustering visualizers.

    Reference: https://www.scikit-yb.org/en/latest/api/cluster/index.html

    Visualizers:
    - KElbowVisualizer: Find optimal K via elbow method (MODERATE)
    - SilhouetteVisualizer: Per-cluster silhouette scores (FAST)
    - InterclusterDistance: Cluster separation visualization (MODERATE)
    """
    kwargs = {
        # KElbowVisualizer - Find optimal K using elbow method
        "KElbowVisualizer": {
            "k": (2, 12),
            "metric": "silhouette",
            "timings": False,
            "locate_elbow": True,
            "force_model": True,  # Required for sklearn 1.4+ compatibility
        },
        # SilhouetteVisualizer - Per-cluster silhouette scores
        "SilhouetteVisualizer": {
            "colors": "yellowbrick",
            "is_fitted": True,  # Use pre-trained model
            "force_model": True,  # Required for sklearn 1.4+ compatibility
        },
        # InterclusterDistance - Cluster separation (MDS/t-SNE)
        "InterclusterDistance": {
            "embedding": "mds",
            "legend": True,
            "force_model": True,  # Required for sklearn 1.4+ compatibility
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_clustering_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series = None,
    model=None,
    n_clusters: int = 5,
):
    """Create and fit YellowBrick clustering visualizer.

    Reference: https://www.scikit-yb.org/en/latest/api/cluster/index.html

    Fit methods:
    - KElbowVisualizer: fit(X) - trains fresh models for each K
    - SilhouetteVisualizer: fit(X, is_fitted=True) - uses pre-trained model
    - InterclusterDistance: fit(X) - uses pre-trained model
    """
    from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
    from sklearn.cluster import KMeans

    visualizer_map = {
        "KElbowVisualizer": KElbowVisualizer,
        "SilhouetteVisualizer": SilhouetteVisualizer,
        "InterclusterDistance": InterclusterDistance,
    }

    for visualizer_name, params in yb_kwargs.items():
        vis_class = visualizer_map.get(visualizer_name)
        if vis_class is None:
            raise ValueError(f"Unknown clustering visualizer: {visualizer_name}")

        if visualizer_name == "KElbowVisualizer":
            # KElbow must create fresh models to test different K values
            visualizer = vis_class(KMeans(random_state=42), **params)
            visualizer.fit(X)
        elif visualizer_name == "SilhouetteVisualizer":
            # Use pre-trained model with is_fitted=True
            if model is None:
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(X)
            kw_copy = params.copy()
            is_fitted = kw_copy.pop("is_fitted", True)
            visualizer = vis_class(model, is_fitted=is_fitted, **kw_copy)
            visualizer.fit(X)
        elif visualizer_name == "InterclusterDistance":
            # Use pre-trained model
            if model is None:
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(X)
            visualizer = vis_class(model, **params)
            visualizer.fit(X)
        else:
            visualizer = vis_class(**params)
            visualizer.fit(X)
        return visualizer
    return None


# =============================================================================
# YellowBrick Text Analysis Visualizers (ECCI)
# Reference: https://www.scikit-yb.org/en/latest/api/text/index.html
# =============================================================================
def yellowbrick_text_analysis_kwargs(
    project_name: str,
    metric_name: str,
) -> dict:
    """Get kwargs for YellowBrick text analysis visualizers.

    Reference: https://www.scikit-yb.org/en/latest/api/text/index.html

    Visualizers:
    - FreqDistVisualizer: Token frequency distribution (FAST)
    - TSNEVisualizer: t-SNE document clustering (SLOW)
    - UMAPVisualizer: UMAP document clustering (MODERATE, optional)
    - DispersionPlot: Word dispersion across documents (FAST)
    - WordCorrelationPlot: Word correlation matrix (FAST)
    - PosTagVisualizer: POS tag distribution (FAST, requires NLTK)
    """
    kwargs = {
        # FreqDistVisualizer - Token frequency distribution
        "FreqDistVisualizer": {
            "n": 50,
            "orient": "h",
            "color": "#3498db",
        },
        # TSNEVisualizer - t-SNE document clustering
        "TSNEVisualizer": {
            "decompose": "svd",
            "decompose_by": 50,
            "random_state": 42,
            "colormap": "viridis",
        },
        # UMAPVisualizer - UMAP document clustering
        "UMAPVisualizer": {
            "random_state": 42,
            "colormap": "plasma",
            "metric": "cosine",
        },
        # DispersionPlot - Word dispersion across documents
        # DispersionPlot - Word dispersion across documents
        # NOTE: Requires TOKENIZED documents (list of word lists)
        "DispersionPlot": {
            "annotate_docs": False,
            "ignore_case": True,
            "colormap": "coolwarm",
        },
        # WordCorrelationPlot - Word correlation matrix
        "WordCorrelationPlot": {
            "colormap": "RdBu",
        },
        # PosTagVisualizer - POS tag distribution
        # PosTagVisualizer - POS tag distribution
        # Uses parser='nltk' to automatically parse raw text
        "PosTagVisualizer": {
            "parser": "nltk",
            "frequency": True,  # Show frequency not raw counts
            "colormap": "tab20",
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_text_analysis_visualizers(
    yb_kwargs: dict,
    search_queries: list[str],
    cluster_labels: np.ndarray = None,
):
    """Create and fit YellowBrick text analysis visualizer.

    Reference: https://www.scikit-yb.org/en/latest/api/text/index.html

    Args:
        yb_kwargs: Visualizer kwargs dict
        search_queries: List of search query strings
        cluster_labels: Optional cluster labels for coloring

    Fit methods per visualizer:
    - FreqDistVisualizer: fit(word_counts) - requires CountVectorizer
    - TSNEVisualizer: fit(tfidf_matrix, cluster_labels)
    - UMAPVisualizer: fit(tfidf_matrix, cluster_labels)
    - DispersionPlot: fit(tokenized_docs)
    - WordCorrelationPlot: fit(raw_docs)
    - PosTagVisualizer: fit(raw_docs)
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # Import text visualizers (handle optional dependencies)
    try:
        from yellowbrick.text import FreqDistVisualizer, TSNEVisualizer, DispersionPlot, WordCorrelationPlot
        FREQ_AVAILABLE = True
    except ImportError:
        FREQ_AVAILABLE = False
        raise ImportError("YellowBrick text module not available")

    try:
        from yellowbrick.text import UMAPVisualizer
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False

    try:
        from yellowbrick.text import PosTagVisualizer
        import nltk
        # Ensure required NLTK data is downloaded for PosTagVisualizer
        # From notebook 018_sklearn_duckdb_sql_clustering.ipynb
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        try:
            nltk.data.find('corpora/treebank')
        except LookupError:
            nltk.download('treebank', quiet=True)
        try:
            nltk.data.find('taggers/universal_tagset')
        except LookupError:
            nltk.download('universal_tagset', quiet=True)
        POS_AVAILABLE = True
    except ImportError:
        POS_AVAILABLE = False

    # Filter empty queries
    valid_queries = [q for q in search_queries if q and q.strip()]
    if len(valid_queries) < 10:
        raise ValueError(f"Insufficient search queries ({len(valid_queries)}). Need at least 10 for visualization.")

    # Log dataset size
    print(f"Text analysis: {len(valid_queries)} valid queries available")

    for visualizer_name, params in yb_kwargs.items():
        if visualizer_name == "FreqDistVisualizer":
            # Vectorize text for frequency distribution - FULL dataset (fast)
            vectorizer = CountVectorizer(
                max_features=params.get("n", 50),
                stop_words="english",
                min_df=2
            )
            word_counts = vectorizer.fit_transform(valid_queries)
            vocab = vectorizer.get_feature_names_out()
            visualizer = FreqDistVisualizer(features=vocab, **params)
            visualizer.fit(word_counts)
            return visualizer

        elif visualizer_name == "TSNEVisualizer":
            # TF-IDF for t-SNE - limited to 2000 samples (slow algorithm)
            sample_size = min(len(valid_queries), 2000)
            sample_queries = valid_queries[:sample_size]
            sample_labels = cluster_labels[:sample_size] if cluster_labels is not None else None

            tfidf = TfidfVectorizer(max_features=100, stop_words="english", min_df=2)
            tfidf_matrix = tfidf.fit_transform(sample_queries)
            visualizer = TSNEVisualizer(**params)
            visualizer.fit(tfidf_matrix, sample_labels)
            return visualizer

        elif visualizer_name == "UMAPVisualizer":
            if not UMAP_AVAILABLE:
                raise ImportError("UMAPVisualizer requires umap-learn: pip install umap-learn")
            # TF-IDF for UMAP - limited to 2000 samples (slow algorithm)
            sample_size = min(len(valid_queries), 2000)
            sample_queries = valid_queries[:sample_size]
            sample_labels = cluster_labels[:sample_size] if cluster_labels is not None else None

            tfidf = TfidfVectorizer(max_features=100, stop_words="english", min_df=2)
            tfidf_matrix = tfidf.fit_transform(sample_queries)
            visualizer = UMAPVisualizer(**params)
            visualizer.fit(tfidf_matrix, sample_labels)
            return visualizer

        elif visualizer_name == "DispersionPlot":
            # Tokenize documents and find target words - FULL dataset (fast)
            from collections import Counter
            tokenized_docs = [q.lower().split() for q in valid_queries]

            # Build a set of all unique words that actually appear in tokenized docs
            all_words_in_corpus = set()
            for doc in tokenized_docs:
                all_words_in_corpus.update(doc)

            # Get word frequencies from tokenized docs
            all_tokens = [word for doc in tokenized_docs for word in doc]
            token_freq = Counter(all_tokens)

            # Get top frequent words that:
            # 1. Have length > 2
            # 2. Appear at least 3 times
            # 3. VERIFIED to exist in the corpus (important for DispersionPlot)
            target_words = []
            for word, count in token_freq.most_common(50):
                if len(word) > 2 and count >= 3 and word in all_words_in_corpus:
                    target_words.append(word)
                    if len(target_words) >= 10:
                        break

            if len(target_words) < 3:
                raise ValueError(f"Insufficient vocabulary for DispersionPlot. Found {len(target_words)} verified words, need at least 3.")

            visualizer = DispersionPlot(target_words, **params)
            visualizer.fit(tokenized_docs)
            return visualizer

        elif visualizer_name == "WordCorrelationPlot":
            # Find most common words for correlation - FULL dataset (fast)
            word_freq = {}
            for q in valid_queries:
                for word in q.lower().split():
                    if len(word) > 2:
                        word_freq[word] = word_freq.get(word, 0) + 1
            target_words = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
            if len(target_words) < 3:
                raise ValueError("Insufficient vocabulary for WordCorrelationPlot")
            visualizer = WordCorrelationPlot(words=target_words, **params)
            visualizer.fit(valid_queries)
            return visualizer

        elif visualizer_name == "PosTagVisualizer":
            if not POS_AVAILABLE:
                raise ImportError("PosTagVisualizer requires NLTK: pip install nltk")
            # Use raw text - limited to 1000 samples (slow NLTK processing)
            sample_size = min(len(valid_queries), 1000)
            visualizer = PosTagVisualizer(**params)
            visualizer.fit(valid_queries[:sample_size])
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
