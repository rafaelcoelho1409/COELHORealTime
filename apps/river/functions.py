"""
River ML Training Helper Functions

Functions for incremental machine learning training using River library.
Reads data from Delta Lake on MinIO (S3-compatible storage) via Polars.
SQL queries use native DuckDB delta extension for direct Delta Lake access.
Models and encoders are loaded from MLflow artifacts (with local fallback).
"""
import pickle
import os
import sys
import time
import tempfile
from typing import Any, Dict, Hashable, Optional, List
from kafka import (
    KafkaConsumer,
    TopicPartition
)
from kafka.errors import NoBrokersAvailable
import json
import datetime as dt
import pandas as pd
import polars as pl
import deltalake
import mlflow
from river import (
    base,
    compose,
    metrics,
    drift,
    forest,
    cluster,
    preprocessing,
    time_series,
    linear_model,
)


KAFKA_HOST = os.environ["KAFKA_HOST"]
# MinIO (S3-compatible) configuration for Delta Lake
MINIO_HOST = os.environ["MINIO_HOST"]
MINIO_ENDPOINT = f"http://{MINIO_HOST}:9000"
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]
# Delta Lake storage options for Polars/deltalake
DELTA_STORAGE_OPTIONS = {
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_ACCESS_KEY_ID": MINIO_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": MINIO_SECRET_KEY,
    "AWS_REGION": "us-east-1",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    "AWS_ALLOW_HTTP": "true",
}
# Configuration
KAFKA_BROKERS = f'{KAFKA_HOST}:9092'
# Delta Lake paths (using s3:// for Polars, not s3a://)
DELTA_PATHS = {
    "Transaction Fraud Detection": "s3://lakehouse/delta/transaction_fraud_detection",
    "Estimated Time of Arrival": "s3://lakehouse/delta/estimated_time_of_arrival",
    "E-Commerce Customer Interactions": "s3://lakehouse/delta/e_commerce_customer_interactions",
    "Sales Forecasting": "s3://lakehouse/delta/sales_forecasting",
}
# MLflow configuration
MLFLOW_HOST = os.environ.get("MLFLOW_HOST", "localhost")
mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
# Model name mapping for MLflow
MLFLOW_MODEL_NAMES = {
    "Transaction Fraud Detection": "ARFClassifier",
    "Estimated Time of Arrival": "ARFRegressor",
    "E-Commerce Customer Interactions": "DBSTREAM",
    "Sales Forecasting": "SNARIMAX",
}
# Encoder artifact names
ENCODER_ARTIFACT_NAMES = {
    "Transaction Fraud Detection": "transaction_fraud_detection.pkl",
    "Estimated Time of Arrival": "estimated_time_of_arrival.pkl",
    "E-Commerce Customer Interactions": "e_commerce_customer_interactions.pkl",
    "Sales Forecasting": "sales_forecasting.pkl",
}
# Kafka offset artifact name (stores last processed offset for continuation)
KAFKA_OFFSET_ARTIFACT = "kafka_offset.json"
# Best metric selection criteria per project
# Format: {"metric_name": str, "maximize": bool}
# - maximize = True: higher is better (e.g., FBeta, F1, MCC, ROCAUC)
# - maximize = False: lower is better (e.g., MAE, RMSE, LogLoss)
#
# TFD uses FBeta (beta=2.0) as primary metric:
#   - Prioritizes Recall over Precision (catching fraud is more important)
#   - Industry standard for fraud detection
#   - Alternative: MCC (best for imbalanced data) or ROCAUC (probability-based)
BEST_METRIC_CRITERIA = {
    "Transaction Fraud Detection": {"metric_name": "FBeta", "maximize": True},
    "Estimated Time of Arrival": {"metric_name": "MAE", "maximize": False},
    "E-Commerce Customer Interactions": None,  # Clustering - no metrics, use latest
    "Sales Forecasting": {"metric_name": "MAE", "maximize": False},
}

# =============================================================================
# Static Dropdown Options (mirrors Kafka producer constants for data validity)
# These ensure form values are always valid for /predict endpoint
# =============================================================================
STATIC_DROPDOWN_OPTIONS = {
    "Transaction Fraud Detection": {
        # Categorical fields (exact values from transaction_fraud_detection.py producer)
        "currency": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "BRL"],
        "transaction_type": ["purchase", "withdrawal", "transfer", "payment", "deposit"],
        "payment_method": ["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"],
        "product_category": [
            "electronics", "clothing", "groceries", "travel", "services",
            "digital_goods", "luxury_items", "gambling", "other"
        ],
        "browser": ["Chrome", "Safari", "Firefox", "Edge", "Opera", "Other"],
        "os": ["iOS", "Android", "Windows", "macOS", "Linux", "Other"],
        # ID field - bounded range from producer: merchant_{1..200}
        "merchant_id": [f"merchant_{i}" for i in range(1, 201)],
    },
    "Estimated Time of Arrival": {
        # Categorical fields (exact values from estimated_time_of_arrival.py producer)
        "weather": ["Clear", "Clouds", "Rain", "Heavy Rain", "Fog", "Thunderstorm"],
        "vehicle_type": ["Sedan", "SUV", "Hatchback", "Motorcycle", "Van"],
        # ID fields - bounded ranges from producer
        "driver_id": [f"driver_{i}" for i in range(1000, 5001)],
        "vehicle_id": [f"vehicle_{i}" for i in range(100, 1000)],
    },
    "E-Commerce Customer Interactions": {
        # Categorical fields (exact values from e_commerce_customer_interactions.py producer)
        "event_type": ["page_view", "add_to_cart", "purchase", "search", "leave_review"],
        "product_category": [
            "Electronics", "Fashion & Apparel", "Home & Garden", "Beauty & Personal Care",
            "Sports & Outdoors", "Books", "Grocery & Gourmet Food", "Automotive",
            "Toys & Games", "Computers", "Pet Supplies", "Health & Household"
        ],
        "browser": ["Chrome", "Safari", "Firefox", "Edge", "Opera", "Other"],
        "device_type": ["Mobile", "Desktop", "Tablet"],
        "os": ["Android", "iOS", "Windows", "macOS", "Linux", "Other"],
        "referrer_url": [
            "direct", "google.com", "facebook.com", "amazon.com", "instagram.com",
            "twitter.com", "youtube.com", "tiktok.com", "pinterest.com", "reddit.com",
            "linkedin.com", "bing.com", "yahoo.com", "email_campaign", "affiliate_link"
        ],
        # ID field - bounded range from producer: prod_{1000..1100}
        "product_id": [f"prod_{i}" for i in range(1000, 1101)],
    },
}


def get_static_dropdown_options(project_name: str) -> Dict[str, List[str]]:
    """Get static dropdown options for a project.

    These are pre-defined values that mirror the Kafka producer constants,
    ensuring all form values are valid for the /predict endpoint.
    Instant access - no I/O or database queries needed.
    """
    return STATIC_DROPDOWN_OPTIONS.get(project_name, {})


# =============================================================================
# Redis Live Model Cache (for real-time predictions during training)
# =============================================================================
import redis

# Redis connection (lazy initialization)
_redis_client: Optional[redis.Redis] = None
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
# Redis key prefixes
REDIS_LIVE_MODEL_PREFIX = "live_model"
REDIS_LIVE_ENCODER_PREFIX = "live_encoder"
REDIS_TRAINING_STATUS_PREFIX = "training_status"
# Live model cache interval (save to Redis every N messages)
REDIS_CACHE_INTERVAL = 100


def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client (lazy initialization with connection pooling)."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=False,  # Keep binary for pickle
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
            )
            # Test connection
            _redis_client.ping()
            print(f"Connected to Redis at {REDIS_URL}")
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
            _redis_client = None
    return _redis_client


def save_live_model_to_redis(
    project_name: str,
    model_name: str,
    model: Any,
    encoders: Dict[str, Any],
    ttl_seconds: int = 3600,
) -> bool:
    """Save live model and encoders to Redis for real-time predictions.

    Called by training scripts every REDIS_CACHE_INTERVAL messages.
    TTL ensures stale models are automatically cleaned up if training crashes.

    Args:
        project_name: Project name (e.g., "Transaction Fraud Detection")
        model_name: Model name (e.g., "ARFClassifier")
        model: The trained River model
        encoders: Dict of encoders (ordinal_encoder, etc.)
        ttl_seconds: Time-to-live for cache entries (default: 1 hour)

    Returns:
        True if saved successfully, False otherwise
    """
    client = get_redis_client()
    if client is None:
        return False
    try:
        # Create Redis keys
        model_key = f"{REDIS_LIVE_MODEL_PREFIX}:{project_name}:{model_name}"
        encoder_key = f"{REDIS_LIVE_ENCODER_PREFIX}:{project_name}:{model_name}"
        status_key = f"{REDIS_TRAINING_STATUS_PREFIX}:{project_name}:{model_name}"
        # Serialize and save model
        model_bytes = pickle.dumps(model)
        client.setex(model_key, ttl_seconds, model_bytes)
        # Serialize and save encoders
        encoder_bytes = pickle.dumps(encoders)
        client.setex(encoder_key, ttl_seconds, encoder_bytes)
        # Update training status with timestamp
        status_data = json.dumps({
            "active": True,
            "last_update": dt.datetime.now().isoformat(),
            "project_name": project_name,
            "model_name": model_name,
        })
        client.setex(status_key, ttl_seconds, status_data.encode())
        return True
    except Exception as e:
        print(f"Error saving live model to Redis: {e}")
        return False


def load_live_model_from_redis(
    project_name: str,
    model_name: str,
) -> Optional[tuple[Any, Dict[str, Any]]]:
    """Load live model and encoders from Redis.

    Called by /predict endpoint when training is active.

    Args:
        project_name: Project name
        model_name: Model name

    Returns:
        Tuple of (model, encoders) if found, None otherwise
    """
    client = get_redis_client()
    if client is None:
        return None
    try:
        model_key = f"{REDIS_LIVE_MODEL_PREFIX}:{project_name}:{model_name}"
        encoder_key = f"{REDIS_LIVE_ENCODER_PREFIX}:{project_name}:{model_name}"
        # Load model
        model_bytes = client.get(model_key)
        if model_bytes is None:
            return None
        model = pickle.loads(model_bytes)
        # Load encoders
        encoder_bytes = client.get(encoder_key)
        encoders = pickle.loads(encoder_bytes) if encoder_bytes else {}
        return model, encoders
    except Exception as e:
        print(f"Error loading live model from Redis: {e}")
        return None


def is_training_active(project_name: str, model_name: str) -> bool:
    """Check if training is currently active for a project/model.

    Args:
        project_name: Project name
        model_name: Model name

    Returns:
        True if training is active (live model exists in Redis)
    """
    client = get_redis_client()
    if client is None:
        return False
    try:
        status_key = f"{REDIS_TRAINING_STATUS_PREFIX}:{project_name}:{model_name}"
        status_bytes = client.get(status_key)
        if status_bytes is None:
            return False
        status = json.loads(status_bytes.decode())
        return status.get("active", False)
    except Exception as e:
        print(f"Error checking training status: {e}")
        return False


def clear_live_model_from_redis(project_name: str, model_name: str) -> bool:
    """Clear live model from Redis when training stops.

    Called when training is stopped gracefully.

    Args:
        project_name: Project name
        model_name: Model name

    Returns:
        True if cleared successfully
    """
    client = get_redis_client()
    if client is None:
        return False
    try:
        model_key = f"{REDIS_LIVE_MODEL_PREFIX}:{project_name}:{model_name}"
        encoder_key = f"{REDIS_LIVE_ENCODER_PREFIX}:{project_name}:{model_name}"
        status_key = f"{REDIS_TRAINING_STATUS_PREFIX}:{project_name}:{model_name}"
        client.delete(model_key, encoder_key, status_key)
        print(f"Cleared live model from Redis: {project_name}/{model_name}")
        return True
    except Exception as e:
        print(f"Error clearing live model from Redis: {e}")
        return False


# =============================================================================
# MLflow Artifact Loading Functions
# =============================================================================
def get_latest_mlflow_run(project_name: str, model_name: str) -> Optional[str]:
    """Get the latest MLflow run ID for a project and model."""
    try:
        experiment = mlflow.get_experiment_by_name(project_name)
        if experiment is None:
            print(f"No MLflow experiment found for {project_name}")
            return None
        runs_df = mlflow.search_runs(
            experiment_ids = [experiment.experiment_id],
            max_results = 100,
            order_by = ["start_time DESC"]
        )
        if runs_df.empty:
            print(f"No runs found in MLflow for {project_name}")
            return None
        # Filter by model name
        filtered_runs = runs_df[runs_df["tags.mlflow.runName"] == model_name]
        if filtered_runs.empty:
            print(f"No runs found for model {model_name} in {project_name}")
            return None
        return filtered_runs.iloc[0]["run_id"]
    except Exception as e:
        print(f"Error getting MLflow run for {project_name}/{model_name}: {e}")
        return None


def _run_has_model_artifact(run_id: str, model_name: str) -> bool:
    """Check if a run has the model artifact."""
    try:
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        return f"{model_name}.pkl" in artifact_names
    except Exception:
        return False


def get_best_mlflow_run(project_name: str, model_name: str) -> Optional[str]:
    """Get the MLflow run ID with the best metrics for a project and model.

    Uses BEST_METRIC_CRITERIA to determine which metric to optimize:
    - For fraud detection (TFD): maximize FBeta (beta=2.0, prioritizes recall)
    - For regression (ETA, Sales): minimize MAE
    - For clustering (ECCI): use latest run (no metrics yet)

    Only selects runs that have the model artifact saved.
    Returns None if no runs exist (new model will be created from scratch).
    """
    try:
        experiment = mlflow.get_experiment_by_name(project_name)
        if experiment is None:
            print(f"No MLflow experiment found for {project_name}")
            return None
        runs_df = mlflow.search_runs(
            experiment_ids = [experiment.experiment_id],
            max_results = 1000,  # Get more runs to find the best
            order_by = ["start_time DESC"],
            filter_string = "attributes.status = 'FINISHED'"  # Only completed runs with full artifacts
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
            # No metric criteria (e.g., clustering) - use latest run with artifacts
            print(f"No metric criteria for {project_name}, using latest run with artifacts")
            for _, row in filtered_runs.iterrows():
                if _run_has_model_artifact(row["run_id"], model_name):
                    return row["run_id"]
            print(f"No runs with model artifact found for {project_name}")
            return None
        metric_name = criteria["metric_name"]
        maximize = criteria["maximize"]
        metric_column = f"metrics.{metric_name}"
        # Check if metric column exists in the runs
        if metric_column not in filtered_runs.columns:
            print(f"Metric {metric_name} not found in runs for {project_name}, using latest run with artifacts")
            for _, row in filtered_runs.iterrows():
                if _run_has_model_artifact(row["run_id"], model_name):
                    return row["run_id"]
            return None
        # Filter out runs without the metric
        runs_with_metric = filtered_runs[filtered_runs[metric_column].notna()]
        if runs_with_metric.empty:
            print(f"No runs with metric {metric_name} for {project_name}, using latest run with artifacts")
            for _, row in filtered_runs.iterrows():
                if _run_has_model_artifact(row["run_id"], model_name):
                    return row["run_id"]
            return None
        # Sort runs by metric (best first)
        ascending = not maximize
        sorted_runs = runs_with_metric.sort_values(by=metric_column, ascending=ascending)
        # Find the best run that has model artifacts
        for _, row in sorted_runs.iterrows():
            run_id = row["run_id"]
            metric_value = row[metric_column]
            if _run_has_model_artifact(run_id, model_name):
                print(f"Best run for {project_name}/{model_name}: {run_id} "
                      f"({metric_name} = {metric_value:.4f}, maximize = {maximize})")
                return run_id
            else:
                print(f"Skipping run {run_id} ({metric_name} = {metric_value:.4f}) - no model artifact")
        print(f"No runs with model artifact found for {project_name}/{model_name}")
        return None
    except Exception as e:
        print(f"Error getting best MLflow run for {project_name}/{model_name}: {e}")
        return None


def load_model_from_mlflow(project_name: str, model_name: str) -> Optional[Any]:
    """Load model from the best MLflow run based on metrics.
    Uses get_best_mlflow_run to find the run with optimal metrics,
    allowing new training to continue from the best existing model.
    """
    try:
        run_id = get_best_mlflow_run(project_name, model_name)
        if run_id is None:
            return None
        # Download model artifact
        artifact_path = f"{model_name}.pkl"
        local_path = mlflow.artifacts.download_artifacts(
            run_id = run_id,
            artifact_path = artifact_path
        )
        with open(local_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from best MLflow run: {project_name}/{model_name} (run_id={run_id})")
        return model
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return None


def load_encoders_from_mlflow(project_name: str) -> Optional[Dict]:
    """Load encoders from the best MLflow run based on metrics.
    Uses get_best_mlflow_run to find the run with optimal metrics,
    ensuring encoders are loaded from the same run as the best model.
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
            run_id = run_id,
            artifact_path = artifact_name
        )
        with open(local_path, 'rb') as f:
            encoders = pickle.load(f)
        print(f"Encoders loaded from best MLflow run: {project_name} (run_id={run_id})")
        return encoders
    except Exception as e:
        print(f"Error loading encoders from MLflow: {e}")
        return None


def load_kafka_offset_from_mlflow(project_name: str) -> Optional[int]:
    """Load Kafka offset from MLflow runs.
    Strategy:
    1. First try to load from the best run (same source as model/encoders)
    2. If best run doesn't have offset, search all FINISHED runs for highest offset

    This handles the case where the best model was trained with old code
    that didn't save offsets, but newer runs do have offset data.

    Returns None if no offset is stored (new training from scratch).
    """
    try:
        model_name = MLFLOW_MODEL_NAMES.get(project_name)
        if not model_name:
            print(f"Unknown project for offset loading: {project_name}")
            return None
        # First, try the best run (same as model/encoder source)
        best_run_id = get_best_mlflow_run(project_name, model_name)
        if best_run_id is not None:
            try:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id = best_run_id,
                    artifact_path = KAFKA_OFFSET_ARTIFACT
                )
                with open(local_path, 'r') as f:
                    offset_data = json.load(f)
                offset = offset_data.get("last_offset")
                if offset is not None:
                    print(f"Kafka offset loaded from best run: {project_name} offset={offset} (run_id={best_run_id})")
                    return offset
            except Exception as e:
                print(f"Best run doesn't have offset artifact, searching other runs...")
        # Fallback: search all FINISHED runs for the highest offset
        experiment = mlflow.get_experiment_by_name(project_name)
        if experiment is None:
            print(f"No MLflow experiment found for {project_name}, starting from offset 0")
            return None
        runs_df = mlflow.search_runs(
            experiment_ids = [experiment.experiment_id],
            max_results = 100,
            order_by = ["start_time DESC"],
            filter_string = "attributes.status = 'FINISHED'"
        )
        if runs_df.empty:
            print(f"No FINISHED runs found for {project_name}, starting from offset 0")
            return None
        # Search runs for highest offset
        highest_offset = None
        source_run_id = None
        for _, row in runs_df.iterrows():
            run_id = row["run_id"]
            try:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id = run_id,
                    artifact_path = KAFKA_OFFSET_ARTIFACT
                )
                with open(local_path, 'r') as f:
                    offset_data = json.load(f)
                offset = offset_data.get("last_offset")
                if offset is not None and (highest_offset is None or offset > highest_offset):
                    highest_offset = offset
                    source_run_id = run_id
            except Exception:
                # This run doesn't have offset artifact, continue searching
                continue
        if highest_offset is not None:
            print(f"Kafka offset loaded from run with highest offset: {project_name} offset={highest_offset} (run_id={source_run_id})")
            return highest_offset
        print(f"No runs with offset artifact found for {project_name}, starting from offset 0")
        return None
    except Exception as e:
        print(f"Error loading Kafka offset from MLflow for {project_name}: {e}")
        return None


# =============================================================================
# Optimized Delta Lake Queries via Polars (Lazy Evaluation)
# =============================================================================
def get_delta_lazyframe(project_name: str) -> Optional[pl.LazyFrame]:
    """Get a lazy Polars LazyFrame for a Delta Lake table.
    LazyFrame doesn't load data until .collect() is called.
    """
    delta_path = DELTA_PATHS.get(project_name)
    if not delta_path:
        print(f"Unknown project: {project_name}")
        return None
    try:
        return pl.scan_delta(delta_path, storage_options = DELTA_STORAGE_OPTIONS)
    except Exception as e:
        print(f"Error loading Delta table for {project_name}: {e}")
        return None


def get_unique_values_polars(project_name: str, column_name: str, limit: int = 100) -> List[str]:
    """Get unique values for a column using Polars (optimized with lazy evaluation).
    Polars will only read the necessary column and compute distinct values efficiently.
    """
    try:
        lf = get_delta_lazyframe(project_name)
        if lf is None:
            return []
        # Optimized query: select only the column, get unique, limit
        # Polars pushes this down for efficient execution
        unique_df = (
            lf.select(pl.col(column_name).cast(pl.Utf8))
            .filter(pl.col(column_name).is_not_null())
            .unique()
            .limit(limit)
            .collect()
        )
        return unique_df[column_name].to_list()
    except Exception as e:
        print(f"Error getting unique values via Polars for {column_name}: {e}")
        return []


def get_sample_polars(project_name: str, n: int = 1) -> Optional[pd.DataFrame]:
    """Get a random sample using Polars (optimized)."""
    try:
        lf = get_delta_lazyframe(project_name)
        if lf is None:
            return None
        # Collect to DataFrame then sample randomly
        df = lf.collect()
        if df.is_empty():
            return None
        # Use Polars sample for true random sampling
        sample_df = df.sample(n = min(n, len(df)))
        return sample_df.to_pandas()
    except Exception as e:
        print(f"Error getting sample via Polars: {e}")
        return None


def precompute_all_unique_values_polars(project_name: str) -> Dict[str, List[str]]:
    """Precompute unique values for all columns of a project using Polars.
    This runs once at startup and caches results for instant access.
    """
    try:
        lf = get_delta_lazyframe(project_name)
        if lf is None:
            return {}
        # Get schema to know column names
        schema = lf.collect_schema()
        columns = schema.names()
        unique_values_cache = {}
        for col_name in columns:
            try:
                unique_df = (
                    lf.select(pl.col(col_name).cast(pl.Utf8))
                    .filter(pl.col(col_name).is_not_null())
                    .unique()
                    .limit(100)
                    .collect()
                )
                unique_values_cache[col_name] = unique_df[col_name].to_list()
            except Exception as e:
                print(f"Error getting unique values for column {col_name}: {e}")
                unique_values_cache[col_name] = []
        print(f"Precomputed unique values for {project_name}: {len(columns)} columns")
        return unique_values_cache
    except Exception as e:
        print(f"Error precomputing unique values for {project_name}: {e}")
        return {}


def get_initial_sample_polars(project_name: str) -> Optional[dict]:
    """Get a single sample row as a dictionary using Polars."""
    try:
        lf = get_delta_lazyframe(project_name)
        if lf is None:
            return None
        # Get first row - very fast operation
        row_df = lf.limit(1).collect()
        if row_df.height > 0:
            return row_df.row(0, named = True)
        return None
    except Exception as e:
        print(f"Error getting initial sample via Polars: {e}")
        return None


# =============================================================================
# DuckDB SQL Query Execution (Delta Lake SQL Tab)
# =============================================================================
import duckdb
import re

# SQL query configuration
SQL_MAX_ROWS = 10000
SQL_DEFAULT_LIMIT = 100
SQL_TIMEOUT_SECONDS = 30.0
SQL_MAX_QUERY_LENGTH = 5000

# Blocked SQL keywords (DDL, DML mutations - security)
BLOCKED_SQL_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL", "MERGE", "REPLACE",
    "COPY", "VACUUM", "ATTACH", "DETACH", "PRAGMA", "SET", "LOAD",
    "INSTALL", "UNINSTALL"
}

def _sanitize_for_json(data: list[dict]) -> list[dict]:
    """
    Sanitize data for JSON serialization.
    Replaces NaN, inf, -inf with None (null in JSON).
    """
    import math
    for row in data:
        for key, value in row.items():
            if value is None:
                continue
            try:
                if isinstance(value, float):
                    if math.isnan(value) or math.isinf(value):
                        row[key] = None
            except (TypeError, ValueError):
                pass
    return data


# Persistent DuckDB connection for Delta Lake queries
# Configured once at module load, reused for all queries (fast)
_duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None


def _get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Get or create a persistent DuckDB connection configured for MinIO."""
    global _duckdb_conn

    if _duckdb_conn is not None:
        try:
            # Test if connection is still valid
            _duckdb_conn.execute("SELECT 1")
            return _duckdb_conn
        except Exception:
            # Connection is stale, recreate it
            _duckdb_conn = None

    # Create new connection and configure it
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


def validate_sql_query(query: str) -> tuple[bool, str]:
    """
    Validate SQL query for security.
    Returns (is_valid, error_message).

    Security checks:
    1. Must start with SELECT
    2. No blocked keywords (DDL/DML mutations)
    3. No multiple statements (semicolons)
    4. No SQL comments
    5. Query length limits
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"

    query_stripped = query.strip()
    query_upper = query_stripped.upper()

    # 1. Must start with SELECT
    if not query_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed"

    # 2. Check for blocked keywords
    # Split by non-word characters to get tokens
    tokens = set(re.findall(r'\b[A-Z]+\b', query_upper))
    blocked_found = tokens.intersection(BLOCKED_SQL_KEYWORDS)
    if blocked_found:
        return False, f"Forbidden keyword(s): {', '.join(sorted(blocked_found))}"

    # 3. Check for multiple statements (semicolons followed by content)
    # Allow trailing semicolon, but not multiple statements
    if re.search(r';\s*\S', query_stripped):
        return False, "Multiple statements are not allowed"

    # 4. Check for SQL comments (could hide malicious code)
    if '--' in query_stripped or '/*' in query_stripped:
        return False, "SQL comments are not allowed"

    # 5. Query length limit
    if len(query_stripped) > SQL_MAX_QUERY_LENGTH:
        return False, f"Query too long (max {SQL_MAX_QUERY_LENGTH} characters)"

    # 6. Check for balanced parentheses
    if query_stripped.count('(') != query_stripped.count(')'):
        return False, "Unbalanced parentheses"

    return True, ""


def _enforce_limit(query: str, max_limit: int = SQL_MAX_ROWS) -> str:
    """Ensure query has a LIMIT clause and it doesn't exceed max_limit."""
    query_upper = query.upper()

    # Remove trailing semicolon for processing
    query_clean = query.rstrip().rstrip(';')

    if "LIMIT" not in query_upper:
        # No LIMIT clause, add one
        return f"{query_clean} LIMIT {max_limit}"

    # Has LIMIT, check if it exceeds max
    # Match LIMIT followed by number
    limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
    if limit_match:
        current_limit = int(limit_match.group(1))
        if current_limit > max_limit:
            # Replace with max_limit
            return re.sub(
                r'LIMIT\s+\d+',
                f'LIMIT {max_limit}',
                query_clean,
                flags=re.IGNORECASE
            )

    return query_clean




def execute_delta_sql(
    project_name: str,
    query: str,
    limit: int = SQL_DEFAULT_LIMIT
) -> dict:
    """
    Execute SQL query against Delta Lake using native DuckDB delta extension.

    Uses DuckDB's native delta_scan() to directly read Delta Lake tables,
    avoiding intermediate copies through Polars.
    The table is available as 'data' via a CTE for query simplicity.

    Args:
        project_name: Project name (determines which Delta table to query)
        query: SQL query string (must be SELECT only)
        limit: Default row limit if not specified in query

    Returns:
        dict with: columns, data, row_count, execution_time_ms, truncated, error
    """
    start_time = time.time()

    # 1. Validate query
    is_valid, error = validate_sql_query(query)
    if not is_valid:
        return {
            "columns": [],
            "data": [],
            "row_count": 0,
            "execution_time_ms": (time.time() - start_time) * 1000,
            "truncated": False,
            "error": error
        }

    # 2. Get Delta table path
    delta_path = DELTA_PATHS.get(project_name)
    if not delta_path:
        return {
            "columns": [],
            "data": [],
            "row_count": 0,
            "execution_time_ms": (time.time() - start_time) * 1000,
            "truncated": False,
            "error": f"Unknown project: {project_name}"
        }

    # 3. Enforce row limit
    effective_limit = min(limit, SQL_MAX_ROWS)
    query_with_limit = _enforce_limit(query, effective_limit)

    # 4. Execute query with native DuckDB delta extension
    try:
        # Get persistent DuckDB connection (already configured for MinIO)
        conn = _get_duckdb_connection()

        # Create a CTE that makes the Delta table available as 'data'
        # This allows users to write simple queries like: SELECT * FROM data
        full_query = f"""
        WITH data AS (
            SELECT * FROM delta_scan('{delta_path}')
        )
        {query_with_limit}
        """

        print(f"Executing DuckDB query for {project_name}: {query_with_limit[:100]}...")

        # Execute the query
        result = conn.execute(full_query).fetchdf()

        print(f"Query returned {len(result)} rows")

        execution_time = (time.time() - start_time) * 1000

        # Check if result was truncated
        truncated = len(result) >= effective_limit

        # Convert to list of dicts and sanitize for JSON
        data = _sanitize_for_json(result.to_dict(orient="records"))

        return {
            "columns": list(result.columns),
            "data": data,
            "row_count": len(result),
            "execution_time_ms": round(execution_time, 2),
            "truncated": truncated,
            "error": None,
            "engine": "duckdb"
        }
    except Exception as e:
        error_msg = f"Query execution error: {str(e)}"
        print(f"DuckDB error for {project_name}: {error_msg}")
        return {
            "columns": [],
            "data": [],
            "row_count": 0,
            "execution_time_ms": (time.time() - start_time) * 1000,
            "truncated": False,
            "error": error_msg,
            "engine": "duckdb"
        }


def execute_delta_sql_polars(
    project_name: str,
    query: str,
    limit: int = SQL_DEFAULT_LIMIT
) -> dict:
    """
    Execute SQL query against Delta Lake using Polars SQLContext.

    Polars loads the Delta Lake table lazily and executes SQL via its built-in SQL engine.
    The table is available as 'data' in your SQL queries.

    Args:
        project_name: Project name (determines which Delta table to query)
        query: SQL query string (must be SELECT only)
        limit: Default row limit if not specified in query

    Returns:
        dict with: columns, data, row_count, execution_time_ms, truncated, error, engine
    """
    start_time = time.time()

    # 1. Validate query
    is_valid, error = validate_sql_query(query)
    if not is_valid:
        return {
            "columns": [],
            "data": [],
            "row_count": 0,
            "execution_time_ms": (time.time() - start_time) * 1000,
            "truncated": False,
            "error": error,
            "engine": "polars"
        }

    # 2. Get Delta table path
    delta_path = DELTA_PATHS.get(project_name)
    if not delta_path:
        return {
            "columns": [],
            "data": [],
            "row_count": 0,
            "execution_time_ms": (time.time() - start_time) * 1000,
            "truncated": False,
            "error": f"Unknown project: {project_name}",
            "engine": "polars"
        }

    # 3. Enforce row limit
    effective_limit = min(limit, SQL_MAX_ROWS)
    query_with_limit = _enforce_limit(query, effective_limit)

    # 4. Execute query with Polars SQLContext
    try:
        print(f"Executing Polars SQL query for {project_name}: {query_with_limit[:100]}...")

        # Load Delta table as LazyFrame
        lf = pl.scan_delta(delta_path, storage_options=DELTA_STORAGE_OPTIONS)

        # Create SQLContext and register the table as 'data'
        ctx = pl.SQLContext()
        ctx.register("data", lf)

        # Execute SQL query (returns LazyFrame)
        result_lf = ctx.execute(query_with_limit)

        # Collect results
        result_df = result_lf.collect()

        print(f"Polars query returned {len(result_df)} rows")

        execution_time = (time.time() - start_time) * 1000

        # Check if result was truncated
        truncated = len(result_df) >= effective_limit

        # Convert to list of dicts and sanitize for JSON
        data = _sanitize_for_json(result_df.to_dicts())

        return {
            "columns": result_df.columns,
            "data": data,
            "row_count": len(result_df),
            "execution_time_ms": round(execution_time, 2),
            "truncated": truncated,
            "error": None,
            "engine": "polars"
        }
    except Exception as e:
        error_msg = f"Query execution error: {str(e)}"
        print(f"Polars error for {project_name}: {error_msg}")
        return {
            "columns": [],
            "data": [],
            "row_count": 0,
            "execution_time_ms": (time.time() - start_time) * 1000,
            "truncated": False,
            "error": error_msg,
            "engine": "polars"
        }


def get_delta_table_schema(project_name: str) -> dict:
    """
    Get schema and metadata for a Delta Lake table using native DuckDB.

    Returns:
        dict with: table_name, delta_path, columns (name, type, nullable),
                   approximate_row_count, error
    """
    delta_path = DELTA_PATHS.get(project_name)
    if not delta_path:
        return {
            "table_name": "",
            "delta_path": "",
            "columns": [],
            "approximate_row_count": 0,
            "error": f"Unknown project: {project_name}"
        }

    try:
        # Get persistent DuckDB connection (already configured for MinIO)
        conn = _get_duckdb_connection()

        # Get schema using DESCRIBE
        schema_query = f"DESCRIBE SELECT * FROM delta_scan('{delta_path}')"
        schema_result = conn.execute(schema_query).fetchdf()

        # Get column info from DESCRIBE result
        columns = []
        for _, row in schema_result.iterrows():
            columns.append({
                "name": row["column_name"],
                "type": row["column_type"],
                "nullable": row.get("null", "YES") == "YES"
            })

        # Get approximate row count
        try:
            count_query = f"SELECT COUNT(*) as cnt FROM delta_scan('{delta_path}')"
            count_result = conn.execute(count_query).fetchone()
            row_count = count_result[0] if count_result else 0
        except Exception:
            row_count = 0

        # Extract table name from path
        table_name = delta_path.split("/")[-1]

        return {
            "table_name": table_name,
            "delta_path": delta_path,
            "columns": columns,
            "approximate_row_count": row_count,
            "error": None
        }
    except Exception as e:
        return {
            "table_name": "",
            "delta_path": delta_path,
            "columns": [],
            "approximate_row_count": 0,
            "error": f"Failed to get schema: {str(e)}"
        }

###---Functions----####
#Data processing functions
class CustomOrdinalEncoder:
    """
    An incremental ordinal encoder that is picklable and processes dictionaries.
    Assigns a unique integer ID to each unique category encountered for each feature.
    """
    def __init__(self):
        # Dictionary to store mappings for each feature.
        # Keys are feature names (from input dictionary), values are dictionaries
        # mapping category value to integer ID for that feature.
        self._feature_mappings: Dict[Hashable, Dict[Any, int]] = {}
        # Dictionary to store the next available integer ID for each feature.
        # Keys are feature names, values are integers.
        self._feature_next_ids: Dict[Hashable, int] = {}
    def learn_one(self, x: Dict[Hashable, Any]):
        """
        Learns categories from a single sample dictionary.
        Iterates through the dictionary's items and learns each category value
        for its corresponding feature.
        Args:
            x: A dictionary representing a single sample.
               Keys are feature names, values are feature values.
               Assumes categorical features are present in this dictionary.
        """
        for feature_name, category_value in x.items():
            # Ensure the category value is hashable (dictionaries/lists are not)
            # You might need more sophisticated type checking or handling
            # if your input dictionaries contain complex unhashable types
            if not isinstance(category_value, Hashable):
                 print(f"Warning: Skipping unhashable value for feature '{feature_name}': {category_value}")
                 continue # Skip this feature for learning
            # If this is the first time we see this feature, initialize its mapping and counter
            if feature_name not in self._feature_mappings:
                self._feature_mappings[feature_name] = {}
                self._feature_next_ids[feature_name] = 0
            # Get the mapping and counter for this specific feature
            feature_map = self._feature_mappings[feature_name]
            feature_next_id = self._feature_next_ids[feature_name]
            # Check if the category value is already in the mapping for this feature
            if category_value not in feature_map:
                # If it's a new category for this feature, assign the next available ID
                feature_map[category_value] = feature_next_id
                # Increment the counter for the next new category for this feature
                self._feature_next_ids[feature_name] += 1
    def transform_one(self, x: Dict[Hashable, Any]) -> Dict[Hashable, int]:
        """
        Transforms categorical features in a single sample dictionary into integer IDs.
        Args:
            x: A dictionary representing a single sample.
               Keys are feature names, values are feature values.
        Returns:
            A new dictionary containing the transformed integer IDs for the
            categorical features that the encoder has seen. Features not
            seen by the encoder are excluded from the output dictionary.
        Raises:
            KeyError: If a feature is seen but a specific category value
                      within that feature has not been seen during learning.
                      You might want to add logic here to handle unseen categories
                      (e.g., return a default value like -1 or NaN for that feature).
        """
        transformed_sample: Dict[Hashable, int] = {}
        for feature_name, category_value in x.items():
            # Only attempt to transform features that the encoder has seen
            if feature_name in self._feature_mappings:
                feature_map = self._feature_mappings[feature_name]
                # Check if the category value for this feature has been seen
                if category_value in feature_map:
                    # Transform the category value using the feature's mapping
                    transformed_sample[feature_name] = feature_map[category_value]
                else:
                    # Handle unseen category values for a known feature
                    # By default, this will raise a KeyError as per the docstring.
                    # Example: return a placeholder value instead of raising error:
                    # transformed_sample[feature_name] = -1 # Or some other indicator
                    # print(f"Warning: Unseen category '{category_value}' for feature '{feature_name}' during transform.")
                    # Or raise the error explicitly:
                    raise KeyError(f"Unseen category '{category_value}' for feature '{feature_name}' during transform.")
            # Features not in self._feature_mappings are ignored in the output.
            # If you need to include them (e.g., original numerical features),
            # you would copy them over here. This encoder only outputs encoded features.
        return transformed_sample
    def get_feature_mappings(self) -> Dict[Hashable, Dict[Any, int]]:
        """Returns the current mappings for all features."""
        return self._feature_mappings
    def get_feature_next_ids(self) -> Dict[Hashable, int]:
        """Returns the next available IDs for all features."""
        return self._feature_next_ids
    def __repr__(self) -> str:
        """String representation of the encoder."""
        num_features = len(self._feature_mappings)
        feature_details = ", ".join([f"{name}: {len(mapping)} categories" for name, mapping in self._feature_mappings.items()])
        return f"CustomPicklableOrdinalEncoder(features={num_features} [{feature_details}])"
    
class DictImputer(base.Transformer):
    """
    Imputes missing values (None or missing keys) for specified features in a dictionary.

    Parameters
    ----------
    on
        List of feature names to impute.
    fill_value
        The value to use for imputation.
    """
    def __init__(self, on: list, fill_value):
        self.on = on
        self.fill_value = fill_value
    def transform_one(self, x: dict):
        x_transformed = x.copy()
        for feature in self.on:
            if x_transformed.get(feature) is None:
                x_transformed[feature] = self.fill_value
        return x_transformed

    

def extract_device_info(x):
    x_ = x['device_info']
    return {
        'os': x_['os'],
        'browser': x_['browser'],
    }

def extract_timestamp_info(x):
    x_ = dt.datetime.strptime(
        x['timestamp'],
        "%Y-%m-%dT%H:%M:%S.%f%z")
    return {
        'year': x_.year,
        'month': x_.month,
        'day': x_.day,
        'hour': x_.hour,
        'minute': x_.minute,
        'second': x_.second
    }

def extract_coordinates(x):
    x_ = x['location']
    return {
        'lat': x_['lat'],
        'lon': x_['lon'],
    }

def load_or_create_encoders(project_name, library="river"):
    """Load encoders from MLflow artifacts or create new ones."""
    # Load from MLflow
    encoders = load_encoders_from_mlflow(project_name)
    if encoders is not None:
        return encoders

    # No encoders in MLflow, create new ones
    print(f"No encoders in MLflow for {project_name}, creating new ones.")
    return _create_default_encoders(project_name)


def _create_default_encoders(project_name):
    """Create default encoders based on project type."""
    if project_name in ["Transaction Fraud Detection", "Estimated Time of Arrival"]:
        return {"ordinal_encoder": CustomOrdinalEncoder()}
    elif project_name == "E-Commerce Customer Interactions":
        return {
            "standard_scaler": preprocessing.StandardScaler(),
            "feature_hasher": preprocessing.FeatureHasher()
        }
    elif project_name == "Sales Forecasting":
        return {
            "one_hot_encoder": preprocessing.OneHotEncoder(),
            "standard_scaler": preprocessing.StandardScaler(),
        }
    else:
        raise ValueError(f"Unknown project: {project_name}")



def process_sample(x, encoders, project_name):
    """Process a single sample for River incremental learning."""
    if project_name == "Transaction Fraud Detection":
        pipe1 = compose.Select(
            "amount",
            "account_age_days",
            "cvv_provided",
            "billing_address_match"
        )
        pipe1.learn_one(x)
        x1 = pipe1.transform_one(x)
        pipe2 = compose.Select(
            "currency",
            "merchant_id",
            "payment_method",
            "product_category",
            "transaction_type",
        )
        pipe2.learn_one(x)
        x_pipe_2 = pipe2.transform_one(x)
        pipe3a = compose.Select("device_info")
        pipe3a.learn_one(x)
        x_pipe_3 = pipe3a.transform_one(x)
        pipe3b = compose.FuncTransformer(extract_device_info)
        pipe3b.learn_one(x_pipe_3)
        x_pipe_3 = pipe3b.transform_one(x_pipe_3)
        pipe4a = compose.Select("timestamp")
        pipe4a.learn_one(x)
        x_pipe_4 = pipe4a.transform_one(x)
        pipe4b = compose.FuncTransformer(extract_timestamp_info)
        pipe4b.learn_one(x_pipe_4)
        x_pipe_4 = pipe4b.transform_one(x_pipe_4)
        x_to_encode = x_pipe_2 | x_pipe_3 | x_pipe_4
        encoders["ordinal_encoder"].learn_one(x_to_encode)
        x2 = encoders["ordinal_encoder"].transform_one(x_to_encode)
        return x1 | x2, {"ordinal_encoder": encoders["ordinal_encoder"]}
    elif project_name == "Estimated Time of Arrival":
        pipe1 = compose.Select(
            'estimated_distance_km',
            'temperature_celsius',
            'hour_of_day',
            'driver_rating',
            'initial_estimated_travel_time_seconds',
            'debug_traffic_factor',
            'debug_weather_factor',
            'debug_incident_delay_seconds',
            'debug_driver_factor'
        )
        pipe1.learn_one(x)
        x1 = pipe1.transform_one(x)
        pipe2 = compose.Select(
            'driver_id',
            'vehicle_id',
            'weather',
            'vehicle_type'
        )
        pipe2.learn_one(x)
        x_pipe_2 = pipe2.transform_one(x)
        pipe3a = compose.Select(
            "timestamp",
        )
        pipe3a.learn_one(x)
        x_pipe_3 = pipe3a.transform_one(x)
        pipe3b = compose.FuncTransformer(
            extract_timestamp_info,
        )
        pipe3b.learn_one(x_pipe_3)
        x_pipe_3 = pipe3b.transform_one(x_pipe_3)
        x_to_encode = x_pipe_2 | x_pipe_3
        encoders["ordinal_encoder"].learn_one(x_to_encode)
        x2 = encoders["ordinal_encoder"].transform_one(x_to_encode)
        return x1 | x2, {
            "ordinal_encoder": encoders["ordinal_encoder"]
        }
    elif project_name == "E-Commerce Customer Interactions":
        pipe1 = compose.Select(
            'price',
            'quantity',
            'session_event_sequence',
            'time_on_page_seconds'
        )
        pipe1.learn_one(x)
        x1 = pipe1.transform_one(x)
        pipe2 = compose.Select(
            'event_type',
            'product_category',
            'product_id',
            'referrer_url',
        )
        pipe2.learn_one(x)
        x_pipe_2 = pipe2.transform_one(x)
        pipe3a = compose.Select(
            "device_info"
        )
        pipe3a.learn_one(x)
        x_pipe_3 = pipe3a.transform_one(x)
        pipe3b = compose.FuncTransformer(
            extract_device_info,
        )
        pipe3b.learn_one(x_pipe_3)
        x_pipe_3 = pipe3b.transform_one(x_pipe_3)
        pipe4a = compose.Select(
            "timestamp",
        )
        pipe4a.learn_one(x)
        x_pipe_4 = pipe4a.transform_one(x)
        pipe4b = compose.FuncTransformer(
            extract_timestamp_info,
        )
        pipe4b.learn_one(x_pipe_4)
        x_pipe_4 = pipe4b.transform_one(x_pipe_4)
        pipe5a = compose.Select(
            "location",
        )
        pipe5a.learn_one(x)
        x_pipe_5 = pipe5a.transform_one(x)
        pipe5b = compose.FuncTransformer(
            extract_coordinates,
        )
        pipe5b.learn_one(x_pipe_5)
        x_pipe_5 = pipe5b.transform_one(x_pipe_5)
        x_to_prep = x1 | x_pipe_2 | x_pipe_3 | x_pipe_4 | x_pipe_5
        x_to_prep = DictImputer(
            fill_value = False, 
            on = list(x_to_prep.keys())).transform_one(
                x_to_prep)
        numerical_features = [
            'price',
            'session_event_sequence',
            'time_on_page_seconds',
            'quantity'
        ]
        categorical_features = [
            'event_type',
            'product_category',
            'product_id',
            'referrer_url',
            'os',
            'browser',
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second'
        ]
        num_pipe = compose.Select(*numerical_features)
        num_pipe.learn_one(x_to_prep)
        x_num = num_pipe.transform_one(x_to_prep)
        cat_pipe = compose.Select(*categorical_features)
        cat_pipe.learn_one(x_to_prep)
        x_cat = cat_pipe.transform_one(x_to_prep)
        encoders["standard_scaler"].learn_one(x_num)
        x_scaled = encoders["standard_scaler"].transform_one(x_num)
        encoders["feature_hasher"].learn_one(x_cat)
        x_hashed = encoders["feature_hasher"].transform_one(x_cat)
        return x_scaled | x_hashed, {
            "standard_scaler": encoders["standard_scaler"], 
            "feature_hasher": encoders["feature_hasher"]
        }
    elif project_name == "Sales Forecasting":
        pipe1 = compose.Select(
            'concept_drift_stage',
            'day_of_week',
            'is_holiday',
            'is_promotion_active',
            'month',
            #'total_sales_amount',
            'unit_price'
        )
        pipe1.learn_one(x)
        x1 = pipe1.transform_one(x)
        pipe2a = compose.Select(
            "timestamp",
        )
        pipe2a.learn_one(x)
        x_pipe_2 = pipe2a.transform_one(x)
        pipe2b = compose.FuncTransformer(
            extract_timestamp_info,
        )
        pipe2b.learn_one(x_pipe_2)
        x2 = pipe2b.transform_one(x_pipe_2)
        pipe3a = compose.Select(
            'product_id',
            'promotion_id',
            'store_id'
        )
        pipe3a.learn_one(x)
        x3 = pipe3a.transform_one(x)
        x_to_process = x1 | x2 | x3
        numerical_features = [
            'unit_price',
            #'total_sales_amount',
        ]
        categorical_features = [
            'is_promotion_active',
            'is_holiday',
            'day_of_week',
            'concept_drift_stage',
            'year',
            'month',
            'day',
            #'hour',
            #'minute',
            #'second',
            'product_id',
            'promotion_id',
            'store_id',
        ]
        pipe_num = compose.Select(*numerical_features)
        pipe_num.learn_one(x_to_process)
        x_num = pipe_num.transform_one(x_to_process)
        pipe_cat = compose.Select(*categorical_features)
        pipe_cat.learn_one(x_to_process)
        x_cat = pipe_cat.transform_one(x_to_process)
        encoders["standard_scaler"].learn_one(x_num)
        x_num = encoders["standard_scaler"].transform_one(x_num)
        encoders["one_hot_encoder"].learn_one(x_cat)
        x_cat = encoders["one_hot_encoder"].transform_one(x_cat)
        return x_num | x_cat, {
            "one_hot_encoder": encoders["one_hot_encoder"],
            "standard_scaler": encoders["standard_scaler"],
        }


def load_or_create_model(project_name, model_name, folder_path=None):
    """Load model from MLflow artifacts or create a new one.

    Args:
        project_name: Name of the project (e.g., "Transaction Fraud Detection")
        model_name: Name of the model (e.g., "ARFClassifier")
        folder_path: Optional path to load model from (legacy, unused)

    Returns:
        The loaded or newly created model.
    """
    # Load from MLflow (best historical model)
    model = load_model_from_mlflow(project_name, model_name)
    if model is not None:
        return model

    # No model in MLflow, create new one
    print(f"No model in MLflow for {project_name}/{model_name}, creating new one.")
    return _create_default_model(project_name)


def _create_default_model(project_name):
    """Create default model based on project type.

    Models are configured based on River ML documentation and best practices.
    All parameters are documented with their River ML defaults and rationale.

    See: https://riverml.xyz/latest/
    """
    if project_name == "Transaction Fraud Detection":
        # =================================================================
        # ARFClassifier - Adaptive Random Forest Classifier
        # For fraud detection with concept drift handling
        # =================================================================
        # OLD CONFIGURATION:
        # return forest.ARFClassifier(
        #     n_models = 10,
        #     drift_detector = drift.ADWIN(),
        #     warning_detector = drift.ADWIN(),
        #     metric = metrics.ROCAUC(),
        #     max_features = "sqrt",
        #     lambda_value = 6,
        #     seed = 42
        # )

        # CONFIGURATION based on River ML documentation:
        # Reference: https://riverml.xyz/latest/api/forest/ARFClassifier/
        # Reference: https://riverml.xyz/latest/examples/imbalanced-learning/
        #
        # - n_models=10: Default number of trees in ensemble
        # - max_features="sqrt": Default, sqrt of features per split
        # - lambda_value=6: Default Leveraging Bagging parameter
        # - metric=ROCAUC(): RECOMMENDED by River for imbalanced fraud detection
        #   (River's imbalanced-learning guide uses ROCAUC for fraud detection)
        # - disable_weighted_vote=False: Enable weighted voting for better accuracy
        # - drift_detector ADWIN(delta=0.002): Default sensitivity (0.002)
        # - warning_detector ADWIN(delta=0.01): Default warning sensitivity
        # - grace_period=50: Default observations between split attempts
        # - max_depth=None: Default, unlimited tree depth
        # - split_criterion="info_gain": Default, information gain criterion
        # - delta=0.01: Default allowed error in split decision
        # - tau=0.05: Default tie-breaking threshold
        # - leaf_prediction="nba": Default, Naive Bayes Adaptive
        # - nb_threshold=0: Default, enable NB immediately
        # - binary_split=False: Default, allow multi-way splits
        # - min_branch_fraction=0.01: Default minimum data per branch
        # - max_share_to_split=0.99: Default majority class proportion
        # - max_size=100.0: Default max memory in MiB
        # - memory_estimate_period=2000000: Default instances between memory checks
        # - merit_preprune=True: Default merit-based pre-pruning
        return forest.ARFClassifier(
            n_models = 10,
            max_features = "sqrt",
            lambda_value = 6,
            metric = metrics.ROCAUC(),
            disable_weighted_vote = False,
            drift_detector = drift.ADWIN(delta = 0.002),
            warning_detector = drift.ADWIN(delta = 0.01),
            grace_period = 50,
            max_depth = None,
            split_criterion = "info_gain",
            delta = 0.01,
            tau = 0.05,
            leaf_prediction = "nba",
            nb_threshold = 0,
            nominal_attributes = None,
            binary_split = False,
            min_branch_fraction = 0.01,
            max_share_to_split = 0.99,
            max_size = 100.0,
            memory_estimate_period = 2000000,
            stop_mem_management = False,
            remove_poor_attrs = False,
            merit_preprune = True,
            seed = 42,
        )
    elif project_name == "Estimated Time of Arrival":
        # =================================================================
        # ARFRegressor - Adaptive Random Forest Regressor
        # For ETA prediction with continuous drift handling
        # =================================================================
        # OLD CONFIGURATION:
        # return forest.ARFRegressor(
        #     n_models = 10,
        #     drift_detector = drift.ADWIN(),
        #     warning_detector = drift.ADWIN(),
        #     metric = metrics.RMSE(),
        #     max_features = "sqrt",
        #     lambda_value = 6,
        #     seed = 42
        # )

        # CONFIGURATION based on River ML documentation:
        # Reference: https://riverml.xyz/latest/api/forest/ARFRegressor/
        #
        # - n_models=10: Default number of trees
        # - max_features="sqrt": Default feature selection
        # - aggregation_method="median": Default, robust to outliers
        # - lambda_value=6: Default Leveraging Bagging parameter
        # - metric=MAE(): Using MAE as it's common for ETA prediction
        # - disable_weighted_vote=True: Default for regressor
        # - drift_detector ADWIN(delta=0.002): Default sensitivity
        # - warning_detector ADWIN(delta=0.01): Default warning sensitivity
        # - grace_period=50: Default observations between split attempts
        # - max_depth=None: Default unlimited depth
        # - delta=0.01: Default allowed error
        # - tau=0.05: Default tie-breaking threshold
        # - leaf_prediction="adaptive": Default, dynamically chooses mean/model
        # - model_selector_decay=0.95: Default decay for leaf model selection
        # - min_samples_split=5: Default minimum samples for split
        # - binary_split=False: Default multi-way splits
        # - max_size=500.0: Default max memory in MiB
        return forest.ARFRegressor(
            n_models=10,
            max_features="sqrt",
            aggregation_method="median",
            lambda_value=6,
            metric=metrics.MAE(),
            disable_weighted_vote=True,
            drift_detector=drift.ADWIN(delta=0.002),
            warning_detector=drift.ADWIN(delta=0.01),
            grace_period=50,
            max_depth=None,
            delta=0.01,
            tau=0.05,
            leaf_prediction="adaptive",
            leaf_model=None,
            model_selector_decay=0.95,
            min_samples_split=5,
            binary_split=False,
            max_size=500.0,
            memory_estimate_period=2000000,
            nominal_attributes=None,
            seed=42,
        )
    elif project_name == "E-Commerce Customer Interactions":
        # =================================================================
        # DBSTREAM - Density-Based Stream Clustering
        # For customer behavior clustering with arbitrary shapes
        # =================================================================
        # OLD CONFIGURATION:
        # return cluster.DBSTREAM(
        #     clustering_threshold = 1.0,
        #     fading_factor = 0.01,
        #     cleanup_interval = 2,
        # )

        # CONFIGURATION based on River ML documentation example:
        # Reference: https://riverml.xyz/latest/api/cluster/DBSTREAM/
        #
        # The River documentation provides this exact example configuration:
        # - clustering_threshold=1.5: Micro-cluster radius
        # - fading_factor=0.05: Historical data importance (must be > 0)
        # - cleanup_interval=4: Time between cleanup processes
        # - intersection_factor=0.5: Cluster overlap ratio for connectivity
        # - minimum_weight=1.0: Threshold for non-noisy cluster classification
        return cluster.DBSTREAM(
            clustering_threshold=1.5,
            fading_factor=0.05,
            cleanup_interval=4,
            intersection_factor=0.5,
            minimum_weight=1.0,
        )
    elif project_name == "Sales Forecasting":
        # =================================================================
        # SNARIMAX - Seasonal Non-linear Auto-Regressive Integrated
        # Moving Average with eXogenous inputs
        # For sales forecasting with weekly seasonality
        # =================================================================
        # OLD CONFIGURATION:
        # regressor_snarimax = linear_model.PARegressor(
        #     C = 0.01,
        #     mode = 1)
        # return time_series.SNARIMAX(
        #     p = 2,
        #     d = 1,
        #     q = 1,
        #     m = 7,
        #     sp = 1,
        #     sd = 0,
        #     sq = 1,
        #     regressor = regressor_snarimax
        # )

        # CONFIGURATION based on River ML documentation:
        # Reference: https://riverml.xyz/latest/api/time-series/SNARIMAX/
        # Reference: https://riverml.xyz/latest/api/linear-model/PARegressor/
        #
        # SNARIMAX parameters for weekly sales data:
        # - p=7: Past 7 days of target values (full week)
        # - d=1: First-order differencing for trend removal
        # - q=2: Past error terms for noise handling
        # - m=7: Weekly seasonality period
        # - sp=1: Seasonal autoregressive order
        # - sd=1: Seasonal differencing (recommended for seasonal data)
        # - sq=1: Seasonal moving average order
        #
        # PARegressor parameters (defaults from River docs):
        # - C=1.0: Default regularization strength
        # - mode=1: Default algorithm mode
        # - eps=0.1: Default tolerance parameter
        # - learn_intercept=True: Default bias learning
        regressor_snarimax = linear_model.PARegressor(
            C=1.0,
            mode=1,
            eps=0.1,
            learn_intercept=True,
        )
        return time_series.SNARIMAX(
            p=7,
            d=1,
            q=2,
            m=7,
            sp=1,
            sd=1,
            sq=1,
            regressor=regressor_snarimax,
        )
    else:
        raise ValueError(f"Unknown project: {project_name}")


def create_consumer(project_name, max_retries: int = 5, retry_delay: float = 5.0, start_offset: Optional[int] = None):
    """Create and return Kafka consumer with manual partition assignment.
    Args:
        project_name: Name of the project for topic selection.
        max_retries: Maximum connection retry attempts.
        retry_delay: Delay between retries in seconds.
        start_offset: If provided, seek to this offset + 1 to continue from last processed.
                     If None, seeks to beginning (replay all messages).
    Note: Uses manual partition assignment instead of group-based subscription
    due to Kafka 4.0 compatibility issues with kafka-python's consumer group protocol.
    """
    consumer_name_dict = {
        "Transaction Fraud Detection": "transaction_fraud_detection",
        "Estimated Time of Arrival": "estimated_time_of_arrival",
        "E-Commerce Customer Interactions": "e_commerce_customer_interactions",
        "Sales Forecasting": "sales_forecasting"
    }
    KAFKA_TOPIC = consumer_name_dict[project_name]
    for attempt in range(max_retries):
        try:
            # Create consumer without topic subscription (manual assignment)
            consumer = KafkaConsumer(
                bootstrap_servers = KAFKA_BROKERS,
                value_deserializer = lambda v: json.loads(v.decode('utf-8')),
                consumer_timeout_ms = 1000,  # 1 second timeout for graceful shutdown checks
                api_version = (3, 7),  # Force API version for Kafka 4.0 compatibility
            )
            # Manually assign partition 0 of the topic
            tp = TopicPartition(KAFKA_TOPIC, 0)
            consumer.assign([tp])
            # Seek to appropriate offset
            if start_offset is not None:
                # Continue from the next message after last processed
                next_offset = start_offset + 1
                consumer.seek(tp, next_offset)
                print(f"Kafka consumer seeking to offset {next_offset} (continuing from {start_offset})")
            else:
                # No stored offset - start from beginning
                consumer.seek_to_beginning(tp)
                print(f"Kafka consumer seeking to beginning (no stored offset)")
            print(f"Kafka consumer created for {project_name} (manual assignment)")
            return consumer
        except NoBrokersAvailable as e:
            if attempt < max_retries - 1:
                print(f"Kafka not available for {project_name}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect to Kafka for {project_name} after {max_retries} attempts. Continuing without consumer.")
                return None
        except Exception as e:
            print(f"Error creating Kafka consumer for {project_name}: {e}")
            return None
    return None


def load_or_create_data(consumer, project_name: str) -> pd.DataFrame:
    """Load data from Delta Lake on MinIO or fallback to Kafka."""
    delta_path_dict = {
        "Transaction Fraud Detection": "s3://lakehouse/delta/transaction_fraud_detection",
        "Estimated Time of Arrival": "s3://lakehouse/delta/estimated_time_of_arrival",
        "E-Commerce Customer Interactions": "s3://lakehouse/delta/e_commerce_customer_interactions",
        "Sales Forecasting": "s3://lakehouse/delta/sales_forecasting",
    }
    DELTA_PATH = delta_path_dict.get(project_name, "")
    # Try loading from Delta Lake on MinIO first
    try:
        print(f"Attempting to load data from Delta Lake: {DELTA_PATH}")
        dt = deltalake.DeltaTable(
            DELTA_PATH, 
            storage_options = DELTA_STORAGE_OPTIONS)
        data_df = dt.to_pandas()
        print(f"Data loaded from Delta Lake for {project_name}: {len(data_df)} rows")
        return data_df
    except Exception as e:
        print(f"Delta Lake not available for {project_name}: {e}")
        print("Falling back to Kafka...")
    # Fallback to Kafka if Delta Lake is not available
    if consumer is not None:
        try:
            transaction = None
            for message in consumer:
                transaction = message.value
                break
            if transaction is not None:
                data_df = pd.DataFrame([transaction])
                print(f"Created data from Kafka for {project_name}")
                return data_df
        except Exception as e:
            print(f"Error loading data from Kafka: {e}")
    print(f"Warning: No data available for {project_name}")
    return pd.DataFrame()
