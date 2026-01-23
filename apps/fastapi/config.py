"""
Shared configuration for the Unified FastAPI Service.

This module centralizes all configuration constants used across routers:
- Project definitions and mappings
- Delta Lake paths
- MLflow model names and metric criteria
- Feature definitions for each project
"""
import os

# =============================================================================
# Environment Variables
# =============================================================================
MLFLOW_HOST = os.getenv("MLFLOW_HOST", "localhost")
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:5000"

# Redis for live model caching (incremental ML)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# AWS/MinIO configuration for Delta Lake
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
MINIO_HOST = os.getenv("MINIO_HOST", "localhost")
MINIO_PORT = os.getenv("MINIO_PORT", "9000")
AWS_S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT", f"{MINIO_HOST}:{MINIO_PORT}")

# =============================================================================
# Project Definitions
# =============================================================================
PROJECT_NAMES = [
    "Transaction Fraud Detection",
    "Estimated Time of Arrival",
    "E-Commerce Customer Interactions",
]

# Project task types (classification, regression, clustering)
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
# Delta Lake Paths (S3/MinIO)
# =============================================================================
DELTA_PATHS = {
    "Transaction Fraud Detection": "s3://lakehouse/delta/transaction_fraud_detection",
    "Estimated Time of Arrival": "s3://lakehouse/delta/estimated_time_of_arrival",
    "E-Commerce Customer Interactions": "s3://lakehouse/delta/e_commerce_customer_interactions",
}

# =============================================================================
# MLflow Model Names
# =============================================================================
# Incremental ML (River)
INCREMENTAL_MODEL_NAMES = {
    "Transaction Fraud Detection": "ARFClassifier",
    "Estimated Time of Arrival": "ARFRegressor",
    "E-Commerce Customer Interactions": "DBSTREAM",
}

# Batch ML (Scikit-Learn/CatBoost)
BATCH_MODEL_NAMES = {
    "Transaction Fraud Detection": "CatBoostClassifier",
    "Estimated Time of Arrival": "CatBoostRegressor",
    "E-Commerce Customer Interactions": "KMeans",
}

# =============================================================================
# Best Metric Criteria (for selecting best MLflow run)
# =============================================================================
BEST_METRIC_CRITERIA = {
    "Transaction Fraud Detection": {
        "metric_name": "fbeta_score",
        "maximize": True,
    },
    "Estimated Time of Arrival": {
        "metric_name": "MAE",
        "maximize": False,  # Lower is better
    },
    "E-Commerce Customer Interactions": {
        "metric_name": "silhouette_score",
        "maximize": True,
    },
}

# =============================================================================
# Training Scripts (for subprocess-based training)
# =============================================================================
INCREMENTAL_TRAINING_SCRIPTS = {
    "Transaction Fraud Detection": "ml_training/river/tfd.py",
    "Estimated Time of Arrival": "ml_training/river/eta.py",
    "E-Commerce Customer Interactions": "ml_training/river/ecci.py",
}

BATCH_TRAINING_SCRIPTS = {
    "Transaction Fraud Detection": "ml_training/sklearn/tfd.py",
    "Estimated Time of Arrival": "ml_training/sklearn/eta.py",
    "E-Commerce Customer Interactions": "ml_training/sklearn/ecci.py",
}

# =============================================================================
# Feature Definitions
# =============================================================================
# Transaction Fraud Detection features
TFD_FEATURES = [
    "amount",
    "account_age_days",
    "currency",
    "merchant_id",
    "product_category",
    "transaction_type",
    "payment_method",
    "location_lat",
    "location_lon",
    "device_type",
    "browser",
    "os",
    "cvv_provided",
    "billing_address_match",
    "hour_of_day",
    "day_of_week",
]

TFD_CATEGORICAL_FEATURES = [
    "currency",
    "merchant_id",
    "product_category",
    "transaction_type",
    "payment_method",
    "device_type",
    "browser",
    "os",
]

# CatBoost categorical feature indices (for batch ML)
TFD_CAT_FEATURE_INDICES = [2, 3, 4, 5, 6, 9, 10, 11]

# Estimated Time of Arrival features
ETA_FEATURES = [
    "estimated_distance_km",
    "weather",
    "temperature_celsius",
    "day_of_week",
    "hour_of_day",
    "driver_rating",
    "vehicle_type",
    "initial_estimated_travel_time_seconds",
    "origin_lat",
    "origin_lon",
    "destination_lat",
    "destination_lon",
]

ETA_CATEGORICAL_FEATURES = [
    "weather",
    "vehicle_type",
]

ETA_CAT_FEATURE_INDICES = [1, 6]

# E-Commerce Customer Interactions features
ECCI_FEATURES = [
    "event_type",
    "product_category",
    "price",
    "quantity",
    "time_on_page_seconds",
    "session_event_sequence",
    "device_type",
    "browser",
    "os",
]

ECCI_CATEGORICAL_FEATURES = [
    "event_type",
    "product_category",
    "device_type",
    "browser",
    "os",
]

# =============================================================================
# SQL Query Defaults
# =============================================================================
SQL_DEFAULT_LIMIT = 10000
SQL_MAX_LIMIT = 100000
SQL_QUERY_TIMEOUT = 60  # seconds

# =============================================================================
# Cache TTL Settings
# =============================================================================
MLFLOW_METRICS_CACHE_TTL = 30  # seconds
MODEL_CACHE_TTL = 30  # seconds
CLUSTER_CACHE_TTL = 30  # seconds
