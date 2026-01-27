"""
Prometheus Metrics Registry - COELHO RealTime FastAPI Service

Central metrics definitions using prometheus_client directly.
All metrics auto-appear at /metrics alongside prometheus-fastapi-instrumentator HTTP metrics.
"""
from prometheus_client import Counter, Histogram, Gauge


# =============================================================================
# Training Metrics
# =============================================================================
TRAINING_ACTIVE = Gauge(
    "fastapi_training_active",
    "Whether training is currently running (1=active, 0=idle)",
    ["project", "model_type"],
)

TRAINING_STARTED_TOTAL = Counter(
    "fastapi_training_started_total",
    "Total number of training sessions started",
    ["project", "model_type"],
)

TRAINING_ERRORS_TOTAL = Counter(
    "fastapi_training_errors_total",
    "Total number of training errors",
    ["project", "model_type"],
)

TRAINING_DURATION_SECONDS = Histogram(
    "fastapi_training_duration_seconds",
    "Training subprocess duration in seconds",
    ["project", "model_type"],
    buckets=[10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
)

# =============================================================================
# Prediction Metrics
# =============================================================================
PREDICTIONS_TOTAL = Counter(
    "fastapi_predictions_total",
    "Total number of predictions made",
    ["project", "source"],
)

PREDICTION_DURATION_SECONDS = Histogram(
    "fastapi_prediction_duration_seconds",
    "Prediction latency in seconds",
    ["project"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "fastapi_prediction_errors_total",
    "Total number of failed predictions",
    ["project"],
)

# =============================================================================
# Caching Metrics
# =============================================================================
MODEL_CACHE_HITS_TOTAL = Counter(
    "fastapi_model_cache_hits_total",
    "Total model/experiment/metrics cache hits",
    ["cache_type"],
)

MODEL_CACHE_MISSES_TOTAL = Counter(
    "fastapi_model_cache_misses_total",
    "Total model/experiment/metrics cache misses",
    ["cache_type"],
)

# =============================================================================
# Model Loading Metrics
# =============================================================================
MODEL_LOAD_DURATION_SECONDS = Histogram(
    "fastapi_model_load_duration_seconds",
    "Model load time in seconds",
    ["project", "source"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# =============================================================================
# MLflow Metrics
# =============================================================================
MLFLOW_OPERATION_DURATION_SECONDS = Histogram(
    "fastapi_mlflow_operation_duration_seconds",
    "MLflow API call latency in seconds",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

MLFLOW_ERRORS_TOTAL = Counter(
    "fastapi_mlflow_errors_total",
    "Total MLflow API errors",
    ["operation"],
)

# =============================================================================
# SQL Metrics
# =============================================================================
SQL_QUERY_DURATION_SECONDS = Histogram(
    "fastapi_sql_query_duration_seconds",
    "SQL query execution time in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

SQL_ROWS_RETURNED = Histogram(
    "fastapi_sql_rows_returned",
    "Number of rows returned per query",
    buckets=[0, 1, 10, 50, 100, 500, 1000, 5000, 10000],
)

SQL_ERRORS_TOTAL = Counter(
    "fastapi_sql_errors_total",
    "Total SQL errors by type",
    ["error_type"],
)

# =============================================================================
# Visualization Metrics
# =============================================================================
VISUALIZATION_DURATION_SECONDS = Histogram(
    "fastapi_visualization_duration_seconds",
    "Visualization generation time in seconds",
    ["viz_type"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

VISUALIZATION_CACHE_HITS_TOTAL = Counter(
    "fastapi_visualization_cache_hits_total",
    "Total visualization MLflow cache hits",
    ["viz_type"],
)
