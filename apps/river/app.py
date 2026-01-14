"""
River ML Training Service

Handles incremental ML model training using River library.
Receives start/stop signals directly from Reflex frontend.
Also handles predictions using trained River models.
Also serves as data service for incremental ML projects (migrated from fastapi app).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
from typing import Optional, Any, Dict
import subprocess
import os
import sys
import json
import math
import time
import asyncio
import pickle
import tempfile
import numpy as np
import pandas as pd
import mlflow
from functions import (
    load_or_create_model,
    load_or_create_encoders,
    process_sample,
    create_consumer,
    load_or_create_data,
    # Polars optimized functions (lightweight, lazy evaluation)
    get_unique_values_polars,
    get_sample_polars,
    get_initial_sample_polars,
    precompute_all_unique_values_polars,
    # Static dropdown options (instant access, mirrors Kafka producer constants)
    get_static_dropdown_options,
    # Redis live model cache (real-time predictions during training)
    load_live_model_from_redis,
    is_training_active,
    # SQL query execution (Delta Lake SQL tab) - DuckDB and Polars engines
    execute_delta_sql,
    execute_delta_sql_polars,
    get_delta_table_schema,
    SQL_DEFAULT_LIMIT,
    # Best model selection (by metrics)
    get_best_mlflow_run,
)


MLFLOW_HOST = os.environ["MLFLOW_HOST"]
PROJECT_NAMES = [
    "Transaction Fraud Detection",
    "Estimated Time of Arrival",
    "E-Commerce Customer Interactions",
]
MODEL_SCRIPTS = {
    f"{name.replace(' ', '_').replace('-', '_').lower()}_river.py": name
    for name in PROJECT_NAMES
}
ENCODER_LIBRARIES = ["river", "sklearn"]

# Global caches for models, encoders, and data
model_dict: dict = {name: {} for name in PROJECT_NAMES}
encoders_dict: dict = {name: {} for name in PROJECT_NAMES}
consumer_dict: dict = {name: None for name in PROJECT_NAMES}
data_dict: dict = {name: None for name in PROJECT_NAMES}
initial_sample_dict: dict = {name: None for name in PROJECT_NAMES}
# Precomputed unique values cache (populated at startup for instant access)
unique_values_cache: dict = {name: {} for name in PROJECT_NAMES}


# =============================================================================
# Pydantic Models (migrated from fastapi app)
# =============================================================================
class Healthcheck(BaseModel):
    model_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    model_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    encoders_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    data_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    data_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    initial_data_sample_loaded: dict[str, bool] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    initial_data_sample_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}


# Initialize healthcheck with default state
healthcheck = Healthcheck(
    model_load = {x: "not_attempted" for x in PROJECT_NAMES},
    encoders_load = {x: "not_attempted" for x in PROJECT_NAMES},
    data_load = {x: "not_attempted" for x in PROJECT_NAMES},
    initial_data_sample_loaded = {x: False for x in PROJECT_NAMES}
)


class DeviceInfo(BaseModel):
    """Pydantic model for device information."""
    device_type: Optional[str] = None
    browser: Optional[str] = None
    os: Optional[str] = None


class Location(BaseModel):
    """Pydantic model for location."""
    lat: Optional[float] = None
    lon: Optional[float] = None

    @field_validator('lat', 'lon', mode = 'before')
    @classmethod
    def check_location_coords_finite(cls, v: Any) -> Optional[float]:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v


class TransactionFraudDetection(BaseModel):
    transaction_id: str
    user_id: str
    timestamp: str
    amount: float
    currency: str
    merchant_id: str
    product_category: str
    transaction_type: str
    payment_method: str
    location: dict
    ip_address: str
    device_info: dict
    user_agent: str
    account_age_days: int
    cvv_provided: bool
    billing_address_match: bool

    @field_validator('location', 'device_info', mode = 'before')
    @classmethod
    def parse_json_string(cls, v: Any) -> dict:
        """Parse JSON string from Delta Lake into dict."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}


class EstimatedTimeOfArrival(BaseModel):
    trip_id: str
    driver_id: str
    vehicle_id: str
    timestamp: str
    origin: dict[str, float]
    destination: dict[str, float]
    estimated_distance_km: float
    weather: str
    temperature_celsius: float
    day_of_week: int
    hour_of_day: int
    driver_rating: float
    vehicle_type: str
    initial_estimated_travel_time_seconds: int
    simulated_actual_travel_time_seconds: int
    debug_traffic_factor: float
    debug_weather_factor: float
    debug_incident_delay_seconds: float
    debug_driver_factor: float

    @field_validator('origin', 'destination', mode = 'before')
    @classmethod
    def parse_json_string(cls, v: Any) -> dict:
        """Parse JSON string from Delta Lake into dict."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}


class ECommerceCustomerInteractions(BaseModel):
    """Pydantic model for validating e-commerce customer interaction events."""
    customer_id: Optional[str] = None
    device_info: Optional[DeviceInfo] = None
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    location: Optional[Location] = None
    page_url: Optional[str] = None
    price: Optional[float] = None
    product_category: Optional[str] = None
    product_id: Optional[str] = None
    quantity: Optional[int] = None
    referrer_url: Optional[str] = None
    search_query: Optional[str] = None
    session_event_sequence: Optional[int] = None
    session_id: Optional[str] = None
    time_on_page_seconds: Optional[int] = None
    timestamp: Optional[str] = None

    @field_validator('device_info', mode = 'before')
    @classmethod
    def parse_device_info_json(cls, v: Any) -> Optional[DeviceInfo]:
        """Parse JSON string from Delta Lake into DeviceInfo."""
        if v is None:
            return None
        if isinstance(v, DeviceInfo):
            return v
        if isinstance(v, dict):
            return DeviceInfo(**v)
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return DeviceInfo(**parsed) if parsed else None
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    @field_validator('location', mode = 'before')
    @classmethod
    def parse_location_json(cls, v: Any) -> Optional[Location]:
        """Parse JSON string from Delta Lake into Location."""
        if v is None:
            return None
        if isinstance(v, Location):
            return v
        if isinstance(v, dict):
            return Location(**v)
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return Location(**parsed) if parsed else None
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    @field_validator('quantity', mode = 'before')
    @classmethod
    def check_quantity_is_finite_int(cls, v: Any) -> Optional[int]:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    @field_validator('price', mode = 'before')
    @classmethod
    def check_price_is_finite(cls, v: Any) -> Optional[float]:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    @field_validator('device_info', 'location', mode = 'before')
    @classmethod
    def ensure_dict_or_none(cls, v: Any) -> Optional[Dict]:
        if v is None or isinstance(v, dict):
            return v
        return v


class SampleRequest(BaseModel):
    project_name: str


class UniqueValuesRequest(BaseModel):
    column_name: str
    project_name: str
    limit: int = 100


class InitialSampleRequest(BaseModel):
    project_name: str


class OrdinalEncoderRequest(BaseModel):
    project_name: str


class ClusterFeatureCountsRequest(BaseModel):
    column_name: str


class MLflowMetricsRequest(BaseModel):
    project_name: str
    model_name: str
    force_refresh: bool = False


class ModelAvailabilityRequest(BaseModel):
    project_name: str
    model_name: str


class SQLQueryRequest(BaseModel):
    """Request model for SQL queries against Delta Lake."""
    project_name: str
    query: str
    limit: int = SQL_DEFAULT_LIMIT
    engine: str = "polars"  # "duckdb" or "polars" - default to polars for speed


class TableSchemaRequest(BaseModel):
    """Request model for table schema."""
    project_name: str


# =============================================================================
# MLflow Metrics Cache
# =============================================================================
class MLflowMetricsCache:
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple[float, dict]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[dict]:
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: dict):
        self._cache[key] = (time.time(), value)

    def clear(self):
        self._cache.clear()


mlflow_cache = MLflowMetricsCache(ttl_seconds = 5)  # Short TTL for responsive live updates


def _sync_get_mlflow_metrics(project_name: str, model_name: str) -> dict:
    """Synchronous MLflow query - hybrid approach for metrics.

    Returns:
    - If training is active (RUNNING status): real-time metrics from running experiment
    - Otherwise: metrics from best FINISHED model (same as predictions)

    This provides live updates during training while showing stable metrics when idle.
    """
    experiment = mlflow.get_experiment_by_name(project_name)
    if experiment is None:
        raise ValueError(f"Experiment '{project_name}' not found in MLflow")

    # First, check for RUNNING experiments (real-time training)
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'RUNNING'",
        max_results=10,
        order_by=["start_time DESC"]
    )
    # Filter by model name
    if not runs_df.empty:
        running_runs = runs_df[runs_df["tags.mlflow.runName"] == model_name]
        if not running_runs.empty:
            # Return real-time metrics from running experiment
            run_id = running_runs.iloc[0]["run_id"]
            run = mlflow.get_run(run_id)
            result = {
                "run_id": run_id,
                "status": "RUNNING",
                "start_time": run.info.start_time,
                "end_time": None,
                "is_live": True,  # Flag for UI to show "live" indicator
            }
            for metric_name, metric_value in run.data.metrics.items():
                result[f"metrics.{metric_name}"] = metric_value
            # Include baseline tags for delta calculation
            for tag_name, tag_value in run.data.tags.items():
                if tag_name.startswith("baseline_"):
                    try:
                        result[tag_name] = float(tag_value)
                    except ValueError:
                        pass
            return result

    # No running experiment - fall back to best FINISHED model
    best_run_id = get_best_mlflow_run(project_name, model_name)
    if best_run_id is None:
        raise ValueError(f"No trained model found for '{model_name}' in '{project_name}'")

    run = mlflow.get_run(best_run_id)
    result = {
        "run_id": best_run_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "is_live": False,  # Flag for UI - showing historical best
    }
    for metric_name, metric_value in run.data.metrics.items():
        result[f"metrics.{metric_name}"] = metric_value
    # Include baseline tags for delta calculation
    for tag_name, tag_value in run.data.tags.items():
        if tag_name.startswith("baseline_"):
            try:
                result[tag_name] = float(tag_value)
            except ValueError:
                pass
    return result


# =============================================================================
# Training State
# =============================================================================
class TrainingState:
    """Tracks current training process state."""
    def __init__(self):
        self.current_process: subprocess.Popen | None = None
        self.current_model_name: str | None = None
        self.status: str = "idle"


state = TrainingState()


def stop_current_model() -> bool:
    """Stop the currently running training process gracefully."""
    if not state.current_process:
        state.status = "No model was active to stop."
        return True
    model_name = state.current_model_name
    process = state.current_process
    pid = process.pid
    print(f"Stopping model '{model_name}' (PID: {pid}) with SIGTERM...")
    state.status = f"Stopping '{model_name}'..."
    try:
        process.terminate()
        process.wait(timeout = 60)
        if process.poll() is not None:
            print(f"Model '{model_name}' stopped gracefully (exit code: {process.returncode})")
            state.status = f"Model '{model_name}' stopped."
        else:
            print(f"SIGTERM failed, sending SIGKILL to PID {pid}")
            process.kill()
            process.wait(timeout = 10)
            state.status = f"Model '{model_name}' force killed."
    except subprocess.TimeoutExpired:
        print(f"Timeout waiting for {model_name}, force killing...")
        process.kill()
        state.status = f"Model '{model_name}' force killed after timeout."
    except Exception as e:
        print(f"Error stopping model: {e}")
        state.status = f"Error stopping model: {e}"
    finally:
        state.current_process = None
        state.current_model_name = None
    return True


# =============================================================================
# Lifespan (data loading at startup)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        consumer_dict, \
        data_dict, \
        model_dict, \
        encoders_dict, \
        initial_sample_dict, \
        unique_values_cache, \
        healthcheck
    # 1. Initialize unique values from static options (instant - no I/O)
    print("Starting River ML service...", flush = True)
    data_load_status = {}
    data_message_dict = {}
    initial_data_sample_loaded_status = {}
    initial_data_sample_message_dict = {}
    project_model_mapping = {
        "Transaction Fraud Detection": TransactionFraudDetection,
        "Estimated Time of Arrival": EstimatedTimeOfArrival,
        "E-Commerce Customer Interactions": ECommerceCustomerInteractions,
    }
    # Load static dropdown options first (instant - mirrors Kafka producer constants)
    # This ensures forms work immediately, even on fresh start with no Delta Lake data
    for project_name in PROJECT_NAMES:
        static_options = get_static_dropdown_options(project_name)
        if static_options:
            unique_values_cache[project_name] = static_options
            data_load_status[project_name] = "success"
            data_message_dict[project_name] = f"Static options loaded ({len(static_options)} fields)"
            print(f"Loaded static dropdown options for {project_name} ({len(static_options)} fields)", flush=True)
        else:
            unique_values_cache[project_name] = {}
            data_load_status[project_name] = "no_static_options"
            data_message_dict[project_name] = "No static options defined"
        # Initial sample loading deferred - forms work without it
        initial_data_sample_loaded_status[project_name] = False
        initial_data_sample_message_dict[project_name] = "Sample loading deferred to on-demand"
    healthcheck.data_load = data_load_status
    healthcheck.data_message = data_message_dict
    healthcheck.initial_data_sample_loaded = initial_data_sample_loaded_status
    healthcheck.initial_data_sample_message = initial_data_sample_message_dict
    # 2. Skip Kafka consumers at startup - create on-demand for training scripts
    print("Kafka consumers will be created on-demand for training scripts.", flush = True)
    # consumer_dict is already initialized with None values
    # 3. Skip model preloading - models are loaded on-demand via MLflow
    model_load_status = {}
    model_message_dict = {}
    for project_name in PROJECT_NAMES:
        model_load_status[project_name] = "deferred"
        model_message_dict[project_name] = "Models loaded on-demand from MLflow"
    healthcheck.model_load = model_load_status
    healthcheck.model_message = model_message_dict
    # 4. Skip encoder preloading - encoders are loaded on-demand via MLflow
    encoders_load_status = {}
    print("Encoders will be loaded on-demand from MLflow.", flush=True)
    for project_name in PROJECT_NAMES:
        encoders_dict[project_name] = {}
        encoders_load_status[project_name] = "deferred"
    healthcheck.encoders_load = encoders_load_status
    # 5. Configure MLflow (non-blocking - just sets URI)
    try:
        print("Configuring MLflow...", flush = True)
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
    except Exception as e:
        print(f"Error configuring MLflow: {e}", file=sys.stderr, flush = True)
    print("River ML service startup complete.", flush=True)
    yield
    print("River ML service shutting down...")


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title = "River ML Training Service - COELHO RealTime",
    description = "Manages incremental ML model training and data services",
    version = "2.0.0",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# Add Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app)


# =============================================================================
# Health & Status Endpoints
# =============================================================================
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "river-ml-training"}


@app.get("/status")
async def get_status():
    """Get current training status."""
    if state.current_model_name and state.current_process:
        if state.current_process.poll() is None:
            return {
                "current_model": state.current_model_name,
                "status": "running",
                "pid": state.current_process.pid
            }
        else:
            return_code = state.current_process.returncode
            stop_current_model()
            return {
                "current_model": state.current_model_name,
                "status": f"stopped (exit code: {return_code})",
                "pid": None
            }
    return {"current_model": None, "status": "idle"}


@app.post("/switch_model")
async def switch_model(payload: dict):
    """
    Start or stop ML model training.

    Payload:
        model_key: str - Script filename (e.g., "transaction_fraud_detection_river.py")
                        or "none" to stop training
        project_name: str - Project name for MLflow
    """
    model_key = payload.get("model_key")

    if model_key == state.current_model_name:
        return {"message": f"Model {model_key} is already running."}
    if state.current_process:
        print(f"Switching from {state.current_model_name} to {model_key}")
        stop_current_model()
    else:
        print(f"No model running, attempting to start {model_key}")
    if model_key == "none" or model_key not in MODEL_SCRIPTS:
        if model_key == "none":
            return {"message": "All models stopped."}
        else:
            raise HTTPException(
                status_code = 404,
                detail = f"Model key '{model_key}' not found. Available: {list(MODEL_SCRIPTS.keys())}"
            )

    command = ["/app/.venv/bin/python3", "-u", model_key]

    try:
        print(f"Starting model: {model_key}")
        log_dir = "/app/logs"
        os.makedirs(log_dir, exist_ok = True)
        log_file_path = f"{log_dir}/{model_key}.log"
        with open(log_file_path, "ab") as log_file:
            process = subprocess.Popen(
                command,
                stdout = log_file,
                stderr = subprocess.STDOUT,
                cwd = "/app"
            )
        state.current_process = process
        state.current_model_name = model_key
        state.status = f"Running {model_key}"
        print(f"Model {model_key} started with PID: {process.pid}")
        return {
            "message": f"Started model: {model_key}",
            "pid": process.pid,
        }
    except Exception as e:
        print(f"Failed to start model {model_key}: {e}")
        state.current_process = None
        state.current_model_name = None
        state.status = f"Failed to start: {e}"
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to start model {model_key}: {str(e)}"
        )


@app.get("/current_model")
async def get_current_model():
    """Get currently running model."""
    return await get_status()


@app.get("/training_status/{project_name}")
async def get_training_status(project_name: str):
    """Check if training is active for a project.

    Returns:
        - is_active: True if live model exists in Redis
        - model_name: The model name being trained (if active)
        - model_source: "live" if training active, "mlflow" otherwise
    """
    from functions import MLFLOW_MODEL_NAMES
    model_name = MLFLOW_MODEL_NAMES.get(project_name)
    if not model_name:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown project: {project_name}"
        )
    is_active = is_training_active(project_name, model_name)
    return {
        "project_name": project_name,
        "model_name": model_name,
        "is_active": is_active,
        "model_source": "live" if is_active else "mlflow",
    }


@app.post("/predict")
async def predict(payload: dict):
    """
    Make predictions using the best available model.

    Priority order:
    1. Live model from Redis (if training is active) - real-time updates
    2. Best model from MLflow (if training is inactive) - production quality

    Payload format: {project_name, model_name, ...feature_data}
    """
    global encoders_dict, model_dict
    project_name = payload.get("project_name")
    model_name = payload.get("model_name")
    if not project_name or not model_name:
        raise HTTPException(
            status_code = 400,
            detail = "Missing required fields: project_name and model_name"
        )
    # Track which model source is being used (for UI indicator)
    model_source = "mlflow"  # default
    encoders = None
    model = None
    # First, check if training is active and try to load live model from Redis
    if is_training_active(project_name, model_name):
        redis_result = load_live_model_from_redis(project_name, model_name)
        if redis_result is not None:
            model, encoders = redis_result
            model_source = "live"
            print(f"Using LIVE model from Redis for {project_name}/{model_name}")
    # If no live model, load from MLflow (best historical model)
    if model is None:
        try:
            model = load_or_create_model(project_name, model_name)
            # Update in-memory cache
            if project_name not in model_dict:
                model_dict[project_name] = {}
            model_dict[project_name][model_name] = model
            model_source = "mlflow"
        except Exception as e:
            print(f"Error loading model from MLflow: {e}", file = sys.stderr)
            # Fall back to cached model if available
            if project_name in model_dict and model_name in model_dict.get(project_name, {}):
                model = model_dict[project_name][model_name]
                model_source = "cache"
                print(f"Using cached model for {project_name}/{model_name}")
            else:
                raise HTTPException(
                    status_code = 503,
                    detail = f"Model '{model_name}' for project '{project_name}' not found in MLflow. Train a model first."
                )
    # Load encoders (from Redis if live, otherwise from MLflow)
    if encoders is None:
        try:
            for library in ENCODER_LIBRARIES:
                encoders_dict[project_name][library] = load_or_create_encoders(project_name, library)
        except Exception as e:
            print(f"Error loading encoders from MLflow: {e}", file=sys.stderr)
            if project_name not in encoders_dict:
                raise HTTPException(
                    status_code = 503,
                    detail = f"Encoders for project '{project_name}' not found in MLflow. Train a model first."
                )
    # Extract feature data (remove metadata fields)
    x = {k: v for k, v in payload.items() if k not in ["project_name", "model_name"]}
    if model_name in ["ARFClassifier", "ARFRegressor", "DBSTREAM"]:
        try:
            # Use encoders (from Redis if live, otherwise from MLflow)
            if encoders is None:
                encoders = encoders_dict[project_name].get("river", {})
            processed_x, _ = process_sample(x, encoders, project_name)
            if project_name == "Transaction Fraud Detection":
                y_pred_proba = model.predict_proba_one(processed_x)
                fraud_probability = y_pred_proba.get(1, 0.0)
                binary_prediction = 1 if fraud_probability >= 0.5 else 0
                return {
                    "fraud_probability": fraud_probability,
                    "prediction": binary_prediction,
                    "model_source": model_source,  # "live" or "mlflow"
                }
            elif project_name == "Estimated Time of Arrival":
                y_pred = model.predict_one(processed_x)
                return {
                    "Estimated Time of Arrival": y_pred,
                    "model_source": model_source,
                }
            elif project_name == "E-Commerce Customer Interactions":
                y_pred = model.predict_one(processed_x)
                return {
                    "cluster": y_pred,
                    "model_source": model_source,
                }
        except Exception as e:
            print(f"Error during prediction: {e}", file=sys.stderr)
            raise HTTPException(
                status_code = 500,
                detail = f"Prediction failed: {e}")
    # XGBClassifier (Batch ML) is now handled by the sklearn service (port 8003)
    elif model_name in ["XGBClassifier"]:
        raise HTTPException(
            status_code = 400,
            detail = "XGBClassifier (Batch ML) is handled by the sklearn service at port 8003"
        )
    raise HTTPException(
        status_code = 400,
        detail = f"Unknown model: {model_name}")


# =============================================================================
# Data Endpoints (migrated from fastapi app)
# =============================================================================
@app.get("/healthcheck", response_model = Healthcheck)
async def get_healthcheck():
    """Get detailed healthcheck status."""
    if ("failed" in healthcheck.model_load.values()) or ("failed" in healthcheck.data_load.values()):
        raise HTTPException(
            status_code = 503,
            detail = "Service Unavailable: Core components failed to load."
        )
    return healthcheck


@app.put("/healthcheck", response_model = Healthcheck)
async def update_healthcheck(update_data: Healthcheck):
    """Update healthcheck status."""
    global healthcheck
    update_dict = update_data.model_dump(exclude_unset = True)
    for field, value in update_dict.items():
        setattr(healthcheck, field, value)
    return healthcheck


@app.post("/sample")
async def get_sample(request: SampleRequest):
    """Get a random sample from the dataset using Spark (fast)."""
    try:
        sample_df = get_sample_polars(request.project_name, n = 1)
        if sample_df is None or sample_df.empty:
            raise HTTPException(
                status_code = 503, 
                detail = "Could not get sample from Spark.")
        sample = sample_df.to_dict(orient = 'records')[0]
        # Validate through Pydantic to parse JSON string fields from Delta Lake
        if request.project_name == "Transaction Fraud Detection":
            sample = TransactionFraudDetection.model_validate(sample)
        elif request.project_name == "Estimated Time of Arrival":
            sample = EstimatedTimeOfArrival.model_validate(sample)
        elif request.project_name == "E-Commerce Customer Interactions":
            sample = ECommerceCustomerInteractions.model_validate(sample)
        return sample
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500, 
            detail = f"Error sampling data: {e}")


@app.post("/unique_values")
async def get_unique_values(request: UniqueValuesRequest):
    """Get unique values for a column (instant from static options, Polars fallback)."""
    global unique_values_cache
    project_cache = unique_values_cache.get(request.project_name, {})
    column = request.column_name
    # First try the cache (populated from static options at startup - instant)
    if column in project_cache:
        unique_values = project_cache[column]
        if len(unique_values) > request.limit:
            unique_values = unique_values[:request.limit]
        return {"unique_values": unique_values}
    # Fallback: try static options directly (in case cache was cleared)
    static_options = get_static_dropdown_options(request.project_name)
    if column in static_options:
        unique_values = static_options[column]
        # Update cache for future requests
        unique_values_cache[request.project_name][column] = unique_values
        if len(unique_values) > request.limit:
            unique_values = unique_values[:request.limit]
        return {"unique_values": unique_values}
    # Last resort: query Polars/Delta Lake for columns not in static options
    try:
        unique_values = get_unique_values_polars(
            request.project_name,
            column,
            request.limit)
        if unique_values:
            # Update cache for future requests
            unique_values_cache[request.project_name][column] = unique_values
            return {"unique_values": unique_values}
        raise HTTPException(
            status_code = 400,
            detail = f"Column '{column}' not found or has no values."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Error getting unique values for column '{column}': {e}"
        )


@app.post("/initial_sample")
async def get_initial_sample(request: InitialSampleRequest):
    """Get initial sample (on-demand from Delta Lake if not cached).

    Returns validated Pydantic models with JSON fields parsed from Delta Lake strings.
    """
    global initial_sample_dict
    project_name = request.project_name
    project_model_mapping = {
        "Transaction Fraud Detection": TransactionFraudDetection,
        "Estimated Time of Arrival": EstimatedTimeOfArrival,
        "E-Commerce Customer Interactions": ECommerceCustomerInteractions,
    }
    # Return cached sample if available
    if initial_sample_dict.get(project_name) is not None:
        return initial_sample_dict[project_name]
    # Load on-demand from Delta Lake via Polars
    try:
        sample_dict = await asyncio.wait_for(
            asyncio.to_thread(get_initial_sample_polars, project_name),
            timeout = 15.0
        )
        if sample_dict:
            model_class = project_model_mapping.get(project_name)
            if model_class:
                validated_sample = model_class(**sample_dict)
                initial_sample_dict[project_name] = validated_sample
                return validated_sample
        raise HTTPException(
            status_code = 503,
            detail = f"No data available for {project_name} in Delta Lake."
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code = 504,
            detail = f"Timeout loading sample for {project_name}. Delta Lake may be unavailable."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Error loading sample for {project_name}: {e}"
        )


@app.post("/get_ordinal_encoder")
async def get_ordinal_encoder(request: OrdinalEncoderRequest):
    """Get ordinal encoder mappings for a project."""
    encoders = load_or_create_encoders(request.project_name, "river")
    if None in encoders.values():
        raise HTTPException(
            status_code = 503,
            detail = "Ordinal encoder is not loaded."
        )
    if request.project_name in ["Transaction Fraud Detection", "Estimated Time of Arrival"]:
        return {
            "ordinal_encoder": encoders["ordinal_encoder"].get_feature_mappings()
        }
    elif request.project_name in ["E-Commerce Customer Interactions"]:
        return {
            "standard_scaler": encoders["standard_scaler"].counts,
        }
    raise HTTPException(
        status_code = 400,
        detail = f"Unknown project: {request.project_name}"
    )


@app.get("/cluster_counts")
async def get_cluster_counts():
    """Get cluster counts from JSON file (for DBSTREAM clustering)."""
    try:
        with open("data/cluster_counts.json", 'r') as f:
            cluster_counts = json.load(f)
        return cluster_counts
    except Exception:
        return {}


@app.post("/cluster_feature_counts")
async def get_cluster_feature_counts(request: ClusterFeatureCountsRequest):
    """Get cluster feature counts for a specific column."""
    try:
        with open("data/cluster_feature_counts.json", 'r') as f:
            cluster_counts = json.load(f)
        clusters = list(cluster_counts.keys())
        return {
            x: cluster_counts[x][request.column_name]
            for x in clusters
        }
    except Exception:
        return {}


@app.post("/mlflow_metrics")
async def get_mlflow_metrics(request: MLflowMetricsRequest):
    """Get MLflow metrics with caching and async execution."""
    cache_key = f"{request.project_name}:{request.model_name}"
    if not request.force_refresh:
        cached_result = mlflow_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_get_mlflow_metrics,
                request.project_name,
                request.model_name
            ),
            timeout = 30.0
        )
        mlflow_cache.set(cache_key, result)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code = 504,
            detail = "MLflow query timed out after 30 seconds"
        )
    except ValueError as e:
        raise HTTPException(
            status_code = 404, 
            detail = str(e))
    except Exception as e:
        print(f"Error fetching MLflow metrics: {e}", file=sys.stderr)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to fetch MLflow metrics: {str(e)}"
        )


def _sync_get_report_metrics(project_name: str, model_name: str) -> dict:
    """Load report_metrics.pkl artifact from MLflow (ConfusionMatrix, ClassificationReport)."""
    # First check for RUNNING experiment, then fall back to best FINISHED
    experiment = mlflow.get_experiment_by_name(project_name)
    if experiment is None:
        return {"available": False, "error": f"Experiment '{project_name}' not found"}

    run_id = None
    # Check for RUNNING experiment first
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'RUNNING'",
        max_results=10,
        order_by=["start_time DESC"]
    )
    if not runs_df.empty:
        running_runs = runs_df[runs_df["tags.mlflow.runName"] == model_name]
        if not running_runs.empty:
            run_id = running_runs.iloc[0]["run_id"]

    # Fall back to best FINISHED model
    if run_id is None:
        run_id = get_best_mlflow_run(project_name, model_name)

    if run_id is None:
        return {"available": False, "error": "No trained model found"}

    # Download and load report_metrics.pkl artifact
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="report_metrics.pkl",
                dst_path=tmpdir
            )
            with open(artifact_path, 'rb') as f:
                report_metrics = pickle.load(f)

            # Extract ConfusionMatrix data
            cm = report_metrics.get("ConfusionMatrix")
            cm_data = {"available": False}
            if cm is not None:
                # River's ConfusionMatrix stores data as nested dict: {actual: {predicted: count}}
                try:
                    # Get the raw confusion matrix data
                    cm_dict = dict(cm.data)
                    # For binary classification: classes are 0 and 1
                    tn = cm_dict.get(0, {}).get(0, 0)
                    fp = cm_dict.get(0, {}).get(1, 0)
                    fn = cm_dict.get(1, {}).get(0, 0)
                    tp = cm_dict.get(1, {}).get(1, 0)
                    total = tn + fp + fn + tp
                    cm_data = {
                        "available": True,
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        "total": total
                    }
                except Exception as e:
                    cm_data = {"available": False, "error": str(e)}

            # Extract ClassificationReport data
            cr = report_metrics.get("ClassificationReport")
            cr_data = {"available": False}
            if cr is not None:
                try:
                    # ClassificationReport has a string representation
                    cr_data = {
                        "available": True,
                        "report": str(cr)
                    }
                except Exception as e:
                    cr_data = {"available": False, "error": str(e)}

            return {
                "available": True,
                "run_id": run_id,
                "confusion_matrix": cm_data,
                "classification_report": cr_data
            }
    except Exception as e:
        return {"available": False, "error": f"Artifact not found: {str(e)}"}


@app.post("/report_metrics")
async def get_report_metrics(request: MLflowMetricsRequest):
    """Get report metrics (ConfusionMatrix, ClassificationReport) from MLflow artifacts."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_get_report_metrics,
                request.project_name,
                request.model_name
            ),
            timeout=30.0
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="MLflow query timed out")
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/model_available")
async def check_model_available(request: ModelAvailabilityRequest):
    """Check if a trained River model is available in MLflow.
    Uses get_best_mlflow_run to find the best model (same as predictions).
    """
    project_name = request.project_name
    model_name = request.model_name
    try:
        experiment = mlflow.get_experiment_by_name(project_name)
        if experiment is None:
            return {
                "available": False,
                "message": f"No experiment found for {project_name}",
                "experiment_url": None
            }
        experiment_url = f"http://localhost:5001/#/experiments/{experiment.experiment_id}"
        # Get the best run (same logic as model loading for predictions)
        best_run_id = get_best_mlflow_run(project_name, model_name)
        if best_run_id is None:
            return {
                "available": False,
                "message": f"No trained model found for {model_name}",
                "experiment_id": experiment.experiment_id,
                "experiment_url": experiment_url
            }
        # Get full run info
        run = mlflow.get_run(best_run_id)
        # Get metrics from the best run
        metrics = {k: float(v) for k, v in run.data.metrics.items()}
        return {
            "available": True,
            "run_id": best_run_id,
            "trained_at": pd.Timestamp(run.info.start_time, unit='ms').isoformat() if run.info.start_time else None,
            "experiment_id": experiment.experiment_id,
            "experiment_url": experiment_url,
            "metrics": metrics
        }
    except Exception as e:
        print(f"Error checking model availability: {e}", file=sys.stderr)
        return {
            "available": False,
            "message": f"Error checking model: {str(e)}",
            "experiment_url": None
        }


# =============================================================================
# Delta Lake SQL Query Endpoints
# =============================================================================
@app.post("/sql_query")
async def sql_query(request: SQLQueryRequest):
    """
    Execute a SQL query against Delta Lake via DuckDB or Polars.

    The query runs against the Delta Lake table for the specified project.
    The table is accessible as 'data' in your SQL queries.

    Example queries:
    - SELECT * FROM data LIMIT 100
    - SELECT COUNT(*) FROM data WHERE amount > 1000
    - SELECT merchant_id, AVG(amount) FROM data GROUP BY merchant_id

    Engines:
    - polars: Fast, uses Polars SQLContext (default)
    - duckdb: Uses DuckDB with native Delta extension

    Security:
    - Only SELECT queries are allowed
    - DDL/DML operations are blocked
    - Row limits are enforced
    """
    try:
        # Debug: log the received engine parameter
        print(f"SQL Query received - project: {request.project_name}, engine: {request.engine}, query: {request.query[:50]}...")

        # Select the appropriate execution function based on engine
        if request.engine == "duckdb":
            execute_fn = execute_delta_sql
        else:
            execute_fn = execute_delta_sql_polars

        # Run the query in a thread pool to avoid blocking
        result = await asyncio.wait_for(
            asyncio.to_thread(
                execute_fn,
                request.project_name,
                request.query,
                request.limit
            ),
            timeout=60.0  # 60 second timeout for complex queries
        )

        # Check for errors
        if result.get("error"):
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )

        return result

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Query timed out after 60 seconds. Try a simpler query or add LIMIT."
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error executing SQL query: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}"
        )


@app.post("/table_schema")
async def table_schema(request: TableSchemaRequest):
    """
    Get schema and metadata for a Delta Lake table.

    Returns:
    - table_name: Name of the Delta Lake table
    - delta_path: S3 path to the Delta Lake table
    - columns: List of column definitions (name, type, nullable)
    - approximate_row_count: Approximate number of rows in the table
    """
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                get_delta_table_schema,
                request.project_name
            ),
            timeout=30.0
        )

        if result.get("error"):
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )

        return result

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Schema query timed out after 30 seconds."
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting table schema: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get table schema: {str(e)}"
        )
