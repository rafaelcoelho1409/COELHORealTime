"""
Incremental ML Router (River)

Handles incremental/streaming ML model training using the River library.
Features:
- Real-time streaming ML with Kafka consumer
- Live model predictions during training (via Redis cache)
- MLflow experiment tracking
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
import subprocess
import asyncio
import time
import mlflow

from models import (
    SwitchModelRequest,
    PageInitRequest,
    ModelAvailabilityRequest,
    MLflowMetricsRequest,
    SampleRequest,
    PredictRequest,
    OrdinalEncoderRequest,
    Healthcheck,
)
from config import (
    PROJECT_NAMES,
    INCREMENTAL_MODEL_NAMES,
    INCREMENTAL_TRAINING_SCRIPTS,
    BEST_METRIC_CRITERIA,
    MLFLOW_METRICS_CACHE_TTL,
)
from utils.river import (
    is_training_active,
    load_live_model_from_redis,
    load_or_create_model,
    load_or_create_encoders,
    process_sample,
    get_sample_polars,
)


router = APIRouter()


# =============================================================================
# Training State
# =============================================================================
class TrainingState:
    """Tracks current incremental training process state."""

    def __init__(self):
        self.current_process: subprocess.Popen | None = None
        self.current_model_name: str | None = None
        self.status: str = "idle"


state = TrainingState()


# =============================================================================
# Healthcheck State
# =============================================================================
# Initialize healthcheck with default state
healthcheck = Healthcheck(
    model_load={x: "not_attempted" for x in PROJECT_NAMES},
    encoders_load={x: "not_attempted" for x in PROJECT_NAMES},
    data_load={x: "not_attempted" for x in PROJECT_NAMES},
)


# =============================================================================
# MLflow Metrics Cache
# =============================================================================
class MLflowMetricsCache:
    def __init__(self, ttl_seconds: int = MLFLOW_METRICS_CACHE_TTL):
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


mlflow_cache = MLflowMetricsCache(ttl_seconds=5)  # Short TTL for responsive live updates


# =============================================================================
# Experiment Cache
# =============================================================================
_experiment_cache: Dict[str, tuple[float, any]] = {}
_EXPERIMENT_CACHE_TTL = 300  # 5 minutes


def get_cached_experiment(project_name: str):
    """Get MLflow experiment with caching."""
    cache_entry = _experiment_cache.get(project_name)
    if cache_entry:
        timestamp, experiment = cache_entry
        if time.time() - timestamp < _EXPERIMENT_CACHE_TTL:
            return experiment

    experiment = mlflow.get_experiment_by_name(project_name)
    if experiment:
        _experiment_cache[project_name] = (time.time(), experiment)
    return experiment


# =============================================================================
# Best Run Selection
# =============================================================================
def get_best_mlflow_run(project_name: str, model_name: str) -> Optional[str]:
    """Get the best MLflow run based on project-specific metric criteria."""
    experiment = get_cached_experiment(project_name)
    if experiment is None:
        return None

    criteria = BEST_METRIC_CRITERIA.get(project_name, {})
    metric_name = criteria.get("metric_name", "fbeta_score")
    maximize = criteria.get("maximize", True)

    # Search for finished runs with this model name
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_name}' AND attributes.status = 'FINISHED'",
        max_results=100,
    )

    if runs_df.empty:
        return None

    # Find best run based on metric
    metric_col = f"metrics.{metric_name}"
    if metric_col not in runs_df.columns:
        # Fallback to first run
        return runs_df.iloc[0]["run_id"]

    runs_df = runs_df.dropna(subset=[metric_col])
    if runs_df.empty:
        return None

    if maximize:
        best_idx = runs_df[metric_col].idxmax()
    else:
        best_idx = runs_df[metric_col].idxmin()

    return runs_df.loc[best_idx, "run_id"]


# =============================================================================
# Helper Functions
# =============================================================================
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
        process.wait(timeout=60)
        if process.poll() is not None:
            print(f"Model '{model_name}' stopped gracefully (exit code: {process.returncode})")
            state.status = f"Model '{model_name}' stopped."
        else:
            print(f"SIGTERM failed, sending SIGKILL to PID {pid}")
            process.kill()
            process.wait(timeout=10)
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
# API Endpoints
# =============================================================================
@router.get("/health")
async def health():
    """Health check for incremental ML router."""
    return {"status": "healthy", "router": "incremental"}


@router.get("/status")
async def get_status():
    """Get current training status."""
    if state.current_model_name and state.current_process:
        if state.current_process.poll() is None:
            return {
                "current_model": state.current_model_name,
                "status": "running",
                "pid": state.current_process.pid,
            }
        else:
            return_code = state.current_process.returncode
            stop_current_model()
            return {
                "current_model": state.current_model_name,
                "status": f"stopped (exit code: {return_code})",
                "pid": None,
            }
    return {"current_model": None, "status": "idle"}


@router.post("/switch-model")
async def switch_model(request: SwitchModelRequest):
    """
    Start or stop incremental ML model training.

    model_key: Script filename (e.g., "transaction_fraud_detection_river.py")
               or "none" to stop training
    """
    model_key = request.model_key

    # Build mapping from script names to project names
    model_scripts = {
        script: project
        for project, script in INCREMENTAL_TRAINING_SCRIPTS.items()
    }

    if model_key == state.current_model_name:
        return {"message": f"Model {model_key} is already running."}

    if state.current_process:
        print(f"Switching from {state.current_model_name} to {model_key}")
        stop_current_model()
    else:
        print(f"No model running, attempting to start {model_key}")

    if model_key == "none" or model_key not in model_scripts:
        if model_key == "none":
            return {"message": "All models stopped."}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Model key '{model_key}' not found. Available: {list(model_scripts.keys())}",
            )

    # Start training subprocess
    command = ["/app/.venv/bin/python3", "-u", model_key]

    try:
        import os

        print(f"Starting model: {model_key}")
        log_dir = "/app/logs"
        os.makedirs(log_dir, exist_ok=True)
        # Flatten the model_key path for log filename (e.g., "ml_training/river/tfd.py" -> "ml_training_river_tfd.py")
        log_filename = model_key.replace("/", "_").replace(".py", "") + ".log"
        log_file_path = f"{log_dir}/{log_filename}"

        # Set PYTHONPATH so training scripts can import from /app (utils, config, etc.)
        env = os.environ.copy()
        env["PYTHONPATH"] = "/app"

        with open(log_file_path, "ab") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd="/app",
                env=env,
            )

        state.current_process = process
        state.current_model_name = model_key
        state.status = f"Running {model_key}"

        print(f"Model {model_key} started with PID: {process.pid}")
        return {"message": f"Started model: {model_key}", "pid": process.pid}

    except Exception as e:
        print(f"Failed to start model {model_key}: {e}")
        state.current_process = None
        state.current_model_name = None
        state.status = f"Failed to start: {e}"
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model {model_key}: {str(e)}",
        )


@router.get("/training-status/{project_name}")
async def get_training_status(project_name: str):
    """Check if training is active for a project."""
    model_name = INCREMENTAL_MODEL_NAMES.get(project_name)
    if not model_name:
        raise HTTPException(status_code=400, detail=f"Unknown project: {project_name}")

    # Check if current training matches this project
    script_name = INCREMENTAL_TRAINING_SCRIPTS.get(project_name)
    is_active = state.current_model_name == script_name and state.current_process is not None

    return {
        "project_name": project_name,
        "model_name": model_name,
        "is_active": is_active,
        "model_source": "live" if is_active else "mlflow",
    }


@router.post("/predict")
async def predict(request: PredictRequest):
    """
    Make predictions using the best available model.

    Priority order:
    1. Live model from Redis (if training is active) - real-time updates
    2. Best model from MLflow (if training is inactive) - production quality
    """
    project_name = request.project_name
    model_name = request.model_name

    if not project_name or not model_name:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: project_name and model_name",
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
            model_source = "mlflow"
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' for project '{project_name}' not available. Train a model first.",
            )

    # Load encoders (from Redis if live, otherwise from MLflow)
    if encoders is None:
        try:
            encoders = load_or_create_encoders(project_name, "river")
        except Exception as e:
            print(f"Error loading encoders: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Encoders for project '{project_name}' not available. Train a model first.",
            )

    # Extract feature data from request (convert Pydantic model to dict)
    payload = request.model_dump(exclude_none=True)
    x = {k: v for k, v in payload.items() if k not in ["project_name", "model_name", "run_id"]}

    # Process and predict based on model type
    if model_name in ["ARFClassifier", "ARFRegressor", "DBSTREAM"]:
        try:
            processed_x, _ = process_sample(x, encoders, project_name)

            if project_name == "Transaction Fraud Detection":
                y_pred_proba = model.predict_proba_one(processed_x)
                fraud_probability = y_pred_proba.get(1, 0.0) if y_pred_proba else 0.0
                binary_prediction = 1 if fraud_probability >= 0.5 else 0
                return {
                    "fraud_probability": fraud_probability,
                    "prediction": binary_prediction,
                    "model_source": model_source,
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
            print(f"Error during prediction: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}",
            )

    raise HTTPException(
        status_code=400,
        detail=f"Unknown model: {model_name}",
    )


@router.post("/model-available")
async def check_model_available(request: ModelAvailabilityRequest):
    """Check if a trained incremental model is available in MLflow."""
    project_name = request.project_name
    model_name = request.model_name

    try:
        experiment = get_cached_experiment(project_name)
        if experiment is None:
            return {
                "available": False,
                "message": f"No experiment found for {project_name}",
                "experiment_url": None,
            }

        experiment_url = f"http://localhost:5001/#/experiments/{experiment.experiment_id}"
        best_run_id = get_best_mlflow_run(project_name, model_name)

        if best_run_id is None:
            return {
                "available": False,
                "message": f"No trained model found for {model_name}",
                "experiment_id": experiment.experiment_id,
                "experiment_url": experiment_url,
            }

        run = mlflow.get_run(best_run_id)
        import pandas as pd

        return {
            "available": True,
            "run_id": best_run_id,
            "trained_at": pd.Timestamp(run.info.start_time, unit="ms").isoformat()
            if run.info.start_time
            else None,
            "experiment_id": experiment.experiment_id,
            "experiment_url": experiment_url,
        }

    except Exception as e:
        print(f"Error checking model availability: {e}")
        return {
            "available": False,
            "message": f"Error checking model: {str(e)}",
            "experiment_url": None,
        }


@router.post("/mlflow-metrics")
async def get_mlflow_metrics(request: MLflowMetricsRequest):
    """Get MLflow metrics with caching."""
    cache_key = f"{request.project_name}:{request.model_name}"

    if not request.force_refresh:
        cached_result = mlflow_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

    try:
        experiment = get_cached_experiment(request.project_name)
        if experiment is None:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{request.project_name}' not found in MLflow",
            )

        # First check for RUNNING experiments (real-time training)
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            max_results=10,
            order_by=["start_time DESC"],
        )

        if not runs_df.empty:
            running_runs = runs_df[runs_df["tags.mlflow.runName"] == request.model_name]
            if not running_runs.empty:
                run_id = running_runs.iloc[0]["run_id"]
                run = mlflow.get_run(run_id)
                result = {
                    "run_id": run_id,
                    "status": "RUNNING",
                    "start_time": run.info.start_time,
                    "is_live": True,
                }
                for metric_name, metric_value in run.data.metrics.items():
                    result[f"metrics.{metric_name}"] = metric_value
                mlflow_cache.set(cache_key, result)
                return result

        # No running experiment - fall back to best FINISHED model
        best_run_id = get_best_mlflow_run(request.project_name, request.model_name)
        if best_run_id is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for '{request.model_name}' in '{request.project_name}'",
            )

        run = mlflow.get_run(best_run_id)
        result = {
            "run_id": best_run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "is_live": False,
        }
        for metric_name, metric_value in run.data.metrics.items():
            result[f"metrics.{metric_name}"] = metric_value

        mlflow_cache.set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching MLflow metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch MLflow metrics: {str(e)}",
        )


# =============================================================================
# Cluster Artifact Cache
# =============================================================================
_cluster_cache: dict = {}
_CLUSTER_CACHE_TTL = 60  # 1 minute cache


# =============================================================================
# Report Metrics Cache
# =============================================================================
_report_metrics_cache: dict = {}
_REPORT_METRICS_CACHE_TTL = 30  # 30 seconds cache


@router.post("/report-metrics")
async def get_report_metrics(request: MLflowMetricsRequest):
    """Get report metrics (ConfusionMatrix, ClassificationReport) from MLflow artifacts.

    Loads report_metrics.pkl artifact which contains River's ConfusionMatrix
    and ClassificationReport metrics for binary classification (TFD).
    """
    import pickle
    import tempfile

    project_name = request.project_name
    model_name = request.model_name

    try:
        experiment = get_cached_experiment(project_name)
        if experiment is None:
            return {"available": False, "error": f"Experiment '{project_name}' not found"}

        run_id = None
        is_live = False

        # Check for RUNNING experiment first
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
            max_results=10,
            order_by=["start_time DESC"],
        )
        if not runs_df.empty:
            running_runs = runs_df[runs_df["tags.mlflow.runName"] == model_name]
            if not running_runs.empty:
                run_id = running_runs.iloc[0]["run_id"]
                is_live = True

        # Fall back to best FINISHED model
        if run_id is None:
            run_id = get_best_mlflow_run(project_name, model_name)

        if run_id is None:
            return {"available": False, "error": "No trained model found"}

        # Check cache (skip for live training)
        if not is_live:
            cache_key = f"report_metrics:{run_id}"
            cache_entry = _report_metrics_cache.get(cache_key)
            if cache_entry:
                timestamp, cached_run_id, data = cache_entry
                if time.time() - timestamp < _REPORT_METRICS_CACHE_TTL and cached_run_id == run_id:
                    return data

        # Download and load report_metrics.pkl artifact
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="report_metrics.pkl",
                    dst_path=tmpdir,
                )
                with open(artifact_path, "rb") as f:
                    report_metrics = pickle.load(f)

                # Extract ConfusionMatrix data
                cm = report_metrics.get("ConfusionMatrix")
                cm_data = {"available": False}
                if cm is not None:
                    try:
                        cm_dict = dict(cm.data)
                        tn = cm_dict.get(0, {}).get(0, 0)
                        fp = cm_dict.get(0, {}).get(1, 0)
                        fn = cm_dict.get(1, {}).get(0, 0)
                        tp = cm_dict.get(1, {}).get(1, 0)
                        total = tn + fp + fn + tp
                        cm_data = {
                            "available": True,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "tp": tp,
                            "total": total,
                        }
                    except Exception as e:
                        cm_data = {"available": False, "error": str(e)}

                # Extract ClassificationReport data
                cr = report_metrics.get("ClassificationReport")
                cr_data = {"available": False}
                if cr is not None:
                    try:
                        cr_data = {"available": True, "report": str(cr)}
                    except Exception as e:
                        cr_data = {"available": False, "error": str(e)}

                result = {
                    "available": True,
                    "run_id": run_id,
                    "confusion_matrix": cm_data,
                    "classification_report": cr_data,
                }

                # Update cache (only for non-live runs)
                if not is_live:
                    cache_key = f"report_metrics:{run_id}"
                    _report_metrics_cache[cache_key] = (time.time(), run_id, result)

                return result

        except Exception as e:
            return {"available": False, "error": f"Artifact not found: {str(e)}"}

    except Exception as e:
        return {"available": False, "error": str(e)}


@router.get("/cluster-counts")
async def get_cluster_counts():
    """Get cluster counts from MLflow artifacts (for DBSTREAM clustering).

    Results are cached for 1 minute to avoid repeated MLflow artifact downloads.
    """
    import json as json_lib

    try:
        project_name = "E-Commerce Customer Interactions"
        model_name = INCREMENTAL_MODEL_NAMES.get(project_name)
        run_id = get_best_mlflow_run(project_name, model_name)

        if run_id is None:
            return {}

        # Check cache
        cache_key = f"cluster_counts:{run_id}"
        cache_entry = _cluster_cache.get(cache_key)
        if cache_entry:
            timestamp, cached_run_id, data = cache_entry
            if time.time() - timestamp < _CLUSTER_CACHE_TTL and cached_run_id == run_id:
                return data

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="cluster_counts.json",
        )
        with open(local_path, "r") as f:
            cluster_counts = json_lib.load(f)

        # Update cache
        _cluster_cache[cache_key] = (time.time(), run_id, cluster_counts)
        return cluster_counts

    except Exception as e:
        print(f"Error fetching cluster counts from MLflow: {e}")
        return {}


@router.post("/cluster-feature-counts")
async def get_cluster_feature_counts(payload: dict):
    """Get cluster feature counts for a specific column from MLflow artifacts.

    Results are cached for 1 minute to avoid repeated MLflow artifact downloads.

    Payload:
        column_name: str - The feature column to get counts for
    """
    import json as json_lib

    column_name = payload.get("column_name")
    if not column_name:
        return {}

    try:
        project_name = "E-Commerce Customer Interactions"
        model_name = INCREMENTAL_MODEL_NAMES.get(project_name)
        run_id = get_best_mlflow_run(project_name, model_name)

        if run_id is None:
            return {}

        # Check cache for full feature counts data
        cache_key = f"cluster_feature_counts:{run_id}"
        cache_entry = _cluster_cache.get(cache_key)
        if cache_entry:
            timestamp, cached_run_id, data = cache_entry
            if time.time() - timestamp < _CLUSTER_CACHE_TTL and cached_run_id == run_id:
                clusters = list(data.keys())
                return {x: data[x].get(column_name, {}) for x in clusters}

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="cluster_feature_counts.json",
        )
        with open(local_path, "r") as f:
            cluster_counts = json_lib.load(f)

        # Update cache with full data
        _cluster_cache[cache_key] = (time.time(), run_id, cluster_counts)

        clusters = list(cluster_counts.keys())
        return {x: cluster_counts[x].get(column_name, {}) for x in clusters}

    except Exception as e:
        print(f"Error fetching cluster feature counts from MLflow: {e}")
        return {}


@router.post("/page-init")
async def page_init(request: PageInitRequest):
    """Combined page initialization endpoint.

    Returns all data needed for page load in a single call:
    - model_available: Check if trained model exists in MLflow
    - mlflow_metrics: Latest metrics from best model run
    - training_status: Whether training is currently active
    """
    project_name = request.project_name
    model_name = request.model_name

    result = {
        "model_available": {"available": False},
        "mlflow_metrics": {},
        "training_status": {"is_training": False},
    }

    try:
        experiment = get_cached_experiment(project_name)
        if experiment:
            experiment_url = f"http://localhost:5001/#/experiments/{experiment.experiment_id}"

            # Check for best model
            best_run_id = get_best_mlflow_run(project_name, model_name)
            if best_run_id:
                run = mlflow.get_run(best_run_id)
                import pandas as pd

                mlflow_metrics = {
                    "run_id": best_run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "is_live": False,
                }
                for k, v in run.data.metrics.items():
                    mlflow_metrics[f"metrics.{k}"] = float(v)

                result["model_available"] = {
                    "available": True,
                    "run_id": best_run_id,
                    "trained_at": pd.Timestamp(run.info.start_time, unit="ms").isoformat()
                    if run.info.start_time
                    else None,
                    "experiment_id": experiment.experiment_id,
                    "experiment_url": experiment_url,
                }
                result["mlflow_metrics"] = mlflow_metrics
            else:
                result["model_available"] = {
                    "available": False,
                    "message": f"No trained model found for {model_name}",
                    "experiment_id": experiment.experiment_id,
                    "experiment_url": experiment_url,
                }

        # Training status
        script_name = INCREMENTAL_TRAINING_SCRIPTS.get(project_name)
        is_training = state.current_model_name == script_name and state.current_process is not None
        result["training_status"] = {"is_training": is_training}

    except Exception as e:
        print(f"Error in page_init for {project_name}: {e}")

    return result


@router.get("/current-model")
async def get_current_model():
    """Get currently running model."""
    return await get_status()


@router.get("/healthcheck", response_model=Healthcheck)
async def get_healthcheck():
    """Get detailed healthcheck status."""
    if ("failed" in healthcheck.model_load.values()) or ("failed" in healthcheck.data_load.values()):
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Core components failed to load."
        )
    return healthcheck


@router.put("/healthcheck", response_model=Healthcheck)
async def update_healthcheck(update_data: Healthcheck):
    """Update healthcheck status."""
    global healthcheck
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(healthcheck, field, value)
    return healthcheck


@router.post("/sample")
async def get_sample(request: SampleRequest):
    """Get a random sample from the dataset using DuckDB."""
    from models import (
        TransactionFraudDetection,
        EstimatedTimeOfArrival,
        ECommerceCustomerInteractions,
    )
    try:
        sample_df = get_sample_polars(request.project_name, n=1)
        if sample_df is None or sample_df.empty:
            raise HTTPException(
                status_code=503,
                detail="Could not get sample from data source."
            )
        sample = sample_df.to_dict(orient='records')[0]
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
            status_code=500,
            detail=f"Error sampling data: {e}"
        )


@router.post("/get-ordinal-encoder")
async def get_ordinal_encoder(request: OrdinalEncoderRequest):
    """Get ordinal encoder mappings for a project."""
    encoders = load_or_create_encoders(request.project_name, "river")
    if None in encoders.values():
        raise HTTPException(
            status_code=503,
            detail="Ordinal encoder is not loaded."
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
        status_code=400,
        detail=f"Unknown project: {request.project_name}"
    )
