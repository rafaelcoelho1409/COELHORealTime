"""
Batch ML Router (Scikit-Learn/CatBoost)

Handles batch ML model training using Scikit-Learn and CatBoost.
Features:
- Subprocess-based batch training
- YellowBrick visualizations
- MLflow experiment tracking and model versioning
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
import subprocess
import asyncio
import time
import os
from datetime import datetime
import mlflow

from models import (
    BatchSwitchModelRequest,
    BatchInitRequest,
    BatchMLflowRunsRequest,
    ModelAvailabilityRequest,
    MLflowMetricsRequest,
    TrainingStatusUpdate,
    YellowBrickRequest,
    PredictRequest,
    SklearnHealthcheck,
)
from config import (
    PROJECT_NAMES,
    BATCH_MODEL_NAMES,
    BATCH_TRAINING_SCRIPTS,
    BEST_METRIC_CRITERIA,
    PROJECT_TASK_TYPES,
    MLFLOW_METRICS_CACHE_TTL,
    DELTA_PATHS,
    AWS_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_S3_ENDPOINT,
)


router = APIRouter()


# =============================================================================
# Training State
# =============================================================================
class BatchTrainingState:
    """Tracks current batch training process state."""

    def __init__(self):
        self.current_process: subprocess.Popen | None = None
        self.current_model_name: str | None = None
        self.status: str = "idle"
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.exit_code: int | None = None
        self.log_file = None
        # Live training status
        self.status_message: str = ""
        self.progress_percent: int = 0
        self.current_stage: str = ""
        self.metrics_preview: Dict[str, float] = {}
        self.total_rows: int = 0

    def close_log_file(self):
        """Close the log file handle if open."""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None

    def update_status(
        self,
        message: str,
        progress: int = None,
        stage: str = None,
        metrics: Dict[str, float] = None,
        total_rows: int = None,
    ):
        """Update training status from training script."""
        self.status_message = message
        if progress is not None:
            self.progress_percent = progress
        if stage is not None:
            self.current_stage = stage
        if metrics is not None:
            self.metrics_preview = metrics
        if total_rows is not None:
            self.total_rows = total_rows

    def reset_status(self):
        """Reset status for new training run."""
        self.status_message = ""
        self.progress_percent = 0
        self.current_stage = ""
        self.metrics_preview = {}
        self.total_rows = 0


batch_state = BatchTrainingState()


# =============================================================================
# Healthcheck State
# =============================================================================
sklearn_healthcheck = SklearnHealthcheck()


# =============================================================================
# Model Cache (for predictions using best model from MLflow)
# =============================================================================
class ModelCache:
    """Cache for loaded models and encoders from MLflow."""
    def __init__(self, ttl_seconds: int = 30):
        self.models: Dict[str, any] = {}
        self.encoders: Dict[str, any] = {}
        self.run_ids: Dict[str, str] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttl_seconds = ttl_seconds

    def _is_expired(self, project_name: str) -> bool:
        """Check if cache entry has expired."""
        if project_name not in self.timestamps:
            return True
        return (time.time() - self.timestamps[project_name]) > self.ttl_seconds

    def get_model(self, project_name: str):
        """Get cached model or load from MLflow if expired/missing."""
        from utils.sklearn import load_model_from_mlflow
        if project_name in self.models and not self._is_expired(project_name):
            return self.models[project_name], self.run_ids.get(project_name)
        # Load from MLflow
        model_name = BATCH_MODEL_NAMES.get(project_name)
        if not model_name:
            return None, None
        run_id = get_best_mlflow_run(project_name, model_name)
        if run_id is None:
            return None, None
        model = load_model_from_mlflow(project_name, model_name, run_id=run_id)
        if model is not None:
            self.models[project_name] = model
            self.run_ids[project_name] = run_id
            self.timestamps[project_name] = time.time()
            print(f"Model cached for {project_name} (run_id={run_id})")
        return model, run_id

    def invalidate(self, project_name: str = None):
        """Invalidate cache for a project or all projects."""
        if project_name:
            self.models.pop(project_name, None)
            self.encoders.pop(project_name, None)
            self.run_ids.pop(project_name, None)
            self.timestamps.pop(project_name, None)
        else:
            self.models.clear()
            self.encoders.clear()
            self.run_ids.clear()
            self.timestamps.clear()


model_cache = ModelCache(ttl_seconds=30)  # 30 second cache


# =============================================================================
# MLflow Caches
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


mlflow_cache = MLflowMetricsCache(ttl_seconds=30)


# =============================================================================
# Helper Functions
# =============================================================================
def get_best_mlflow_run(project_name: str, model_name: str) -> Optional[str]:
    """Get the best MLflow run based on project-specific metric criteria."""
    experiment = mlflow.get_experiment_by_name(project_name)
    if experiment is None:
        return None

    criteria = BEST_METRIC_CRITERIA.get(project_name, {})
    metric_name = criteria.get("metric_name", "fbeta_score")
    maximize = criteria.get("maximize", True)

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_name}' AND attributes.status = 'FINISHED'",
        max_results=100,
    )

    if runs_df.empty:
        return None

    metric_col = f"metrics.{metric_name}"
    if metric_col not in runs_df.columns:
        return runs_df.iloc[0]["run_id"]

    runs_df = runs_df.dropna(subset=[metric_col])
    if runs_df.empty:
        return None

    if maximize:
        best_idx = runs_df[metric_col].idxmax()
    else:
        best_idx = runs_df[metric_col].idxmin()

    return runs_df.loc[best_idx, "run_id"]


def get_all_mlflow_runs(project_name: str, model_name: str) -> list:
    """Get all MLflow runs for a project, ordered by metric criteria (best first)."""
    experiment = mlflow.get_experiment_by_name(project_name)
    if experiment is None:
        return []

    criteria = BEST_METRIC_CRITERIA.get(project_name, {})
    metric_name = criteria.get("metric_name", "fbeta_score")
    maximize = criteria.get("maximize", True)

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_name}'",
        max_results=100,
        order_by=["start_time DESC"],
    )

    if runs_df.empty:
        return []

    # Sort by metric if available
    metric_col = f"metrics.{metric_name}"
    if metric_col in runs_df.columns:
        runs_df = runs_df.sort_values(by=metric_col, ascending=not maximize, na_position="last")

    # Mark best run
    best_run_id = get_best_mlflow_run(project_name, model_name)

    runs = []
    for _, row in runs_df.iterrows():
        run_info = {
            "run_id": row["run_id"],
            "run_name": row.get("tags.mlflow.runName", model_name),
            "start_time": row["start_time"].isoformat() if row.get("start_time") else None,
            "end_time": row["end_time"].isoformat() if row.get("end_time") else None,
            "metrics": {},
            "params": {},
            "total_rows": 0,
            "is_best": row["run_id"] == best_run_id,
        }

        # Add metrics
        for col in runs_df.columns:
            if col.startswith("metrics."):
                metric_key = col.replace("metrics.", "")
                value = row[col]
                if value is not None and str(value) != "nan":
                    run_info["metrics"][metric_key] = float(value)

        # Add params
        for col in runs_df.columns:
            if col.startswith("params."):
                param_key = col.replace("params.", "")
                value = row[col]
                if value is not None and str(value) != "nan":
                    run_info["params"][param_key] = value

        # Calculate total rows from params
        train_samples = run_info["params"].get("train_samples")
        test_samples = run_info["params"].get("test_samples")
        if train_samples and test_samples:
            run_info["total_rows"] = int(train_samples) + int(test_samples)

        runs.append(run_info)

    return runs


def stop_current_training() -> bool:
    """Stop the currently running batch training process gracefully."""
    if not batch_state.current_process:
        batch_state.status = "No training was active to stop."
        return True

    model_name = batch_state.current_model_name
    process = batch_state.current_process
    pid = process.pid

    print(f"Stopping batch training '{model_name}' (PID: {pid}) with SIGTERM...")
    batch_state.status = f"Stopping '{model_name}'..."

    try:
        process.terminate()
        process.wait(timeout=60)
        if process.poll() is not None:
            print(f"Training '{model_name}' stopped gracefully (exit code: {process.returncode})")
            batch_state.status = f"Training '{model_name}' stopped."
            batch_state.exit_code = process.returncode
        else:
            print(f"SIGTERM failed, sending SIGKILL to PID {pid}")
            process.kill()
            process.wait(timeout=10)
            batch_state.status = f"Training '{model_name}' force killed."
            batch_state.exit_code = -9
    except subprocess.TimeoutExpired:
        print(f"Timeout waiting for {model_name}, force killing...")
        process.kill()
        batch_state.status = f"Training '{model_name}' force killed after timeout."
        batch_state.exit_code = -9
    except Exception as e:
        print(f"Error stopping training: {e}")
        batch_state.status = f"Error stopping training: {e}"
    finally:
        batch_state.close_log_file()
        batch_state.current_process = None
        batch_state.current_model_name = None
        batch_state.completed_at = datetime.utcnow().isoformat() + "Z"

    return True


# =============================================================================
# API Endpoints
# =============================================================================
@router.get("/health")
async def health():
    """Health check for batch ML router."""
    return {"status": "healthy", "router": "batch"}


@router.get("/status")
async def get_batch_status():
    """Get current batch training status including live progress updates."""
    live_status = {
        "status_message": batch_state.status_message,
        "progress_percent": batch_state.progress_percent,
        "current_stage": batch_state.current_stage,
        "metrics_preview": batch_state.metrics_preview,
        "total_rows": batch_state.total_rows,
    }

    if batch_state.current_model_name and batch_state.current_process:
        poll_result = batch_state.current_process.poll()
        if poll_result is None:
            return {
                "current_model": batch_state.current_model_name,
                "status": "running",
                "pid": batch_state.current_process.pid,
                "started_at": batch_state.started_at,
                **live_status,
            }
        else:
            exit_code = batch_state.current_process.returncode
            model_name = batch_state.current_model_name
            batch_state.exit_code = exit_code
            batch_state.completed_at = datetime.utcnow().isoformat() + "Z"
            batch_state.close_log_file()
            batch_state.current_process = None
            batch_state.current_model_name = None

            if exit_code == 0:
                batch_state.status = "completed"
                return {
                    "current_model": model_name,
                    "status": "completed",
                    "exit_code": exit_code,
                    "started_at": batch_state.started_at,
                    "completed_at": batch_state.completed_at,
                    **live_status,
                }
            else:
                batch_state.status = "failed"
                return {
                    "current_model": model_name,
                    "status": "failed",
                    "exit_code": exit_code,
                    "started_at": batch_state.started_at,
                    "completed_at": batch_state.completed_at,
                    **live_status,
                }

    return {
        "current_model": None,
        "status": batch_state.status,
        "exit_code": batch_state.exit_code,
        "started_at": batch_state.started_at,
        "completed_at": batch_state.completed_at,
        **live_status,
    }


@router.post("/switch-model")
async def switch_batch_model(request: BatchSwitchModelRequest):
    """
    Start or stop batch ML model training.

    model_key: Script filename (e.g., "transaction_fraud_detection_sklearn.py")
               or "none" to stop training
    sample_frac: Optional fraction of data to use (0.0-1.0)
    max_rows: Optional maximum number of rows to use
    """
    model_key = request.model_key
    sample_frac = request.sample_frac
    max_rows = request.max_rows

    # Build mapping from script names to project names
    model_scripts = {script: project for project, script in BATCH_TRAINING_SCRIPTS.items()}

    if model_key == "none":
        if batch_state.current_process:
            stop_current_training()
            return {"message": "Training stopped."}
        return {"message": "No training was running."}

    if model_key == batch_state.current_model_name and batch_state.current_process:
        poll_result = batch_state.current_process.poll()
        if poll_result is None:
            return {
                "message": f"Training {model_key} is already running.",
                "pid": batch_state.current_process.pid,
            }

    if batch_state.current_process:
        print(f"Stopping current training before starting {model_key}")
        stop_current_training()

    if model_key not in model_scripts:
        raise HTTPException(
            status_code=404,
            detail=f"Model key '{model_key}' not found. Available: {list(model_scripts.keys())}",
        )

    command = ["/app/.venv/bin/python3", "-u", model_key]

    if sample_frac is not None and 0.0 < sample_frac <= 1.0:
        command.extend(["--sample-frac", str(sample_frac)])
    elif max_rows is not None and max_rows > 0:
        command.extend(["--max-rows", str(max_rows)])

    try:
        print(f"Starting batch training: {model_key}")
        log_dir = "/app/logs"
        os.makedirs(log_dir, exist_ok=True)
        # Flatten the model_key path for log filename (e.g., "ml_training/sklearn/tfd.py" -> "ml_training_sklearn_tfd.log")
        log_filename = model_key.replace("/", "_").replace(".py", "") + ".log"
        log_file_path = f"{log_dir}/{log_filename}"

        # Set PYTHONPATH so training scripts can import from /app (utils, config, etc.)
        env = os.environ.copy()
        env["PYTHONPATH"] = "/app"

        log_file = open(log_file_path, "a", buffering=1)
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd="/app",
            env=env,
        )

        batch_state.log_file = log_file
        batch_state.current_process = process
        batch_state.current_model_name = model_key
        batch_state.status = "running"
        batch_state.started_at = datetime.utcnow().isoformat() + "Z"
        batch_state.completed_at = None
        batch_state.exit_code = None
        batch_state.reset_status()
        batch_state.update_status("Starting training...", progress=0, stage="initializing")

        print(f"Batch training {model_key} started with PID: {process.pid}")
        return {"message": f"Started training: {model_key}", "pid": process.pid}

    except Exception as e:
        print(f"Failed to start training {model_key}: {e}")
        batch_state.close_log_file()
        batch_state.current_process = None
        batch_state.current_model_name = None
        batch_state.status = f"Failed to start: {e}"
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training {model_key}: {str(e)}",
        )


@router.post("/stop-training")
async def stop_training():
    """Stop the current batch training process and reset state."""
    if not batch_state.current_process:
        return {"status": "idle", "message": "No training is currently running."}

    model_name = batch_state.current_model_name
    stop_current_training()
    batch_state.reset_status()
    batch_state.status = "idle"

    return {"status": "stopped", "message": f"Training '{model_name}' has been stopped."}


@router.post("/training-status")
async def update_training_status(update: TrainingStatusUpdate):
    """Receive live training status updates from training script."""
    batch_state.update_status(
        message=update.message,
        progress=update.progress,
        stage=update.stage,
        metrics=update.metrics,
        total_rows=update.total_rows,
    )
    return {"status": "ok"}


@router.post("/predict")
async def predict(request: PredictRequest):
    """Make batch ML prediction using models from MLflow.

    Supports all three project types:
    - Transaction Fraud Detection (TFD): Binary classification
    - Estimated Time of Arrival (ETA): Regression
    - E-Commerce Customer Interactions (ECCI): Clustering

    The model is loaded from MLflow artifacts for the specified run_id,
    or the best run if no run_id is provided.
    """
    import pickle
    import pandas as pd
    import numpy as np

    project_name = request.project_name
    model_name = request.model_name
    run_id = request.run_id

    # Get run_id if not specified
    if not run_id:
        run_id = get_best_mlflow_run(project_name, model_name)
        if not run_id:
            raise HTTPException(status_code=404, detail="No trained model found")

    task_type = PROJECT_TASK_TYPES.get(project_name, "classification")

    try:
        # Load model from MLflow
        def load_model():
            # Try loading via MLflow's catboost flavor first
            try:
                import mlflow.catboost
                model_uri = f"runs:/{run_id}/model"
                return mlflow.catboost.load_model(model_uri)
            except Exception:
                pass
            # Fallback: load from pickle artifact
            artifact_path = f"{model_name}.pkl"
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
            )
            with open(local_path, 'rb') as f:
                return pickle.load(f)

        model = await asyncio.to_thread(load_model)

        # Process sample based on project type
        if project_name == "Transaction Fraud Detection":
            # Build feature dict
            device_info = request.device_info or {}
            timestamp = pd.to_datetime(request.timestamp)
            features = {
                "amount": request.amount,
                "account_age_days": request.account_age_days,
                "cvv_provided": int(request.cvv_provided or 0),
                "billing_address_match": int(request.billing_address_match or 0),
                "currency": request.currency,
                "merchant_id": request.merchant_id,
                "payment_method": request.payment_method,
                "product_category": request.product_category,
                "transaction_type": request.transaction_type,
                "browser": device_info.get("browser"),
                "os": device_info.get("os"),
                "year": timestamp.year,
                "month": timestamp.month,
                "day": timestamp.day,
                "hour": timestamp.hour,
                "minute": timestamp.minute,
                "second": timestamp.second,
            }
            df = pd.DataFrame([features])
            # Predict
            prediction = int(model.predict(df)[0])
            proba = model.predict_proba(df)[0]
            fraud_probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            return {
                "prediction": prediction,
                "fraud_probability": fraud_probability,
                "model_source": "mlflow",
                "run_id": run_id,
            }

        elif project_name == "Estimated Time of Arrival":
            # Build feature dict
            timestamp = pd.to_datetime(request.timestamp)
            features = {
                "estimated_distance_km": request.estimated_distance_km,
                "temperature_celsius": request.temperature_celsius,
                "driver_rating": request.driver_rating,
                "hour_of_day": request.hour_of_day,
                "initial_estimated_travel_time_seconds": request.initial_estimated_travel_time_seconds,
                "debug_traffic_factor": request.debug_traffic_factor,
                "debug_weather_factor": request.debug_weather_factor,
                "debug_incident_delay_seconds": request.debug_incident_delay_seconds,
                "debug_driver_factor": request.debug_driver_factor,
                "trip_id": request.trip_id,
                "driver_id": request.driver_id,
                "vehicle_id": request.vehicle_id,
                "origin": str(request.origin),
                "destination": str(request.destination),
                "weather": request.weather,
                "day_of_week": request.day_of_week,
                "vehicle_type": request.vehicle_type,
                "year": timestamp.year,
                "month": timestamp.month,
                "day": timestamp.day,
                "hour": timestamp.hour,
                "minute": timestamp.minute,
                "second": timestamp.second,
            }
            df = pd.DataFrame([features])
            # Predict
            prediction = float(model.predict(df)[0])
            return {
                "estimated_travel_time_seconds": prediction,
                "model_source": "mlflow",
                "run_id": run_id,
            }

        elif project_name == "E-Commerce Customer Interactions":
            # Build feature dict for clustering
            device_info = request.device_info or {}
            timestamp = pd.to_datetime(request.timestamp) if request.timestamp else pd.Timestamp.now()
            features = {
                "price": request.price or 0,
                "quantity": request.quantity or 0,
                "session_event_sequence": request.session_event_sequence or 0,
                "time_on_page_seconds": request.time_on_page_seconds or 0,
                "event_type": request.event_type or "unknown",
                "product_category": request.product_category or "unknown",
                "product_id": request.product_id or "unknown",
                "referrer_url": request.referrer_url or "unknown",
                "browser": device_info.get("browser", "unknown"),
                "os": device_info.get("os", "unknown"),
                "year": timestamp.year,
                "month": timestamp.month,
                "day": timestamp.day,
                "hour": timestamp.hour,
                "minute": timestamp.minute,
                "second": timestamp.second,
            }
            df = pd.DataFrame([features])
            # For clustering, predict returns cluster assignment
            cluster_id = int(model.predict(df)[0])
            return {
                "cluster_id": cluster_id,
                "model_source": "mlflow",
                "run_id": run_id,
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown project: {project_name}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/model-available")
async def check_model_available(request: ModelAvailabilityRequest):
    """Check if a trained batch model is available in MLflow."""
    project_name = request.project_name
    model_name = request.model_name

    try:
        experiment = mlflow.get_experiment_by_name(project_name)
        if experiment is None:
            return {
                "available": False,
                "message": f"No experiment found for {project_name}",
                "experiment_url": None,
            }

        experiment_url = f"http://localhost:5001/#/experiments/{experiment.experiment_id}"

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            return {
                "available": False,
                "message": f"No trained model found for {model_name}",
                "experiment_url": experiment_url,
            }

        run = runs.iloc[0]
        import pandas as pd

        return {
            "available": True,
            "run_id": run["run_id"],
            "trained_at": run["start_time"].isoformat() if pd.notna(run["start_time"]) else None,
            "experiment_url": experiment_url,
        }

    except Exception as e:
        print(f"Error checking model availability: {e}")
        return {"available": False, "error": str(e), "experiment_url": None}


@router.post("/mlflow-runs")
async def list_mlflow_runs(request: BatchMLflowRunsRequest):
    """List all MLflow runs for a project, ordered by metric criteria (best first)."""
    project_name = request.project_name
    model_name = request.model_name or BATCH_MODEL_NAMES.get(project_name)

    if not model_name:
        raise HTTPException(status_code=400, detail=f"Unknown project: {project_name}")

    try:
        runs = await asyncio.wait_for(
            asyncio.to_thread(get_all_mlflow_runs, project_name, model_name),
            timeout=30.0,
        )
        return runs
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="MLflow query timed out")
    except Exception as e:
        print(f"Error listing MLflow runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list MLflow runs: {str(e)}")


@router.post("/mlflow-metrics")
async def get_mlflow_metrics(request: MLflowMetricsRequest):
    """Get MLflow metrics for a specific run or best run."""
    cache_run_key = request.run_id or "best"
    cache_key = f"{request.project_name}:{request.model_name}:{cache_run_key}"

    if not request.force_refresh:
        cached_result = mlflow_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

    try:
        if request.run_id:
            run = mlflow.get_run(request.run_id)
            experiment_id = run.info.experiment_id
            result = {
                "run_id": run.info.run_id,
                "experiment_id": experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "run_url": f"http://localhost:5001/#/experiments/{experiment_id}/runs/{run.info.run_id}",
            }
            for key, value in run.data.metrics.items():
                result[f"metrics.{key}"] = value
            for key, value in run.data.params.items():
                result[f"params.{key}"] = value
        else:
            best_run_id = get_best_mlflow_run(request.project_name, request.model_name)
            if best_run_id is None:
                return {"_no_runs": True, "message": "No runs found"}

            run = mlflow.get_run(best_run_id)
            experiment_id = run.info.experiment_id
            result = {
                "run_id": best_run_id,
                "experiment_id": experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "run_url": f"http://localhost:5001/#/experiments/{experiment_id}/runs/{best_run_id}",
            }
            for key, value in run.data.metrics.items():
                result[f"metrics.{key}"] = value
            for key, value in run.data.params.items():
                result[f"params.{key}"] = value

        mlflow_cache.set(cache_key, result)
        return result

    except Exception as e:
        print(f"Error fetching MLflow metrics: {e}")
        return {"_no_runs": True, "message": str(e)}


@router.post("/init")
async def batch_init(request: BatchInitRequest):
    """Initialize batch ML page with all required data in a single request."""
    project_name = request.project_name
    model_name = BATCH_MODEL_NAMES.get(project_name)

    if not model_name:
        raise HTTPException(status_code=400, detail=f"Unknown project: {project_name}")

    async def fetch_runs():
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(get_all_mlflow_runs, project_name, model_name),
                timeout=15.0,
            )
        except Exception as e:
            print(f"Error fetching runs: {e}")
            return []

    async def check_model():
        try:
            experiment = mlflow.get_experiment_by_name(project_name)
            if experiment is None:
                return {"available": False, "experiment_url": None}
            experiment_url = f"http://localhost:5001/#/experiments/{experiment.experiment_id}"
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{model_name}'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            return {"available": not runs.empty, "experiment_url": experiment_url}
        except Exception as e:
            print(f"Error checking model: {e}")
            return {"available": False, "experiment_url": None}

    async def fetch_metrics(run_id: str = None):
        try:
            if run_id:
                run = mlflow.get_run(run_id)
            else:
                best_run_id = get_best_mlflow_run(project_name, model_name)
                if not best_run_id:
                    return {"_no_runs": True}
                run = mlflow.get_run(best_run_id)

            result = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            }
            for key, value in run.data.metrics.items():
                result[f"metrics.{key}"] = value
            for key, value in run.data.params.items():
                result[f"params.{key}"] = value
            return result
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return {"_no_runs": True}

    runs_result, model_result, metrics_result = await asyncio.gather(
        fetch_runs(),
        check_model(),
        fetch_metrics(request.run_id),
        return_exceptions=True,
    )

    if isinstance(runs_result, Exception):
        runs_result = []
    if isinstance(model_result, Exception):
        model_result = {"available": False, "experiment_url": None}
    if isinstance(metrics_result, Exception):
        metrics_result = {"_no_runs": True}

    total_rows = 0
    if not metrics_result.get("_no_runs"):
        train_samples = metrics_result.get("params.train_samples")
        test_samples = metrics_result.get("params.test_samples")
        if train_samples and test_samples:
            total_rows = int(train_samples) + int(test_samples)

    best_run_id = runs_result[0]["run_id"] if runs_result else None

    return {
        "runs": runs_result,
        "model_available": model_result.get("available", False),
        "experiment_url": model_result.get("experiment_url"),
        "metrics": metrics_result if not metrics_result.get("_no_runs") else {},
        "total_rows": total_rows,
        "best_run_id": best_run_id,
    }


# =============================================================================
# Cluster Feature Counts Cache (ECCI)
# =============================================================================
_cluster_feature_counts_cache: Dict[str, dict] = {}
_cluster_feature_counts_cache_time: Dict[str, float] = {}
CLUSTER_FEATURE_COUNTS_CACHE_TTL = 30.0  # seconds


@router.post("/cluster-feature-counts")
async def get_cluster_feature_counts(payload: dict):
    """Get cluster feature counts for batch ML (ECCI clustering).

    Computes feature value distribution per cluster from MLflow training data artifact.
    Uses X_events.parquet which contains original features + cluster_label.

    Payload:
        project_name: str - Must be "E-Commerce Customer Interactions"
        feature_name: str - Feature to analyze (e.g., "event_type")
        run_id: str (optional) - Specific MLflow run ID, or None for best

    Returns:
        feature_counts: Dict mapping cluster_id -> {value: count}
        run_id: str - The MLflow run ID used
        total_samples: int - Total number of samples
    """
    import pandas as pd

    project_name = payload.get("project_name", "E-Commerce Customer Interactions")
    feature_name = payload.get("feature_name")
    run_id = payload.get("run_id")

    if project_name != "E-Commerce Customer Interactions":
        return {"feature_counts": {}, "error": "Only ECCI clustering is supported"}

    if not feature_name:
        return {"feature_counts": {}, "error": "feature_name is required"}

    # Get best run if not specified
    if not run_id:
        model_name = BATCH_MODEL_NAMES.get(project_name)
        run_id = get_best_mlflow_run(project_name, model_name)
        if not run_id:
            return {"feature_counts": {}, "message": "No MLflow runs found"}

    # Check cache
    cache_key = f"{run_id}:{feature_name}"
    cache_time = _cluster_feature_counts_cache_time.get(cache_key, 0)
    if time.time() - cache_time < CLUSTER_FEATURE_COUNTS_CACHE_TTL:
        cached = _cluster_feature_counts_cache.get(cache_key)
        if cached:
            return cached

    try:
        def compute_counts():
            # Download X_events.parquet from MLflow
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="training_data/X_events.parquet",
            )
            df = pd.read_parquet(local_path)

            if feature_name not in df.columns:
                return {"feature_counts": {}, "error": f"Feature '{feature_name}' not found"}
            if "cluster_label" not in df.columns:
                return {"feature_counts": {}, "error": "cluster_label not found in data"}

            # Compute counts per cluster
            feature_counts = {}
            for cluster_id in sorted(df["cluster_label"].unique()):
                cluster_data = df[df["cluster_label"] == cluster_id][feature_name]
                value_counts = cluster_data.value_counts().to_dict()
                # Convert keys to strings for JSON serialization
                feature_counts[str(cluster_id)] = {str(k): int(v) for k, v in value_counts.items()}

            return {
                "feature_counts": feature_counts,
                "run_id": run_id,
                "total_samples": len(df),
            }

        result = await asyncio.wait_for(
            asyncio.to_thread(compute_counts),
            timeout=30.0
        )

        # Cache result
        _cluster_feature_counts_cache[cache_key] = result
        _cluster_feature_counts_cache_time[cache_key] = time.time()

        return result

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out")
    except Exception as e:
        print(f"Error getting cluster feature counts: {e}")
        return {"feature_counts": {}, "error": str(e)}


# =============================================================================
# YellowBrick Visualization Helpers
# =============================================================================
def _get_visualization_artifact_path(metric_type: str, metric_name: str) -> str:
    """Get MLflow artifact path for a YellowBrick visualization.

    Maps metric_type to YellowBrick module name:
    - Classification → classifier
    - Regression → regressor
    - Feature Analysis → features
    - Target → target
    - Model Selection → model_selection
    - Clustering → cluster
    - Text Analysis → text

    Returns: visualizations/{module}/{metric_name}.png
    """
    module_map = {
        "Classification": "classifier",
        "Regression": "regressor",
        "Feature Analysis": "features",
        "Target": "target",
        "Model Selection": "model_selection",
        "Clustering": "cluster",
        "Text Analysis": "text",
    }
    module_name = module_map.get(metric_type, metric_type.lower().replace(" ", "_"))
    return f"visualizations/{module_name}/{metric_name}.png"


def _check_visualization_artifact(run_id: str, artifact_path: str) -> Optional[bytes]:
    """Check if visualization artifact exists in MLflow run.

    Returns: PNG bytes if found, None otherwise.
    """
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
        with open(local_path, "rb") as f:
            return f.read()
    except Exception:
        # Artifact doesn't exist
        return None


def _save_artifact(run_id: str, artifact_path: str, data: bytes) -> bool:
    """Save data as MLflow artifact to an existing run.

    Uses MlflowClient.log_artifact() to add artifacts to finished runs
    without changing run status. Can be used for any file type.

    Args:
        run_id: MLflow run ID to save artifact to
        artifact_path: Path within artifacts (e.g., "visualizations/classifier/ConfusionMatrix.png")
        data: Binary data to save

    Returns: True if saved successfully, False otherwise.
    """
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract directory and filename from artifact_path
            artifact_dir = os.path.dirname(artifact_path)  # e.g., "visualizations/classifier"
            artifact_filename = os.path.basename(artifact_path)  # e.g., "ConfusionMatrix.png"
            # Write data to temp file
            local_file = os.path.join(tmpdir, artifact_filename)
            with open(local_file, "wb") as f:
                f.write(data)
            # Use MlflowClient to log artifact (works on finished runs)
            client = mlflow.MlflowClient()
            client.log_artifact(run_id, local_file, artifact_path=artifact_dir)
            print(f"Saved artifact: {artifact_path} (run_id={run_id[:8]}...)")
            return True
    except Exception as e:
        print(f"Failed to save artifact {artifact_path}: {e}")
        return False


def _load_search_queries_from_mlflow(run_id: str, project_name: str) -> list:
    """Load search queries from MLflow artifact for text analysis."""
    import pandas as pd
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="training_data/search_queries.parquet",
        )
        df = pd.read_parquet(local_path)
        return df["search_query"].dropna().tolist()
    except Exception as e:
        print(f"Failed to load search queries: {e}")
        return []


def _sync_generate_yellowbrick_plot(
    project_name: str,
    metric_type: str,
    metric_name: str,
    run_id: str = None,
) -> tuple:
    """Synchronous yellowbrick plot generation with MLflow artifact caching.

    First checks if visualization exists as MLflow artifact. If not, generates
    the plot and saves it as an artifact for future requests.

    Supports classification (TFD), regression (ETA), and clustering (ECCI) projects.

    Args:
        project_name: MLflow experiment name
        metric_type: YellowBrick category (Classification, Regression, Feature Analysis, etc.)
        metric_name: Specific visualizer name (ConfusionMatrix, ResidualsPlot, RadViz, etc.)
        run_id: Optional specific run ID. If None, uses best run.

    Returns:
        Tuple of (image_bytes, run_id, cache_hit)
    """
    import io
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.sklearn import (
        load_model_from_mlflow,
        load_training_data_from_mlflow,
        yellowbrick_classification_kwargs,
        yellowbrick_classification_visualizers,
        yellowbrick_regression_kwargs,
        yellowbrick_regression_visualizers,
        yellowbrick_feature_analysis_kwargs,
        yellowbrick_feature_analysis_visualizers,
        yellowbrick_target_kwargs,
        yellowbrick_target_visualizers,
        yellowbrick_model_selection_kwargs,
        yellowbrick_model_selection_visualizers,
        yellowbrick_clustering_kwargs,
        yellowbrick_clustering_visualizers,
        yellowbrick_text_analysis_kwargs,
        yellowbrick_text_analysis_visualizers,
    )

    # Validate project
    task_type = PROJECT_TASK_TYPES.get(project_name)
    if task_type is None:
        raise ValueError(f"Unsupported project: {project_name}")

    # Get run_id if not provided
    if run_id is None:
        model_name = BATCH_MODEL_NAMES.get(project_name)
        run_id = get_best_mlflow_run(project_name, model_name)
        if run_id is None:
            raise ValueError("No trained model found in MLflow.")

    # Check for cached visualization in MLflow artifacts
    artifact_path = _get_visualization_artifact_path(metric_type, metric_name)
    cached_image = _check_visualization_artifact(run_id, artifact_path)
    if cached_image is not None:
        print(f"MLflow artifact cache HIT: {artifact_path} (run_id={run_id[:8]}...)")
        return cached_image, run_id, True

    print(f"MLflow artifact cache MISS: {artifact_path} (run_id={run_id[:8]}...) - generating...")

    # Load training data from MLflow artifacts (from selected or best run)
    result = load_training_data_from_mlflow(project_name, run_id=run_id)
    if result is None:
        raise ValueError(
            "No training data found in MLflow. Train a model first to generate visualizations."
        )

    X_train, X_test, y_train, y_test, feature_names = result

    # Combined data for visualizers that need full dataset
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    # Classes for classification/clustering (None for regression)
    classes = None
    if task_type == "classification":
        classes = sorted(list(set(y_train.unique().tolist() + y_test.unique().tolist())))
    elif task_type == "clustering":
        # For clustering, generate labels for each unique cluster
        unique_clusters = sorted(list(set(y_train.unique().tolist() + y_test.unique().tolist())))
        classes = [f"Cluster {c}" for c in unique_clusters]

    fig_buf = io.BytesIO()
    yb_vis = None

    try:
        if metric_type == "Classification":
            # Classification visualizers (TFD only)
            model_name = BATCH_MODEL_NAMES.get(project_name)
            model = load_model_from_mlflow(project_name, model_name, run_id=run_id)
            yb_kwargs = yellowbrick_classification_kwargs(
                project_name, metric_name, y_train, classes
            )
            yb_vis = yellowbrick_classification_visualizers(
                yb_kwargs, X_train, X_test, y_train, y_test, model=model
            )
        elif metric_type == "Regression":
            # Regression visualizers (ETA only)
            model_name = BATCH_MODEL_NAMES.get(project_name)
            model = load_model_from_mlflow(project_name, model_name, run_id=run_id)
            yb_kwargs = yellowbrick_regression_kwargs(
                project_name, metric_name
            )
            yb_vis = yellowbrick_regression_visualizers(
                yb_kwargs, X_train, X_test, y_train, y_test, model=model
            )
        elif metric_type == "Feature Analysis":
            yb_kwargs = yellowbrick_feature_analysis_kwargs(
                project_name, metric_name, classes, feature_names
            )
            yb_vis = yellowbrick_feature_analysis_visualizers(
                yb_kwargs, X, y
            )
        elif metric_type == "Target":
            labels = classes if classes else None
            features_list = X_train.columns.tolist()
            yb_kwargs = yellowbrick_target_kwargs(
                project_name, metric_name, labels, features_list
            )
            yb_vis = yellowbrick_target_visualizers(yb_kwargs, X, y)
        elif metric_type == "Model Selection":
            # Load model from MLflow for FeatureImportances
            model_name = BATCH_MODEL_NAMES.get(project_name)
            model = load_model_from_mlflow(project_name, model_name, run_id=run_id)
            yb_kwargs = yellowbrick_model_selection_kwargs(
                project_name, metric_name, feature_names=feature_names
            )
            yb_vis = yellowbrick_model_selection_visualizers(
                yb_kwargs, X_train, X_test, y_train, y_test, model=model,
                project_name=project_name
            )
        elif metric_type == "Clustering":
            # Clustering visualizers (ECCI only)
            model_name = BATCH_MODEL_NAMES.get(project_name)
            model = load_model_from_mlflow(project_name, model_name, run_id=run_id)
            n_clusters = model.n_clusters if hasattr(model, 'n_clusters') else 5
            yb_kwargs = yellowbrick_clustering_kwargs(
                project_name, metric_name, n_clusters=n_clusters
            )
            yb_vis = yellowbrick_clustering_visualizers(
                yb_kwargs, X, y, model=model, n_clusters=n_clusters
            )
        elif metric_type == "Text Analysis":
            # Text analysis visualizers (ECCI only - search queries)
            try:
                search_queries = _load_search_queries_from_mlflow(run_id, project_name)
            except Exception as e:
                raise ValueError(f"Failed to load search queries: {e}")
            cluster_labels = y.values if y is not None else None
            yb_kwargs = yellowbrick_text_analysis_kwargs(project_name, metric_name)
            yb_vis = yellowbrick_text_analysis_visualizers(
                yb_kwargs, search_queries, cluster_labels=cluster_labels
            )
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        if yb_vis is not None:
            # CRITICAL: Call show() to finalize visualization before saving
            yb_vis.show()
            yb_vis.fig.savefig(fig_buf, format="png")
            fig_buf.seek(0)
            image_bytes = fig_buf.getvalue()
            # Save to MLflow artifacts for future requests
            _save_artifact(run_id, artifact_path, image_bytes)
            return image_bytes, run_id, False

        raise ValueError("Failed to generate visualization")

    finally:
        plt.clf()
        plt.close('all')
        if fig_buf:
            fig_buf.close()


@router.post("/yellowbrick-metric")
async def yellowbrick_metric(request: YellowBrickRequest):
    """Generate YellowBrick visualizations for a specific MLflow run.

    Uses MLflow artifact caching: first request generates and saves to MLflow,
    subsequent requests load from MLflow artifacts (fast!).
    """
    import base64

    project_name = request.project_name
    metric_type = request.metric_type
    metric_name = request.metric_name
    run_id = request.run_id

    print(f"[DEBUG] yellowbrick_metric called: project={project_name}, type={metric_type}, name={metric_name}, run_id={run_id}")

    if not all([project_name, metric_type, metric_name]):
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: project_name, metric_type, metric_name"
        )

    try:
        # Generate or load from MLflow artifact cache
        image_bytes, actual_run_id, cache_hit = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_generate_yellowbrick_plot,
                project_name,
                metric_type,
                metric_name,
                run_id,
            ),
            timeout=300.0  # 5 minutes for slow visualizations
        )

        # Return base64 encoded image for Reflex frontend
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "image_base64": image_base64,
            "cache": "HIT" if cache_hit else "MISS",
            "cache_type": "mlflow_artifact",
            "metric_type": metric_type,
            "metric_name": metric_name,
            "run_id": actual_run_id,
        }

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Visualization generation timed out")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error generating yellowbrick plot: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {str(e)}")


@router.get("/healthcheck", response_model=SklearnHealthcheck)
async def get_healthcheck():
    """Get detailed healthcheck status for sklearn batch ML service."""
    return sklearn_healthcheck


@router.put("/healthcheck", response_model=SklearnHealthcheck)
async def update_healthcheck(update_data: SklearnHealthcheck):
    """Update healthcheck status."""
    global sklearn_healthcheck
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(sklearn_healthcheck, field, value)
    return sklearn_healthcheck


@router.get("/best-model/{project_name}")
async def get_best_model_info(project_name: str):
    """Get information about the current best model for a project.

    Returns model info from cache or loads from MLflow if needed.
    """
    model_name = BATCH_MODEL_NAMES.get(project_name)
    if not model_name:
        raise HTTPException(status_code=400, detail=f"Unknown project: {project_name}")
    model, run_id = model_cache.get_model(project_name)
    if model is None:
        return {
            "project_name": project_name,
            "model_name": model_name,
            "status": "no_model",
            "message": "No trained model found. Train a model first."
        }
    return {
        "project_name": project_name,
        "model_name": model_name,
        "run_id": run_id,
        "status": "available",
        "cached": project_name in model_cache.models
    }


@router.post("/invalidate-cache")
async def invalidate_model_cache(project_name: str = None):
    """Invalidate model cache for a project or all projects.

    Use this to force reload of models from MLflow.
    """
    model_cache.invalidate(project_name)
    if project_name:
        return {"message": f"Cache invalidated for {project_name}"}
    return {"message": "All caches invalidated"}


@router.post("/delta-total-rows")
async def get_delta_total_rows(payload: dict):
    """Get total number of rows available in Delta Lake for a project.

    This is used to set the maximum value for the max_rows training option.
    Uses DuckDB with Delta Lake extension to query the row count.

    Payload:
        project_name: str - Project name to query

    Returns:
        total_rows: int - Total rows in Delta Lake table
    """
    project_name = payload.get("project_name")
    if not project_name:
        raise HTTPException(status_code=400, detail="project_name is required")

    delta_path = DELTA_PATHS.get(project_name)
    if not delta_path:
        raise HTTPException(status_code=400, detail=f"Unknown project: {project_name}")

    try:
        import duckdb

        def query_count():
            conn = duckdb.connect()
            conn.execute("INSTALL delta; LOAD delta;")
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            # Configure S3 credentials from environment
            conn.execute(f"""
                SET s3_region = '{AWS_REGION}';
                SET s3_access_key_id = '{AWS_ACCESS_KEY_ID}';
                SET s3_secret_access_key = '{AWS_SECRET_ACCESS_KEY}';
                SET s3_endpoint = '{AWS_S3_ENDPOINT}';
                SET s3_use_ssl = false;
                SET s3_url_style = 'path';
            """)
            result = conn.execute(f"SELECT COUNT(*) FROM delta_scan('{delta_path}')").fetchone()
            conn.close()
            return result[0] if result else 0

        total_rows = await asyncio.wait_for(
            asyncio.to_thread(query_count),
            timeout=30.0
        )

        return {"total_rows": total_rows, "project_name": project_name}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out")
    except Exception as e:
        print(f"Error querying Delta Lake total rows: {e}")
        # Return 0 on error - UI will use default max
        return {"total_rows": 0, "project_name": project_name, "error": str(e)}
