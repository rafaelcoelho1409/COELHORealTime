"""
Scikit-Learn Batch ML Service

FastAPI service for batch machine learning predictions and YellowBrick visualizations.
Manages batch ML training via subprocess (like River service).
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Dict
import subprocess
import sys
import mlflow
import mlflow.catboost
import os
import numpy as np
import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import asyncio
import time
from datetime import datetime
from functions import (
    ModelDataManager,
    process_sklearn_sample,
    load_or_create_sklearn_encoders,
    yellowbrick_classification_kwargs,
    yellowbrick_classification_visualizers,
    yellowbrick_feature_analysis_kwargs,
    yellowbrick_feature_analysis_visualizers,
    yellowbrick_target_kwargs,
    yellowbrick_target_visualizers,
    yellowbrick_model_selection_kwargs,
    yellowbrick_model_selection_visualizers,
    TFD_CAT_FEATURE_INDICES,
    MLFLOW_MODEL_NAMES,
    load_model_from_mlflow,
    load_encoders_from_mlflow,
    get_best_mlflow_run,
)


MLFLOW_HOST = os.getenv("MLFLOW_HOST", "localhost")


# =============================================================================
# Subprocess-based Training (like River service)
# =============================================================================
PROJECT_NAMES_BATCH = [
    "Transaction Fraud Detection",
]

MODEL_SCRIPTS = {
    "transaction_fraud_detection_sklearn.py": "Transaction Fraud Detection",
}


class BatchTrainingState:
    """Tracks current batch training process state."""
    def __init__(self):
        self.current_process: subprocess.Popen | None = None
        self.current_model_name: str | None = None
        self.status: str = "idle"
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.exit_code: int | None = None
        self.log_file = None  # Log file handle for subprocess output
        # Live training status (updated by training script via POST)
        self.status_message: str = ""
        self.progress_percent: int = 0
        self.current_stage: str = ""
        self.metrics_preview: Dict[str, float] = {}

    def close_log_file(self):
        """Close the log file handle if open."""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None

    def update_status(self, message: str, progress: int = None, stage: str = None, metrics: Dict[str, float] = None):
        """Update training status from training script."""
        self.status_message = message
        if progress is not None:
            self.progress_percent = progress
        if stage is not None:
            self.current_stage = stage
        if metrics is not None:
            self.metrics_preview = metrics

    def reset_status(self):
        """Reset status for new training run."""
        self.status_message = ""
        self.progress_percent = 0
        self.current_stage = ""
        self.metrics_preview = {}


batch_state = BatchTrainingState()


# =============================================================================
# Model Cache (for predictions using best model from MLflow)
# =============================================================================
class ModelCache:
    """Cache for loaded models and encoders from MLflow."""
    def __init__(self, ttl_seconds: int = 300):
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
        if project_name in self.models and not self._is_expired(project_name):
            return self.models[project_name], self.run_ids.get(project_name)

        # Load from MLflow
        model_name = MLFLOW_MODEL_NAMES.get(project_name)
        if not model_name:
            return None, None

        run_id = get_best_mlflow_run(project_name, model_name)
        if run_id is None:
            return None, None

        model = load_model_from_mlflow(project_name, model_name)
        if model is not None:
            self.models[project_name] = model
            self.run_ids[project_name] = run_id
            self.timestamps[project_name] = time.time()
            print(f"Model cached for {project_name} (run_id={run_id})")

        return model, run_id

    def get_encoders(self, project_name: str):
        """Get cached encoders or load from MLflow if expired/missing."""
        if project_name in self.encoders and not self._is_expired(project_name):
            return self.encoders[project_name]

        encoders = load_encoders_from_mlflow(project_name)
        if encoders is not None:
            self.encoders[project_name] = encoders
            print(f"Encoders cached for {project_name}")

        return encoders

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


model_cache = ModelCache(ttl_seconds=300)  # 5 minute cache


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
# TTL Cache for MLflow Metrics
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


mlflow_cache = MLflowMetricsCache(ttl_seconds=30)


def _sync_get_mlflow_metrics(project_name: str, model_name: str) -> dict:
    """Synchronous MLflow query - to be run in thread pool."""
    experiment = mlflow.get_experiment_by_name(project_name)
    if experiment is None:
        raise ValueError(f"Experiment '{project_name}' not found in MLflow")
    experiment_id = experiment.experiment_id
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        max_results=100,
        order_by=["start_time DESC"]
    )
    runs_df = runs_df[runs_df["tags.mlflow.runName"] == model_name]
    if runs_df.empty:
        raise ValueError(f"No runs found for model '{model_name}'")
    run_df = runs_df.iloc[0]
    return run_df.replace({np.nan: None}).to_dict()


# =============================================================================
# Project Configuration
# =============================================================================
PROJECT_NAMES = [
    "Transaction Fraud Detection",
]

# Global data manager
data_manager = ModelDataManager()
encoders_dict = {x: None for x in PROJECT_NAMES}


# =============================================================================
# Pydantic Models
# =============================================================================
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


class PredictRequest(BaseModel):
    project_name: str
    model_name: str
    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    merchant_id: Optional[str] = None
    product_category: Optional[str] = None
    transaction_type: Optional[str] = None
    payment_method: Optional[str] = None
    location: Optional[dict] = None
    ip_address: Optional[str] = None
    device_info: Optional[dict] = None
    user_agent: Optional[str] = None
    account_age_days: Optional[int] = None
    cvv_provided: Optional[bool] = None
    billing_address_match: Optional[bool] = None


class MLflowMetricsRequest(BaseModel):
    project_name: str
    model_name: str
    force_refresh: bool = False


class SklearnHealthcheck(BaseModel):
    """Healthcheck for sklearn batch ML service."""
    model_available: dict[str, bool] = {}
    encoders_load: dict[str, str] = {}


# Initialize sklearn healthcheck
sklearn_healthcheck = SklearnHealthcheck()


class ModelAvailabilityRequest(BaseModel):
    project_name: str
    model_name: str = "CatBoostClassifier"


# =============================================================================
# Lifespan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoders_dict, sklearn_healthcheck

    encoders_load_status = {}

    # Load sklearn encoders from MLflow (or create new if not available)
    print("Loading sklearn encoders...")
    for project_name in PROJECT_NAMES:
        try:
            encoders_dict[project_name] = load_or_create_sklearn_encoders(project_name)
            encoders_load_status[project_name] = "success"
            print(f"Encoders loaded for {project_name}")
        except Exception as e:
            encoders_load_status[project_name] = f"failed: {e}"
            print(f"Error loading encoders for {project_name}: {e}", file=sys.stderr)

    sklearn_healthcheck.encoders_load = encoders_load_status

    # Configure MLflow
    try:
        print("Configuring MLflow...")
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
        print("MLflow configured successfully")
    except Exception as e:
        print(f"Error configuring MLflow: {e}", file=sys.stderr)

    print("Sklearn service startup complete.")
    yield
    print("Sklearn service shutting down...")


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Scikit-Learn Batch ML Service",
    description="Batch ML predictions and YellowBrick visualizations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/")
async def root():
    return {"message": "Scikit-Learn Batch ML Service is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/healthcheck", response_model=SklearnHealthcheck)
async def get_healthcheck():
    """Get detailed healthcheck status for sklearn batch ML service."""
    return sklearn_healthcheck


@app.put("/healthcheck", response_model=SklearnHealthcheck)
async def update_healthcheck(update_data: SklearnHealthcheck):
    """Update healthcheck status."""
    global sklearn_healthcheck
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(sklearn_healthcheck, field, value)
    return sklearn_healthcheck


@app.post("/predict")
async def predict(request: PredictRequest):
    """Make batch ML prediction using best CatBoostClassifier from MLflow.

    Uses model cache to avoid loading model on every request.
    Selects best model based on fbeta_score (beta=2.0) for fraud detection.
    """
    project_name = request.project_name
    model_name = request.model_name

    if model_name != "CatBoostClassifier":
        raise HTTPException(status_code=400, detail=f"Model {model_name} not supported")

    if project_name == "Transaction Fraud Detection":
        sample = {
            "transaction_id": request.transaction_id,
            "user_id": request.user_id,
            "timestamp": request.timestamp,
            "amount": request.amount,
            "currency": request.currency,
            "merchant_id": request.merchant_id,
            "product_category": request.product_category,
            "transaction_type": request.transaction_type,
            "payment_method": request.payment_method,
            "location": request.location,
            "ip_address": request.ip_address,
            "device_info": request.device_info,
            "user_agent": request.user_agent,
            "account_age_days": request.account_age_days,
            "cvv_provided": request.cvv_provided,
            "billing_address_match": request.billing_address_match,
        }

        try:
            X = process_sklearn_sample(sample, project_name)

            # Get best model from cache (loads from MLflow if expired/missing)
            model, run_id = model_cache.get_model(project_name)
            if model is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No trained model found for {project_name}. Train a model first."
                )

            prediction = model.predict(X)[0]
            fraud_probability = model.predict_proba(X)[0][1]

            return {
                "prediction": int(prediction),
                "fraud_probability": float(fraud_probability),
                "model_name": model_name,
                "run_id": run_id,
                "best_model": True  # Indicates this is from best model selection
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Project {project_name} not supported")


@app.get("/best_model/{project_name}")
async def get_best_model_info(project_name: str):
    """Get information about the current best model for a project.

    Returns model info from cache or loads from MLflow if needed.
    """
    model_name = MLFLOW_MODEL_NAMES.get(project_name)
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


@app.post("/invalidate_cache")
async def invalidate_model_cache(project_name: str = None):
    """Invalidate model cache for a project or all projects.

    Use this to force reload of models from MLflow.
    """
    model_cache.invalidate(project_name)
    if project_name:
        return {"message": f"Cache invalidated for {project_name}"}
    return {"message": "All caches invalidated"}


@app.post("/mlflow_metrics")
async def get_mlflow_metrics(request: MLflowMetricsRequest):
    """Get MLflow metrics with caching."""
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
            timeout=30.0
        )
        mlflow_cache.set(cache_key, result)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="MLflow query timed out")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error fetching MLflow metrics: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to fetch MLflow metrics: {str(e)}")


def _sync_generate_yellowbrick_plot(
    project_name: str,
    metric_type: str,
    metric_name: str,
    dm: ModelDataManager
) -> bytes:
    """Synchronous yellowbrick plot generation."""
    dm.load_data(project_name)

    if project_name != "Transaction Fraud Detection":
        raise ValueError(f"Unsupported project: {project_name}")

    classes = list(set(dm.y_train.unique().tolist() + dm.y_test.unique().tolist()))
    classes.sort()

    fig_buf = io.BytesIO()
    yb_vis = None

    try:
        if metric_type == "Classification":
            yb_kwargs = yellowbrick_classification_kwargs(
                project_name, metric_name, dm.y_train, classes
            )
            yb_vis = yellowbrick_classification_visualizers(
                yb_kwargs, dm.X_train, dm.X_test, dm.y_train, dm.y_test
            )
        elif metric_type == "Feature Analysis":
            yb_kwargs = yellowbrick_feature_analysis_kwargs(
                project_name, metric_name, classes
            )
            yb_vis = yellowbrick_feature_analysis_visualizers(
                yb_kwargs, dm.X, dm.y
            )
        elif metric_type == "Target":
            labels = list(set(dm.y_train.unique().tolist() + dm.y_test.unique().tolist()))
            features_list = dm.X_train.columns.tolist()
            yb_kwargs = yellowbrick_target_kwargs(
                project_name, metric_name, labels, features_list
            )
            yb_vis = yellowbrick_target_visualizers(yb_kwargs, dm.X, dm.y)
        elif metric_type == "Model Selection":
            yb_kwargs = yellowbrick_model_selection_kwargs(
                project_name, metric_name, dm.y_train
            )
            yb_vis = yellowbrick_model_selection_visualizers(yb_kwargs, dm.X, dm.y)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        if yb_vis is not None:
            yb_vis.fig.savefig(fig_buf, format="png", bbox_inches='tight')
            fig_buf.seek(0)
            return fig_buf.getvalue()
        raise ValueError("Failed to generate visualization")
    finally:
        plt.clf()
        plt.close('all')
        if fig_buf:
            fig_buf.close()


@app.post("/yellowbrick_metric")
async def yellowbrick_metric(payload: dict):
    """Generate YellowBrick visualizations."""
    project_name = payload.get("project_name")
    metric_type = payload.get("metric_type")
    metric_name = payload.get("metric_name")

    if not all([project_name, metric_type, metric_name]):
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: project_name, metric_type, metric_name"
        )

    try:
        image_bytes = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_generate_yellowbrick_plot,
                project_name,
                metric_type,
                metric_name,
                data_manager
            ),
            timeout=120.0
        )
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png"
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Visualization generation timed out")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error generating yellowbrick plot: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {str(e)}")


@app.post("/model_available")
async def check_model_available(request: ModelAvailabilityRequest):
    """Check if a trained model is available in MLflow."""
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

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_name}'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if runs.empty:
            return {
                "available": False,
                "message": f"No trained model found for {model_name}",
                "experiment_url": experiment_url
            }

        run = runs.iloc[0]
        return {
            "available": True,
            "run_id": run["run_id"],
            "trained_at": run["start_time"].isoformat() if pd.notna(run["start_time"]) else None,
            "experiment_url": experiment_url,
            "metrics": {
                "Accuracy": run.get("metrics.Accuracy"),
                "Precision": run.get("metrics.Precision"),
                "Recall": run.get("metrics.Recall"),
                "F1": run.get("metrics.F1"),
                "ROCAUC": run.get("metrics.ROCAUC"),
                "GeometricMean": run.get("metrics.GeometricMean"),
            }
        }
    except Exception as e:
        print(f"Error checking model availability: {e}", file=sys.stderr)
        return {
            "available": False,
            "error": str(e),
            "experiment_url": None
        }


# =============================================================================
# Subprocess-based Training Endpoints (like River service)
# =============================================================================
def get_latest_catboost_log_line(model_name: str) -> Dict[str, str]:
    """Read and parse the latest CatBoost iteration line from the training log file.

    Returns a dict with parsed fields: iteration, test/learn, best, total, remaining
    """
    empty_result = {}
    if not model_name:
        return empty_result
    log_file_path = f"/app/logs/{model_name}.log"
    try:
        with open(log_file_path, "rb") as f:
            # Seek to end and read last 8KB (enough for many log lines)
            f.seek(0, 2)  # End of file
            file_size = f.tell()
            read_size = min(8192, file_size)
            f.seek(max(0, file_size - read_size))
            content = f.read().decode("utf-8", errors="ignore")

        # Split into lines and find the latest CatBoost iteration line
        # CatBoost format: "67:\ttest: 0.9947585\tbest: 0.9949188 (22)\ttotal: 25.7s\tremaining: 5m 52s"
        lines = content.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            # CatBoost iteration lines start with a number followed by colon
            if line and line[0].isdigit() and ("test:" in line or "learn:" in line) and "total:" in line:
                # Parse the line into structured data
                result = {}
                # Split by tabs
                parts = line.replace("\t", "  ").split("  ")
                parts = [p.strip() for p in parts if p.strip()]

                for part in parts:
                    if part and part[0].isdigit() and ":" in part and "test" not in part and "learn" not in part:
                        # This is the iteration number (e.g., "67:")
                        result["iteration"] = part.rstrip(":")
                    elif "test:" in part:
                        result["test"] = part.replace("test:", "").strip()
                    elif "learn:" in part:
                        result["learn"] = part.replace("learn:", "").strip()
                    elif "best:" in part:
                        result["best"] = part.replace("best:", "").strip()
                    elif "total:" in part:
                        result["total"] = part.replace("total:", "").strip()
                    elif "remaining:" in part:
                        result["remaining"] = part.replace("remaining:", "").strip()

                return result
        return empty_result
    except Exception as e:
        print(f"Error reading catboost log: {e}")
        return empty_result


@app.get("/batch_status")
async def get_batch_status():
    """Get current batch training status including live progress updates."""
    # Get latest CatBoost log line if training
    catboost_log = ""
    # Read log during training stage (30-70% progress)
    if batch_state.current_model_name and batch_state.progress_percent >= 30 and batch_state.progress_percent < 70:
        catboost_log = get_latest_catboost_log_line(batch_state.current_model_name)

    # Common status fields
    live_status = {
        "status_message": batch_state.status_message,
        "progress_percent": batch_state.progress_percent,
        "current_stage": batch_state.current_stage,
        "metrics_preview": batch_state.metrics_preview,
        "catboost_log": catboost_log,
    }

    if batch_state.current_model_name and batch_state.current_process:
        poll_result = batch_state.current_process.poll()
        if poll_result is None:
            # Process is still running
            return {
                "current_model": batch_state.current_model_name,
                "status": "running",
                "pid": batch_state.current_process.pid,
                "started_at": batch_state.started_at,
                **live_status
            }
        else:
            # Process has finished
            exit_code = batch_state.current_process.returncode
            model_name = batch_state.current_model_name
            batch_state.exit_code = exit_code
            batch_state.completed_at = datetime.utcnow().isoformat() + "Z"
            batch_state.close_log_file()
            batch_state.current_process = None
            batch_state.current_model_name = None

            if exit_code == 0:
                batch_state.status = "completed"
                # Invalidate model cache so next prediction uses the new best model
                project_name = MODEL_SCRIPTS.get(model_name)
                if project_name:
                    model_cache.invalidate(project_name)
                    print(f"Model cache invalidated for {project_name} after training completed")
                return {
                    "current_model": model_name,
                    "status": "completed",
                    "exit_code": exit_code,
                    "started_at": batch_state.started_at,
                    "completed_at": batch_state.completed_at,
                    **live_status
                }
            else:
                batch_state.status = "failed"
                return {
                    "current_model": model_name,
                    "status": "failed",
                    "exit_code": exit_code,
                    "started_at": batch_state.started_at,
                    "completed_at": batch_state.completed_at,
                    **live_status
                }

    # No process running - return last known state
    return {
        "current_model": None,
        "status": batch_state.status,
        "exit_code": batch_state.exit_code,
        "started_at": batch_state.started_at,
        "completed_at": batch_state.completed_at,
        **live_status
    }


class TrainingStatusUpdate(BaseModel):
    """Request body for training status updates."""
    message: str
    progress: Optional[int] = None
    stage: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


@app.post("/training_status")
async def update_training_status(update: TrainingStatusUpdate):
    """Receive live training status updates from training script."""
    batch_state.update_status(
        message=update.message,
        progress=update.progress,
        stage=update.stage,
        metrics=update.metrics
    )
    return {"status": "ok"}


@app.post("/switch_model")
async def switch_batch_model(payload: dict):
    """
    Start or stop batch ML model training.

    Payload:
        model_key: str - Script filename (e.g., "transaction_fraud_detection_sklearn.py")
                        or "none" to stop training
        sample_frac: float (optional) - Fraction of data to use (0.0-1.0)
    """
    model_key = payload.get("model_key")
    sample_frac = payload.get("sample_frac")  # Optional: 0.0-1.0

    # If requesting to stop
    if model_key == "none":
        if batch_state.current_process:
            stop_current_training()
            return {"message": "Training stopped."}
        return {"message": "No training was running."}

    # Check if already running this model
    if model_key == batch_state.current_model_name and batch_state.current_process:
        poll_result = batch_state.current_process.poll()
        if poll_result is None:
            return {"message": f"Training {model_key} is already running.", "pid": batch_state.current_process.pid}

    # Stop any current training before starting new one
    if batch_state.current_process:
        print(f"Stopping current training before starting {model_key}")
        stop_current_training()

    # Validate model key
    if model_key not in MODEL_SCRIPTS:
        raise HTTPException(
            status_code=404,
            detail=f"Model key '{model_key}' not found. Available: {list(MODEL_SCRIPTS.keys())}"
        )

    command = ["/app/.venv/bin/python3", "-u", model_key]
    # Add --sample-frac if provided
    if sample_frac is not None and 0.0 < sample_frac <= 1.0:
        command.extend(["--sample-frac", str(sample_frac)])

    try:
        print(f"Starting batch training: {model_key}" + (f" with sample_frac={sample_frac}" if sample_frac else ""))

        log_dir = "/app/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = f"{log_dir}/{model_key}.log"

        # Open log file without context manager - subprocess will inherit the handle
        # Use line buffering (buffering=1) for real-time log output
        log_file = open(log_file_path, "a", buffering=1)
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd="/app"
        )
        # Store log file handle to close later
        batch_state.log_file = log_file

        batch_state.current_process = process
        batch_state.current_model_name = model_key
        batch_state.status = "running"
        batch_state.started_at = datetime.utcnow().isoformat() + "Z"
        batch_state.completed_at = None
        batch_state.exit_code = None
        batch_state.reset_status()  # Clear previous training status
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
            detail=f"Failed to start training {model_key}: {str(e)}"
        )
