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
import mlflow.sklearn
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.metrics import geometric_mean_score

from functions import (
    ModelDataManager,
    process_sklearn_sample,
    load_or_create_data,
    create_consumer,
    load_sklearn_encoders,
    create_batch_model,
    process_batch_data,
    yellowbrick_classification_kwargs,
    yellowbrick_classification_visualizers,
    yellowbrick_feature_analysis_kwargs,
    yellowbrick_feature_analysis_visualizers,
    yellowbrick_target_kwargs,
    yellowbrick_target_visualizers,
    yellowbrick_model_selection_kwargs,
    yellowbrick_model_selection_visualizers
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


batch_state = BatchTrainingState()


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
data_dict = {x: None for x in PROJECT_NAMES}
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


class SampleRequest(BaseModel):
    project_name: str


class SklearnHealthcheck(BaseModel):
    """Healthcheck for sklearn batch ML service."""
    model_available: dict[str, bool] = {}
    data_load: dict[str, str] = {}
    encoders_load: dict[str, str] = {}


# Initialize sklearn healthcheck
sklearn_healthcheck = SklearnHealthcheck()


class TrainRequest(BaseModel):
    project_name: str
    model_name: str = "XGBClassifier"
    force_retrain: bool = False


class ModelAvailabilityRequest(BaseModel):
    project_name: str
    model_name: str = "XGBClassifier"


# =============================================================================
# Training State
# =============================================================================
training_state: Dict[str, dict] = {}


def _sync_train_model(project_name: str, model_name: str) -> dict:
    """Synchronous model training - to be run in thread pool."""
    start_time = time.time()

    # Load and process data
    print(f"Loading data for training: {project_name}")
    consumer = create_consumer(project_name)
    data_df = load_or_create_data(consumer, project_name)

    if data_df.empty:
        raise ValueError(f"No data available for training {project_name}")

    print(f"Processing data for {project_name}...")
    X_train, X_test, y_train, y_test = process_batch_data(data_df, project_name)

    print(f"Creating model: {model_name}")
    model = create_batch_model(project_name, y_train=y_train)

    print("Training model...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROCAUC": float(roc_auc_score(y_test, y_pred_proba)),
        "GeometricMean": float(geometric_mean_score(y_test, y_pred)),
    }

    print(f"Metrics calculated: {metrics}")

    # Log to MLflow
    print("Logging to MLflow...")
    experiment = mlflow.get_experiment_by_name(project_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(project_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name):
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Clear cache to ensure fresh metrics
    mlflow_cache.clear()

    return {
        "status": "success",
        "project_name": project_name,
        "model_name": model_name,
        "run_id": run_id,
        "metrics": metrics,
        "training_time_seconds": round(training_time, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "trained_at": datetime.utcnow().isoformat() + "Z"
    }


# =============================================================================
# Lifespan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_dict, encoders_dict, sklearn_healthcheck

    data_load_status = {}
    encoders_load_status = {}

    # Load data for batch ML projects
    print("Loading data for Batch ML projects...")
    for project_name in PROJECT_NAMES:
        try:
            consumer = create_consumer(project_name)
            data_dict[project_name] = load_or_create_data(consumer, project_name)
            data_load_status[project_name] = "success"
            print(f"Data loaded for {project_name}")
        except Exception as e:
            data_load_status[project_name] = f"failed: {e}"
            print(f"Error loading data for {project_name}: {e}", file=sys.stderr)

    sklearn_healthcheck.data_load = data_load_status

    # Load sklearn encoders
    print("Loading sklearn encoders...")
    for project_name in PROJECT_NAMES:
        try:
            encoders_dict[project_name] = load_sklearn_encoders(project_name)
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


@app.post("/sample")
async def get_sample(request: SampleRequest):
    """Get a random sample from the dataset."""
    if data_dict.get(request.project_name) is None:
        raise HTTPException(status_code=503, detail="Data is not loaded.")
    try:
        sample = data_dict[request.project_name].sample(1).to_dict(orient='records')[0]
        return sample
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling data: {e}")


@app.post("/predict")
async def predict(request: PredictRequest):
    """Make batch ML prediction using XGBClassifier from MLflow."""
    project_name = request.project_name
    model_name = request.model_name

    if model_name != "XGBClassifier":
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

            # Load model from MLflow
            experiment = mlflow.get_experiment_by_name(project_name)
            if experiment is None:
                raise HTTPException(status_code=404, detail=f"MLflow experiment not found")

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{model_name}'",
                order_by=["start_time DESC"],
                max_results=1
            )

            if runs.empty:
                raise HTTPException(status_code=404, detail=f"No MLflow runs found for {model_name}")

            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

            prediction = model.predict(X)[0]
            fraud_probability = model.predict_proba(X)[0][1]

            return {
                "prediction": int(prediction),
                "fraud_probability": float(fraud_probability),
                "model_name": model_name,
                "run_id": run_id
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Project {project_name} not supported")


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


@app.post("/train")
async def train_model(request: TrainRequest):
    """Train a batch ML model and log to MLflow."""
    project_name = request.project_name
    model_name = request.model_name
    state_key = f"{project_name}:{model_name}"

    # Check if already training
    if state_key in training_state and training_state[state_key].get("status") == "training":
        return {
            "status": "already_training",
            "message": f"Model {model_name} for {project_name} is already being trained"
        }

    # Check if model exists and force_retrain is False
    if not request.force_retrain:
        try:
            experiment = mlflow.get_experiment_by_name(project_name)
            if experiment is not None:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{model_name}'",
                    max_results=1
                )
                if not runs.empty:
                    return {
                        "status": "model_exists",
                        "message": f"Model {model_name} already exists. Set force_retrain=true to retrain.",
                        "run_id": runs.iloc[0]["run_id"]
                    }
        except Exception:
            pass  # Proceed with training if check fails

    # Update training state
    training_state[state_key] = {
        "status": "training",
        "started_at": datetime.utcnow().isoformat() + "Z"
    }

    try:
        # Run training in thread pool to avoid blocking
        result = await asyncio.wait_for(
            asyncio.to_thread(_sync_train_model, project_name, model_name),
            timeout=600.0  # 10 minute timeout for training
        )

        # Update training state with result
        training_state[state_key] = {
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "result": result
        }

        return result

    except asyncio.TimeoutError:
        training_state[state_key] = {
            "status": "timeout",
            "error": "Training timed out after 10 minutes"
        }
        raise HTTPException(status_code=504, detail="Training timed out after 10 minutes")
    except ValueError as e:
        training_state[state_key] = {
            "status": "failed",
            "error": str(e)
        }
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        training_state[state_key] = {
            "status": "failed",
            "error": str(e)
        }
        print(f"Error training model: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


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
                "message": f"No experiment found for {project_name}"
            }

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{model_name}'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if runs.empty:
            return {
                "available": False,
                "message": f"No trained model found for {model_name}"
            }

        run = runs.iloc[0]
        return {
            "available": True,
            "run_id": run["run_id"],
            "trained_at": run["start_time"].isoformat() if pd.notna(run["start_time"]) else None,
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
            "error": str(e)
        }


@app.get("/training_status/{project_name}/{model_name}")
async def get_training_status(project_name: str, model_name: str):
    """Get the current training status for a model."""
    state_key = f"{project_name}:{model_name}"
    if state_key in training_state:
        return training_state[state_key]
    return {"status": "not_started"}


# =============================================================================
# Subprocess-based Training Endpoints (like River service)
# =============================================================================
@app.get("/batch_status")
async def get_batch_status():
    """Get current batch training status."""
    if batch_state.current_model_name and batch_state.current_process:
        poll_result = batch_state.current_process.poll()
        if poll_result is None:
            # Process is still running
            return {
                "current_model": batch_state.current_model_name,
                "status": "running",
                "pid": batch_state.current_process.pid,
                "started_at": batch_state.started_at
            }
        else:
            # Process has finished
            exit_code = batch_state.current_process.returncode
            model_name = batch_state.current_model_name
            batch_state.exit_code = exit_code
            batch_state.completed_at = datetime.utcnow().isoformat() + "Z"
            batch_state.current_process = None
            batch_state.current_model_name = None

            if exit_code == 0:
                batch_state.status = "completed"
                return {
                    "current_model": model_name,
                    "status": "completed",
                    "exit_code": exit_code,
                    "started_at": batch_state.started_at,
                    "completed_at": batch_state.completed_at
                }
            else:
                batch_state.status = "failed"
                return {
                    "current_model": model_name,
                    "status": "failed",
                    "exit_code": exit_code,
                    "started_at": batch_state.started_at,
                    "completed_at": batch_state.completed_at
                }

    # No process running - return last known state
    return {
        "current_model": None,
        "status": batch_state.status,
        "exit_code": batch_state.exit_code,
        "started_at": batch_state.started_at,
        "completed_at": batch_state.completed_at
    }


@app.post("/switch_model")
async def switch_batch_model(payload: dict):
    """
    Start or stop batch ML model training.

    Payload:
        model_key: str - Script filename (e.g., "transaction_fraud_detection_sklearn.py")
                        or "none" to stop training
    """
    model_key = payload.get("model_key")

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

    try:
        print(f"Starting batch training: {model_key}")

        log_dir = "/app/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = f"{log_dir}/{model_key}.log"

        with open(log_file_path, "ab") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd="/app"
            )

        batch_state.current_process = process
        batch_state.current_model_name = model_key
        batch_state.status = f"Running {model_key}"
        batch_state.started_at = datetime.utcnow().isoformat() + "Z"
        batch_state.completed_at = None
        batch_state.exit_code = None

        print(f"Batch training {model_key} started with PID: {process.pid}")
        return {"message": f"Started training: {model_key}", "pid": process.pid}

    except Exception as e:
        print(f"Failed to start training {model_key}: {e}")
        batch_state.current_process = None
        batch_state.current_model_name = None
        batch_state.status = f"Failed to start: {e}"
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training {model_key}: {str(e)}"
        )
