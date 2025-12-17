"""
River ML Training Service

Handles incremental ML model training using River library.
Receives start/stop signals directly from Reflex frontend.
Also handles predictions using trained River models.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import sys
import pandas as pd
from functions import (
    load_or_create_model,
    load_or_create_encoders,
    process_sample,
)


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

# Global caches for models and encoders
model_dict: dict = {name: {} for name in PROJECT_NAMES}
encoders_dict: dict = {name: {} for name in PROJECT_NAMES}


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


app = FastAPI(
    title="River ML Training Service",
    description="Manages incremental ML model training",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
                status_code=404,
                detail=f"Model key '{model_key}' not found. Available: {list(MODEL_SCRIPTS.keys())}"
            )

    command = ["/app/.venv/bin/python3", "-u", model_key]

    try:
        print(f"Starting model: {model_key}")

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
            detail=f"Failed to start model {model_key}: {str(e)}"
        )


@app.get("/current_model")
async def get_current_model():
    """Get currently running model."""
    return await get_status()


@app.post("/predict")
async def predict(payload: dict):
    """
    Make predictions using the latest model from disk.
    Reloads model before each prediction to stay synchronized with training.
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
    # Reload model from disk to get the latest trained version
    model_folder_name = project_name.lower().replace(' ', '_').replace("-", "_")
    model_folder = f"models/{model_folder_name}"
    try:
        model = load_or_create_model(project_name, model_name, model_folder)
        # Update in-memory cache
        if project_name not in model_dict:
            model_dict[project_name] = {}
        model_dict[project_name][model_name] = model
    except Exception as e:
        print(f"Error reloading model from disk: {e}", file=sys.stderr)
        # Fall back to cached model if available
        if project_name in model_dict and model_name in model_dict.get(project_name, {}):
            model = model_dict[project_name][model_name]
            print(f"Using cached model for {project_name}/{model_name}")
        else:
            raise HTTPException(
                status_code = 503,
                detail = f"Model '{model_name}' for project '{project_name}' could not be loaded: {e}"
            )
    # Reload encoders from disk to stay synchronized with training
    try:
        for library in ENCODER_LIBRARIES:
            encoders_dict[project_name][library] = load_or_create_encoders(project_name, library)
    except Exception as e:
        print(f"Error reloading encoders: {e}", file=sys.stderr)
        if project_name not in encoders_dict:
            raise HTTPException(
                status_code = 503,
                detail = f"Encoders for project '{project_name}' could not be loaded: {e}"
            )
    # Extract feature data (remove metadata fields)
    x = {k: v for k, v in payload.items() if k not in ["project_name", "model_name"]}
    if model_name in ["ARFClassifier", "ARFRegressor", "DBSTREAM"]:
        try:
            # Use pre-loaded encoders
            encoders = encoders_dict[project_name].get("river", {})
            processed_x, _ = process_sample(x, encoders, project_name)
            if project_name == "Transaction Fraud Detection":
                y_pred_proba = model.predict_proba_one(processed_x)
                fraud_probability = y_pred_proba.get(1, 0.0)
                binary_prediction = 1 if fraud_probability >= 0.5 else 0
                return {
                    "fraud_probability": fraud_probability,
                    "prediction": binary_prediction,
                }
            elif project_name == "Estimated Time of Arrival":
                y_pred = model.predict_one(processed_x)
                return {"Estimated Time of Arrival": y_pred}
            elif project_name == "E-Commerce Customer Interactions":
                y_pred = model.predict_one(processed_x)
                return {"cluster": y_pred}
        except Exception as e:
            print(f"Error during prediction: {e}", file=sys.stderr)
            raise HTTPException(
                status_code = 500,
                detail = f"Prediction failed: {e}")
    elif model_name in ["XGBClassifier"]:
        try:
            # Use pre-loaded encoders
            encoders = encoders_dict[project_name].get("sklearn", {})
            if project_name == "Transaction Fraud Detection":
                processed_x = process_sample(x, ..., project_name, library="sklearn")
                preprocessor = encoders.get("preprocessor")
                if preprocessor is None:
                    raise HTTPException(
                        status_code = 503,
                        detail = "Preprocessor not loaded")
                processed_x = pd.DataFrame({k: [v] for k, v in processed_x.items()})
                processed_x = preprocessor.transform(processed_x)
                y_pred_proba = model.predict_proba(processed_x).tolist()[0]
                fraud_probability = y_pred_proba[1]
                binary_prediction = 1 if fraud_probability >= 0.5 else 0
                return {
                    "fraud_probability": fraud_probability,
                    "prediction": binary_prediction,
                }
        except Exception as e:
            print(f"Error during prediction: {e}", file=sys.stderr)
            raise HTTPException(
                status_code = 500,
                detail = f"Prediction failed: {e}")
    raise HTTPException(
        status_code = 400,
        detail = f"Unknown model: {model_name}")
