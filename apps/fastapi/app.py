from fastapi import (
    FastAPI,
    HTTPException
)
from fastapi.responses import (
    FileResponse, 
    StreamingResponse
)
from contextlib import asynccontextmanager
from pydantic import (
    BaseModel, 
    Field,
    field_validator
)
import sys
import mlflow
from typing import (
    Optional,
    Any,
    Dict
)
import math
from pprint import pprint
import json
import subprocess
import os
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from functions import (
    ModelDataManager,
    process_sample,
    process_sklearn_sample,
    load_or_create_model,
    load_or_create_data,
    create_consumer,
    load_or_create_encoders,
    extract_device_info,
    extract_timestamp_info,
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


MLFLOW_HOST = os.environ["MLFLOW_HOST"]


PROJECT_NAMES = [
    "Transaction Fraud Detection", 
    "Estimated Time of Arrival",
    "E-Commerce Customer Interactions",
    #"Sales Forecasting"
]
MODEL_SCRIPTS = {
    f"{project_name} - River": f"{project_name.replace(' ', '_').replace('-', '_').lower()}_river.py"
    for project_name in PROJECT_NAMES
} | {
    f"{project_name} - Scikit Learn": f"{project_name.replace(' ', '_').replace('-', '_').lower()}_sklearn.py"
    for project_name in PROJECT_NAMES
}
ENCODER_LIBRARIES = [
    "river",
    "sklearn"
]


def stop_current_model():
    """
    Stops the currently running model process using Popen.terminate() (SIGTERM),
    then Popen.kill() (SIGKILL) if necessary.
    Manages state via the global healthcheck object.
    Returns True if the process is confirmed stopped or was already stopped.
    """
    global healthcheck
    if not healthcheck.current_training_process:
        healthcheck.current_training_status = "No model was active to stop."
        return True
    active_model_name = healthcheck.current_model_name
    process_to_stop = healthcheck.current_training_process # This is the Popen object
    if not hasattr(process_to_stop, 'pid') or process_to_stop.pid is None:
        print(f"Model '{active_model_name}' has invalid/stale process object. Cleaning up.")
        healthcheck.current_training_status = f"Model '{active_model_name}' had invalid process object. State cleared."
        healthcheck.current_training_process = None
        healthcheck.current_model_name = None
        return True
    pid_to_stop = process_to_stop.pid
    print(f"Attempting to stop model: '{active_model_name}' (PID: {pid_to_stop}) using Popen.terminate() (SIGTERM).")
    healthcheck.current_training_status = f"Sending SIGTERM to model '{active_model_name}' (PID: {pid_to_stop}). Waiting for graceful shutdown (60s)..."
    try:
        process_to_stop.terminate() # Sends SIGTERM
        process_to_stop.wait(timeout=60) # Wait for process to terminate
        if process_to_stop.poll() is not None: # Check if process has terminated
            print(f"Model '{active_model_name}' (PID: {pid_to_stop}) terminated gracefully after SIGTERM (exit code: {process_to_stop.returncode}).")
            healthcheck.current_training_status = f"Model '{active_model_name}' stopped gracefully (SIGTERM)."
        else:
            # This case should ideally not be reached if wait() returned without TimeoutExpired
            # and poll() is still None, but we handle it defensively.
            print(f"Model '{active_model_name}' (PID: {pid_to_stop}) did not terminate after SIGTERM and wait(). Status unclear. Proceeding to SIGKILL.")
            healthcheck.current_training_status = f"Model '{active_model_name}' SIGTERM sent, status unclear. Proceeding to SIGKILL."
            # Fall through to SIGKILL logic below (implicitly, as poll() will be None)
    except subprocess.TimeoutExpired:
        print(f"Model '{active_model_name}' (PID: {pid_to_stop}) timed out (60s) after SIGTERM. Sending SIGKILL.")
        healthcheck.current_training_status = f"Model '{active_model_name}' timed out after SIGTERM, will send SIGKILL."
        # Fall through to SIGKILL logic
    except Exception as e_terminate: # Catch other errors during terminate/wait
        print(f"Error during SIGTERM/wait for model '{active_model_name}' (PID: {pid_to_stop}): {e_terminate}")
        healthcheck.current_training_status = f"Error during SIGTERM/wait for '{active_model_name}': {e_terminate}. Proceeding to SIGKILL."
        # Fall through to SIGKILL logic
    # If process is still running (poll() is None after SIGTERM attempt or if an error occurred)
    if process_to_stop.poll() is None:
        try:
            print(f"Sending SIGKILL to model '{active_model_name}' (PID: {pid_to_stop}).")
            healthcheck.current_training_status = f"Sending SIGKILL to model '{active_model_name}' (PID: {pid_to_stop})."
            process_to_stop.kill() # Sends SIGKILL
            process_to_stop.wait(timeout=10) # Wait for SIGKILL to take effect

            if process_to_stop.poll() is not None:
                print(f"Model '{active_model_name}' (PID: {pid_to_stop}) terminated after SIGKILL (exit code: {process_to_stop.returncode}).")
                healthcheck.current_training_status = f"Model '{active_model_name}' stopped via SIGKILL."
            else:
                print(f"ERROR: Model '{active_model_name}' (PID: {pid_to_stop}) did NOT terminate after SIGKILL and wait. Process may be unkillable.")
                healthcheck.current_training_status = f"ERROR: Model '{active_model_name}' failed to stop even after SIGKILL."
                # Do not clear process/model name if we failed to kill it, to reflect this state.
                return False # Indicate stop failed
        except Exception as e_kill:
            print(f"Error during SIGKILL/wait for model '{active_model_name}' (PID: {pid_to_stop}): {e_kill}")
            healthcheck.current_training_status = f"Error during SIGKILL for '{active_model_name}': {e_kill}"
            # Do not clear process/model name if we failed to kill it.
            return False # Indicate stop failed
    elif process_to_stop.poll() is not None and not healthcheck.current_training_status.startswith(f"Model '{active_model_name}' stopped gracefully"):
         # Process already stopped before SIGKILL attempt, and not by successful SIGTERM
         print(f"Model '{active_model_name}' was already stopped before SIGKILL stage (exit code: {process_to_stop.returncode}).")
         healthcheck.current_training_status = f"Model '{active_model_name}' confirmed stopped before SIGKILL stage."
    # Final state cleanup if process is confirmed stopped or we've exhausted attempts and it's considered "done"
    healthcheck.current_training_process = None
    healthcheck.current_model_name = None
    return True # Signifies completion of the stop sequence.


# Initialize global variables as None or placeholder BEFORE startup
consumer_dict = {x: None for x in PROJECT_NAMES}
data_dict = {x: None for x in PROJECT_NAMES}
model_dict = {x: {} for x in PROJECT_NAMES}
encoders_dict = {x: {} for x in PROJECT_NAMES}
initial_sample_dict = {x: None for x in PROJECT_NAMES} # Also move this initialization
data_manager = ModelDataManager()


class Healthcheck(BaseModel):
    model_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES} # "success", "failed", "not_attempted"
    model_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    encoders_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES} # "success", "failed", "not_attempted"
    data_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES} # Added data load status
    data_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES} # Added data load message
    initial_data_sample_loaded: dict[str, bool] | dict[str, None] = {x: None for x in PROJECT_NAMES} # Added sample status
    initial_data_sample_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    current_training_process: Optional[Any] = None
    current_model_name: Optional[str] = None
    current_training_status: Optional[str] = Field(
        "Initializing",
        description = "Describes the current state of the managed training process."
    )
    model_config = {
        "arbitrary_types_allowed": True
    }

# Initialize healthcheck with default state
healthcheck = Healthcheck(
    model_load = {x: "not_attempted" for x in PROJECT_NAMES},
    encoders_load = {x: "not_attempted" for x in PROJECT_NAMES},
    data_load = {x: "not_attempted" for x in PROJECT_NAMES},
    initial_data_sample_loaded = {x: False for x in PROJECT_NAMES}
)


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


class DeviceInfo(BaseModel):
    """Pydantic model for device information."""
    device_type: Optional[str] = None
    browser: Optional[str] = None
    os: Optional[str] = None 

class Location(BaseModel):
    """Pydantic model for location."""
    lat: Optional[float] = None
    lon: Optional[float] = None

    # Add validators within the nested model for its float fields
    @field_validator('lat', 'lon', mode='before')
    @classmethod
    def check_location_coords_finite(cls, v: Any) -> Optional[float]:
        """
        Convert non-finite float values (NaN, Infinity) to None for lat/lon
        before standard validation.
        """
        # Use math.isfinite() which checks for both NaN and +/- Infinity
        if isinstance(v, float) and not math.isfinite(v):
            return None
        # Return the original value if it's finite, None, or not a float type
        # Pydantic's standard validation will still run afterwards.
        return v

# --- Main Model with Fixes ---

class ECommerceCustomerInteractions(BaseModel):
    """Pydantic model for validating e-commerce customer interaction events."""
    customer_id: Optional[str] = None
    # Use the nested DeviceInfo model
    device_info: Optional[DeviceInfo] = None
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    # Use the nested Location model
    location: Optional[Location] = None
    page_url: Optional[str] = None
    price: Optional[float] = None # Allow float or None for price
    product_category: Optional[str] = None
    product_id: Optional[str] = None
    # Keep quantity as Optional[int]
    quantity: Optional[int] = None
    referrer_url: Optional[str] = None
    search_query: Optional[str] = None
    session_event_sequence: Optional[int] = None
    session_id: Optional[str] = None
    time_on_page_seconds: Optional[int] = None
    # Consider using datetime type if you process timestamps
    timestamp: Optional[str] = None

    # Pydantic v2 validator syntax using @field_validator
    # Validator for quantity (handle potential float NaN input before int validation)
    @field_validator('quantity', mode='before')
    @classmethod
    def check_quantity_is_finite_int(cls, v: Any) -> Optional[int]:
        """Convert float NaN to None before checking if input is valid Optional[int]."""
        # Check specifically for float('nan') as infinity doesn't make sense for quantity
        if isinstance(v, float) and math.isnan(v):
            return None
        return v # Let Pydantic validate if it's int or None
    # Add validator for the price field
    @field_validator('price', mode='before')
    @classmethod
    def check_price_is_finite(cls, v: Any) -> Optional[float]:
        """
        Convert non-finite float values (NaN, Infinity) to None for price
        before standard validation.
        """
        # Use math.isfinite() which checks for both NaN and +/- Infinity
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v # Return original value for standard Pydantic float validation
    # Validator to ensure nested fields are dicts or None before processing
    @field_validator('device_info', 'location', mode='before')
    @classmethod
    def ensure_dict_or_none(cls, v: Any) -> Optional[Dict]:
        """Ensure input is a dictionary or None before nested model validation."""
        if v is None or isinstance(v, dict):
            return v
        # Let Pydantic raise the appropriate error if it's not a dict/None
        return v
    class Config:
        """Pydantic model configuration."""
        # Example: If you need to allow extra fields not defined in the model
        # extra = 'ignore'
        pass


#class SalesForecasting(BaseModel):
#    concept_drift_stage: Optional[int] = None
#    day_of_week: Optional[int] = None
#    event_id: Optional[str] = None
#    is_holiday: Optional[bool] = None
#    is_promotion_active: Optional[bool] = None
#    month: Optional[int] = None
#    product_id: Optional[str] = None
#    promotion_id: Optional[str] = None
#    quantity_sold: Optional[int] = None
#    store_id: Optional[str] = None
#    timestamp: Optional[str] = None
#    total_sales_amount: Optional[float] = None
#    unit_price: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        consumer_dict, \
        data_dict, \
        model_dict, \
        encoders_dict, \
        initial_sample_dict, \
        healthcheck
    # 1. Load data
    print("Loading data...")
    consumer_dict = {
        x: create_consumer(x)
        for x in PROJECT_NAMES
    }
    data_dict = {
        x: load_or_create_data(
            consumer_dict[x],
            x
        ) for x in PROJECT_NAMES}
    data_load_status = {}
    data_message_dict = {}
    for project_name in PROJECT_NAMES:
        try:
            data_load_status[project_name] = "success"
            data_message_dict[project_name] = "Data loaded successfully"
        except Exception as e:
            data_load_status[project_name] = "failed"
            data_message_dict[project_name] = f"Error: {e}"
            print(f"Error loading data: {e}", file = sys.stderr)
    healthcheck.data_load = data_load_status
    healthcheck.data_message = data_message_dict
    # 2. Create initial transaction data sample (requires data)
    initial_data_sample_loaded_status = {}
    initial_data_sample_message_dict = {}
    for project_name in PROJECT_NAMES:
        try:
            if data_dict[project_name] is not None and not data_dict[project_name].empty:
                project_name_pascal = project_name.replace("_", " ").replace("-", " ").title().replace(" ", "")
                initial_sample_dict[project_name] = globals()[project_name_pascal](
                    **data_dict[project_name].sample(1).to_dict(orient = 'records')[0]
                )
                initial_data_sample_loaded_status[project_name] = True
                initial_data_sample_message_dict[project_name] = "Initial sample created."
            else:
                initial_data_sample_loaded_status[project_name] = False
                initial_data_sample_message_dict[project_name] = "Data was not loaded or is empty, cannot create initial sample."
        except Exception as e:
            initial_data_sample_loaded_status[project_name] = False
            initial_data_sample_message_dict[project_name] = f"Error creating initial sample: {e}"
            print(f"Error creating initial sample: {e}", file = sys.stderr)
    healthcheck.initial_data_sample_loaded = initial_data_sample_loaded_status
    healthcheck.initial_data_sample_message = initial_data_sample_message_dict
    # 3. Load or create the model
    model_load_status = {}
    model_message_dict = {}
    for project_name in PROJECT_NAMES:
        try:
            print("Loading model...")
            model_folder_name = project_name.lower().replace(' ', '_').replace("-", "_")
            model_folder = f"models/{model_folder_name}"
            model_names = [
                x.replace(".pkl", "") 
                for x 
                in os.listdir(model_folder) 
                if x.endswith(".pkl")
            ]
            for model_name in model_names:
                model_dict[project_name][model_name] = load_or_create_model(
                    project_name,
                    model_name,
                    model_folder
                )
            model_load_status[project_name] = "success"
            model_message_dict[project_name] = "Model loaded successfully"
            print("Model loaded successfully.")
        except Exception as e:
            model_load_status[project_name] = "failed"
            model_message_dict[project_name] = f"Error: {e}"
            print(f"Error loading model: {e}", file = sys.stderr)
    healthcheck.model_load = model_load_status
    healthcheck.model_message = model_message_dict
    # 4. Create or load the ordinal encoders
    encoders_dict = {}
    encoders_load_status = {}
    print("Loading encoders...")
    for project_name in PROJECT_NAMES:
        try:
            encoders_dict[project_name] = {}
            for library in ENCODER_LIBRARIES:
                encoders_dict[project_name][library] = load_or_create_encoders(
                    project_name,
                    library
                )
            encoders_load_status[project_name] = "success"
            print("Encoder loaded successfully.")
        except Exception as e:
            encoders_load_status[project_name] = "failed"
            print(f"Error loading encoder: {e}", file = sys.stderr)
    healthcheck.encoders_load = encoders_load_status
    try:
        print("Loading MLflow...")
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
    except Exception as e:
        print(f"Error loading MLflow: {e}", file = sys.stderr)
    print("Application setup finished. Yielding control...")
    yield # <-- Application is now ready to serve requests
    # --- Shutdown Logic (Equivalent to @app.on_event("shutdown")) ---
    print("FastAPI application shutting down. Stopping any active model...")
    stop_current_model()
    print("Starting application shutdown (lifespan)...")
    print("Application shutdown finished.")


###---FastAPI App---###
app = FastAPI(lifespan = lifespan)


# IMPORTANT OBSERVATION: you can put the PUT only after the GET
@app.get("/healthcheck", response_model = Healthcheck, response_model_exclude = {"current_training_process"})
async def get_healthcheck():
    # You can add more detailed checks here if needed, e.g., if model is None
    if ("failed" in healthcheck.model_load.values()) or ("failed" in healthcheck.data_load.values()):
        raise HTTPException(
            status_code = 503, 
            detail = "Service Unavailable: Core components failed to load.")
    return healthcheck

@app.put("/healthcheck", response_model = Healthcheck, response_model_exclude = {"current_training_process"})
async def update_healthcheck(update_data: Healthcheck):
    global healthcheck # Declare that you intend to modify the global variable
    update_dict = update_data.model_dump(exclude_unset = True)
    for field, value in update_dict.items():
        setattr(healthcheck, field, value)
    return healthcheck


class SampleRequest(BaseModel):
    project_name: str


# IMPORTANT OBSERVATION: you can put the PUT only after the GET
@app.post("/sample")
async def get_sample(request: SampleRequest):
    global data_dict # Need to access the global data
    if data_dict[request.project_name] is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Data is not loaded.")
    try:
        sample = data_dict[request.project_name].sample(1).to_dict(orient = 'records')[0]
        if request.project_name ==  "E-Commerce Customer Interactions":
            sample = ECommerceCustomerInteractions.model_validate(sample)
        pprint(sample)
        return sample
    except Exception as e:
        raise HTTPException(
            status_code = 500, 
            detail = f"Error sampling data: {e}")


class UniqueValuesRequest(BaseModel):
    column_name: str
    project_name: str


@app.post("/unique_values")
async def get_unique_values(request: UniqueValuesRequest):
    global data_dict # Need to access the global data
    if data_dict[request.project_name] is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Data is not loaded.")
    column = request.column_name
    if column not in data_dict[request.project_name].columns:
        raise HTTPException(
            status_code = 400,
            detail = f"Column '{column}' not found in dataset.")
    try:
        # Use apply(str) for robustness against mixed types or non-string/numeric data
        unique_values = data_dict[request.project_name][column].apply(str).unique().tolist()
        return {"unique_values": unique_values}
    except Exception as e:
        raise HTTPException(
            status_code = 500, 
            detail = f"Error getting unique values for column '{column}': {e}")
    

class InitialSampleRequest(BaseModel):
    project_name: str


@app.post("/initial_sample")
async def get_initial_sample(request: InitialSampleRequest):
    global initial_sample_dict # Need to access the global variable
    if initial_sample_dict[request.project_name] is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Initial transaction data sample is not loaded.")
    if request.project_name == "E-Commerce Customer Interactions":
        initial_sample = ECommerceCustomerInteractions.model_validate(initial_sample_dict[request.project_name])
        return initial_sample
    else:
        return initial_sample_dict[request.project_name]


@app.post("/predict")
async def predict(payload: dict):
    #IMPORTANT OBSERVATION: payload must be something in this format:
    #{project_name: PROJECT_NAME, model_name: MODEL_NAME} | {transaction: TransactionFraudDetection}
    #{project_name: PROJECT_NAME, model_name: MODEL_NAME} | {eta_event: EstimatedTimeOfArrival}
    #{project_name: PROJECT_NAME, model_name: MODEL_NAME} | {customer: ECommerceCustomerInteractions}
    global encoders_dict, model_dict
    x = payload
    project_name = x["project_name"]
    model_name = x["model_name"]
    if model_dict[payload["project_name"]][payload["model_name"]] is None or encoders_dict[payload["project_name"]] is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Model or encoders are not loaded.")
    folder_name = project_name.lower().replace(' ', '_').replace("-", "_")
    del x["project_name"]
    del x["model_name"]
    model = load_or_create_model(
        project_name,
        model_name,
        f"models/{folder_name}")
    if model_name in [
        "ARFClassifier",
        "ARFRegressor",
        "DBSTREAM"
    ]:
        try:
            encoders = load_or_create_encoders(
                project_name,
                "river"
            )
            processed_x, _ = process_sample(
                x, 
                encoders,
                project_name) # Discard returned encoders if they are meant to be global state
            if project_name == "Transaction Fraud Detection":
                y_pred_proba = model.predict_proba_one(processed_x) # Use processed data
                fraud_probability = y_pred_proba.get(1, 0.0) # Use 0.0 as default
                binary_prediction = 1 if fraud_probability >= 0.5 else 0
                return {
                    "fraud_probability": fraud_probability,
                    "prediction": binary_prediction,
                }
            elif project_name == "Estimated Time of Arrival":
                y_pred = model.predict_one(processed_x) # Use processed data
                return {
                    "Estimated Time of Arrival": y_pred
                }
            elif project_name == "E-Commerce Customer Interactions":
                y_pred = model.predict_one(processed_x) # Use processed data
                return {
                    "cluster": y_pred
                }
            #elif project_name == "Sales Forecasting":
            #    y_pred = model.forecast(horizon = 1, xs = [processed_x]) # Use processed data
            #    return {
            #        "sales": y_pred
            #    }
        except Exception as e:
            print(f"Error during prediction: {e}", file = sys.stderr)
            raise HTTPException(
                status_code = 500, 
                detail = f"Prediction failed: {e}")
    elif model_name in [
        "XGBClassifier"
    ]:
        try:
            encoders = load_or_create_encoders(
                project_name,
                "sklearn"
            )
            if project_name == "Transaction Fraud Detection":
                processed_x = process_sample(x, ..., project_name, library = "sklearn")
                preprocessor = encoders["preprocessor"]
                processed_x = pd.DataFrame(
                    {x: [y] for x, y in processed_x.items()})
                processed_x = preprocessor.transform(processed_x)
                y_pred_proba = model.predict_proba(processed_x).tolist()[0] # Use processed data
                fraud_probability = y_pred_proba[1]
                binary_prediction = 1 if fraud_probability >= 0.5 else 0
                return {
                    "fraud_probability": fraud_probability,
                    "prediction": binary_prediction,
                }
        except Exception as e:
            print(f"Error during prediction: {e}", file = sys.stderr)
            raise HTTPException(
                status_code = 500, 
                detail = f"Prediction failed: {e}")
        
        
class OrdinalEncoderRequest(BaseModel):
    project_name: str

@app.post("/get_ordinal_encoder")
async def get_ordinal_encoder(request: OrdinalEncoderRequest):
    global encoders
    encoders = load_or_create_encoders(
        request.project_name,
        "river"
    )
    if None in encoders.values():
        raise HTTPException(
            status_code = 503, 
            detail = "Ordinal encoder is not loaded.")
    if request.project_name in ["Transaction Fraud Detection", "Estimated Time of Arrival"]:
        return {
            "ordinal_encoder": encoders["ordinal_encoder"].get_feature_mappings()
        }
    elif request.project_name in ["E-Commerce Customer Interactions"]:
        return {
            "standard_scaler": encoders["standard_scaler"].counts,
        }



class MLflowMetricsRequest(BaseModel):
    project_name: str
    model_name: str


@app.post("/mlflow_metrics")
async def get_mlflow_metrics(request: MLflowMetricsRequest):
    experiment = mlflow.get_experiment_by_name(request.project_name)
    experiment_id = experiment.experiment_id
    runs_df = mlflow.search_runs(
        experiment_ids = [experiment_id]
    )
    runs_df = runs_df[
        runs_df["tags.mlflow.runName"] == request.model_name]
    run_df = runs_df.iloc[0]
    run_df = run_df.replace({np.nan: None}).to_dict()
    return run_df


@app.get("/cluster_counts")
async def get_cluster_counts():
    try:
        with open("data/cluster_counts.json", 'r') as f:
            cluster_counts = json.load(f)
        return cluster_counts
    except:
        return {}
    

class ClusterFeatureCountsRequest(BaseModel):
    column_name: str


@app.post("/cluster_feature_counts")
async def get_cluster_counts(request: ClusterFeatureCountsRequest):
    try:
        with open("data/cluster_feature_counts.json", 'r') as f:
            cluster_counts = json.load(f)
        clusters = list(cluster_counts.keys())
        return {
            x: cluster_counts[x][request.column_name]
            for x in clusters
        }
    except:
        return {}
    

@app.post("/switch_model")
async def switch_model(payload: dict):
    global healthcheck
    if payload["model_key"] == healthcheck.current_model_name:
        return {"message": f"Model {payload["model_key"]} is already running."}
    # Stop any currently running model
    if healthcheck.current_training_process:
        print(f"Switching from {healthcheck.current_model_name} to {payload["model_key"]}")
        stop_current_model()
    else:
        print(f"No model running, attempting to start {payload["model_key"]}")
    if payload["model_key"] == "none" or payload["model_key"] not in MODEL_SCRIPTS.values():
        if payload["model_key"] == "none":
            return {"message": "All models stopped."}
        else:
            raise HTTPException(
                status_code = 404, 
                detail = f"Model key '{payload["model_key"]}' not found.")
    script_to_run = payload["model_key"]
    command = ["python3", script_to_run] # Use the venv python
    try:
        print(f"Starting model: {payload["model_key"]} with script: {script_to_run}")
        # Start the new training script
        # Make sure logs from these scripts go somewhere, or capture stdout/stderr
        # For simplicity, letting them log to their own files or stdout for now.
        # If you redirect stdout/stderr, ensure non-blocking reads or use tempfiles.
        log_file_path = f"/app/logs/{payload["model_key"]}.log" # Ensure /app/logs exists
        os.makedirs(
            os.path.dirname(log_file_path), 
            exist_ok = True)
        with open(log_file_path, 'ab') as log_file: # Append mode, binary
            # `start_new_session=True` or `preexec_fn=os.setsid` can be useful if you need to kill process groups
            # For SIGTERM on the direct child, it's usually not strictly necessary but good practice.
            process = subprocess.Popen(
                command,
                stdout = log_file,
                stderr = subprocess.STDOUT, # Redirect stderr to stdout (goes to same log_file)
                cwd = "/app" # Assuming scripts are in /app and paths are relative to it
                          # or use absolute paths for scripts
            )
        healthcheck.current_training_process = process
        healthcheck.current_model_name = payload["model_key"]
        print(f"Model {payload["model_key"]} started with PID: {process.pid}. Logs at {log_file_path}")
        if payload["model_key"].endswith("sklearn.py"):
            data_manager.load_data(payload["project_name"])
        return {"message": f"Switched to model: {payload["model_key"]}"}
    except Exception as e:
        print(f"Failed to start model {payload["model_key"]}: {e}")
        healthcheck.current_training_process = None # Ensure state is clean
        healthcheck.current_model_name = None
        raise HTTPException(
            status_code = 500, 
            detail = f"Failed to start model {payload["model_key"]}: {str(e)}")
    

@app.get("/current_model")
async def get_current_model():
    if healthcheck.current_model_name and healthcheck.current_training_process:
        # Check if process is still alive
        if healthcheck.current_training_process.poll() is None: # None means still running
            return {"current_model": healthcheck.current_model_name, "status": "running", "pid": healthcheck.current_training_process.pid}
        else: # Process has terminated
            return_code = healthcheck.current_training_process.returncode
            stop_current_model() # Clean up state
            return {"current_model": healthcheck.current_model_name, "status": f"stopped (exit code: {return_code})", "pid": None }
    return {"current_model": None, "status": "none running"}


@app.post("/yellowbrick_metric", response_class = FileResponse)
async def yellowbrick_metric(payload: dict):
    PROJECT_NAME = payload["project_name"]
    METRIC_TYPE = payload["metric_type"]
    METRIC_NAME = payload["metric_name"]
    # Ensure data is loaded for the requested project
    data_manager.load_data(PROJECT_NAME)
    if PROJECT_NAME == "Transaction Fraud Detection":
        classes = list(set(data_manager.y_train.unique().tolist() + data_manager.y_test.unique().tolist()))
        classes.sort()
        fig_buf = io.BytesIO()
        if METRIC_TYPE == "Classification":
            yb_kwargs = yellowbrick_classification_kwargs(
                PROJECT_NAME,
                METRIC_NAME,
                data_manager.y_train,
                classes
            )
            yb_vis = yellowbrick_classification_visualizers(
                yb_kwargs,
                data_manager.X_train,
                data_manager.X_test,
                data_manager.y_train,
                data_manager.y_test,
            )
            yb_vis.fig.savefig(fig_buf, format = "png")
        elif METRIC_TYPE == "Feature Analysis":
            yb_kwargs = yellowbrick_feature_analysis_kwargs(
                PROJECT_NAME,
                METRIC_NAME,
                classes
            )
            yb_vis = yellowbrick_feature_analysis_visualizers(
                yb_kwargs,
                data_manager.X,
                data_manager.y,
            )
            yb_vis.fig.savefig(fig_buf, format = "png")
        elif METRIC_TYPE == "Target":
            labels = list(set(data_manager.y_train.unique().tolist() + data_manager.y_test.unique().tolist()))
            features = data_manager.X_train.columns.tolist()
            yb_kwargs = yellowbrick_target_kwargs(
                PROJECT_NAME,
                METRIC_NAME,
                labels,
                features
            )
            yb_vis = yellowbrick_target_visualizers(
                yb_kwargs,
                data_manager.X,
                data_manager.y,
            )
            yb_vis.fig.savefig(fig_buf, format = "png")
        elif METRIC_TYPE == "Model Selection":
            yb_kwargs = yellowbrick_model_selection_kwargs(
                PROJECT_NAME,
                METRIC_NAME,
                data_manager.y_train
            )
            yb_vis = yellowbrick_model_selection_visualizers(
                yb_kwargs,
                data_manager.X,
                data_manager.y,
            )
            yb_vis.fig.savefig(fig_buf, format = "png")
        fig_buf.seek(0)
        plt.clf()
        return StreamingResponse(
            fig_buf, 
            media_type = "image/png")

