from fastapi import (
    FastAPI,
    HTTPException
)
from contextlib import asynccontextmanager
from pydantic import (
    BaseModel, 
    validator, 
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
import pickle
import json
from functions import (
    process_sample,
    load_or_create_model,
    load_or_create_data,
    create_consumer,
    load_or_create_encoders,
)

PROJECT_NAMES = [
    "Transaction Fraud Detection", 
    "Estimated Time of Arrival",
    "E-Commerce Customer Interactions"
]


# Initialize global variables as None or placeholder BEFORE startup
consumer_dict = {x: None for x in PROJECT_NAMES}
data_dict = {x: None for x in PROJECT_NAMES}
model_dict = {x: None for x in PROJECT_NAMES}
encoders_dict = {x: None for x in PROJECT_NAMES}
initial_sample_dict = {x: None for x in PROJECT_NAMES} # Also move this initialization


class Healthcheck(BaseModel):
    model_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES} # "success", "failed", "not_attempted"
    model_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}
    encoders_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES} # "success", "failed", "not_attempted"
    data_load: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES} # Added data load status
    data_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES} # Added data load message
    initial_data_sample_loaded: dict[str, bool] | dict[str, None] = {x: None for x in PROJECT_NAMES} # Added sample status
    initial_data_sample_message: dict[str, str] | dict[str, None] = {x: None for x in PROJECT_NAMES}


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
            model_dict[project_name] = load_or_create_model(
                project_name,
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
            encoders_dict[project_name] = load_or_create_encoders(
                project_name
            )
            encoders_load_status[project_name] = "success"
            print("Encoder loaded successfully.")
        except Exception as e:
            encoders_load_status[project_name] = "failed"
            print(f"Error loading encoder: {e}", file = sys.stderr)
    healthcheck.encoders_load = encoders_load_status
    try:
        print("Loading MLflow...")
        mlflow.set_tracking_uri("http://mlflow:5000")
    except Exception as e:
        print(f"Error loading MLflow: {e}", file = sys.stderr)
    print("Application setup finished. Yielding control...")
    yield # <-- Application is now ready to serve requests
    # --- Shutdown Logic (Equivalent to @app.on_event("shutdown")) ---
    print("Starting application shutdown (lifespan)...")
    print("Application shutdown finished.")


###---FastAPI App---###
app = FastAPI(lifespan = lifespan)


# IMPORTANT OBSERVATION: you can put the PUT only after the GET
@app.get("/healthcheck")
async def get_healthcheck():
    # You can add more detailed checks here if needed, e.g., if model is None
    if ("failed" in healthcheck.model_load.values()) or ("failed" in healthcheck.data_load.values()):
        raise HTTPException(
            status_code = 503, 
            detail = "Service Unavailable: Core components failed to load.")
    return healthcheck

@app.put("/healthcheck", response_model = Healthcheck)
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
    #{project_name: PROJECT_NAME} | {transaction: TransactionFraudDetection}
    #{project_name: PROJECT_NAME} | {eta_event: EstimatedTimeOfArrival}
    global encoders_dict, model_dict
    if model_dict[payload["project_name"]] is None or encoders_dict[payload["project_name"]] is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Model or encoders are not loaded.")
    x = payload
    project_name = x["project_name"]
    folder_name = project_name.lower().replace(' ', '_').replace("-", "_")
    del x["project_name"]
    try:
        model = load_or_create_model(
            project_name,
            f"models/{folder_name}")
        encoders = load_or_create_encoders(
            project_name
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
        request.project_name
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


@app.post("/mlflow_metrics")
async def get_mlflow_metrics(request: MLflowMetricsRequest):
    experiment = mlflow.get_experiment_by_name(request.project_name)
    experiment_id = experiment.experiment_id
    runs_df = mlflow.search_runs(
        experiment_ids = [experiment_id]
    )
    run_df = runs_df.iloc[0]
    return run_df.to_dict()


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