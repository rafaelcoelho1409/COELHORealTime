from fastapi import (
    FastAPI,
    HTTPException,
    Request
)
from fastapi.responses import (
    FileResponse,
    StreamingResponse
)
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
from pydantic import (
    BaseModel,
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
import os
import numpy as np
import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import asyncio
from functools import lru_cache
import time
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


MLFLOW_HOST = os.getenv("MLFLOW_HOST", "localhost")


# Simple TTL cache for MLflow metrics
class MLflowMetricsCache:
    def __init__(self, ttl_seconds: int = 300):  # 5 minute TTL
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


mlflow_cache = MLflowMetricsCache(ttl_seconds = 30)  # Reduced from 300 to 30 seconds


def _sync_get_mlflow_metrics(project_name: str, model_name: str) -> dict:
    """Synchronous MLflow query - to be run in thread pool."""
    experiment = mlflow.get_experiment_by_name(project_name)
    if experiment is None:
        raise ValueError(f"Experiment '{project_name}' not found in MLflow")
    experiment_id = experiment.experiment_id
    runs_df = mlflow.search_runs(
        experiment_ids = [experiment_id],
        max_results = 100,  # Limit results to reduce memory
        order_by = ["start_time DESC"]  # Get most recent runs first
    )
    runs_df = runs_df[runs_df["tags.mlflow.runName"] == model_name]
    if runs_df.empty:
        raise ValueError(f"No runs found for model '{model_name}'")
    # Get the most recent run (already sorted by start_time DESC)
    run_df = runs_df.iloc[0]
    return run_df.replace({np.nan: None}).to_dict()


PROJECT_NAMES = [
    "Transaction Fraud Detection", 
    "Estimated Time of Arrival",
    "E-Commerce Customer Interactions",
    #"Sales Forecasting"
]
# NOTE: MODEL_SCRIPTS removed - training moved to River app
ENCODER_LIBRARIES = [
    "river",
    "sklearn"
]


# NOTE: stop_current_model() removed - training moved to River app


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
    # NOTE: Training-related fields removed - training moved to River app

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
    @field_validator('quantity', mode = 'before')
    @classmethod
    def check_quantity_is_finite_int(cls, v: Any) -> Optional[int]:
        """Convert float NaN to None before checking if input is valid Optional[int]."""
        # Check specifically for float('nan') as infinity doesn't make sense for quantity
        if isinstance(v, float) and math.isnan(v):
            return None
        return v # Let Pydantic validate if it's int or None
    # Add validator for the price field
    @field_validator('price', mode = 'before')
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
    # --- Shutdown Logic ---
    print("FastAPI application shutting down...")
    print("Application shutdown finished.")


###---FastAPI App---###
app = FastAPI(lifespan = lifespan)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],  # Configure appropriately for production
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# Add Prometheus metrics instrumentation
# Exposes /metrics endpoint for Prometheus scraping
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint with rate limiting.
    Returns 200 OK if the service is running.
    Rate limited to 100 requests/minute per IP to prevent abuse.
    """
    return {"status": "ok"}


# IMPORTANT OBSERVATION: you can put the PUT only after the GET
@app.get("/healthcheck", response_model = Healthcheck)
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
    limit: int = 100  # Default limit to prevent dropdown performance issues


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
        # Apply limit to prevent performance issues with high-cardinality fields
        if len(unique_values) > request.limit:
            unique_values = unique_values[:request.limit]
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
        initial_sample = ECommerceCustomerInteractions.model_validate(
            initial_sample_dict[request.project_name])
        return initial_sample
    else:
        return initial_sample_dict[request.project_name]


# NOTE: /predict endpoint moved to River app
# Predictions are now handled by the River ML Training Service


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
    force_refresh: bool = False  # Bypass cache when True


@app.post("/mlflow_metrics")
async def get_mlflow_metrics(request: MLflowMetricsRequest):
    """Get MLflow metrics with caching and async execution."""
    cache_key = f"{request.project_name}:{request.model_name}"
    # Check cache first (unless force_refresh is True)
    if not request.force_refresh:
        cached_result = mlflow_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
    try:
        # Run blocking MLflow call in thread pool with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_get_mlflow_metrics,
                request.project_name,
                request.model_name
            ),
            timeout = 30.0  # 30 second timeout
        )
        # Cache the result
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
        print(f"Error fetching MLflow metrics: {e}", file = sys.stderr)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to fetch MLflow metrics: {str(e)}"
        )


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
    

# NOTE: /switch_model and /current_model endpoints moved to River app
# Training is now handled by the River ML Training Service


def _sync_generate_yellowbrick_plot(
    project_name: str,
    metric_type: str,
    metric_name: str,
    dm: ModelDataManager
) -> bytes:
    """Synchronous yellowbrick plot generation - runs in thread pool."""
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
            features = dm.X_train.columns.tolist()
            yb_kwargs = yellowbrick_target_kwargs(
                project_name, metric_name, labels, features
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
        # CRITICAL: Proper matplotlib cleanup to prevent memory leaks
        plt.clf()
        plt.close('all')
        if fig_buf:
            fig_buf.close()


@app.post("/yellowbrick_metric", response_class = FileResponse)
async def yellowbrick_metric(payload: dict):
    """Generate yellowbrick visualizations asynchronously."""
    project_name = payload.get("project_name")
    metric_type = payload.get("metric_type")
    metric_name = payload.get("metric_name")
    if not all([project_name, metric_type, metric_name]):
        raise HTTPException(
            status_code = 400,
            detail = "Missing required fields: project_name, metric_type, metric_name"
        )
    try:
        # Run blocking visualization in thread pool with timeout
        image_bytes = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_generate_yellowbrick_plot,
                project_name,
                metric_type,
                metric_name,
                data_manager
            ),
            timeout = 120.0  # 2 minute timeout for complex visualizations
        )
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type = "image/png"
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code = 504,
            detail = "Visualization generation timed out after 120 seconds"
        )
    except ValueError as e:
        raise HTTPException(
            status_code = 400, 
            detail = str(e))
    except Exception as e:
        print(f"Error generating yellowbrick plot: {e}", file = sys.stderr)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to generate visualization: {str(e)}"
        )

