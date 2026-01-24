"""
Shared Pydantic models for the Unified FastAPI Service.

This module contains all request/response models used across routers:
- Domain models (TFD, ETA, ECCI data structures)
- API request models
- API response models
"""
from pydantic import BaseModel, field_validator
from typing import Optional, Any, Dict
import math
import json

from config import PROJECT_NAMES, SQL_DEFAULT_LIMIT


# =============================================================================
# Healthcheck Models
# =============================================================================
class Healthcheck(BaseModel):
    """Healthcheck status for incremental ML service (River)."""
    model_load: Dict[str, Optional[str]] = {}
    model_message: Dict[str, Optional[str]] = {}
    encoders_load: Dict[str, Optional[str]] = {}
    data_load: Dict[str, Optional[str]] = {}
    data_message: Dict[str, Optional[str]] = {}


class SklearnHealthcheck(BaseModel):
    """Healthcheck status for sklearn batch ML service."""
    model_available: Dict[str, bool] = {}
    encoders_load: Dict[str, str] = {}


class UnifiedHealthcheck(BaseModel):
    """Healthcheck status for the unified service."""
    status: str = "healthy"
    service: str = "unified-ml-service"
    incremental: dict = {}
    batch: dict = {}
    sql: dict = {}


# =============================================================================
# Nested Domain Models
# =============================================================================
class DeviceInfo(BaseModel):
    """Device information for TFD and ECCI."""
    device_type: Optional[str] = None
    browser: Optional[str] = None
    os: Optional[str] = None


class Location(BaseModel):
    """Geographic location with lat/lon coordinates."""
    lat: Optional[float] = None
    lon: Optional[float] = None

    @field_validator('lat', 'lon', mode='before')
    @classmethod
    def check_location_coords_finite(cls, v: Any) -> Optional[float]:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v


# =============================================================================
# Transaction Fraud Detection Models
# =============================================================================
class TransactionFraudDetection(BaseModel):
    """Transaction data for fraud detection."""
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

    @field_validator('location', 'device_info', mode='before')
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


# =============================================================================
# Estimated Time of Arrival Models
# =============================================================================
class EstimatedTimeOfArrival(BaseModel):
    """Trip data for ETA prediction."""
    trip_id: str
    driver_id: str
    vehicle_id: str
    timestamp: str
    origin: dict  # {"lat": float, "lon": float}
    destination: dict  # {"lat": float, "lon": float}
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

    @field_validator('origin', 'destination', mode='before')
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


# =============================================================================
# E-Commerce Customer Interactions Models
# =============================================================================
class ECommerceCustomerInteractions(BaseModel):
    """Customer interaction event for clustering."""
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

    @field_validator('device_info', mode='before')
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

    @field_validator('location', mode='before')
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

    @field_validator('quantity', mode='before')
    @classmethod
    def check_quantity_is_finite_int(cls, v: Any) -> Optional[int]:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    @field_validator('price', mode='before')
    @classmethod
    def check_price_is_finite(cls, v: Any) -> Optional[float]:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v


# =============================================================================
# API Request Models - Common
# =============================================================================
class ProjectRequest(BaseModel):
    """Base request with project_name."""
    project_name: str


class SampleRequest(ProjectRequest):
    """Request for random sample from dataset."""
    n: int = 1


class OrdinalEncoderRequest(BaseModel):
    """Request for ordinal encoder mappings."""
    project_name: str


class ModelAvailabilityRequest(BaseModel):
    """Request to check model availability."""
    project_name: str
    model_name: str


class MLflowMetricsRequest(BaseModel):
    """Request for MLflow metrics."""
    project_name: str
    model_name: str
    run_id: Optional[str] = None
    force_refresh: bool = False


# =============================================================================
# API Request Models - Incremental ML
# =============================================================================
class SwitchModelRequest(BaseModel):
    """Request to start/stop incremental ML training."""
    model_key: str  # Script filename or "none" to stop
    project_name: str = ""


class PageInitRequest(BaseModel):
    """Request for combined page initialization (incremental ML)."""
    project_name: str
    model_name: str = "ARFClassifier"


# =============================================================================
# API Request Models - Batch ML
# =============================================================================
class BatchSwitchModelRequest(BaseModel):
    """Request to start/stop batch ML training."""
    model_key: str  # Script filename or "none" to stop
    sample_frac: Optional[float] = None  # 0.0-1.0
    max_rows: Optional[int] = None


class BatchInitRequest(BaseModel):
    """Request for batch page initialization."""
    project_name: str
    run_id: Optional[str] = None  # Optional: specific run, or None for best


class BatchMLflowRunsRequest(BaseModel):
    """Request for listing MLflow runs."""
    project_name: str
    model_name: Optional[str] = None


class TrainingStatusUpdate(BaseModel):
    """Request body for training status updates from training scripts."""
    message: str
    progress: Optional[int] = None
    stage: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    total_rows: Optional[int] = None
    # KMeans K-search log (for ECCI clustering)
    kmeans_log: Optional[Dict[str, Any]] = None


class YellowBrickRequest(BaseModel):
    """Request for YellowBrick visualization generation."""
    project_name: str
    metric_type: str  # Classification, Regression, Feature Analysis, etc.
    metric_name: str  # ConfusionMatrix, ResidualsPlot, RadViz, etc.
    run_id: Optional[str] = None  # Optional: specific run, or None for best


class SklearnVisualizationRequest(BaseModel):
    """Request for sklearn visualization generation."""
    project_name: str
    metric_type: str  # Classification, Feature Analysis, Model Selection
    metric_name: str  # ConfusionMatrixDisplay, RocCurveDisplay, etc.
    run_id: Optional[str] = None  # Optional: specific run, or None for best


# =============================================================================
# API Request Models - Delta Lake SQL
# =============================================================================
class SQLQueryRequest(BaseModel):
    """Request for SQL queries against Delta Lake using DuckDB."""
    project_name: str
    query: str
    limit: int = SQL_DEFAULT_LIMIT


class TableSchemaRequest(BaseModel):
    """Request for table schema."""
    project_name: str


# =============================================================================
# API Request Models - Prediction
# =============================================================================
class PredictRequest(BaseModel):
    """Unified prediction request supporting all projects."""
    project_name: str
    model_name: str
    run_id: Optional[str] = None  # Optional: specific run, or None for best

    # TFD fields
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

    # ETA fields
    trip_id: Optional[str] = None
    driver_id: Optional[str] = None
    vehicle_id: Optional[str] = None
    origin: Optional[dict] = None
    destination: Optional[dict] = None
    estimated_distance_km: Optional[float] = None
    weather: Optional[str] = None
    temperature_celsius: Optional[float] = None
    day_of_week: Optional[int] = None
    hour_of_day: Optional[int] = None
    driver_rating: Optional[float] = None
    vehicle_type: Optional[str] = None
    initial_estimated_travel_time_seconds: Optional[int] = None
    debug_traffic_factor: Optional[float] = None
    debug_weather_factor: Optional[float] = None
    debug_incident_delay_seconds: Optional[int] = None
    debug_driver_factor: Optional[float] = None

    # ECCI fields
    customer_id: Optional[str] = None
    event_id: Optional[str] = None
    session_id: Optional[str] = None
    event_type: Optional[str] = None
    product_id: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    page_url: Optional[str] = None
    referrer_url: Optional[str] = None
    search_query: Optional[str] = None
    time_on_page_seconds: Optional[int] = None
    session_event_sequence: Optional[int] = None


# =============================================================================
# API Response Models
# =============================================================================
class PredictResponse(BaseModel):
    """Base prediction response."""
    model_source: str  # "live", "mlflow", or "cache"
    run_id: Optional[str] = None


class TFDPredictResponse(PredictResponse):
    """Transaction Fraud Detection prediction response."""
    fraud_probability: float
    prediction: int  # 0 or 1


class ETAPredictResponse(PredictResponse):
    """Estimated Time of Arrival prediction response."""
    estimated_travel_time_seconds: float


class ECCIPredictResponse(PredictResponse):
    """E-Commerce Customer Interactions prediction response."""
    cluster_id: int


class SQLQueryResponse(BaseModel):
    """SQL query execution response."""
    columns: list[str]
    data: list[dict]
    row_count: int
    execution_time_ms: float


class TableSchemaResponse(BaseModel):
    """Table schema response."""
    table_name: str
    delta_path: str
    columns: list[dict]  # {"name": str, "type": str, "nullable": bool}
    approximate_row_count: int
