"""
Shared state module - constants, helpers, and common utilities.

This module contains:
- API base URLs (RIVER_BASE_URL, SKLEARN_BASE_URL)
- Pre-loaded constants (METRIC_INFO, DROPDOWN_OPTIONS)
- Helper functions (safe_str, safe_int_str, etc.)
"""
import reflex as rx
import os
import asyncio
import datetime as dt
import json
import plotly.graph_objects as go
import folium
import orjson
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from ..utils import httpx_client_post, httpx_client_get


# =============================================================================
# MLflow Run Type Definition (for typed foreach in Reflex)
# =============================================================================
class MLflowRunInfo(BaseModel):
    """Pydantic model for MLflow run info returned by /mlflow_runs endpoint."""
    run_id: str
    run_name: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    metrics: dict = {}
    params: dict = {}
    total_rows: int = 0
    is_best: bool = False

# =============================================================================
# API Base URLs
# =============================================================================
RIVER_HOST = os.getenv("RIVER_HOST", "localhost")
RIVER_BASE_URL = f"http://{RIVER_HOST}:8002"
SKLEARN_HOST = os.getenv("SKLEARN_HOST", "localhost")
SKLEARN_BASE_URL = f"http://{SKLEARN_HOST}:8003"


# =============================================================================
# Metric Info Loader (LaTeX formulas and contextual explanations)
# =============================================================================
def load_metric_info(project_key: str) -> dict:
    """Load metric info JSON for a project using orjson."""
    data_dir = Path(__file__).parent.parent / "data"
    file_path = data_dir / f"metric_info_{project_key}.json"
    if file_path.exists():
        with open(file_path, "rb") as f:
            return orjson.loads(f.read())
    return {"metrics": {}}

# Load metric info at module startup
METRIC_INFO = {
    "tfd": load_metric_info("tfd"),
    "eta": load_metric_info("eta"),
    "ecci": load_metric_info("ecci"),
}


# =============================================================================
# YellowBrick Visualizer Info Loader (descriptions and interpretations)
# =============================================================================
def load_yellowbrick_info(project_key: str) -> dict:
    """Load YellowBrick visualizer info JSON for a project using orjson."""
    data_dir = Path(__file__).parent.parent / "data"
    file_path = data_dir / f"yellowbrick_info_{project_key}.json"
    if file_path.exists():
        with open(file_path, "rb") as f:
            return orjson.loads(f.read())
    return {"visualizers": {}}

# Load yellowbrick info at module startup
YELLOWBRICK_INFO = {
    "tfd": load_yellowbrick_info("tfd"),
}


# =============================================================================
# Dropdown Options Loader (pre-loaded for instant form rendering)
# =============================================================================
def load_dropdown_options(project_key: str) -> dict:
    """Load dropdown options JSON for a project using orjson."""
    data_dir = Path(__file__).parent.parent / "data"
    file_path = data_dir / f"dropdown_options_{project_key}.json"
    if file_path.exists():
        with open(file_path, "rb") as f:
            return orjson.loads(f.read())
    return {}

# Load dropdown options at module startup for instant form rendering
DROPDOWN_OPTIONS = {
    "Transaction Fraud Detection": load_dropdown_options("tfd"),
    "Estimated Time of Arrival": load_dropdown_options("eta"),
    "E-Commerce Customer Interactions": load_dropdown_options("ecci"),
}


# =============================================================================
# Form Value Helpers
# =============================================================================
# These helpers ensure form values display correctly in Reflex inputs.
# JavaScript treats 0, 0.0, None, and "" as falsy, which causes issues with
# Reflex's .get() method returning defaults instead of actual values.
# Converting numeric values to strings ensures they display properly.

def safe_str(val, default: str = "") -> str:
    """Safely convert value to string, handling None."""
    return str(val) if val is not None else default

def safe_int_str(val, default: int = 0) -> str:
    """Safely convert to int then string, handling None. Returns string for form display."""
    return str(int(val)) if val is not None else str(default)

def safe_float_str(val, default: float = 0.0) -> str:
    """Safely convert to float then string, handling None. Returns string for form display."""
    return str(float(val)) if val is not None else str(default)

def safe_bool(val, default: bool = False) -> bool:
    """Safely convert to bool, handling None."""
    return bool(val) if val is not None else default

def get_str(data: dict, key: str, default: str = "") -> str:
    """Get string value from dict, converting None to default."""
    return data.get(key) or default

def get_nested_str(data: dict, key1: str, key2: str, default: str = "") -> str:
    """Get nested string value from dict, converting None to default."""
    return data.get(key1, {}).get(key2) or default

def parse_json_field(data: dict, key: str) -> dict:
    """Parse a JSON string field from Delta Lake into a dict.

    Delta Lake stores nested objects (location, device_info, origin, destination)
    as JSON strings. This function safely parses them back to dicts.
    """
    value = data.get(key)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


# =============================================================================
# SharedState - Base state class with common variables and methods
# =============================================================================
class SharedState(rx.State):
    """Base state class with shared variables and methods for all pages.

    Domain-specific states (TFDState, ETAState, ECCIState, SQLState) inherit
    from this class to access common state variables while adding their own
    domain-specific logic.
    """

    # ==========================================================================
    # COMMON STATE VARIABLES
    # ==========================================================================
    tab_name: str = "incremental_ml"
    page_name_mapping: dict = {
        "/": "Home",
        "/transaction-fraud-detection": "Transaction Fraud Detection",
        "/estimated-time-of-arrival": "Estimated Time of Arrival",
        "/e-commerce-customer-interactions": "E-Commerce Customer Interactions"
    }
    project_name: str = "Home"

    # Track the currently active Kafka producer/model
    # Only one model runs at a time to conserve resources
    activated_model: str = ""
    model_switch_message: str = ""
    model_switch_error: str = ""

    # ML training switch state - user-controlled toggle
    ml_training_enabled: bool = False
    _current_page_model_key: str = ""  # Track which model key belongs to current page

    # Incremental ML state (shared across all projects)
    incremental_ml_state: dict = {
        "Transaction Fraud Detection": False,
        "Estimated Time of Arrival": False,
        "E-Commerce Customer Interactions": False
    }
    incremental_ml_model_name: dict = {
        "Transaction Fraud Detection": "Adaptive Random Forest Classifier (River)",
        "Estimated Time of Arrival": "Adaptive Random Forest Regressor (River)",
        "E-Commerce Customer Interactions": "DBSTREAM Clustering (River)",
    }
    incremental_ml_sample: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }

    # Form data, dropdown options, prediction results (shared structure)
    form_data: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    # Pre-loaded from JSON files for instant form rendering
    dropdown_options: dict = DROPDOWN_OPTIONS.copy()
    prediction_results: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    mlflow_metrics: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }

    # Report metrics from MLflow artifacts (ConfusionMatrix, ClassificationReport)
    report_metrics: dict = {
        "Transaction Fraud Detection": {},
    }

    # Incremental ML model availability (checked from MLflow)
    incremental_model_available: dict = {
        "Transaction Fraud Detection": False,
        "Estimated Time of Arrival": False,
        "E-Commerce Customer Interactions": False
    }
    incremental_model_last_trained: dict = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }

    # MLflow experiment URLs for each project
    mlflow_experiment_url: dict = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }

    # Model name mapping for MLflow metrics
    _mlflow_model_names: dict = {
        "Transaction Fraud Detection": "ARFClassifier",
        "Estimated Time of Arrival": "ARFRegressor",
        "E-Commerce Customer Interactions": "DBSTREAM",
    }

    # Incremental model name mapping for MLflow
    _incremental_model_names: dict = {
        "Transaction Fraud Detection": "ARFClassifier",
        "Estimated Time of Arrival": "ARFRegressor",
        "E-Commerce Customer Interactions": "DBSTREAM",
    }

    # ==========================================================================
    # BATCH ML STATE VARIABLES (Scikit-Learn)
    # ==========================================================================
    # Batch ML training loading state per project
    batch_training_loading: dict = {
        "Transaction Fraud Detection": False,
        "Estimated Time of Arrival": False,
        "E-Commerce Customer Interactions": False
    }

    # Batch ML model availability (checked from MLflow)
    batch_model_available: dict = {
        "Transaction Fraud Detection": False,
        "Estimated Time of Arrival": False,
        "E-Commerce Customer Interactions": False
    }

    # Batch ML model names for display
    batch_ml_model_name: dict = {
        "Transaction Fraud Detection": "CatBoost Classifier",
        "Estimated Time of Arrival": "CatBoost Regressor",
        "E-Commerce Customer Interactions": "KMeans (Scikit-Learn)",
    }

    # Batch model name mapping for MLflow
    _batch_model_names: dict = {
        "Transaction Fraud Detection": "CatBoostClassifier",
        "Estimated Time of Arrival": "CatBoostRegressor",
        "E-Commerce Customer Interactions": "KMeans",
    }

    # Batch training script mapping (for subprocess-based training)
    _batch_training_scripts: dict = {
        "Transaction Fraud Detection": "transaction_fraud_detection_sklearn.py",
        "Estimated Time of Arrival": "estimated_time_of_arrival_sklearn.py",
        "E-Commerce Customer Interactions": "e_commerce_customer_interactions_sklearn.py",
    }

    # Batch MLflow experiment URLs for each project
    batch_mlflow_experiment_url: dict = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }

    # Batch MLflow metrics
    batch_mlflow_metrics: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }

    # Batch prediction results
    batch_prediction_results: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }

    # Batch training data percentage (0-100, where 100 = use all data)
    batch_training_data_percentage: dict[str, int] = {
        "Transaction Fraud Detection": 100,
        "Estimated Time of Arrival": 100,
        "E-Commerce Customer Interactions": 100
    }

    # Live training status (updated during batch training)
    batch_training_status: dict[str, str] = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }
    batch_training_progress: dict[str, int] = {
        "Transaction Fraud Detection": 0,
        "Estimated Time of Arrival": 0,
        "E-Commerce Customer Interactions": 0
    }
    batch_training_stage: dict[str, str] = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }
    batch_training_metrics_preview: dict[str, dict] = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    # CatBoost training log (parsed fields from latest iteration)
    batch_training_catboost_log: dict[str, dict] = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    # Total rows loaded from DuckDB (set once after data loading)
    batch_training_total_rows: dict[str, int] = {
        "Transaction Fraud Detection": 0,
        "Estimated Time of Arrival": 0,
        "E-Commerce Customer Interactions": 0
    }

    # ==========================================================================
    # MLFLOW RUN SELECTION STATE VARIABLES
    # ==========================================================================
    # All available MLflow runs for each project (ordered by criteria, best first)
    batch_mlflow_runs: dict[str, list[MLflowRunInfo]] = {
        "Transaction Fraud Detection": [],
        "Estimated Time of Arrival": [],
        "E-Commerce Customer Interactions": []
    }
    # Currently selected run_id for each project (empty string = use best/first)
    selected_batch_run: dict[str, str] = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }
    # Loading state for fetching runs list
    batch_runs_loading: dict[str, bool] = {
        "Transaction Fraud Detection": False,
        "Estimated Time of Arrival": False,
        "E-Commerce Customer Interactions": False
    }

    # ==========================================================================
    # SQL / DELTA LAKE STATE VARIABLES
    # ==========================================================================
    sql_query_input: dict = {
        "Transaction Fraud Detection": "SELECT * FROM data LIMIT 100",
        "Estimated Time of Arrival": "SELECT * FROM data LIMIT 100",
        "E-Commerce Customer Interactions": "SELECT * FROM data LIMIT 100"
    }
    sql_query_results: dict = {
        "Transaction Fraud Detection": {"columns": [], "data": [], "row_count": 0},
        "Estimated Time of Arrival": {"columns": [], "data": [], "row_count": 0},
        "E-Commerce Customer Interactions": {"columns": [], "data": [], "row_count": 0}
    }
    sql_loading: dict = {
        "Transaction Fraud Detection": False,
        "Estimated Time of Arrival": False,
        "E-Commerce Customer Interactions": False
    }
    sql_error: dict = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }
    sql_execution_time: dict = {
        "Transaction Fraud Detection": 0.0,
        "Estimated Time of Arrival": 0.0,
        "E-Commerce Customer Interactions": 0.0
    }
    sql_engine: dict = {
        "Transaction Fraud Detection": "polars",
        "Estimated Time of Arrival": "polars",
        "E-Commerce Customer Interactions": "polars"
    }
    sql_search_filter: dict = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }
    sql_sort_column: dict = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }
    sql_sort_direction: dict = {
        "Transaction Fraud Detection": "asc",
        "Estimated Time of Arrival": "asc",
        "E-Commerce Customer Interactions": "asc"
    }
    sql_table_metadata: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    sql_query_templates: dict = {
        "Transaction Fraud Detection": [
            {"name": "Sample Data", "query": "SELECT * FROM data LIMIT 100"},
            {"name": "Fraud Cases", "query": "SELECT * FROM data WHERE is_fraud = 1 LIMIT 100"},
            {"name": "High Value (>$1000)", "query": "SELECT * FROM data WHERE amount > 1000 LIMIT 100"},
            {"name": "Fraud by Merchant", "query": "SELECT merchant_id, COUNT(*) as fraud_count FROM data WHERE is_fraud = 1 GROUP BY merchant_id ORDER BY fraud_count DESC LIMIT 20"},
            {"name": "Transaction Types", "query": "SELECT transaction_type, COUNT(*) as count, AVG(amount) as avg_amount FROM data GROUP BY transaction_type ORDER BY count DESC"},
            {"name": "Row Count", "query": "SELECT COUNT(*) as total_rows FROM data"},
        ],
        "Estimated Time of Arrival": [
            {"name": "Sample Data", "query": "SELECT * FROM data LIMIT 100"},
            {"name": "Long Trips (>50km)", "query": "SELECT * FROM data WHERE estimated_distance_km > 50 LIMIT 100"},
            {"name": "Weather Impact", "query": "SELECT weather, COUNT(*) as trips, AVG(simulated_actual_travel_time_seconds) as avg_time FROM data GROUP BY weather ORDER BY trips DESC"},
            {"name": "Driver Performance", "query": "SELECT driver_id, COUNT(*) as trips, AVG(driver_rating) as avg_rating FROM data GROUP BY driver_id ORDER BY trips DESC LIMIT 20"},
            {"name": "Vehicle Types", "query": "SELECT vehicle_type, COUNT(*) as count, AVG(estimated_distance_km) as avg_distance FROM data GROUP BY vehicle_type ORDER BY count DESC"},
            {"name": "Row Count", "query": "SELECT COUNT(*) as total_rows FROM data"},
        ],
        "E-Commerce Customer Interactions": [
            {"name": "Sample Data", "query": "SELECT * FROM data LIMIT 100"},
            {"name": "Purchases", "query": "SELECT * FROM data WHERE event_type = 'purchase' LIMIT 100"},
            {"name": "By Category", "query": "SELECT product_category, COUNT(*) as count FROM data GROUP BY product_category ORDER BY count DESC"},
            {"name": "Event Types", "query": "SELECT event_type, COUNT(*) as count FROM data GROUP BY event_type ORDER BY count DESC"},
            {"name": "Top Products", "query": "SELECT product_id, COUNT(*) as views, SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases FROM data GROUP BY product_id ORDER BY views DESC LIMIT 20"},
            {"name": "Row Count", "query": "SELECT COUNT(*) as total_rows FROM data"},
        ]
    }

    # ==========================================================================
    # SHARED COMPUTED VARIABLES
    # ==========================================================================
    @rx.var
    def page_name(self) -> str:
        """Get the current page name based on the route."""
        current_path = self.router.url.path
        return self.page_name_mapping.get(current_path, "Home")

    @rx.var
    def ml_training_switch_checked(self) -> bool:
        """Check if ML training is enabled for the current page's model."""
        return self.ml_training_enabled and self.activated_model == self._current_page_model_key

    @rx.var
    def is_batch_ml_tab(self) -> bool:
        """Check if Batch ML tab is active."""
        return self.tab_name == "batch_ml"

    @rx.var
    def is_incremental_ml_tab(self) -> bool:
        """Check if Incremental ML tab is active."""
        return self.tab_name == "incremental_ml"

    @rx.var
    def is_delta_lake_sql_tab(self) -> bool:
        """Check if Delta Lake SQL tab is active."""
        return self.tab_name == "delta_lake_sql"

    @rx.var
    def mlflow_run_info(self) -> dict[str, dict]:
        """Get MLflow run info for all projects (run_id, status, is_live, start_time)."""
        result = {}
        for project_name in ["Transaction Fraud Detection", "Estimated Time of Arrival", "E-Commerce Customer Interactions"]:
            metrics = self.mlflow_metrics.get(project_name, {})
            if not isinstance(metrics, dict):
                result[project_name] = {"run_id": "", "run_id_full": "", "status": "", "is_live": False, "start_time": ""}
                continue
            run_id = metrics.get("run_id", "")
            status = metrics.get("status", "")
            is_live = metrics.get("is_live", False)
            start_time = metrics.get("start_time")
            # Format start_time if available
            start_time_str = ""
            if start_time:
                try:
                    start_time_str = dt.datetime.fromtimestamp(start_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    start_time_str = str(start_time)
            result[project_name] = {
                "run_id": run_id[:8] if run_id else "",  # Short ID for display
                "run_id_full": run_id,
                "status": status,
                "is_live": is_live,
                "start_time": start_time_str,
            }
        return result

    # ==========================================================================
    # SQL COMPUTED VARIABLES
    # ==========================================================================
    @rx.var
    def current_sql_query(self) -> str:
        """Get SQL query for current page."""
        return self.sql_query_input.get(self.page_name, "SELECT * FROM data LIMIT 100")

    @rx.var
    def current_sql_results(self) -> dict:
        """Get SQL results for current page."""
        return self.sql_query_results.get(self.page_name, {"columns": [], "data": [], "row_count": 0})

    @rx.var
    def current_sql_loading(self) -> bool:
        """Check if SQL query is loading for current page."""
        return self.sql_loading.get(self.page_name, False)

    @rx.var
    def current_sql_error(self) -> str:
        """Get SQL error for current page."""
        return self.sql_error.get(self.page_name, "")

    @rx.var
    def current_sql_execution_time(self) -> float:
        """Get SQL execution time for current page."""
        return self.sql_execution_time.get(self.page_name, 0.0)

    @rx.var
    def current_sql_engine(self) -> str:
        """Get selected SQL engine for current page."""
        return self.sql_engine.get(self.page_name, "polars")

    @rx.var
    def current_sql_search_filter(self) -> str:
        """Get search filter for current page."""
        return self.sql_search_filter.get(self.page_name, "")

    @rx.var
    def current_sql_sort_column(self) -> str:
        """Get sort column for current page."""
        return self.sql_sort_column.get(self.page_name, "")

    @rx.var
    def current_sql_sort_direction(self) -> str:
        """Get sort direction for current page."""
        return self.sql_sort_direction.get(self.page_name, "asc")

    @rx.var
    def sql_results_filtered(self) -> list[dict]:
        """Get filtered and sorted data rows from SQL results."""
        results = self.current_sql_results
        data = results.get("data", [])
        search = self.current_sql_search_filter.lower().strip()
        sort_col = self.current_sql_sort_column
        sort_dir = self.current_sql_sort_direction

        # Step 1: Filter
        if search:
            filtered = []
            for row in data:
                for value in row.values():
                    if search in str(value).lower():
                        filtered.append(row)
                        break
            data = filtered

        # Step 2: Sort
        if sort_col and data:
            try:
                def sort_key(row):
                    val = row.get(sort_col, "")
                    try:
                        return (0, float(val))
                    except (ValueError, TypeError):
                        return (1, str(val).lower())
                data = sorted(data, key=sort_key, reverse=(sort_dir == "desc"))
            except Exception:
                pass

        return data

    @rx.var
    def sql_filtered_row_count(self) -> int:
        """Get count of filtered rows."""
        return len(self.sql_results_filtered)

    @rx.var
    def sql_results_columns(self) -> list[str]:
        """Get column names from SQL results."""
        results = self.current_sql_results
        return results.get("columns", [])

    @rx.var
    def sql_results_data(self) -> list[dict]:
        """Get data rows from SQL results."""
        results = self.current_sql_results
        return results.get("data", [])

    @rx.var
    def sql_results_row_count(self) -> int:
        """Get row count from SQL results."""
        results = self.current_sql_results
        return results.get("row_count", 0)

    @rx.var
    def sql_has_results(self) -> bool:
        """Check if there are SQL results to display."""
        return len(self.sql_results_data) > 0

    @rx.var
    def sql_has_error(self) -> bool:
        """Check if there's an SQL error."""
        return bool(self.current_sql_error)

    @rx.var
    def current_sql_templates(self) -> list[dict[str, str]]:
        """Get SQL query templates for current page."""
        return self.sql_query_templates.get(self.page_name, [])

    @rx.var
    def current_table_metadata(self) -> dict:
        """Get table metadata for current page."""
        return self.sql_table_metadata.get(self.page_name, {})

    @rx.var
    def current_table_columns(self) -> list[dict[str, str]]:
        """Get table column definitions for current page."""
        metadata = self.current_table_metadata
        return metadata.get("columns", [])

    @rx.var
    def current_table_row_count(self) -> int:
        """Get approximate row count for current table."""
        metadata = self.current_table_metadata
        return metadata.get("approximate_row_count", 0)

    # ==========================================================================
    # SHARED EVENT HANDLERS
    # ==========================================================================
    @rx.event
    def set_tab(self, tab_value: str):
        """Set the active tab (incremental_ml, batch_ml, or delta_lake_sql)."""
        self.tab_name = tab_value

    @rx.event
    def set_current_page_model(self, model_key: str):
        """Set the model key for the current page (called on page mount)."""
        self._current_page_model_key = model_key
        # Reset switch state when entering a new page
        # Switch will be ON only if this page's model is already running
        self.ml_training_enabled = (self.activated_model == model_key)

    @rx.event(background=True)
    async def toggle_ml_training(self, enabled: bool, model_key: str, project_name: str):
        """
        Toggle ML training on/off via the switch component.
        This is the main user-facing control for starting/stopping Kafka streams.
        """
        async with self:
            self.ml_training_enabled = enabled
        if enabled:
            # Start the model
            if self.activated_model == model_key:
                yield rx.toast.info(
                    f"Already running",
                    description=f"ML training for {project_name} is already active",
                    duration=3000,
                )
                return
            try:
                response = await httpx_client_post(
                    url=f"{RIVER_BASE_URL}/switch_model",
                    json={
                        "model_key": model_key,
                        "project_name": project_name,
                    },
                    timeout=30.0
                )
                result = response.json()
                message = result.get("message", "Model switched successfully")
                async with self:
                    self.activated_model = model_key
                    self.model_switch_message = message
                    self.model_switch_error = ""
                    self.ml_training_enabled = True
                yield rx.toast.success(
                    f"Real-time ML training started",
                    description=f"Processing live data for {project_name}",
                    duration=5000,
                    close_button=True,
                )
                print(f"Model start successful: {message}")
            except Exception as e:
                error_msg = f"Error starting model: {e}"
                print(error_msg)
                async with self:
                    self.model_switch_error = error_msg
                    self.model_switch_message = ""
                    self.ml_training_enabled = False
                    if self.activated_model == model_key:
                        self.activated_model = ""
                yield rx.toast.error(
                    f"Failed to start ML training",
                    description=str(e),
                    duration=8000,
                    close_button=True,
                )
        else:
            # Stop the model
            if not self.activated_model:
                return
            try:
                await httpx_client_post(
                    url=f"{RIVER_BASE_URL}/switch_model",
                    json={
                        "model_key": "none",
                        "project_name": ""
                    },
                    timeout=30.0
                )
                async with self:
                    self.activated_model = ""
                    self.model_switch_message = "Model stopped"
                    self.ml_training_enabled = False
                # Re-check model availability after training stopped (model may have been saved)
                yield SharedState.check_incremental_model_available(project_name)
                yield SharedState.get_mlflow_metrics(project_name)
                yield rx.toast.info(
                    "Real-time ML training stopped",
                    description=f"Stopped processing for {project_name}",
                    duration=3000,
                )
            except Exception as e:
                print(f"Error stopping model: {e}")
                yield rx.toast.warning(
                    "Could not stop ML training",
                    description=str(e),
                    duration=5000,
                )

    @rx.event(background=True)
    async def cleanup_on_page_leave(self, project_name: str):
        """
        Called when user navigates away from a page.
        Stops the model if it was running for this page.
        """
        if self.activated_model and self._current_page_model_key:
            # Only stop if leaving the page that owns the running model
            if self.activated_model == self._current_page_model_key:
                try:
                    await httpx_client_post(
                        url=f"{RIVER_BASE_URL}/switch_model",
                        json={
                            "model_key": "none",
                            "project_name": ""
                        },
                        timeout=30.0
                    )
                    async with self:
                        self.activated_model = ""
                        self.model_switch_message = "Model stopped"
                        self.ml_training_enabled = False
                    yield rx.toast.info(
                        "Real-time ML training stopped",
                        description=f"Stopped processing for {project_name}",
                        duration=3000,
                    )
                except Exception as e:
                    print(f"Error stopping model on page leave: {e}")

    @rx.event(background=True)
    async def get_mlflow_metrics(self, project_name: str):
        """Fetch MLflow metrics for a project (runs in background to avoid lock expiration)."""
        model_name = self._mlflow_model_names.get(project_name, "ARFClassifier")
        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": model_name
                },
                timeout=60.0
            )
            async with self:
                self.mlflow_metrics = {
                    **self.mlflow_metrics,
                    project_name: response.json()
                }
            # Also fetch report metrics (ConfusionMatrix, ClassificationReport)
            if project_name == "Transaction Fraud Detection":
                await self._fetch_report_metrics_internal(project_name, model_name)
        except Exception as e:
            print(f"Error fetching MLflow metrics: {e}")
            async with self:
                self.mlflow_metrics = {
                    **self.mlflow_metrics,
                    project_name: {}
                }

    @rx.event(background=True)
    async def refresh_mlflow_metrics(self, project_name: str):
        """Force refresh MLflow metrics bypassing cache."""
        model_name = self._mlflow_model_names.get(project_name, "ARFClassifier")
        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": model_name,
                    "force_refresh": True
                },
                timeout=60.0
            )
            async with self:
                self.mlflow_metrics = {
                    **self.mlflow_metrics,
                    project_name: response.json()
                }
            # Also refresh report metrics (ConfusionMatrix, ClassificationReport)
            if project_name == "Transaction Fraud Detection":
                await self._fetch_report_metrics_internal(project_name, model_name)
            yield rx.toast.success(
                "Metrics refreshed",
                description=f"Latest metrics loaded for {project_name}",
                duration=2000
            )
        except Exception as e:
            print(f"Error refreshing MLflow metrics: {e}")
            yield rx.toast.error(
                "Refresh failed",
                description=str(e),
                duration=3000
            )

    async def _fetch_mlflow_metrics_internal(self, project_name: str, force_refresh: bool = True):
        """Internal helper to fetch MLflow metrics (called from other async methods)."""
        model_name = self._mlflow_model_names.get(project_name, "ARFClassifier")
        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": model_name,
                    "force_refresh": force_refresh
                },
                timeout=30.0
            )
            async with self:
                self.mlflow_metrics = {
                    **self.mlflow_metrics,
                    project_name: response.json()
                }
            # Also fetch report metrics (ConfusionMatrix, ClassificationReport)
            if project_name == "Transaction Fraud Detection":
                await self._fetch_report_metrics_internal(project_name, model_name)
        except Exception as e:
            print(f"Error fetching MLflow metrics: {e}")

    async def _fetch_report_metrics_internal(self, project_name: str, model_name: str):
        """Internal helper to fetch report metrics from MLflow artifacts."""
        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/report_metrics",
                json={"project_name": project_name, "model_name": model_name},
                timeout=30.0
            )
            async with self:
                self.report_metrics = {
                    **self.report_metrics,
                    project_name: response.json()
                }
        except Exception as e:
            print(f"Error fetching report metrics: {e}")

    @rx.event(background=True)
    async def check_incremental_model_available(self, project_name: str):
        """Check if a trained incremental (River) model is available in MLflow."""
        model_name = self._incremental_model_names.get(project_name, "ARFClassifier")
        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/model_available",
                json={
                    "project_name": project_name,
                    "model_name": model_name
                },
                timeout=30.0
            )
            result = response.json()

            async with self:
                self.incremental_model_available[project_name] = result.get("available", False)
                if result.get("available"):
                    self.incremental_model_last_trained[project_name] = result.get("trained_at", "")
                # Store experiment URL (available even if no model trained yet)
                experiment_url = result.get("experiment_url", "")
                if experiment_url:
                    self.mlflow_experiment_url[project_name] = experiment_url
        except Exception as e:
            print(f"Error checking incremental model availability: {e}")
            async with self:
                self.incremental_model_available[project_name] = False

    @rx.event(background=True)
    async def init_page(self, model_key: str, project_name: str):
        """Combined page initialization - fetches MLflow metrics and training status.

        Forms are populated via randomize_*_form on page mount (local, instant).
        Dropdown options are loaded from local JSON at startup via orjson.
        """
        # Synchronous state update first
        async with self:
            self._current_page_model_key = model_key
            self.ml_training_enabled = (self.activated_model == model_key)

        model_name = self._incremental_model_names.get(project_name, "ARFClassifier")

        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/page_init",
                json={
                    "project_name": project_name,
                    "model_name": model_name
                },
                timeout=10.0
            )
            data = response.json()

            async with self:
                # Model availability
                model_avail = data.get("model_available", {})
                self.incremental_model_available[project_name] = model_avail.get("available", False)
                if model_avail.get("available"):
                    self.incremental_model_last_trained[project_name] = model_avail.get("trained_at", "")
                experiment_url = model_avail.get("experiment_url", "")
                if experiment_url:
                    self.mlflow_experiment_url[project_name] = experiment_url

                # MLflow metrics
                metrics = data.get("mlflow_metrics", {})
                if metrics:
                    self.mlflow_metrics[project_name] = metrics

                # NOTE: initial_sample removed - forms are populated locally
                # NOTE: dropdown_options removed - loaded from local JSON at startup
                # Forms are randomized on page load via on_mount (see pages/*.py)

                # Training status
                training = data.get("training_status", {})
                if training.get("is_training"):
                    self.activated_model = model_key
                    self.ml_training_enabled = True

        except Exception as e:
            print(f"Error in init_page for {project_name}: {e}")
            async with self:
                self.incremental_model_available[project_name] = False

        # Also fetch batch ML model availability and metrics from sklearn service
        batch_model_name = self._batch_model_names.get(project_name, "CatBoostClassifier")
        try:
            # Check batch model availability (also returns experiment_url)
            avail_response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/model_available",
                json={
                    "project_name": project_name,
                    "model_name": batch_model_name
                },
                timeout=10.0
            )
            avail_result = avail_response.json()
            async with self:
                self.batch_model_available[project_name] = avail_result.get("available", False)
                experiment_url = avail_result.get("experiment_url", "")
                if experiment_url:
                    self.batch_mlflow_experiment_url[project_name] = experiment_url

            # Fetch batch MLflow metrics
            batch_response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": batch_model_name
                },
                timeout=10.0
            )
            async with self:
                self.batch_mlflow_metrics[project_name] = batch_response.json()
        except Exception as e:
            print(f"Error fetching batch ML data for {project_name}: {e}")

    def _update_form_from_sample(self, project_name: str, sample: dict):
        """Update form_data from sample values (helper method)."""
        if project_name == "Transaction Fraud Detection":
            if "timestamp" in sample and sample["timestamp"]:
                try:
                    ts = dt.datetime.fromisoformat(str(sample["timestamp"]).replace("Z", "+00:00"))
                    sample["timestamp_date"] = ts.strftime("%Y-%m-%d")
                    sample["timestamp_time"] = ts.strftime("%H:%M")
                except:
                    pass
            self.form_data = {
                **self.form_data,
                project_name: {
                    "transaction_id": safe_str(sample.get("transaction_id")),
                    "user_id": safe_str(sample.get("user_id")),
                    "timestamp_date": sample.get("timestamp_date", ""),
                    "timestamp_time": sample.get("timestamp_time", ""),
                    "amount": safe_float_str(sample.get("amount")),
                    "merchant_id": safe_str(sample.get("merchant_id")),
                    "merchant_category": safe_str(sample.get("merchant_category")),
                    "location": safe_str(sample.get("location")),
                    "transaction_type": safe_str(sample.get("transaction_type")),
                    "device_id": safe_str(sample.get("device_id")),
                    "ip_address": safe_str(sample.get("ip_address")),
                    "account_age_days": safe_int_str(sample.get("account_age_days")),
                    "is_international": str(sample.get("is_international", False)).lower(),
                }
            }

    # ==========================================================================
    # SQL EVENT HANDLERS
    # ==========================================================================
    @rx.event
    def update_sql_query(self, query: str):
        """Update SQL query input for current page."""
        project_name = self.page_name
        self.sql_query_input = {
            **self.sql_query_input,
            project_name: query
        }

    @rx.event
    def set_sql_query_from_template(self, template_query: str):
        """Set SQL query from a template."""
        project_name = self.page_name
        self.sql_query_input = {
            **self.sql_query_input,
            project_name: template_query
        }

    @rx.event
    def clear_sql_query(self):
        """Clear SQL query and results for current page."""
        project_name = self.page_name
        default_query = "SELECT * FROM data LIMIT 100"
        self.sql_query_input = {
            **self.sql_query_input,
            project_name: default_query
        }
        self.sql_query_results = {
            **self.sql_query_results,
            project_name: {"columns": [], "data": [], "row_count": 0}
        }
        self.sql_error = {
            **self.sql_error,
            project_name: ""
        }
        self.sql_execution_time = {
            **self.sql_execution_time,
            project_name: 0.0
        }

    @rx.event
    def set_sql_engine(self, engine: str):
        """Set the SQL engine for current page (polars or duckdb)."""
        project_name = self.page_name
        self.sql_engine = {
            **self.sql_engine,
            project_name: engine
        }

    @rx.event
    def set_sql_search_filter(self, search: str):
        """Set the search filter for SQL results."""
        project_name = self.page_name
        self.sql_search_filter = {
            **self.sql_search_filter,
            project_name: search
        }

    @rx.event
    def toggle_sql_sort(self, column: str):
        """Toggle sorting for a column."""
        project_name = self.page_name
        current_col = self.sql_sort_column.get(project_name, "")
        current_dir = self.sql_sort_direction.get(project_name, "asc")

        if current_col != column:
            new_col = column
            new_dir = "asc"
        elif current_dir == "asc":
            new_col = column
            new_dir = "desc"
        else:
            new_col = ""
            new_dir = "asc"

        self.sql_sort_column = {
            **self.sql_sort_column,
            project_name: new_col
        }
        self.sql_sort_direction = {
            **self.sql_sort_direction,
            project_name: new_dir
        }

    @rx.event(background=True)
    async def execute_sql_query(self):
        """Execute SQL query against Delta Lake via River service."""
        async with self:
            project_name = self.page_name
            query = self.sql_query_input.get(project_name, "")
            engine = self.sql_engine.get(project_name, "polars")

        if not query.strip():
            yield rx.toast.warning(
                "Empty query",
                description="Please enter a SQL query to execute",
                duration=3000
            )
            return

        async with self:
            self.sql_loading[project_name] = True
            self.sql_error[project_name] = ""

        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/sql_query",
                json={
                    "project_name": project_name,
                    "query": query,
                    "limit": 100,
                    "engine": engine
                },
                timeout=65.0
            )
            result = response.json()

            async with self:
                self.sql_query_results[project_name] = {
                    "columns": result.get("columns", []),
                    "data": result.get("data", []),
                    "row_count": result.get("row_count", 0)
                }
                self.sql_execution_time[project_name] = result.get("execution_time_ms", 0.0)
                self.sql_loading[project_name] = False

            row_count = result.get("row_count", 0)
            exec_time = result.get("execution_time_ms", 0.0)
            engine_used = result.get("engine", engine).upper()
            yield rx.toast.success(
                f"Query executed ({engine_used})",
                description=f"{row_count} rows in {exec_time:.0f}ms",
                duration=3000
            )

        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'response'):
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("detail", str(e))
                except Exception:
                    pass

            async with self:
                self.sql_loading[project_name] = False
                self.sql_error[project_name] = error_msg
                self.sql_query_results[project_name] = {"columns": [], "data": [], "row_count": 0}

            yield rx.toast.error(
                "Query failed",
                description=error_msg[:100] if len(error_msg) > 100 else error_msg,
                duration=5000
            )

    @rx.event(background=True)
    async def fetch_table_schema(self):
        """Fetch table schema and metadata for current page."""
        project_name = self.page_name

        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/table_schema",
                json={"project_name": project_name},
                timeout=30.0
            )
            result = response.json()

            async with self:
                self.sql_table_metadata = {
                    **self.sql_table_metadata,
                    project_name: result
                }

        except Exception as e:
            print(f"Error fetching table schema for {project_name}: {e}")

    # ==========================================================================
    # BATCH ML EVENT HANDLERS (Scikit-Learn)
    # ==========================================================================
    @rx.event
    def set_batch_training_percentage(self, project_name: str, value: str):
        """Set the training data percentage for a project (1-100)."""
        try:
            percentage = int(value)
            if 1 <= percentage <= 100:
                self.batch_training_data_percentage[project_name] = percentage
        except (ValueError, TypeError):
            pass  # Ignore invalid values

    @rx.event(background=True)
    async def train_batch_model(self, model_key: str, project_name: str):
        """Train a batch ML model using Scikit-Learn service via subprocess.

        Uses /switch_model to start training script, then polls /batch_status
        until training completes. Updates live status for UI display.
        """
        # Get the training script for this project
        training_script = self._batch_training_scripts.get(
            project_name, "transaction_fraud_detection_sklearn.py"
        )
        # Get the training data percentage (convert to fraction for --sample-frac)
        data_percentage = self.batch_training_data_percentage.get(project_name, 100)

        async with self:
            self.batch_training_loading[project_name] = True
            # Reset live status
            self.batch_training_status[project_name] = "Starting training..."
            self.batch_training_progress[project_name] = 0
            self.batch_training_stage[project_name] = "init"
            self.batch_training_metrics_preview[project_name] = {}
            self.batch_training_catboost_log[project_name] = {}
            self.batch_training_total_rows[project_name] = 0

        # Show toast with percentage info
        pct_info = f" using {data_percentage}% of data" if data_percentage < 100 else ""
        yield rx.toast.info(
            "Batch ML training started",
            description=f"Training {self.batch_ml_model_name.get(project_name, 'model')}{pct_info}...",
            duration=5000,
        )

        try:
            # Start training via /switch_model endpoint with data percentage
            payload = {"model_key": training_script}
            if data_percentage < 100:
                payload["sample_frac"] = data_percentage / 100.0  # Convert to 0.0-1.0
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/switch_model",
                json=payload,
                timeout=30.0
            )
            result = response.json()

            if result.get("status") == "error":
                raise Exception(result.get("message", "Failed to start training"))

            # Poll /batch_status until training completes (faster interval for live updates)
            max_polls = 300  # 10 minutes max (2 second intervals)
            poll_interval = 2.0

            for _ in range(max_polls):
                await asyncio.sleep(poll_interval)

                status_response = await httpx_client_get(
                    url=f"{SKLEARN_BASE_URL}/batch_status",
                    timeout=10.0
                )
                status = status_response.json()

                # Update live status from response
                async with self:
                    if status.get("status_message"):
                        self.batch_training_status[project_name] = status.get("status_message", "")
                    if status.get("progress_percent") is not None:
                        self.batch_training_progress[project_name] = status.get("progress_percent", 0)
                    if status.get("current_stage"):
                        self.batch_training_stage[project_name] = status.get("current_stage", "")
                    if status.get("metrics_preview"):
                        self.batch_training_metrics_preview[project_name] = status.get("metrics_preview", {})
                    # CatBoost training log (parsed dict)
                    self.batch_training_catboost_log[project_name] = status.get("catboost_log", {})
                    # Total rows (set once after data loading, persists through training)
                    if status.get("total_rows") and status.get("total_rows") > 0:
                        self.batch_training_total_rows[project_name] = status.get("total_rows", 0)

                if status.get("status") == "completed":
                    # Training completed successfully
                    async with self:
                        self.batch_training_loading[project_name] = False
                        self.batch_model_available[project_name] = True
                        self.batch_training_status[project_name] = "Training complete!"
                        self.batch_training_progress[project_name] = 100
                        self.batch_training_stage[project_name] = "complete"

                    yield rx.toast.success(
                        "Batch ML training complete",
                        description=f"Model trained successfully for {project_name}",
                        duration=5000,
                        close_button=True,
                    )

                    # Refresh MLflow runs dropdown, check model availability, fetch metrics
                    yield SharedState.fetch_mlflow_runs(project_name)
                    yield SharedState.check_batch_model_available(project_name)
                    yield SharedState.get_batch_mlflow_metrics(project_name)
                    return

                elif status.get("status") == "failed":
                    error_msg = status.get("error", "Training failed")
                    async with self:
                        self.batch_training_status[project_name] = f"Failed: {error_msg}"
                        self.batch_training_stage[project_name] = "error"
                    raise Exception(error_msg)

                elif status.get("status") != "running":
                    # Unknown status or idle - training may have finished quickly
                    break

            # If we exit the loop, check if model is available
            async with self:
                self.batch_training_loading[project_name] = False
                self.batch_training_status[project_name] = ""

            yield SharedState.fetch_mlflow_runs(project_name)
            yield SharedState.check_batch_model_available(project_name)
            yield SharedState.get_batch_mlflow_metrics(project_name)

        except Exception as e:
            error_msg = str(e)
            print(f"Error training batch model: {error_msg}")
            async with self:
                self.batch_training_loading[project_name] = False
                self.batch_training_status[project_name] = f"Error: {error_msg}"
                self.batch_training_stage[project_name] = "error"

            yield rx.toast.error(
                "Batch ML training failed",
                description=error_msg[:100] if len(error_msg) > 100 else error_msg,
                duration=8000,
                close_button=True,
            )

    @rx.event(background=True)
    async def stop_batch_training(self, project_name: str):
        """Stop the current batch training process and reset state."""
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/stop_training",
                json={},
                timeout=30.0
            )
            result = response.json()

            async with self:
                # Reset all training state to initial values
                self.batch_training_loading[project_name] = False
                self.batch_training_status[project_name] = ""
                self.batch_training_progress[project_name] = 0
                self.batch_training_stage[project_name] = ""
                self.batch_training_metrics_preview[project_name] = {}
                self.batch_training_catboost_log[project_name] = {}
                self.batch_training_total_rows[project_name] = 0

            if result.get("status") == "stopped":
                yield rx.toast.info(
                    "Batch ML training stopped",
                    description=f"Stopped training for {project_name}",
                    duration=3000,
                )
            else:
                yield rx.toast.info(
                    "No training running",
                    description="No batch training was active.",
                    duration=3000,
                )

        except Exception as e:
            print(f"Error stopping batch training: {e}")
            yield rx.toast.error(
                "Error stopping training",
                description=str(e)[:100],
                duration=5000,
            )

    @rx.event(background=True)
    async def check_batch_model_available(self, project_name: str):
        """Check if a trained batch (Scikit-Learn) model is available in MLflow."""
        model_name = self._batch_model_names.get(project_name, "XGBClassifier")
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/model_available",
                json={
                    "project_name": project_name,
                    "model_name": model_name
                },
                timeout=30.0
            )
            result = response.json()

            async with self:
                self.batch_model_available[project_name] = result.get("available", False)
                # Store experiment URL (available even if no model trained yet)
                experiment_url = result.get("experiment_url", "")
                if experiment_url:
                    self.batch_mlflow_experiment_url[project_name] = experiment_url

        except Exception as e:
            print(f"Error checking batch model availability: {e}")
            async with self:
                self.batch_model_available[project_name] = False

    @rx.event(background=True)
    async def get_batch_mlflow_metrics(self, project_name: str):
        """Fetch MLflow metrics for selected run (no toast - for background fetches)."""
        model_name = self._batch_model_names.get(project_name, "CatBoostClassifier")
        # Use selected run_id if set, otherwise endpoint uses best
        run_id = self.selected_batch_run.get(project_name) or None
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": model_name,
                    "run_id": run_id,
                },
                timeout=60.0
            )
            async with self:
                metrics_data = response.json()
                # Check if no runs exist (e.g., all runs deleted)
                if metrics_data.get("_no_runs"):
                    print(f"[DEBUG] get_batch_mlflow_metrics: No runs found for {project_name}")
                    self.batch_mlflow_metrics[project_name] = {}
                    self.batch_training_total_rows[project_name] = 0
                    return
                self.batch_mlflow_metrics[project_name] = metrics_data
                # Set total_rows from MLflow params (persists across page refresh)
                train_samples = metrics_data.get("params.train_samples")
                test_samples = metrics_data.get("params.test_samples")
                print(f"[DEBUG] get_batch_mlflow_metrics: run_id={run_id}, train_samples={train_samples}, test_samples={test_samples}")
                if train_samples and test_samples:
                    total = int(train_samples) + int(test_samples)
                    print(f"[DEBUG] Setting batch_training_total_rows[{project_name}] = {total}")
                    self.batch_training_total_rows[project_name] = total
                # Update MLflow URL to link directly to the selected run
                run_url = metrics_data.get("run_url")
                if run_url:
                    self.batch_mlflow_experiment_url[project_name] = run_url

        except Exception as e:
            print(f"Error fetching batch MLflow metrics: {e}")
            async with self:
                self.batch_mlflow_metrics[project_name] = {}

    @rx.event(background=True)
    async def refresh_batch_mlflow_metrics(self, project_name: str):
        """Force refresh batch MLflow metrics with toast notification."""
        model_name = self._batch_model_names.get(project_name, "CatBoostClassifier")
        # Use selected run_id if set, otherwise endpoint uses best
        run_id = self.selected_batch_run.get(project_name) or None
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": model_name,
                    "run_id": run_id,
                    "force_refresh": True
                },
                timeout=60.0
            )
            async with self:
                metrics_data = response.json()
                # Check if no runs exist (e.g., all runs deleted)
                if metrics_data.get("_no_runs"):
                    self.batch_mlflow_metrics[project_name] = {}
                    self.batch_training_total_rows[project_name] = 0
                    yield rx.toast.info(
                        "No runs found",
                        description="No MLflow runs exist for this experiment. Train a model first.",
                        duration=3000
                    )
                    return
                self.batch_mlflow_metrics[project_name] = metrics_data
                # Set total_rows from MLflow params (persists across page refresh)
                train_samples = metrics_data.get("params.train_samples")
                test_samples = metrics_data.get("params.test_samples")
                if train_samples and test_samples:
                    self.batch_training_total_rows[project_name] = int(train_samples) + int(test_samples)
                # Update MLflow URL to link directly to the selected run
                run_url = metrics_data.get("run_url")
                if run_url:
                    self.batch_mlflow_experiment_url[project_name] = run_url

            yield rx.toast.success(
                "Batch metrics refreshed",
                description=f"Latest metrics loaded for {project_name}",
                duration=2000
            )
        except Exception as e:
            print(f"Error refreshing batch MLflow metrics: {e}")
            yield rx.toast.error(
                "Refresh failed",
                description=str(e),
                duration=3000
            )

    @rx.event(background=True)
    async def init_batch_page(self, project_name: str):
        """Initialize batch ML page - fetch runs list and metrics."""
        print(f"[DEBUG] init_batch_page called for: {project_name}")
        # Fetch list of available MLflow runs (ordered by criteria, best first)
        yield SharedState.fetch_mlflow_runs(project_name)
        # Check model availability
        yield SharedState.check_batch_model_available(project_name)
        # Fetch metrics for selected run (or best if none selected)
        yield SharedState.get_batch_mlflow_metrics(project_name)

    @rx.event(background=True)
    async def fetch_mlflow_runs(self, project_name: str):
        """Fetch list of all MLflow runs for a project (ordered by criteria, best first)."""
        async with self:
            self.batch_runs_loading[project_name] = True

        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/mlflow_runs",
                json={"project_name": project_name},
                timeout=30.0
            )
            runs_data = response.json()
            # Convert to Pydantic models for typed foreach in Reflex
            runs = [MLflowRunInfo(**run) for run in runs_data]
            async with self:
                self.batch_mlflow_runs[project_name] = runs
                self.batch_runs_loading[project_name] = False
                if runs:
                    # If no run selected and runs exist, select the best (first)
                    if not self.selected_batch_run[project_name]:
                        self.selected_batch_run[project_name] = runs[0].run_id
                else:
                    # No runs - clear selected run and metrics
                    self.selected_batch_run[project_name] = ""
                    self.batch_mlflow_metrics[project_name] = {}
                    self.batch_training_total_rows[project_name] = 0
                print(f"[DEBUG] Fetched {len(runs)} MLflow runs for {project_name}")

        except Exception as e:
            print(f"Error fetching MLflow runs: {e}")
            async with self:
                self.batch_mlflow_runs[project_name] = []
                self.selected_batch_run[project_name] = ""
                self.batch_mlflow_metrics[project_name] = {}
                self.batch_training_total_rows[project_name] = 0
                self.batch_runs_loading[project_name] = False

    @rx.event(background=True)
    async def select_batch_run(self, project_name: str, run_id: str):
        """Select a specific MLflow run and refresh metrics/visualizations."""
        async with self:
            self.selected_batch_run[project_name] = run_id
            # Clear current metrics while loading new ones
            self.batch_mlflow_metrics[project_name] = {}

        # Refresh metrics for the newly selected run
        yield SharedState.get_batch_mlflow_metrics(project_name)

        yield rx.toast.info(
            "Run selected",
            description=f"Switched to run {run_id[:8]}...",
            duration=2000
        )
