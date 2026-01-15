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
from ..utils import httpx_client_post, httpx_client_get

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
    async def update_sample(self, project_name: str):
        """Fetch initial sample from FastAPI (runs in background to avoid lock expiration)."""
        if project_name == "Home":
            async with self:
                self.incremental_ml_sample[project_name] = {}
            return
        try:
            sample = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/initial_sample",
                json={"project_name": project_name},
                timeout=30.0
            )
            sample_data = sample.json()
            async with self:
                self.incremental_ml_sample[project_name] = sample_data
            # Initialize form data with sample
            if project_name == "Transaction Fraud Detection":
                await self._init_tfd_form_internal(sample_data)
            elif project_name == "Estimated Time of Arrival":
                await self._init_eta_form_internal(sample_data)
            elif project_name == "E-Commerce Customer Interactions":
                await self._init_ecci_form_internal(sample_data)
        except Exception as e:
            print(f"Error fetching initial sample for {project_name}: {e}")
            async with self:
                self.incremental_ml_sample[project_name] = {}

    async def _init_tfd_form_internal(self, sample: dict):
        """Internal helper to initialize TFD form (called from background events)."""
        # Parse timestamp
        timestamp_str = sample.get("timestamp") or ""
        timestamp_date = ""
        timestamp_time = ""
        if timestamp_str:
            try:
                timestamp = dt.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                timestamp_date = timestamp.strftime("%Y-%m-%d")
                timestamp_time = timestamp.strftime("%H:%M")
            except:
                timestamp_date = dt.datetime.now().strftime("%Y-%m-%d")
                timestamp_time = dt.datetime.now().strftime("%H:%M")

        # Parse JSON string fields from Delta Lake
        location = parse_json_field(sample, "location")
        device_info = parse_json_field(sample, "device_info")

        form_data = {
            # Numeric fields - convert to string for proper display
            "amount": safe_float_str(sample.get("amount"), 0.0),
            "account_age_days": safe_int_str(sample.get("account_age_days"), 0),
            "lat": safe_float_str(location.get("lat"), 0.0),
            "lon": safe_float_str(location.get("lon"), 0.0),
            # Timestamp fields
            "timestamp_date": timestamp_date,
            "timestamp_time": timestamp_time,
            # String fields - use or "" to handle None
            "currency": get_str(sample, "currency"),
            "merchant_id": get_str(sample, "merchant_id"),
            "product_category": get_str(sample, "product_category"),
            "transaction_type": get_str(sample, "transaction_type"),
            "payment_method": get_str(sample, "payment_method"),
            "browser": device_info.get("browser") or "",
            "os": device_info.get("os") or "",
            # Boolean fields
            "cvv_provided": safe_bool(sample.get("cvv_provided"), False),
            "billing_address_match": safe_bool(sample.get("billing_address_match"), False),
            # ID fields
            "transaction_id": get_str(sample, "transaction_id"),
            "user_id": get_str(sample, "user_id"),
            "ip_address": get_str(sample, "ip_address"),
            "user_agent": get_str(sample, "user_agent"),
        }
        async with self:
            self.form_data["Transaction Fraud Detection"] = form_data

    async def _init_eta_form_internal(self, sample: dict):
        """Internal helper to initialize ETA form (called from background events)."""
        # Parse timestamp
        timestamp_str = sample.get("timestamp") or ""
        timestamp_date = ""
        timestamp_time = ""
        if timestamp_str:
            try:
                timestamp = dt.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                timestamp_date = timestamp.strftime("%Y-%m-%d")
                timestamp_time = timestamp.strftime("%H:%M")
            except:
                timestamp_date = dt.datetime.now().strftime("%Y-%m-%d")
                timestamp_time = dt.datetime.now().strftime("%H:%M")

        # Parse JSON string fields from Delta Lake
        origin = parse_json_field(sample, "origin")
        destination = parse_json_field(sample, "destination")

        form_data = {
            # Coordinate fields
            "origin_lat": safe_float_str(origin.get("lat"), 0.0),
            "origin_lon": safe_float_str(origin.get("lon"), 0.0),
            "destination_lat": safe_float_str(destination.get("lat"), 0.0),
            "destination_lon": safe_float_str(destination.get("lon"), 0.0),
            # Numeric fields
            "distance_km": safe_float_str(sample.get("distance_km"), 0.0),
            "hour_of_day": safe_int_str(sample.get("hour_of_day"), 0),
            "day_of_week": safe_int_str(sample.get("day_of_week"), 0),
            "is_rush_hour": safe_bool(sample.get("is_rush_hour"), False),
            "is_weekend": safe_bool(sample.get("is_weekend"), False),
            # Timestamp fields
            "timestamp_date": timestamp_date,
            "timestamp_time": timestamp_time,
            # String fields
            "driver_id": get_str(sample, "driver_id"),
            "vehicle_id": get_str(sample, "vehicle_id"),
            "weather": get_str(sample, "weather"),
            "vehicle_type": get_str(sample, "vehicle_type"),
            "trip_id": get_str(sample, "trip_id"),
        }
        async with self:
            self.form_data["Estimated Time of Arrival"] = form_data

    async def _init_ecci_form_internal(self, sample: dict):
        """Internal helper to initialize ECCI form (called from background events)."""
        # Parse timestamp
        timestamp_str = sample.get("timestamp") or ""
        timestamp_date = ""
        timestamp_time = ""
        if timestamp_str:
            try:
                timestamp = dt.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                timestamp_date = timestamp.strftime("%Y-%m-%d")
                timestamp_time = timestamp.strftime("%H:%M")
            except:
                timestamp_date = dt.datetime.now().strftime("%Y-%m-%d")
                timestamp_time = dt.datetime.now().strftime("%H:%M")

        # Parse JSON string fields from Delta Lake
        location = parse_json_field(sample, "location")

        form_data = {
            # Numeric fields
            "session_duration_sec": safe_int_str(sample.get("session_duration_sec"), 0),
            "pages_viewed": safe_int_str(sample.get("pages_viewed"), 0),
            "total_clicks": safe_int_str(sample.get("total_clicks"), 0),
            "add_to_cart_count": safe_int_str(sample.get("add_to_cart_count"), 0),
            "checkout_initiated": safe_bool(sample.get("checkout_initiated"), False),
            "purchase_completed": safe_bool(sample.get("purchase_completed"), False),
            "cart_abandonment": safe_bool(sample.get("cart_abandonment"), False),
            "total_spent": safe_float_str(sample.get("total_spent"), 0.0),
            # Coordinate fields
            "lat": safe_float_str(location.get("lat"), 0.0),
            "lon": safe_float_str(location.get("lon"), 0.0),
            # Timestamp fields
            "timestamp_date": timestamp_date,
            "timestamp_time": timestamp_time,
            # String fields
            "customer_id": get_str(sample, "customer_id"),
            "session_id": get_str(sample, "session_id"),
            "device_type": get_str(sample, "device_type"),
            "traffic_source": get_str(sample, "traffic_source"),
            "product_category_viewed": get_str(sample, "product_category_viewed"),
            "customer_segment": get_str(sample, "customer_segment"),
        }
        async with self:
            self.form_data["E-Commerce Customer Interactions"] = form_data

    @rx.event(background=True)
    async def init_page(self, model_key: str, project_name: str):
        """Combined page initialization - replaces multiple HTTP calls with one.

        This single endpoint replaces:
        - set_current_page_model
        - update_sample
        - check_incremental_model_available
        - get_mlflow_metrics (partial)
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

                # Initial sample for form fields
                sample = data.get("initial_sample", {})
                if sample:
                    self.incremental_ml_sample[project_name] = sample
                    # Update form data with sample values
                    self._update_form_from_sample(project_name, sample)

                # Dropdown options (can update pre-loaded options with fresh data)
                options = data.get("dropdown_options", {})
                if options:
                    self.dropdown_options[project_name] = options

                # Training status
                training = data.get("training_status", {})
                if training.get("is_training"):
                    self.activated_model = model_key
                    self.ml_training_enabled = True

        except Exception as e:
            print(f"Error in init_page for {project_name}: {e}")
            async with self:
                self.incremental_model_available[project_name] = False

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
