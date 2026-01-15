import reflex as rx
import os
import asyncio
import datetime as dt
import json
import plotly.graph_objects as go
import folium
import orjson
from pathlib import Path
from .utils import httpx_client_post, httpx_client_get

RIVER_HOST = os.getenv("RIVER_HOST", "localhost")
RIVER_BASE_URL = f"http://{RIVER_HOST}:8002"
SKLEARN_HOST = os.getenv("SKLEARN_HOST", "localhost")
SKLEARN_BASE_URL = f"http://{SKLEARN_HOST}:8003"

# =============================================================================
# Metric Info Loader (LaTeX formulas and contextual explanations)
# =============================================================================
def load_metric_info(project_key: str) -> dict:
    """Load metric info JSON for a project using orjson."""
    data_dir = Path(__file__).parent / "data"
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


class State(rx.State):
    tab_name: str = "incremental_ml"
    page_name_mapping: dict = {
        "/": "Home",
        "/transaction-fraud-detection": "Transaction Fraud Detection",
        "/estimated-time-of-arrival": "Estimated Time of Arrival",
        "/e-commerce-customer-interactions": "E-Commerce Customer Interactions"
    }

    # ==========================================================================
    # BATCH ML STATE
    # ==========================================================================
    batch_ml_model_name: dict = {
        "Transaction Fraud Detection": "XGBoost Classifier (Scikit-Learn)",
    }
    # YellowBrick visualization state
    yellowbrick_metric_type: str = "Classification"
    yellowbrick_metric_name: str = ""
    yellowbrick_image_base64: str = ""
    yellowbrick_loading: bool = False
    yellowbrick_error: str = ""
    # Detailed metrics options for YellowBrick
    yellowbrick_metrics_options: dict = {
        "Classification": [
            "",
            "ClassificationReport",
            "ConfusionMatrix",
            "ROCAUC",
            "PrecisionRecallCurve",
            "ClassPredictionError"
        ],
        "Feature Analysis": [
            "",
        ],
        "Target": [
            "",
            "BalancedBinningReference",
            "ClassBalance"
        ],
        "Model Selection": [
            "",
            "ValidationCurve",
            "LearningCurve",
            "CVScores",
            "FeatureImportances",
            "DroppingCurve"
        ]
    }
    # Batch ML prediction results (separate from incremental ML)
    batch_prediction_results: dict = {
        "Transaction Fraud Detection": {},
    }
    # Batch ML MLflow metrics (separate from incremental ML)
    batch_mlflow_metrics: dict = {
        "Transaction Fraud Detection": {},
    }
    # Batch ML training state
    batch_training_loading: bool = False
    batch_model_available: dict = {
        "Transaction Fraud Detection": False,
    }
    batch_training_error: str = ""
    batch_last_trained: dict = {
        "Transaction Fraud Detection": "",
    }
    batch_training_metrics: dict = {
        "Transaction Fraud Detection": {},
    }
    # Batch ML toggle state (like incremental ML)
    batch_ml_state: dict = {
        "Transaction Fraud Detection": False,
    }
    batch_ml_model_key: dict = {
        "Transaction Fraud Detection": "transaction_fraud_detection_sklearn.py",
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
    form_data: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    dropdown_options: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
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

    # ==========================================================================
    # DELTA LAKE SQL TAB STATE
    # ==========================================================================
    # SQL query input per project (default queries)
    sql_query_input: dict = {
        "Transaction Fraud Detection": "SELECT * FROM data ORDER BY timestamp DESC LIMIT 100",
        "Estimated Time of Arrival": "SELECT * FROM data ORDER BY timestamp DESC LIMIT 100",
        "E-Commerce Customer Interactions": "SELECT * FROM data ORDER BY timestamp DESC LIMIT 100"
    }
    # SQL query results per project
    sql_query_results: dict = {
        "Transaction Fraud Detection": {"columns": [], "data": [], "row_count": 0},
        "Estimated Time of Arrival": {"columns": [], "data": [], "row_count": 0},
        "E-Commerce Customer Interactions": {"columns": [], "data": [], "row_count": 0}
    }
    # SQL execution state
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
    # SQL engine selection per project ("polars" or "duckdb")
    sql_engine: dict = {
        "Transaction Fraud Detection": "polars",
        "Estimated Time of Arrival": "polars",
        "E-Commerce Customer Interactions": "polars"
    }
    # SQL results search filter per project
    sql_search_filter: dict = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }
    # SQL results sort column per project
    sql_sort_column: dict = {
        "Transaction Fraud Detection": "",
        "Estimated Time of Arrival": "",
        "E-Commerce Customer Interactions": ""
    }
    # SQL results sort direction per project ("asc" or "desc")
    sql_sort_direction: dict = {
        "Transaction Fraud Detection": "asc",
        "Estimated Time of Arrival": "asc",
        "E-Commerce Customer Interactions": "asc"
    }
    # Table metadata cache (schema info)
    sql_table_metadata: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    # Query templates per project
    sql_query_templates: dict = {
        "Transaction Fraud Detection": [
            {"name": "Recent Transactions", "query": "SELECT * FROM data ORDER BY timestamp DESC LIMIT 100"},
            {"name": "Fraud Cases", "query": "SELECT * FROM data WHERE is_fraud = 1 LIMIT 100"},
            {"name": "High Value Transactions", "query": "SELECT * FROM data WHERE amount > 1000 ORDER BY amount DESC LIMIT 100"},
            {"name": "Fraud by Merchant", "query": "SELECT merchant_id, COUNT(*) as fraud_count FROM data WHERE is_fraud = 1 GROUP BY merchant_id ORDER BY fraud_count DESC LIMIT 20"},
            {"name": "Transaction Types", "query": "SELECT transaction_type, COUNT(*) as count, AVG(amount) as avg_amount FROM data GROUP BY transaction_type ORDER BY count DESC"},
        ],
        "Estimated Time of Arrival": [
            {"name": "Recent Trips", "query": "SELECT * FROM data ORDER BY timestamp DESC LIMIT 100"},
            {"name": "Long Trips", "query": "SELECT * FROM data WHERE estimated_distance_km > 50 ORDER BY estimated_distance_km DESC LIMIT 100"},
            {"name": "Weather Impact", "query": "SELECT weather, COUNT(*) as trips, AVG(simulated_actual_travel_time_seconds) as avg_time FROM data GROUP BY weather ORDER BY trips DESC"},
            {"name": "Driver Performance", "query": "SELECT driver_id, COUNT(*) as trips, AVG(driver_rating) as avg_rating FROM data GROUP BY driver_id ORDER BY trips DESC LIMIT 20"},
            {"name": "Vehicle Types", "query": "SELECT vehicle_type, COUNT(*) as count, AVG(estimated_distance_km) as avg_distance FROM data GROUP BY vehicle_type ORDER BY count DESC"},
        ],
        "E-Commerce Customer Interactions": [
            {"name": "Recent Events", "query": "SELECT * FROM data ORDER BY timestamp DESC LIMIT 100"},
            {"name": "Purchases", "query": "SELECT * FROM data WHERE event_type = 'purchase' LIMIT 100"},
            {"name": "By Category", "query": "SELECT product_category, COUNT(*) as count FROM data GROUP BY product_category ORDER BY count DESC"},
            {"name": "Event Types", "query": "SELECT event_type, COUNT(*) as count FROM data GROUP BY event_type ORDER BY count DESC"},
            {"name": "Top Products", "query": "SELECT product_id, COUNT(*) as views, SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases FROM data GROUP BY product_id ORDER BY views DESC LIMIT 20"},
        ]
    }

    ##==========================================================================
    ## VARS
    ##==========================================================================
    @rx.var
    def page_name(self) -> str:
        """Get the current page name based on the route."""
        current_path = self.router.url.path
        return self.page_name_mapping.get(current_path, "Home")

    @rx.var
    def tfd_form_data(self) -> dict:
        """Get Transaction Fraud Detection form data."""
        return self.form_data.get("Transaction Fraud Detection", {})

    # Transaction Fraud Detection dropdown options (consolidated)
    @rx.var
    def tfd_options(self) -> dict[str, list[str]]:
        """Get all TFD dropdown options as a dict."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        return opts if isinstance(opts, dict) else {}

    @rx.var
    def tfd_prediction_show(self) -> bool:
        """Check if prediction results should be shown."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return results.get("show", False)
        return False

    @rx.var
    def tfd_fraud_probability(self) -> float:
        """Get fraud probability."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return results.get("fraud_probability", 0.0)
        return 0.0

    @rx.var
    def tfd_prediction_text(self) -> str:
        """Get prediction result text."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return "Fraud" if results.get("prediction", 0) == 1 else "Not Fraud"
        return ""

    @rx.var
    def tfd_prediction_color(self) -> str:
        """Get prediction result color."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return "red" if results.get("prediction", 0) == 1 else "green"
        return "gray"

    @rx.var(cache=True)
    def tfd_fraud_gauge(self) -> go.Figure:
        """Generate Plotly gauge chart for fraud probability."""
        prob = self.tfd_fraud_probability * 100

        # Determine risk level and colors
        if prob < 30:
            risk_text = "LOW RISK"
            bar_color = "#22c55e"  # green
        elif prob < 70:
            risk_text = "MEDIUM RISK"
            bar_color = "#eab308"  # yellow
        else:
            risk_text = "HIGH RISK"
            bar_color = "#ef4444"  # red

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={
                'suffix': "%",
                'font': {'size': 40, 'color': bar_color}
            },
            title={
                'text': f"<b>{risk_text}</b><br><span style='font-size:14px;color:gray'>Fraud Probability</span>",
                'font': {'size': 18}
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "#666",
                    'tickvals': [0, 25, 50, 75, 100],
                    'ticktext': ['0%', '25%', '50%', '75%', '100%']
                },
                'bar': {'color': bar_color, 'thickness': 0.75},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.3)'},    # green zone
                    {'range': [30, 70], 'color': 'rgba(234, 179, 8, 0.3)'},   # yellow zone
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}   # red zone
                ],
                'threshold': {
                    'line': {'color': "#333", 'width': 4},
                    'thickness': 0.8,
                    'value': prob
                }
            }
        ))

        fig.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=80, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#888'}
        )

        return fig

    # =========================================================================
    # MLflow Run Info (shared across all pages)
    # =========================================================================
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

    # Transaction Fraud Detection metrics (consolidated)
    # All 13 metrics from River ML training:
    # Class-based (10): Recall, Precision, F1, FBeta, Accuracy, BalancedAccuracy, MCC, GeometricMean, CohenKappa, Jaccard
    # Proba-based (3): ROCAUC, RollingROCAUC, LogLoss
    @rx.var
    def tfd_metrics(self) -> dict[str, str]:
        """Get all TFD metrics as formatted percentage strings."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if not isinstance(metrics, dict):
            return {
                # Primary metrics
                "fbeta": "0.00%", "rocauc": "0.00%", "precision": "0.00%", "recall": "0.00%",
                # Secondary metrics
                "mcc": "0.00", "balanced_accuracy": "0.00%",
                # Additional metrics
                "f1": "0.00%", "accuracy": "0.00%", "geometric_mean": "0.00%",
                "cohen_kappa": "0.00", "jaccard": "0.00%",
                "rolling_rocauc": "0.00%", "logloss": "0.000"
            }
        return {
            # Primary metrics (KPI indicators)
            "fbeta": f"{(metrics.get('metrics.FBeta') or 0) * 100:.2f}%",
            "rocauc": f"{(metrics.get('metrics.ROCAUC') or 0) * 100:.2f}%",
            "precision": f"{(metrics.get('metrics.Precision') or 0) * 100:.2f}%",
            "recall": f"{(metrics.get('metrics.Recall') or 0) * 100:.2f}%",
            # Secondary metrics (gauges) - MCC/CohenKappa range is -1 to 1, not percentage
            "mcc": f"{(metrics.get('metrics.MCC') or 0):.3f}",
            "balanced_accuracy": f"{(metrics.get('metrics.BalancedAccuracy') or 0) * 100:.2f}%",
            # Additional metrics (grid)
            "f1": f"{(metrics.get('metrics.F1') or 0) * 100:.2f}%",
            "accuracy": f"{(metrics.get('metrics.Accuracy') or 0) * 100:.2f}%",
            "geometric_mean": f"{(metrics.get('metrics.GeometricMean') or 0) * 100:.2f}%",
            "cohen_kappa": f"{(metrics.get('metrics.CohenKappa') or 0):.3f}",
            "jaccard": f"{(metrics.get('metrics.Jaccard') or 0) * 100:.2f}%",
            "rolling_rocauc": f"{(metrics.get('metrics.RollingROCAUC') or 0) * 100:.2f}%",
            "logloss": f"{(metrics.get('metrics.LogLoss') or 0):.4f}",
        }

    @rx.var
    def tfd_metrics_raw(self) -> dict[str, float]:
        """Get all TFD metrics as raw float values for Plotly charts."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if not isinstance(metrics, dict):
            return {
                "fbeta": 0.0, "rocauc": 0.0, "precision": 0.0, "recall": 0.0,
                "mcc": 0.0, "balanced_accuracy": 0.0, "f1": 0.0, "accuracy": 0.0,
                "geometric_mean": 0.0, "cohen_kappa": 0.0, "jaccard": 0.0,
                "rolling_rocauc": 0.0, "logloss": 0.0
            }
        return {
            "fbeta": float(metrics.get('metrics.FBeta') or 0),
            "rocauc": float(metrics.get('metrics.ROCAUC') or 0),
            "precision": float(metrics.get('metrics.Precision') or 0),
            "recall": float(metrics.get('metrics.Recall') or 0),
            "mcc": float(metrics.get('metrics.MCC') or 0),
            "balanced_accuracy": float(metrics.get('metrics.BalancedAccuracy') or 0),
            "f1": float(metrics.get('metrics.F1') or 0),
            "accuracy": float(metrics.get('metrics.Accuracy') or 0),
            "geometric_mean": float(metrics.get('metrics.GeometricMean') or 0),
            "cohen_kappa": float(metrics.get('metrics.CohenKappa') or 0),
            "jaccard": float(metrics.get('metrics.Jaccard') or 0),
            "rolling_rocauc": float(metrics.get('metrics.RollingROCAUC') or 0),
            "logloss": float(metrics.get('metrics.LogLoss') or 0),
        }

    @rx.var
    def tfd_dashboard_figures(self) -> dict:
        """Generate all TFD dashboard Plotly figures (KPI indicators, gauges, confusion matrix)."""
        raw = self.tfd_metrics_raw
        report_data = self.report_metrics.get("Transaction Fraud Detection", {})
        mlflow_data = self.mlflow_metrics.get("Transaction Fraud Detection", {})

        # Extract baseline metrics for delta calculation
        baseline = {
            "fbeta": mlflow_data.get("baseline_FBeta", 0),
            "rocauc": mlflow_data.get("baseline_ROCAUC", 0),
            "precision": mlflow_data.get("baseline_Precision", 0),
            "recall": mlflow_data.get("baseline_Recall", 0),
            "rolling_rocauc": mlflow_data.get("baseline_RollingROCAUC", 0),
        }

        def create_kpi(value: float, title: str, baseline_val: float = 0) -> go.Figure:
            """Create KPI indicator with percentage display and delta from baseline."""
            display_value = value * 100
            if value >= 0.85:
                color = "#3b82f6"  # blue - excellent
            elif value >= 0.70:
                color = "#22c55e"  # green - good
            elif value >= 0.50:
                color = "#eab308"  # yellow - fair
            else:
                color = "#ef4444"  # red - poor

            # Configure delta if baseline exists
            delta_config = None
            if baseline_val > 0:
                delta_config = {
                    "reference": baseline_val * 100,
                    "relative": True,
                    "valueformat": ".1%",
                    "increasing": {"color": "#22c55e"},
                    "decreasing": {"color": "#ef4444"}
                }

            fig = go.Figure(go.Indicator(
                mode="number+delta" if delta_config else "number",
                value=display_value,
                delta=delta_config,
                title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                number={"suffix": "%", "font": {"size": 28, "color": color}, "valueformat": ".1f"}
            ))
            fig.update_layout(
                height=110, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_gauge(value: float, title: str, min_val: float = 0, max_val: float = 1) -> go.Figure:
            """Create gauge with colored ranges."""
            if min_val == -1:  # MCC/CohenKappa range
                steps = [
                    {"range": [-1, 0], "color": "#ef4444"},
                    {"range": [0, 0.4], "color": "#eab308"},
                    {"range": [0.4, 0.6], "color": "#22c55e"},
                    {"range": [0.6, 1], "color": "#3b82f6"}
                ]
                threshold_val = 0.5
            else:  # 0-1 range
                steps = [
                    {"range": [0, 0.5], "color": "#ef4444"},
                    {"range": [0.5, 0.7], "color": "#eab308"},
                    {"range": [0.7, 0.85], "color": "#22c55e"},
                    {"range": [0.85, 1], "color": "#3b82f6"}
                ]
                threshold_val = 0.8

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                number={"valueformat": ".3f", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [min_val, max_val], "tickwidth": 1},
                    "bar": {"color": "#1e40af"},
                    "steps": steps,
                    "threshold": {"value": threshold_val, "line": {"color": "black", "width": 2}, "thickness": 0.75}
                }
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_confusion_matrix() -> go.Figure:
            """Create confusion matrix heatmap."""
            cm = report_data.get("confusion_matrix", {})
            if not cm.get("available", False):
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available yet",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )
                fig.update_layout(
                    title={"text": "<b>Confusion Matrix</b>", "font": {"size": 14}},
                    height=250, margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False), yaxis=dict(visible=False)
                )
                return fig

            tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
            z = [[tn, fp], [fn, tp]]
            text = [[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]]

            fig = go.Figure(go.Heatmap(
                z=z, x=["Pred: 0", "Pred: 1"], y=["Actual: 0", "Actual: 1"],
                colorscale="Blues", text=text, texttemplate="%{text}",
                textfont={"size": 12}, showscale=False
            ))
            fig.update_layout(
                title={"text": "<b>Confusion Matrix</b>", "font": {"size": 14}},
                height=250, margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis={"autorange": "reversed"}
            )
            return fig

        def create_classification_report() -> go.Figure:
            """Create YellowBrick-style classification report heatmap."""
            cm = report_data.get("confusion_matrix", {})
            if not cm.get("available", False):
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available yet",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )
                fig.update_layout(
                    title={"text": "<b>Classification Report</b>", "font": {"size": 14}},
                    height=250, margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False), yaxis=dict(visible=False)
                )
                return fig

            tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

            # Calculate per-class metrics
            # Class 0 (Not Fraud): TN=correct, FN=missed (predicted 1 when actual 0)
            prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
            rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
            support_0 = tn + fp

            # Class 1 (Fraud): TP=correct, FP=false alarm
            prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
            support_1 = tp + fn

            # Build heatmap data (rows=classes, cols=metrics)
            z = [[prec_0, rec_0, f1_0], [prec_1, rec_1, f1_1]]
            text = [[f"{v:.2f}" for v in row] for row in z]

            fig = go.Figure(go.Heatmap(
                z=z,
                x=["Precision", "Recall", "F1"],
                y=[f"0 (n={support_0:,})", f"1 (n={support_1:,})"],
                colorscale="YlOrRd",  # YellowBrick default
                text=text,
                texttemplate="%{text}",
                textfont={"size": 14, "color": "black"},
                showscale=True,
                zmin=0, zmax=1,
                colorbar={"len": 0.8, "thickness": 10}
            ))
            fig.update_layout(
                title={"text": "<b>Classification Report</b>", "font": {"size": 14}},
                height=250, margin=dict(l=20, r=80, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis={"autorange": "reversed"}
            )
            return fig

        return {
            # ROW 1: KPI Indicators (primary metrics with delta from baseline)
            "kpi_fbeta": create_kpi(raw["fbeta"], "FBeta (β=2)", baseline["fbeta"]),
            "kpi_rocauc": create_kpi(raw["rocauc"], "ROC AUC", baseline["rocauc"]),
            "kpi_precision": create_kpi(raw["precision"], "Precision", baseline["precision"]),
            "kpi_recall": create_kpi(raw["recall"], "Recall", baseline["recall"]),
            "kpi_rolling_rocauc": create_kpi(raw["rolling_rocauc"], "Rolling AUC", baseline["rolling_rocauc"]),
            # ROW 2: Gauges (secondary metrics)
            "gauge_mcc": create_gauge(raw["mcc"], "MCC", min_val=-1, max_val=1),
            "gauge_balanced_accuracy": create_gauge(raw["balanced_accuracy"], "Balanced Accuracy"),
            # Confusion Matrix + Classification Report
            "confusion_matrix": create_confusion_matrix(),
            "classification_report": create_classification_report(),
        }

    @rx.var
    def ml_training_switch_checked(self) -> bool:
        """Check if ML training is enabled for the current page's model."""
        return self.ml_training_enabled and self.activated_model == self._current_page_model_key

    # =========================================================================
    # Batch ML (Scikit-Learn) computed vars
    # =========================================================================
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
    def tfd_batch_ml_enabled(self) -> bool:
        """Check if TFD batch ML training toggle is enabled."""
        return self.batch_ml_state.get("Transaction Fraud Detection", False)

    @rx.var
    def tfd_batch_model_available(self) -> bool:
        """Check if TFD batch model is available for prediction."""
        return self.batch_model_available.get("Transaction Fraud Detection", False)

    @rx.var
    def tfd_batch_last_trained(self) -> str:
        """Get the last trained timestamp for TFD batch model."""
        return self.batch_last_trained.get("Transaction Fraud Detection", "")

    @rx.var
    def tfd_batch_training_metrics_display(self) -> dict[str, str]:
        """Get TFD batch training metrics for display."""
        metrics = self.batch_training_metrics.get("Transaction Fraud Detection", {})
        if not isinstance(metrics, dict) or not metrics:
            return {}
        return {
            "f1": f"{(metrics.get('F1') or 0) * 100:.2f}%",
            "accuracy": f"{(metrics.get('Accuracy') or 0) * 100:.2f}%",
            "recall": f"{(metrics.get('Recall') or 0) * 100:.2f}%",
            "precision": f"{(metrics.get('Precision') or 0) * 100:.2f}%",
            "rocauc": f"{(metrics.get('ROCAUC') or 0) * 100:.2f}%",
            "geometric_mean": f"{(metrics.get('GeometricMean') or 0) * 100:.2f}%",
        }

    @rx.var
    def yellowbrick_metric_options(self) -> list[str]:
        """Get available YellowBrick metric names for current metric type."""
        return self.yellowbrick_metrics_options.get(self.yellowbrick_metric_type, [""])

    @rx.var
    def yellowbrick_metric_types(self) -> list[str]:
        """Get available YellowBrick metric types."""
        return list(self.yellowbrick_metrics_options.keys())

    @rx.var
    def tfd_batch_prediction_show(self) -> bool:
        """Check if batch prediction results should be shown for TFD."""
        results = self.batch_prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return results.get("show", False)
        return False

    @rx.var
    def tfd_batch_fraud_probability(self) -> float:
        """Get batch fraud probability for TFD."""
        results = self.batch_prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return results.get("fraud_probability", 0.0)
        return 0.0

    @rx.var
    def tfd_batch_prediction_text(self) -> str:
        """Get batch prediction result text for TFD."""
        results = self.batch_prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return "Fraud" if results.get("prediction", 0) == 1 else "Not Fraud"
        return ""

    @rx.var
    def tfd_batch_prediction_color(self) -> str:
        """Get batch prediction result color for TFD."""
        results = self.batch_prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return "red" if results.get("prediction", 0) == 1 else "green"
        return "gray"

    @rx.var
    def tfd_batch_fraud_gauge(self) -> go.Figure:
        """Generate Plotly gauge chart for batch fraud probability."""
        prob = self.tfd_batch_fraud_probability * 100

        # Determine risk level and colors
        if prob < 30:
            risk_text = "LOW RISK"
            bar_color = "#22c55e"  # green
        elif prob < 70:
            risk_text = "MEDIUM RISK"
            bar_color = "#eab308"  # yellow
        else:
            risk_text = "HIGH RISK"
            bar_color = "#ef4444"  # red

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={
                'suffix': "%",
                'font': {'size': 40, 'color': bar_color}
            },
            title={
                'text': f"<b>{risk_text}</b><br><span style='font-size:14px;color:gray'>Fraud Probability</span>",
                'font': {'size': 18}
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "#666",
                    'tickvals': [0, 25, 50, 75, 100],
                    'ticktext': ['0%', '25%', '50%', '75%', '100%']
                },
                'bar': {'color': bar_color, 'thickness': 0.75},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.3)'},
                    {'range': [30, 70], 'color': 'rgba(234, 179, 8, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#333", 'width': 4},
                    'thickness': 0.8,
                    'value': prob
                }
            }
        ))

        fig.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=80, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#888'}
        )

        return fig

    @rx.var
    def tfd_batch_metrics(self) -> dict[str, str]:
        """Get all TFD batch ML metrics as formatted percentage strings."""
        metrics = self.batch_mlflow_metrics.get("Transaction Fraud Detection", {})
        if not isinstance(metrics, dict):
            return {}
        # Return all metrics dynamically
        result = {}
        for key, value in metrics.items():
            if key.startswith("metrics."):
                metric_name = key.replace("metrics.", "")
                # Format as percentage for classification metrics
                if isinstance(value, (int, float)):
                    result[metric_name] = f"{value * 100:.2f}%"
                else:
                    result[metric_name] = str(value)
        return result

    @rx.var
    def tfd_batch_metric_names(self) -> list[str]:
        """Get list of batch ML metric names for TFD."""
        return list(self.tfd_batch_metrics.keys())

    # =========================================================================
    # Estimated Time of Arrival (ETA) computed vars
    # =========================================================================
    @rx.var
    def eta_form_data(self) -> dict:
        """Get Estimated Time of Arrival form data."""
        return self.form_data.get("Estimated Time of Arrival", {})

    @rx.var
    def eta_options(self) -> dict[str, list[str]]:
        """Get all ETA dropdown options as a dict."""
        opts = self.dropdown_options.get("Estimated Time of Arrival", {})
        return opts if isinstance(opts, dict) else {}

    @rx.var
    def eta_prediction_show(self) -> bool:
        """Check if ETA prediction results should be shown."""
        results = self.prediction_results.get("Estimated Time of Arrival", {})
        if isinstance(results, dict):
            return results.get("show", False)
        return False

    @rx.var
    def eta_prediction_seconds(self) -> float:
        """Get ETA prediction in seconds."""
        results = self.prediction_results.get("Estimated Time of Arrival", {})
        if isinstance(results, dict):
            return results.get("eta_seconds", 0.0)
        return 0.0

    @rx.var
    def eta_prediction_minutes(self) -> float:
        """Get ETA prediction in minutes."""
        return round(self.eta_prediction_seconds / 60, 2) if self.eta_prediction_seconds > 0 else 0.0

    @rx.var
    def eta_metrics(self) -> dict[str, str]:
        """Get all ETA regression metrics as formatted strings."""
        metrics = self.mlflow_metrics.get("Estimated Time of Arrival", {})
        if not isinstance(metrics, dict):
            return {
                "mae": "0.00", "mape": "0.00", "mse": "0.00", "r2": "0.00",
                "rmse": "0.00", "rmsle": "0.00", "smape": "0.00",
                "rolling_mae": "0.00", "rolling_rmse": "0.00", "time_rolling_mae": "0.00"
            }
        return {
            "mae": f"{metrics.get('metrics.MAE', 0):.2f}",
            "mape": f"{metrics.get('metrics.MAPE', 0):.2f}",
            "mse": f"{metrics.get('metrics.MSE', 0):.2f}",
            "r2": f"{metrics.get('metrics.R2', 0):.2f}",
            "rmse": f"{metrics.get('metrics.RMSE', 0):.2f}",
            "rmsle": f"{metrics.get('metrics.RMSLE', 0):.2f}",
            "smape": f"{metrics.get('metrics.SMAPE', 0):.2f}",
            "rolling_mae": f"{metrics.get('metrics.RollingMAE', 0):.2f}",
            "rolling_rmse": f"{metrics.get('metrics.RollingRMSE', 0):.2f}",
            "time_rolling_mae": f"{metrics.get('metrics.TimeRollingMAE', 0):.2f}",
        }

    @rx.var
    def eta_metrics_raw(self) -> dict[str, float]:
        """Get raw ETA metrics as floats for dashboard calculations."""
        metrics = self.mlflow_metrics.get("Estimated Time of Arrival", {})
        if not isinstance(metrics, dict):
            return {
                "mae": 0, "mape": 0, "mse": 0, "r2": 0, "rmse": 0, "rmsle": 0, "smape": 0,
                "rolling_mae": 0, "rolling_rmse": 0, "time_rolling_mae": 0
            }
        return {
            "mae": metrics.get("metrics.MAE", 0),
            "mape": metrics.get("metrics.MAPE", 0),
            "mse": metrics.get("metrics.MSE", 0),
            "r2": metrics.get("metrics.R2", 0),
            "rmse": metrics.get("metrics.RMSE", 0),
            "rmsle": metrics.get("metrics.RMSLE", 0),
            "smape": metrics.get("metrics.SMAPE", 0),
            "rolling_mae": metrics.get("metrics.RollingMAE", 0),
            "rolling_rmse": metrics.get("metrics.RollingRMSE", 0),
            "time_rolling_mae": metrics.get("metrics.TimeRollingMAE", 0),
        }

    @rx.var
    def eta_dashboard_figures(self) -> dict:
        """Generate all ETA dashboard Plotly figures (KPI indicators, gauges)."""
        raw = self.eta_metrics_raw
        mlflow_data = self.mlflow_metrics.get("Estimated Time of Arrival", {})

        # Extract baseline metrics for delta calculation
        baseline = {
            "mae": float(mlflow_data.get("baseline_MAE", 0) or 0),
            "rmse": float(mlflow_data.get("baseline_RMSE", 0) or 0),
            "r2": float(mlflow_data.get("baseline_R2", 0) or 0),
            "rolling_mae": float(mlflow_data.get("baseline_RollingMAE", 0) or 0),
        }

        def create_kpi_regression(value: float, title: str, unit: str = "s",
                                   baseline_val: float = 0, lower_is_better: bool = True) -> go.Figure:
            """Create KPI indicator for regression metrics with delta from baseline."""
            # Color coding based on metric type
            if lower_is_better:
                # For MAE, RMSE, MAPE - lower is better
                if value <= 30:
                    color = "#3b82f6"  # blue - excellent
                elif value <= 60:
                    color = "#22c55e"  # green - good
                elif value <= 120:
                    color = "#eab308"  # yellow - fair
                else:
                    color = "#ef4444"  # red - poor
            else:
                # For R² - higher is better
                if value >= 0.9:
                    color = "#3b82f6"  # blue - excellent
                elif value >= 0.7:
                    color = "#22c55e"  # green - good
                elif value >= 0.5:
                    color = "#eab308"  # yellow - fair
                else:
                    color = "#ef4444"  # red - poor

            # Configure delta if baseline exists
            delta_config = None
            if baseline_val > 0:
                delta_config = {
                    "reference": baseline_val,
                    "relative": True,
                    "valueformat": ".1%",
                    # For lower_is_better metrics, decreasing (negative delta) is good
                    "increasing": {"color": "#ef4444" if lower_is_better else "#22c55e"},
                    "decreasing": {"color": "#22c55e" if lower_is_better else "#ef4444"}
                }

            fig = go.Figure(go.Indicator(
                mode="number+delta" if delta_config else "number",
                value=value,
                delta=delta_config,
                title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                number={"suffix": unit, "font": {"size": 28, "color": color}, "valueformat": ".1f"}
            ))
            fig.update_layout(
                height=110, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_gauge_r2(value: float) -> go.Figure:
            """Create R² gauge (-1 to 1 scale, higher is better)."""
            steps = [
                {"range": [-1, 0], "color": "#ef4444"},    # red - negative R²
                {"range": [0, 0.5], "color": "#eab308"},   # yellow - poor fit
                {"range": [0.5, 0.7], "color": "#22c55e"}, # green - moderate fit
                {"range": [0.7, 1], "color": "#3b82f6"}    # blue - good fit
            ]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": "<b>R² (Goodness of Fit)</b>", "font": {"size": 14}},
                number={"valueformat": ".3f", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 1},
                    "bar": {"color": "#1e40af"},
                    "steps": steps,
                    "threshold": {"value": 0.7, "line": {"color": "black", "width": 2}, "thickness": 0.75}
                }
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_gauge_mape(value: float) -> go.Figure:
            """Create MAPE gauge (0 to 100% scale, lower is better)."""
            # Cap at 100 for display purposes
            display_value = min(value, 100)
            steps = [
                {"range": [0, 10], "color": "#3b82f6"},    # blue - excellent (<10%)
                {"range": [10, 25], "color": "#22c55e"},   # green - good (10-25%)
                {"range": [25, 50], "color": "#eab308"},   # yellow - fair (25-50%)
                {"range": [50, 100], "color": "#ef4444"}   # red - poor (>50%)
            ]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=display_value,
                title={"text": "<b>MAPE (% Error)</b>", "font": {"size": 14}},
                number={"suffix": "%", "valueformat": ".1f", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#1e40af"},
                    "steps": steps,
                    "threshold": {"value": 25, "line": {"color": "black", "width": 2}, "thickness": 0.75}
                }
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        return {
            # ROW 1: KPI Indicators (primary metrics)
            "kpi_mae": create_kpi_regression(raw["mae"], "MAE", "s", baseline["mae"], lower_is_better=True),
            "kpi_rmse": create_kpi_regression(raw["rmse"], "RMSE", "s", baseline["rmse"], lower_is_better=True),
            "kpi_r2": create_kpi_regression(raw["r2"], "R²", "", baseline["r2"], lower_is_better=False),
            "kpi_rolling_mae": create_kpi_regression(raw["rolling_mae"], "Rolling MAE", "s", baseline["rolling_mae"], lower_is_better=True),
            # ROW 3: Gauges
            "gauge_r2": create_gauge_r2(raw["r2"]),
            "gauge_mape": create_gauge_mape(raw["mape"]),
        }

    @rx.var
    def eta_prediction_figure(self) -> go.Figure:
        """Generate Plotly figure for ETA prediction display."""
        seconds = self.eta_prediction_seconds
        minutes = self.eta_prediction_minutes

        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=seconds,
                title={'text': "<b>Seconds</b>", 'font': {'size': 18}},
                number={'font': {'size': 48, 'color': '#3b82f6'}},
                domain={'row': 0, 'column': 0}
            )
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=minutes,
                title={'text': "<b>Minutes</b>", 'font': {'size': 18}},
                number={'font': {'size': 48, 'color': '#22c55e'}},
                domain={'row': 1, 'column': 0}
            )
        )
        fig.update_layout(
            grid={'rows': 2, 'columns': 1, 'pattern': "independent"},
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    # =========================================================================
    # ECCI Clustering Metrics
    # =========================================================================
    @rx.var
    def ecci_metrics(self) -> dict[str, str]:
        """Get all ECCI clustering metrics as formatted strings."""
        metrics = self.mlflow_metrics.get("E-Commerce Customer Interactions", {})
        if not isinstance(metrics, dict):
            return {
                "silhouette": "0.00",
                "rolling_silhouette": "0.00",
                "time_rolling_silhouette": "0.00",
                "n_clusters": "0",
                "n_micro_clusters": "0",
            }
        return {
            "silhouette": f"{metrics.get('metrics.Silhouette', 0):.4f}",
            "rolling_silhouette": f"{metrics.get('metrics.RollingSilhouette', 0):.4f}",
            "time_rolling_silhouette": f"{metrics.get('metrics.TimeRollingSilhouette', 0):.4f}",
            "n_clusters": f"{int(metrics.get('metrics.n_clusters', 0))}",
            "n_micro_clusters": f"{int(metrics.get('metrics.n_micro_clusters', 0))}",
        }

    @rx.var
    def ecci_metrics_raw(self) -> dict[str, float]:
        """Get raw ECCI metrics as floats for dashboard calculations."""
        metrics = self.mlflow_metrics.get("E-Commerce Customer Interactions", {})
        if not isinstance(metrics, dict):
            return {
                "silhouette": 0, "rolling_silhouette": 0, "time_rolling_silhouette": 0,
                "n_clusters": 0, "n_micro_clusters": 0,
            }
        return {
            "silhouette": metrics.get("metrics.Silhouette", 0),
            "rolling_silhouette": metrics.get("metrics.RollingSilhouette", 0),
            "time_rolling_silhouette": metrics.get("metrics.TimeRollingSilhouette", 0),
            "n_clusters": metrics.get("metrics.n_clusters", 0),
            "n_micro_clusters": metrics.get("metrics.n_micro_clusters", 0),
        }

    @rx.var
    def ecci_dashboard_figures(self) -> dict:
        """Generate all ECCI dashboard Plotly figures (KPI indicators, gauges)."""
        raw = self.ecci_metrics_raw
        mlflow_data = self.mlflow_metrics.get("E-Commerce Customer Interactions", {})

        # Extract baseline metrics for delta calculation
        baseline = {
            "silhouette": float(mlflow_data.get("baseline_Silhouette", 0) or 0),
            "rolling_silhouette": float(mlflow_data.get("baseline_RollingSilhouette", 0) or 0),
        }

        def create_kpi_silhouette(value: float, title: str, baseline_val: float = 0) -> go.Figure:
            """Create KPI indicator for Silhouette metrics (higher is better)."""
            # Color coding: Silhouette ranges from -1 to 1, higher is better
            if value >= 0.7:
                color = "#3b82f6"  # blue - excellent
            elif value >= 0.5:
                color = "#22c55e"  # green - good
            elif value >= 0.25:
                color = "#eab308"  # yellow - fair
            elif value >= 0:
                color = "#f97316"  # orange - poor
            else:
                color = "#ef4444"  # red - negative (misclassification)

            # Configure delta if baseline exists
            delta_config = None
            if baseline_val != 0:
                delta_config = {
                    "reference": baseline_val,
                    "relative": True,
                    "valueformat": ".1%",
                    # For Silhouette, increasing is good
                    "increasing": {"color": "#22c55e"},
                    "decreasing": {"color": "#ef4444"}
                }

            fig = go.Figure(go.Indicator(
                mode="number+delta" if delta_config else "number",
                value=value,
                delta=delta_config,
                title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                number={"font": {"size": 28, "color": color}, "valueformat": ".4f"}
            ))
            fig.update_layout(
                height=110, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_kpi_count(value: float, title: str) -> go.Figure:
            """Create KPI indicator for cluster counts."""
            fig = go.Figure(go.Indicator(
                mode="number",
                value=value,
                title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                number={"font": {"size": 28, "color": "#3b82f6"}, "valueformat": ".0f"}
            ))
            fig.update_layout(
                height=110, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_gauge_silhouette(value: float) -> go.Figure:
            """Create Silhouette gauge (-1 to 1 scale, higher is better)."""
            steps = [
                {"range": [-1, 0], "color": "#ef4444"},     # red - misclassification
                {"range": [0, 0.25], "color": "#f97316"},   # orange - weak structure
                {"range": [0.25, 0.5], "color": "#eab308"}, # yellow - reasonable
                {"range": [0.5, 0.7], "color": "#22c55e"},  # green - good
                {"range": [0.7, 1], "color": "#3b82f6"}     # blue - excellent
            ]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": "<b>Silhouette Score</b>", "font": {"size": 14}},
                number={"valueformat": ".4f", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 1},
                    "bar": {"color": "#1e40af"},
                    "steps": steps,
                    "threshold": {"value": 0.5, "line": {"color": "black", "width": 2}, "thickness": 0.75}
                }
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_cluster_stats_indicator(n_clusters: float, n_micro: float) -> go.Figure:
            """Create dual indicator for cluster statistics (same height as gauge)."""
            fig = go.Figure()
            # Left indicator: Macro Clusters
            fig.add_trace(go.Indicator(
                mode="number",
                value=n_clusters,
                title={"text": "<b>Macro Clusters</b>", "font": {"size": 12}},
                number={"font": {"size": 36, "color": "#3b82f6"}, "valueformat": ".0f"},
                domain={"x": [0, 0.45], "y": [0.1, 0.9]}
            ))
            # Right indicator: Micro Clusters
            fig.add_trace(go.Indicator(
                mode="number",
                value=n_micro,
                title={"text": "<b>Micro Clusters</b>", "font": {"size": 12}},
                number={"font": {"size": 36, "color": "#8b5cf6"}, "valueformat": ".0f"},
                domain={"x": [0.55, 1], "y": [0.1, 0.9]}
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        return {
            # ROW 1: KPI Indicators (primary metrics)
            "kpi_silhouette": create_kpi_silhouette(raw["silhouette"], "Silhouette", baseline["silhouette"]),
            "kpi_rolling_silhouette": create_kpi_silhouette(raw["rolling_silhouette"], "Rolling Silhouette", baseline["rolling_silhouette"]),
            "kpi_n_clusters": create_kpi_count(raw["n_clusters"], "Clusters"),
            "kpi_n_micro_clusters": create_kpi_count(raw["n_micro_clusters"], "Micro Clusters"),
            # ROW 2: Gauge + Cluster Stats
            "gauge_silhouette": create_gauge_silhouette(raw["silhouette"]),
            "cluster_stats": create_cluster_stats_indicator(raw["n_clusters"], raw["n_micro_clusters"]),
        }

    # Average speed for initial ETA estimate (same as Kafka producer)
    _eta_avg_speed_kmh = 40

    def _safe_float(self, value, default: float) -> float:
        """Safely convert value to float, returning default on any error."""
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # =========================================================================
    # Map coordinate computed vars (for reflex-map)
    # =========================================================================
    @rx.var
    def eta_origin_lat(self) -> float:
        """Get ETA origin latitude."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        return self._safe_float(form_data.get("origin_lat"), 29.8)

    @rx.var
    def eta_origin_lon(self) -> float:
        """Get ETA origin longitude."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        return self._safe_float(form_data.get("origin_lon"), -95.4)

    @rx.var
    def eta_destination_lat(self) -> float:
        """Get ETA destination latitude."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        return self._safe_float(form_data.get("destination_lat"), 29.8)

    @rx.var
    def eta_destination_lon(self) -> float:
        """Get ETA destination longitude."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        return self._safe_float(form_data.get("destination_lon"), -95.4)

    @rx.var
    def ecci_lat(self) -> float:
        """Get ECCI location latitude."""
        form_data = self.form_data.get("E-Commerce Customer Interactions") or {}
        return self._safe_float(form_data.get("lat"), 29.8)

    @rx.var
    def ecci_lon(self) -> float:
        """Get ECCI location longitude."""
        form_data = self.form_data.get("E-Commerce Customer Interactions") or {}
        return self._safe_float(form_data.get("lon"), -95.4)

    # =========================================================================
    # Folium Map HTML generation (for rx.html embedding)
    # =========================================================================
    @rx.var
    def eta_folium_map_html(self) -> str:
        """Generate Folium map HTML for ETA origin/destination display."""
        # Get coordinates
        origin_lat = self.eta_origin_lat
        origin_lon = self.eta_origin_lon
        dest_lat = self.eta_destination_lat
        dest_lon = self.eta_destination_lon

        # Calculate center
        center_lat = (origin_lat + dest_lat) / 2
        center_lon = (origin_lon + dest_lon) / 2

        # Create map centered between origin and destination
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='cartodbpositron'
        )

        # Add origin marker (blue)
        folium.Marker(
            location=[origin_lat, origin_lon],
            popup='Origin',
            icon=folium.Icon(color='blue', icon='play')
        ).add_to(m)

        # Add destination marker (red)
        folium.Marker(
            location=[dest_lat, dest_lon],
            popup='Destination',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

        # Add line connecting origin and destination
        folium.PolyLine(
            locations=[[origin_lat, origin_lon], [dest_lat, dest_lon]],
            color='#333333',
            weight=3,
            opacity=0.8
        ).add_to(m)

        # Generate HTML with proper sizing
        html = m._repr_html_()
        # Make iframe fill container completely
        html = html.replace(
            'style="position:relative;width:100%;height:0;padding-bottom:60%;"',
            'style="width:100%;height:300px;"'
        )
        html = html.replace(
            'style="position:absolute;width:100%;height:100%;left:0;top:0;',
            'style="width:100%;height:100%;'
        )
        # Remove Jupyter notebook trust message
        html = html.replace(
            'Make this Notebook Trusted to load map: File -> Trust Notebook',
            ''
        )
        return html

    @rx.var
    def ecci_folium_map_html(self) -> str:
        """Generate Folium map HTML for ECCI customer location display."""
        # Get coordinates
        lat = self.ecci_lat
        lon = self.ecci_lon

        # Create map centered on customer location
        m = folium.Map(
            location=[lat, lon],
            zoom_start=10,
            tiles='cartodbpositron'
        )

        # Add customer location marker (purple - using darkpurple)
        folium.Marker(
            location=[lat, lon],
            popup='Customer Location',
            icon=folium.Icon(color='purple', icon='user')
        ).add_to(m)

        # Generate HTML with proper sizing
        html = m._repr_html_()
        # Make iframe fill container completely
        html = html.replace(
            'style="position:relative;width:100%;height:0;padding-bottom:60%;"',
            'style="width:100%;height:250px;"'
        )
        html = html.replace(
            'style="position:absolute;width:100%;height:100%;left:0;top:0;',
            'style="width:100%;height:100%;'
        )
        # Remove Jupyter notebook trust message
        html = html.replace(
            'Make this Notebook Trusted to load map: File -> Trust Notebook',
            ''
        )
        return html

    @rx.var
    def eta_estimated_distance_km(self) -> float:
        """Calculate estimated distance using Haversine formula."""
        import math
        # Safely get form data dict
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        if not isinstance(form_data, dict):
            return 0.0

        # Safely convert coordinates with defaults
        origin_lat = self._safe_float(form_data.get("origin_lat"), 0)
        origin_lon = self._safe_float(form_data.get("origin_lon"), 0)
        dest_lat = self._safe_float(form_data.get("destination_lat"), 0)
        dest_lon = self._safe_float(form_data.get("destination_lon"), 0)

        # Haversine formula
        lon1, lat1, lon2, lat2 = map(math.radians, [origin_lon, origin_lat, dest_lon, dest_lat])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return round(c * r, 2)

    @rx.var
    def eta_initial_estimated_travel_time_seconds(self) -> int:
        """Calculate initial estimated travel time based on distance and average speed."""
        distance = self.eta_estimated_distance_km
        if distance <= 0:
            return 60  # Minimum 1 minute
        # Same formula as Kafka producer: (distance / AVG_SPEED_KMH) * 3600
        travel_time = int((distance / self._eta_avg_speed_kmh) * 3600)
        return max(60, travel_time)  # Minimum 1 minute

    # =========================================================================
    # E-Commerce Customer Interactions (ECCI) computed vars
    # =========================================================================
    @rx.var
    def ecci_form_data(self) -> dict:
        """Get E-Commerce Customer Interactions form data."""
        return self.form_data.get("E-Commerce Customer Interactions", {})

    @rx.var
    def ecci_options(self) -> dict[str, list[str]]:
        """Get all ECCI dropdown options as a dict."""
        opts = self.dropdown_options.get("E-Commerce Customer Interactions", {})
        return opts if isinstance(opts, dict) else {}

    @rx.var
    def ecci_prediction_show(self) -> bool:
        """Check if ECCI prediction results should be shown."""
        results = self.prediction_results.get("E-Commerce Customer Interactions", {})
        if isinstance(results, dict):
            return results.get("show", False)
        return False

    @rx.var
    def ecci_predicted_cluster(self) -> int:
        """Get predicted cluster number."""
        results = self.prediction_results.get("E-Commerce Customer Interactions", {})
        if isinstance(results, dict):
            return results.get("cluster", 0)
        return 0

    @rx.var
    def ecci_prediction_figure(self) -> go.Figure:
        """Generate Plotly figure for ECCI cluster prediction display."""
        cluster = self.ecci_predicted_cluster

        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=cluster,
                title={'text': "<b>Cluster</b>", 'font': {'size': 24}},
                number={'font': {'size': 72, 'color': '#8b5cf6'}},  # purple
                domain={'row': 0, 'column': 0}
            )
        )
        fig.update_layout(
            grid={'rows': 1, 'columns': 1},
            height=250,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    # ECCI Cluster Analytics state vars
    ecci_cluster_counts: dict = {}
    ecci_selected_feature: str = "event_type"
    ecci_cluster_feature_counts: dict = {}

    @rx.var
    def ecci_feature_options(self) -> list[str]:
        """Get available features for cluster analysis."""
        return [
            "event_type",
            "product_category",
            "referrer_url",
            "quantity",
            "time_on_page_seconds",
            "session_event_sequence",
            "device_type",
            "browser",
            "os",
        ]

    @rx.var
    def ecci_cluster_counts_figure(self) -> go.Figure:
        """Generate bar chart for samples per cluster."""
        cluster_counts = self.ecci_cluster_counts
        fig = go.Figure()

        if cluster_counts:
            clusters = list(cluster_counts.keys())
            counts = list(cluster_counts.values())

            fig.add_trace(
                go.Bar(
                    x=[f"Cluster {c}" for c in clusters],
                    y=counts,
                    marker_color='#8b5cf6',
                    text=counts,
                    textposition='auto',
                )
            )

        fig.update_layout(
            title='Samples per Cluster',
            xaxis_title='Cluster',
            yaxis_title='Count',
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    @rx.var
    def ecci_selected_cluster_feature_figure(self) -> go.Figure:
        """Generate bar chart for predicted cluster's feature distribution."""
        fig = go.Figure()
        predicted_cluster = self.ecci_predicted_cluster
        feature_counts = self.ecci_cluster_feature_counts
        selected_feature = self.ecci_selected_feature

        cluster_key = str(predicted_cluster)
        if cluster_key in feature_counts:
            cluster_data = feature_counts[cluster_key]
            if isinstance(cluster_data, dict):
                # Sort by count descending and take top 10
                sorted_items = sorted(cluster_data.items(), key=lambda x: x[1], reverse=True)[:10]
                labels = [str(item[0]) for item in sorted_items]
                values = [item[1] for item in sorted_items]

                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color='#3b82f6',
                        text=values,
                        textposition='auto',
                    )
                )

        fig.update_layout(
            title=f'Cluster {predicted_cluster} - {selected_feature}',
            xaxis_title=selected_feature,
            yaxis_title='Count',
            height=300,
            margin=dict(l=40, r=20, t=50, b=80),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45,
        )
        return fig

    @rx.var
    def ecci_all_clusters_feature_figure(self) -> go.Figure:
        """Generate grouped bar chart for all clusters' feature distribution."""
        fig = go.Figure()
        feature_counts = self.ecci_cluster_feature_counts
        selected_feature = self.ecci_selected_feature

        if feature_counts:
            # Get all unique feature values across clusters
            all_values = set()
            for cluster_data in feature_counts.values():
                if isinstance(cluster_data, dict):
                    all_values.update(cluster_data.keys())

            # Sort and limit to top 10 by total count
            value_totals = {}
            for val in all_values:
                total = sum(
                    cluster_data.get(val, 0)
                    for cluster_data in feature_counts.values()
                    if isinstance(cluster_data, dict)
                )
                value_totals[val] = total

            top_values = sorted(value_totals.keys(), key=lambda x: value_totals[x], reverse=True)[:10]

            # Add a bar for each cluster
            colors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899']
            for i, (cluster_id, cluster_data) in enumerate(sorted(feature_counts.items())):
                if isinstance(cluster_data, dict):
                    values = [cluster_data.get(val, 0) for val in top_values]
                    fig.add_trace(
                        go.Bar(
                            name=f'Cluster {cluster_id}',
                            x=[str(v) for v in top_values],
                            y=values,
                            marker_color=colors[i % len(colors)],
                        )
                    )

        fig.update_layout(
            title=f'Feature Distribution: {selected_feature}',
            xaxis_title=selected_feature,
            yaxis_title='Count',
            barmode='group',
            height=500,
            margin=dict(l=40, r=20, t=50, b=150),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45,
            legend=dict(orientation='h', yanchor='top', y=-0.35, xanchor='center', x=0.5),
        )
        return fig

    # ==========================================================================
    # DELTA LAKE SQL TAB COMPUTED VARS
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
                # Try numeric sort first, fall back to string sort
                def sort_key(row):
                    val = row.get(sort_col, "")
                    # Try to convert to float for numeric sorting
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

    ##==========================================================================
    ## EVENTS
    ##==========================================================================
    @rx.event
    def set_current_page_model(self, model_key: str):
        """Set the model key for the current page (called on page mount)."""
        self._current_page_model_key = model_key
        # Reset switch state when entering a new page
        # Switch will be ON only if this page's model is already running
        self.ml_training_enabled = (self.activated_model == model_key)

    @rx.event(background = True)
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
                    description = f"ML training for {project_name} is already active",
                    duration = 3000,
                )
                return
            try:
                response = await httpx_client_post(
                    url = f"{RIVER_BASE_URL}/switch_model",
                    json = {
                        "model_key": model_key,
                        "project_name": project_name,
                    },
                    timeout = 30.0
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
                    description = f"Processing live data for {project_name}",
                    duration = 5000,
                    close_button = True,
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
                    description = str(e),
                    duration = 8000,
                    close_button = True,
                )
        else:
            # Stop the model
            if not self.activated_model:
                return
            try:
                await httpx_client_post(
                    url = f"{RIVER_BASE_URL}/switch_model",
                    json = {
                        "model_key": "none",
                        "project_name": ""
                    },
                    timeout = 30.0
                )
                async with self:
                    self.activated_model = ""
                    self.model_switch_message = "Model stopped"
                    self.ml_training_enabled = False
                # Re-check model availability after training stopped (model may have been saved)
                yield State.check_incremental_model_available(project_name)
                yield State.get_mlflow_metrics(project_name)
                yield rx.toast.info(
                    "Real-time ML training stopped",
                    description = f"Stopped processing for {project_name}",
                    duration = 3000,
                )
            except Exception as e:
                print(f"Error stopping model: {e}")
                yield rx.toast.warning(
                    "Could not stop ML training",
                    description = str(e),
                    duration = 5000,
                )

    @rx.event(background = True)
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
                        url = f"{RIVER_BASE_URL}/switch_model",
                        json = {
                            "model_key": "none",
                            "project_name": ""
                        },
                        timeout = 30.0
                    )
                    async with self:
                        self.activated_model = ""
                        self.model_switch_message = "Model stopped"
                        self.ml_training_enabled = False
                    yield rx.toast.info(
                        "Real-time ML training stopped",
                        description = f"Stopped processing for {project_name}",
                        duration = 3000,
                    )
                except Exception as e:
                    print(f"Error stopping model on page leave: {e}")

    @rx.event(background = True)
    async def update_sample(self, project_name: str):
        """Fetch initial sample from FastAPI (runs in background to avoid lock expiration)."""
        if project_name == "Home":
            async with self:
                self.incremental_ml_sample[project_name] = {}
            return
        try:
            sample = await httpx_client_post(
                url = f"{RIVER_BASE_URL}/initial_sample",
                json = {"project_name": project_name},
                timeout = 30.0
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
            # Read-only string fields
            "transaction_id": get_str(sample, "transaction_id"),
            "user_id": get_str(sample, "user_id"),
            "ip_address": get_str(sample, "ip_address"),
            "user_agent": get_str(sample, "user_agent"),
        }
        async with self:
            self.form_data["Transaction Fraud Detection"] = form_data
        # Fetch dropdown options in parallel
        await self._fetch_tfd_options_internal()

    async def _fetch_tfd_options_internal(self):
        """Internal helper to fetch dropdown options in parallel."""
        project_name = "Transaction Fraud Detection"
        try:
            # Fetch all unique values in parallel for better performance
            responses = await asyncio.gather(
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "currency", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "merchant_id", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "product_category", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "transaction_type", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "payment_method", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "device_info", "project_name": project_name},
                    timeout = 30.0
                ),
                return_exceptions = True
            )
            # Unpack responses
            currency_response, merchant_response, product_response, \
                transaction_response, payment_response, device_response = responses
            # Parse device info
            browsers = set()
            oses = set()
            if not isinstance(device_response, Exception):
                device_info_options = device_response.json().get("unique_values", [])
                for device_str in device_info_options:
                    try:
                        device_dict = eval(device_str)
                        browsers.add(device_dict.get("browser", ""))
                        oses.add(device_dict.get("os", ""))
                    except:
                        pass
            dropdown_options = {
                "currency": currency_response.json().get("unique_values", []) if not isinstance(currency_response, Exception) else [],
                "merchant_id": merchant_response.json().get("unique_values", []) if not isinstance(merchant_response, Exception) else [],
                "product_category": product_response.json().get("unique_values", []) if not isinstance(product_response, Exception) else [],
                "transaction_type": transaction_response.json().get("unique_values", []) if not isinstance(transaction_response, Exception) else [],
                "payment_method": payment_response.json().get("unique_values", []) if not isinstance(payment_response, Exception) else [],
                "browser": sorted(list(browsers)),
                "os": sorted(list(oses))
            }
            async with self:
                self.dropdown_options["Transaction Fraud Detection"] = dropdown_options
        except Exception as e:
            print(f"Error fetching dropdown options: {e}")
            async with self:
                self.dropdown_options["Transaction Fraud Detection"] = {}

    # TFD Form field update handler (consolidated)
    # Field type mappings for automatic conversion
    _tfd_float_fields = {"amount", "lat", "lon"}
    _tfd_int_fields = {"account_age_days"}
    _tfd_bool_fields = {"cvv_provided", "billing_address_match"}

    @rx.event
    def update_tfd(self, field: str, value):
        """Update a TFD form field with automatic type conversion."""
        try:
            if field in self._tfd_float_fields:
                value = float(value) if value else 0.0
            elif field in self._tfd_int_fields:
                value = int(value) if value else 0
            elif field in self._tfd_bool_fields:
                value = bool(value)
            # str fields need no conversion
        except (ValueError, TypeError):
            return  # Ignore invalid conversions
        if "Transaction Fraud Detection" not in self.form_data:
            self.form_data["Transaction Fraud Detection"] = {}
        self.form_data["Transaction Fraud Detection"][field] = value

    @rx.event(background = True)
    async def predict_transaction_fraud_detection(self):
        """Make prediction for transaction fraud detection using current form state."""
        project_name = "Transaction Fraud Detection"
        current_form = self.form_data.get(project_name, {})
        # Combine date and time
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"
        # Prepare request payload from current form state
        payload = {
            "project_name": project_name,
            "model_name": "ARFClassifier",
            "transaction_id": current_form.get("transaction_id", ""),
            "user_id": current_form.get("user_id", ""),
            "timestamp": timestamp,
            "amount": float(current_form.get("amount", 0)),
            "currency": current_form.get("currency", ""),
            "merchant_id": current_form.get("merchant_id", ""),
            "product_category": current_form.get("product_category", ""),
            "transaction_type": current_form.get("transaction_type", ""),
            "payment_method": current_form.get("payment_method", ""),
            "location": {
                "lat": float(current_form.get("lat", 0)),
                "lon": float(current_form.get("lon", 0))
            },
            "ip_address": current_form.get("ip_address", ""),
            "device_info": {
                "os": current_form.get("os", ""),
                "browser": current_form.get("browser", "")
            },
            "user_agent": current_form.get("user_agent", ""),
            "account_age_days": int(current_form.get("account_age_days", 0)),
            "cvv_provided": bool(current_form.get("cvv_provided", False)),
            "billing_address_match": bool(current_form.get("billing_address_match", False))
        }
        # Make prediction
        try:
            print(f"Making prediction with payload: {payload}")
            response = await httpx_client_post(
                url = f"{RIVER_BASE_URL}/predict",
                json = payload,
                timeout = 30.0
            )
            result = response.json()
            print(f"Prediction result: {result}")
            # Create new dict to trigger reactivity
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "prediction": result.get("prediction"),
                        "fraud_probability": result.get("fraud_probability"),
                        "model_source": result.get("model_source", "mlflow"),
                        "show": True
                    }
                }
            # Refresh MLflow metrics after prediction (to show real-time updates during training)
            await self._fetch_mlflow_metrics_internal(project_name)
        except Exception as e:
            print(f"Error making prediction: {e}")
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "prediction": None,
                        "fraud_probability": 0.0,
                        "model_source": "mlflow",
                        "show": False
                    }
                }

    # Model name mapping for MLflow metrics
    _mlflow_model_names: dict = {
        "Transaction Fraud Detection": "ARFClassifier",
        "Estimated Time of Arrival": "ARFRegressor",
        "E-Commerce Customer Interactions": "DBSTREAM",
    }

    async def _fetch_mlflow_metrics_internal(self, project_name: str, force_refresh: bool = True):
        """Internal helper to fetch MLflow metrics (called from other async methods)."""
        model_name = self._mlflow_model_names.get(project_name, "ARFClassifier")
        try:
            response = await httpx_client_post(
                url = f"{RIVER_BASE_URL}/mlflow_metrics",
                json = {
                    "project_name": project_name,
                    "model_name": model_name,
                    "force_refresh": force_refresh
                },
                timeout = 30.0
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

    @rx.event(background = True)
    async def get_mlflow_metrics(self, project_name: str):
        """Fetch MLflow metrics for a project (runs in background to avoid lock expiration)."""
        model_name = self._mlflow_model_names.get(project_name, "ARFClassifier")
        try:
            response = await httpx_client_post(
                url = f"{RIVER_BASE_URL}/mlflow_metrics",
                json = {
                    "project_name": project_name,
                    "model_name": model_name
                },
                timeout = 60.0
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

    @rx.event(background = True)
    async def refresh_mlflow_metrics(self, project_name: str):
        """Force refresh MLflow metrics bypassing cache."""
        model_name = self._mlflow_model_names.get(project_name, "ARFClassifier")
        try:
            response = await httpx_client_post(
                url = f"{RIVER_BASE_URL}/mlflow_metrics",
                json = {
                    "project_name": project_name,
                    "model_name": model_name,
                    "force_refresh": True
                },
                timeout = 60.0
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
                description = f"Latest metrics loaded for {project_name}",
                duration = 2000
            )
        except Exception as e:
            print(f"Error refreshing MLflow metrics: {e}")
            yield rx.toast.error(
                "Refresh failed",
                description = str(e),
                duration = 3000
            )

    # =========================================================================
    # Estimated Time of Arrival (ETA) event handlers
    # =========================================================================
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

        # Get select field values for dropdown options
        driver_id = get_str(sample, "driver_id")
        vehicle_id = get_str(sample, "vehicle_id")
        weather = get_str(sample, "weather")
        vehicle_type = get_str(sample, "vehicle_type")

        # Fetch dropdown options first, ensuring sample values are included
        await self._fetch_eta_options_internal(
            driver_id=driver_id,
            vehicle_id=vehicle_id,
            weather=weather,
            vehicle_type=vehicle_type
        )

        # Parse JSON string fields from Delta Lake
        origin = parse_json_field(sample, "origin")
        destination = parse_json_field(sample, "destination")

        # Then set form data (after options are loaded so selects work correctly)
        form_data = {
            # Select fields (string)
            "driver_id": driver_id,
            "vehicle_id": vehicle_id,
            "weather": weather,
            "vehicle_type": vehicle_type,
            # Timestamp fields
            "timestamp_date": timestamp_date,
            "timestamp_time": timestamp_time,
            # Coordinate fields (float as string) - parsed from JSON
            "origin_lat": safe_float_str(origin.get("lat"), 29.8),
            "origin_lon": safe_float_str(origin.get("lon"), -95.4),
            "destination_lat": safe_float_str(destination.get("lat"), 29.8),
            "destination_lon": safe_float_str(destination.get("lon"), -95.4),
            # Integer fields (as string for proper display)
            "hour_of_day": safe_int_str(sample.get("hour_of_day"), 12),
            "debug_incident_delay_seconds": safe_int_str(sample.get("debug_incident_delay_seconds"), 0),
            "initial_estimated_travel_time_seconds": safe_int_str(sample.get("initial_estimated_travel_time_seconds"), 60),
            "day_of_week": safe_int_str(sample.get("day_of_week"), 0),
            # Float fields (as string for proper display)
            "driver_rating": safe_float_str(sample.get("driver_rating"), 4.5),
            "debug_traffic_factor": safe_float_str(sample.get("debug_traffic_factor"), 1.0),
            "debug_weather_factor": safe_float_str(sample.get("debug_weather_factor"), 1.0),
            "debug_driver_factor": safe_float_str(sample.get("debug_driver_factor"), 1.0),
            "temperature_celsius": safe_float_str(sample.get("temperature_celsius"), 25.0),
            "estimated_distance_km": safe_float_str(sample.get("estimated_distance_km"), 0.0),
            # Read-only string fields
            "trip_id": get_str(sample, "trip_id"),
        }
        async with self:
            self.form_data["Estimated Time of Arrival"] = form_data

    async def _fetch_eta_options_internal(
        self,
        driver_id: str = None,
        vehicle_id: str = None,
        weather: str = None,
        vehicle_type: str = None
    ):
        """Internal helper to fetch ETA dropdown options in parallel.

        Args:
            driver_id: If provided, ensures this value is in the driver_id options
            vehicle_id: If provided, ensures this value is in the vehicle_id options
            weather: If provided, ensures this value is in the weather options
            vehicle_type: If provided, ensures this value is in the vehicle_type options
        """
        project_name = "Estimated Time of Arrival"
        try:
            responses = await asyncio.gather(
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "driver_id", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "vehicle_id", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "weather", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{RIVER_BASE_URL}/unique_values",
                    json = {"column_name": "vehicle_type", "project_name": project_name},
                    timeout = 30.0
                ),
                return_exceptions = True
            )
            driver_response, vehicle_response, weather_response, vehicle_type_response = responses

            # Get options and ensure sample values are included
            driver_id_options = driver_response.json().get("unique_values", []) if not isinstance(driver_response, Exception) else []
            if driver_id and driver_id not in driver_id_options:
                driver_id_options = [driver_id] + driver_id_options

            vehicle_id_options = vehicle_response.json().get("unique_values", []) if not isinstance(vehicle_response, Exception) else []
            if vehicle_id and vehicle_id not in vehicle_id_options:
                vehicle_id_options = [vehicle_id] + vehicle_id_options

            weather_options = weather_response.json().get("unique_values", []) if not isinstance(weather_response, Exception) else []
            if weather and weather not in weather_options:
                weather_options = [weather] + weather_options

            vehicle_type_options = vehicle_type_response.json().get("unique_values", []) if not isinstance(vehicle_type_response, Exception) else []
            if vehicle_type and vehicle_type not in vehicle_type_options:
                vehicle_type_options = [vehicle_type] + vehicle_type_options

            dropdown_options = {
                "driver_id": driver_id_options,
                "vehicle_id": vehicle_id_options,
                "weather": weather_options,
                "vehicle_type": vehicle_type_options,
            }
            async with self:
                self.dropdown_options["Estimated Time of Arrival"] = dropdown_options
        except Exception as e:
            print(f"Error fetching ETA dropdown options: {e}")
            async with self:
                self.dropdown_options["Estimated Time of Arrival"] = {}

    # ETA coordinate bounds (Houston metro area)
    _eta_lat_bounds = (29.5, 30.1)
    _eta_lon_bounds = (-95.8, -95.0)

    # ETA Form field update handler (consolidated)
    _eta_float_fields = {
        "origin_lat", "origin_lon", "destination_lat", "destination_lon",
        "driver_rating", "debug_traffic_factor", "debug_weather_factor",
        "debug_driver_factor", "temperature_celsius", "estimated_distance_km"
    }
    _eta_int_fields = {"hour_of_day", "debug_incident_delay_seconds", "initial_estimated_travel_time_seconds", "day_of_week"}

    # Coordinate fields that should reset prediction when changed
    _eta_coordinate_fields = {"origin_lat", "origin_lon", "destination_lat", "destination_lon"}

    @rx.event
    def update_eta(self, field: str, value):
        """Update an ETA form field with automatic type conversion."""
        try:
            if field in self._eta_float_fields:
                value = float(value) if value else 0.0
            elif field in self._eta_int_fields:
                value = int(value) if value else 0
            # str fields need no conversion
        except (ValueError, TypeError):
            return  # Ignore invalid conversions
        if "Estimated Time of Arrival" not in self.form_data:
            self.form_data["Estimated Time of Arrival"] = {}
        self.form_data["Estimated Time of Arrival"][field] = value
        # Reset prediction if coordinate field changed
        if field in self._eta_coordinate_fields:
            self.prediction_results["Estimated Time of Arrival"] = {"show": False, "eta_seconds": 0.0}

    @rx.event
    def generate_random_eta_coordinates(self):
        """Generate random coordinates within Houston metro bounds."""
        import random
        if "Estimated Time of Arrival" not in self.form_data:
            self.form_data["Estimated Time of Arrival"] = {}
        form = self.form_data["Estimated Time of Arrival"]
        # Generate random origin coordinates
        form["origin_lat"] = round(random.uniform(self._eta_lat_bounds[0], self._eta_lat_bounds[1]), 6)
        form["origin_lon"] = round(random.uniform(self._eta_lon_bounds[0], self._eta_lon_bounds[1]), 6)
        # Generate random destination coordinates (ensure not too close to origin)
        while True:
            dest_lat = round(random.uniform(self._eta_lat_bounds[0], self._eta_lat_bounds[1]), 6)
            dest_lon = round(random.uniform(self._eta_lon_bounds[0], self._eta_lon_bounds[1]), 6)
            if abs(form["origin_lat"] - dest_lat) >= 0.01 or abs(form["origin_lon"] - dest_lon) >= 0.01:
                break
        form["destination_lat"] = dest_lat
        form["destination_lon"] = dest_lon
        # Reset prediction when coordinates change
        self.prediction_results["Estimated Time of Arrival"] = {"show": False, "eta_seconds": 0.0}

    @rx.event(background = True)
    async def predict_eta(self):
        """Make prediction for Estimated Time of Arrival using current form state."""
        project_name = "Estimated Time of Arrival"
        current_form = self.form_data.get(project_name, {})
        # Combine date and time
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"
        # Prepare request payload from current form state
        payload = {
            "project_name": project_name,
            "model_name": "ARFRegressor",
            "trip_id": current_form.get("trip_id", ""),
            "driver_id": current_form.get("driver_id", ""),
            "vehicle_id": current_form.get("vehicle_id", ""),
            "timestamp": timestamp,
            "origin": {
                "lat": float(current_form.get("origin_lat", 0)),
                "lon": float(current_form.get("origin_lon", 0))
            },
            "destination": {
                "lat": float(current_form.get("destination_lat", 0)),
                "lon": float(current_form.get("destination_lon", 0))
            },
            "estimated_distance_km": self.eta_estimated_distance_km,
            "weather": current_form.get("weather", ""),
            "temperature_celsius": float(current_form.get("temperature_celsius", 0)),
            "day_of_week": int(current_form.get("day_of_week", 0)),
            "hour_of_day": int(current_form.get("hour_of_day", 0)),
            "driver_rating": float(current_form.get("driver_rating", 0)),
            "vehicle_type": current_form.get("vehicle_type", ""),
            "initial_estimated_travel_time_seconds": self.eta_initial_estimated_travel_time_seconds,
            "debug_traffic_factor": float(current_form.get("debug_traffic_factor", 0)),
            "debug_weather_factor": float(current_form.get("debug_weather_factor", 0)),
            "debug_incident_delay_seconds": int(current_form.get("debug_incident_delay_seconds", 0)),
            "debug_driver_factor": float(current_form.get("debug_driver_factor", 0))
        }
        # Make prediction
        try:
            print(f"Making ETA prediction with payload: {payload}")
            response = await httpx_client_post(
                url = f"{RIVER_BASE_URL}/predict",
                json = payload,
                timeout = 30.0
            )
            result = response.json()
            print(f"ETA Prediction result: {result}")
            eta_seconds = result.get("Estimated Time of Arrival", 0.0)
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "eta_seconds": eta_seconds,
                        "model_source": result.get("model_source", "mlflow"),
                        "show": True
                    }
                }
            # Refresh MLflow metrics after prediction (to show real-time updates during training)
            await self._fetch_mlflow_metrics_internal(project_name)
        except Exception as e:
            print(f"Error making ETA prediction: {e}")
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "eta_seconds": 0.0,
                        "model_source": "mlflow",
                        "show": False
                    }
                }

    # =========================================================================
    # E-Commerce Customer Interactions (ECCI) event handlers
    # =========================================================================
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
        device_info = parse_json_field(sample, "device_info")
        location = parse_json_field(sample, "location")

        form_data = {
            # Device info (parsed from JSON string) - select fields
            "browser": device_info.get("browser") or "",
            "device_type": device_info.get("device_type") or "",
            "os": device_info.get("os") or "",
            # Location (parsed from JSON string)
            "lat": safe_float_str(location.get("lat"), 29.8),
            "lon": safe_float_str(location.get("lon"), -95.4),
            # Select/string fields
            "event_type": get_str(sample, "event_type"),
            "product_category": get_str(sample, "product_category"),
            # Text input fields
            "product_id": get_str(sample, "product_id"),
            "referrer_url": get_str(sample, "referrer_url"),
            # Numeric fields (as string for proper display)
            "price": safe_float_str(sample.get("price"), 0.0),
            "session_event_sequence": safe_int_str(sample.get("session_event_sequence"), 1),
            "quantity": safe_int_str(sample.get("quantity"), 1),
            "time_on_page_seconds": safe_int_str(sample.get("time_on_page_seconds"), 0),
            # Timestamp fields
            "timestamp_date": timestamp_date,
            "timestamp_time": timestamp_time,
            # Read-only string fields
            "customer_id": get_str(sample, "customer_id"),
            "event_id": get_str(sample, "event_id"),
            "page_url": get_str(sample, "page_url"),
            "search_query": get_str(sample, "search_query"),
            "session_id": get_str(sample, "session_id"),
        }
        async with self:
            self.form_data["E-Commerce Customer Interactions"] = form_data
        # Fetch dropdown options in parallel
        await self._fetch_ecci_options_internal()

    async def _fetch_ecci_options_internal(self):
        """Internal helper to fetch ECCI dropdown options in parallel.
        Note: product_id and referrer_url are excluded - they use text inputs due to high cardinality.
        """
        project_name = "E-Commerce Customer Interactions"
        try:
            # Only fetch low-cardinality fields for dropdowns
            responses = await asyncio.gather(
                httpx_client_post(
                    url=f"{RIVER_BASE_URL}/unique_values",
                    json={"column_name": "device_info", "project_name": project_name, "limit": 50},
                    timeout=30.0
                ),
                httpx_client_post(
                    url=f"{RIVER_BASE_URL}/unique_values",
                    json={"column_name": "event_type", "project_name": project_name, "limit": 20},
                    timeout=30.0
                ),
                httpx_client_post(
                    url=f"{RIVER_BASE_URL}/unique_values",
                    json={"column_name": "product_category", "project_name": project_name, "limit": 50},
                    timeout=30.0
                ),
                return_exceptions=True
            )
            (device_response, event_type_response, product_category_response) = responses

            # Parse device info to extract browser, device_type, os options
            browsers = set()
            device_types = set()
            oses = set()
            if not isinstance(device_response, Exception):
                device_info_options = device_response.json().get("unique_values", [])
                for device_str in device_info_options:
                    try:
                        device_dict = eval(device_str)
                        browsers.add(device_dict.get("browser", ""))
                        device_types.add(device_dict.get("device_type", ""))
                        oses.add(device_dict.get("os", ""))
                    except:
                        pass

            dropdown_options = {
                "browser": sorted(list(browsers - {""})),
                "device_type": sorted(list(device_types - {""})),
                "os": sorted(list(oses - {""})),
                "event_type": event_type_response.json().get("unique_values", []) if not isinstance(event_type_response, Exception) else [],
                "product_category": product_category_response.json().get("unique_values", []) if not isinstance(product_category_response, Exception) else [],
            }
            async with self:
                self.dropdown_options["E-Commerce Customer Interactions"] = dropdown_options
        except Exception as e:
            print(f"Error fetching ECCI dropdown options: {e}")
            async with self:
                self.dropdown_options["E-Commerce Customer Interactions"] = {}

    # ECCI coordinate bounds (Houston metro area - same as ETA)
    _ecci_lat_bounds = (29.5, 30.1)
    _ecci_lon_bounds = (-95.8, -95.0)

    # ECCI Form field update handler (consolidated)
    _ecci_float_fields = {"lat", "lon", "price"}
    _ecci_int_fields = {"session_event_sequence", "quantity", "time_on_page_seconds"}

    @rx.event
    def update_ecci(self, field: str, value):
        """Update an ECCI form field with automatic type conversion."""
        try:
            if field in self._ecci_float_fields:
                value = float(value) if value else 0.0
            elif field in self._ecci_int_fields:
                value = int(value) if value else 0
            # str fields need no conversion
        except (ValueError, TypeError):
            return  # Ignore invalid conversions
        if "E-Commerce Customer Interactions" not in self.form_data:
            self.form_data["E-Commerce Customer Interactions"] = {}
        self.form_data["E-Commerce Customer Interactions"][field] = value

    @rx.event
    def generate_random_ecci_coordinates(self):
        """Generate random coordinates within Houston metro bounds for ECCI."""
        import random
        if "E-Commerce Customer Interactions" not in self.form_data:
            self.form_data["E-Commerce Customer Interactions"] = {}
        form = self.form_data["E-Commerce Customer Interactions"]
        form["lat"] = round(random.uniform(self._ecci_lat_bounds[0], self._ecci_lat_bounds[1]), 3)
        form["lon"] = round(random.uniform(self._ecci_lon_bounds[0], self._ecci_lon_bounds[1]), 3)
        # Reset prediction when coordinates change
        self.prediction_results["E-Commerce Customer Interactions"] = {"show": False, "cluster": 0}

    @rx.event(background=True)
    async def predict_ecci(self):
        """Make prediction for E-Commerce Customer Interactions using current form state."""
        project_name = "E-Commerce Customer Interactions"
        current_form = self.form_data.get(project_name, {})
        # Combine date and time
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"
        # Prepare request payload from current form state
        payload = {
            "project_name": project_name,
            "model_name": "DBSTREAM",
            "customer_id": current_form.get("customer_id", ""),
            "device_info": {
                "device_type": current_form.get("device_type", ""),
                "browser": current_form.get("browser", ""),
                "os": current_form.get("os", "")
            },
            "event_id": current_form.get("event_id", ""),
            "event_type": current_form.get("event_type", ""),
            "location": {
                "lat": float(current_form.get("lat", 0)),
                "lon": float(current_form.get("lon", 0))
            },
            "page_url": current_form.get("page_url", ""),
            "price": float(current_form.get("price", 0)),
            "product_category": current_form.get("product_category", ""),
            "product_id": current_form.get("product_id", ""),
            "quantity": int(current_form.get("quantity", 1)) if current_form.get("quantity") else None,
            "referrer_url": current_form.get("referrer_url", ""),
            "search_query": current_form.get("search_query", ""),
            "session_event_sequence": int(current_form.get("session_event_sequence", 1)),
            "session_id": current_form.get("session_id", ""),
            "time_on_page_seconds": int(current_form.get("time_on_page_seconds", 0)),
            "timestamp": timestamp
        }
        # Make prediction
        try:
            print(f"Making ECCI prediction with payload: {payload}")
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/predict",
                json=payload,
                timeout=30.0
            )
            result = response.json()
            print(f"ECCI Prediction result: {result}")
            cluster = result.get("cluster", 0)
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "cluster": cluster,
                        "model_source": result.get("model_source", "mlflow"),
                        "show": True
                    }
                }
            # Refresh MLflow metrics after prediction
            await self._fetch_mlflow_metrics_internal(project_name)
        except Exception as e:
            print(f"Error making ECCI prediction: {e}")
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "cluster": 0,
                        "show": False
                    }
                }

    @rx.event(background=True)
    async def fetch_ecci_cluster_counts(self):
        """Fetch cluster counts from FastAPI."""
        try:
            response = await httpx_client_get(
                url=f"{RIVER_BASE_URL}/cluster_counts",
                timeout=30.0
            )
            cluster_counts = response.json()
            async with self:
                self.ecci_cluster_counts = cluster_counts
        except Exception as e:
            print(f"Error fetching ECCI cluster counts: {e}")
            async with self:
                self.ecci_cluster_counts = {}

    @rx.event(background=True)
    async def fetch_ecci_cluster_feature_counts(self, feature: str = None):
        """Fetch feature counts per cluster from FastAPI."""
        if feature is None:
            feature = self.ecci_selected_feature
        # Handle device_info sub-features
        device_info_features = ["device_type", "browser", "os"]
        column_name = "device_info" if feature in device_info_features else feature
        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/cluster_feature_counts",
                json={"column_name": column_name},
                timeout=30.0
            )
            raw_data = response.json()
            # Process device_info sub-features
            if feature in device_info_features:
                processed_data = {}
                for cluster_id, device_counts in raw_data.items():
                    processed_data[cluster_id] = {}
                    for device_str, count in device_counts.items():
                        try:
                            import ast
                            device_dict = ast.literal_eval(device_str)
                            feature_value = device_dict.get(feature, "unknown")
                            if feature_value in processed_data[cluster_id]:
                                processed_data[cluster_id][feature_value] += count
                            else:
                                processed_data[cluster_id][feature_value] = count
                        except:
                            pass
                async with self:
                    self.ecci_cluster_feature_counts = processed_data
            else:
                async with self:
                    self.ecci_cluster_feature_counts = raw_data
        except Exception as e:
            print(f"Error fetching ECCI cluster feature counts: {e}")
            async with self:
                self.ecci_cluster_feature_counts = {}

    @rx.event
    def set_ecci_selected_feature(self, feature: str):
        """Set the selected feature and trigger data fetch."""
        self.ecci_selected_feature = feature
        return State.fetch_ecci_cluster_feature_counts(feature)

    # =========================================================================
    # Randomize Form Event Handlers (local generation - instant, no network call)
    # =========================================================================
    @rx.event
    def randomize_tfd_form(self):
        """Generate random values locally using loaded dropdown options (instant)."""
        import random
        import uuid
        project_name = "Transaction Fraud Detection"
        opts = self.dropdown_options.get(project_name, {})
        now = dt.datetime.now()
        # Generate random form data using loaded dropdown options
        # Use `or` to handle both missing keys AND empty lists
        form_data = {
            # Dropdown fields - pick random from loaded options
            "currency": random.choice(opts.get("currency") or ["USD"]),
            "merchant_id": random.choice(opts.get("merchant_id") or ["merchant_1"]),
            "product_category": random.choice(opts.get("product_category") or ["electronics"]),
            "transaction_type": random.choice(opts.get("transaction_type") or ["purchase"]),
            "payment_method": random.choice(opts.get("payment_method") or ["credit_card"]),
            "browser": random.choice(opts.get("browser") or ["Chrome"]),
            "os": random.choice(opts.get("os") or ["Windows"]),
            # Numeric fields - random within realistic bounds
            "amount": str(round(random.uniform(10.0, 5000.0), 2)),
            "account_age_days": str(random.randint(1, 3650)),
            # Coordinate fields (Houston metro area)
            "lat": str(round(random.uniform(29.5, 30.1), 6)),
            "lon": str(round(random.uniform(-95.8, -95.0), 6)),
            # Timestamp fields
            "timestamp_date": now.strftime("%Y-%m-%d"),
            "timestamp_time": now.strftime("%H:%M"),
            # Boolean fields
            "cvv_provided": random.choice([True, False]),
            "billing_address_match": random.choice([True, False]),
            # Generated IDs
            "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
            "user_id": f"user_{random.randint(1000, 9999)}",
            "ip_address": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
            "user_agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Safari/605.1",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) Firefox/120.0",
            ]),
        }
        self.form_data[project_name] = form_data
        self.prediction_results[project_name] = {"prediction": None, "probability": None, "show": False}

    @rx.event
    def randomize_eta_form(self):
        """Generate random values locally using loaded dropdown options (instant)."""
        import random
        import uuid
        project_name = "Estimated Time of Arrival"
        opts = self.dropdown_options.get(project_name, {})
        now = dt.datetime.now()
        # Generate random coordinates (Houston metro area)
        origin_lat = round(random.uniform(29.5, 30.1), 6)
        origin_lon = round(random.uniform(-95.8, -95.0), 6)
        dest_lat = round(random.uniform(29.5, 30.1), 6)
        dest_lon = round(random.uniform(-95.8, -95.0), 6)
        # Estimate distance based on coordinates
        distance_km = round(abs(origin_lat - dest_lat) * 111 + abs(origin_lon - dest_lon) * 85, 2)
        # Use `or` to handle both missing keys AND empty lists
        form_data = {
            # Dropdown fields - pick random from loaded options
            "driver_id": random.choice(opts.get("driver_id") or ["driver_1000"]),
            "vehicle_id": random.choice(opts.get("vehicle_id") or ["vehicle_100"]),
            "weather": random.choice(opts.get("weather") or ["Clear"]),
            "vehicle_type": random.choice(opts.get("vehicle_type") or ["Sedan"]),
            # Timestamp fields
            "timestamp_date": now.strftime("%Y-%m-%d"),
            "timestamp_time": now.strftime("%H:%M"),
            # Coordinate fields
            "origin_lat": str(origin_lat),
            "origin_lon": str(origin_lon),
            "destination_lat": str(dest_lat),
            "destination_lon": str(dest_lon),
            # Numeric fields
            "hour_of_day": str(random.randint(0, 23)),
            "day_of_week": str(random.randint(0, 6)),
            "driver_rating": str(round(random.uniform(3.5, 5.0), 1)),
            "temperature_celsius": str(round(random.uniform(15.0, 35.0), 1)),
            "estimated_distance_km": str(distance_km),
            "initial_estimated_travel_time_seconds": str(int(distance_km * 60)),
            # Debug factors
            "debug_traffic_factor": str(round(random.uniform(0.8, 1.5), 2)),
            "debug_weather_factor": str(round(random.uniform(0.9, 1.3), 2)),
            "debug_incident_delay_seconds": str(random.choice([0, 0, 0, 60, 120, 300])),
            "debug_driver_factor": str(round(random.uniform(0.9, 1.1), 2)),
            # Generated ID
            "trip_id": f"trip_{uuid.uuid4().hex[:12]}",
        }
        self.form_data[project_name] = form_data
        self.prediction_results[project_name] = {"eta_seconds": 0.0, "show": False}

    @rx.event
    def randomize_ecci_form(self):
        """Generate random values locally using loaded dropdown options (instant)."""
        import random
        import uuid
        project_name = "E-Commerce Customer Interactions"
        opts = self.dropdown_options.get(project_name, {})
        now = dt.datetime.now()
        # Use `or` to handle both missing keys AND empty lists
        form_data = {
            # Dropdown fields - pick random from loaded options
            "browser": random.choice(opts.get("browser") or ["Chrome"]),
            "device_type": random.choice(opts.get("device_type") or ["Desktop"]),
            "os": random.choice(opts.get("os") or ["Windows"]),
            "event_type": random.choice(opts.get("event_type") or ["page_view"]),
            "product_category": random.choice(opts.get("product_category") or ["Electronics"]),
            # Text fields with realistic values
            "product_id": f"prod_{random.randint(1000, 1100)}",
            "referrer_url": random.choice([
                "direct", "google.com", "facebook.com", "instagram.com", "email_campaign"
            ]),
            # Coordinate fields (Houston metro area)
            "lat": str(round(random.uniform(29.5, 30.1), 3)),
            "lon": str(round(random.uniform(-95.8, -95.0), 3)),
            # Numeric fields
            "price": str(round(random.uniform(9.99, 999.99), 2)),
            "quantity": str(random.randint(1, 5)),
            "session_event_sequence": str(random.randint(1, 20)),
            "time_on_page_seconds": str(random.randint(5, 300)),
            # Timestamp fields
            "timestamp_date": now.strftime("%Y-%m-%d"),
            "timestamp_time": now.strftime("%H:%M"),
            # Generated IDs
            "customer_id": f"cust_{uuid.uuid4().hex[:8]}",
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "session_id": f"sess_{uuid.uuid4().hex[:10]}",
            "page_url": f"https://shop.example.com/products/{random.randint(1000, 9999)}",
            "search_query": random.choice(["", "", "laptop", "phone", "headphones", "shoes"]),
        }
        self.form_data[project_name] = form_data
        self.prediction_results[project_name] = {"cluster": 0, "show": False}

    # =========================================================================
    # Batch ML (Scikit-Learn) Event Handlers
    # =========================================================================
    @rx.event
    def set_tab(self, tab_value: str):
        """Set the active tab (incremental_ml or batch_ml)."""
        self.tab_name = tab_value
        # Reset YellowBrick state when switching tabs
        if tab_value == "batch_ml":
            self.yellowbrick_metric_name = ""
            self.yellowbrick_image_base64 = ""
            self.yellowbrick_error = ""

    @rx.event
    def set_yellowbrick_metric_type(self, metric_type: str):
        """Set YellowBrick metric type and reset metric name."""
        self.yellowbrick_metric_type = metric_type
        self.yellowbrick_metric_name = ""
        self.yellowbrick_image_base64 = ""
        self.yellowbrick_error = ""

    @rx.event
    def set_yellowbrick_metric_name(self, metric_name: str):
        """Set YellowBrick metric name."""
        self.yellowbrick_metric_name = metric_name
        if metric_name:
            return State.fetch_yellowbrick_metric("Transaction Fraud Detection")

    @rx.event(background=True)
    async def fetch_yellowbrick_metric(self, project_name: str):
        """Fetch YellowBrick visualization from FastAPI."""
        metric_type = self.yellowbrick_metric_type
        metric_name = self.yellowbrick_metric_name

        if not metric_name:
            async with self:
                self.yellowbrick_image_base64 = ""
                self.yellowbrick_error = ""
            return

        async with self:
            self.yellowbrick_loading = True
            self.yellowbrick_error = ""

        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/yellowbrick_metric",
                json={
                    "project_name": project_name,
                    "metric_type": metric_type,
                    "metric_name": metric_name
                },
                timeout=120.0  # YellowBrick can take time to generate
            )
            result = response.json()

            if "error" in result:
                async with self:
                    self.yellowbrick_image_base64 = ""
                    self.yellowbrick_error = result.get("error", "Unknown error")
                    self.yellowbrick_loading = False
            else:
                async with self:
                    self.yellowbrick_image_base64 = result.get("image_base64", "")
                    self.yellowbrick_error = ""
                    self.yellowbrick_loading = False
        except Exception as e:
            print(f"Error fetching YellowBrick metric: {e}")
            async with self:
                self.yellowbrick_image_base64 = ""
                self.yellowbrick_error = str(e)
                self.yellowbrick_loading = False

    @rx.event(background=True)
    async def toggle_batch_ml_training(self, project_name: str):
        """Toggle batch ML training on/off using subprocess (like incremental ML)."""
        import asyncio

        async with self:
            current_state = self.batch_ml_state.get(project_name, False)
            model_key = self.batch_ml_model_key.get(project_name, "")

        if current_state:
            # Turn OFF - stop training
            try:
                response = await httpx_client_post(
                    url=f"{SKLEARN_BASE_URL}/switch_model",
                    json={"model_key": "none"},
                    timeout=30.0
                )
                result = response.json()
                print(f"Batch training stopped: {result}")

                async with self:
                    self.batch_ml_state = {
                        **self.batch_ml_state,
                        project_name: False
                    }
                    self.batch_training_loading = False

                yield rx.toast.info(
                    "Training Stopped",
                    description=f"Batch training for {project_name} has been stopped",
                    duration=3000
                )
            except Exception as e:
                print(f"Error stopping batch training: {e}")
                yield rx.toast.error("Failed to stop training", description=str(e)[:100])
        else:
            # Turn ON - start training and poll for completion
            try:
                response = await httpx_client_post(
                    url=f"{SKLEARN_BASE_URL}/switch_model",
                    json={"model_key": model_key},
                    timeout=30.0
                )
                result = response.json()
                print(f"Batch training started: {result}")

                async with self:
                    self.batch_ml_state = {
                        **self.batch_ml_state,
                        project_name: True
                    }
                    self.batch_training_loading = True
                    self.batch_training_error = ""

                yield rx.toast.info(
                    "Training Started",
                    description=f"Batch training for {project_name} has started",
                    duration=3000
                )

                # Poll for completion
                max_polls = 300  # 10 minutes max (2s interval)
                for _ in range(max_polls):
                    await asyncio.sleep(2)

                    try:
                        status_response = await httpx_client_get(
                            url=f"{SKLEARN_BASE_URL}/batch_status",
                            timeout=10.0
                        )
                        status_result = status_response.json()
                        status = status_result.get("status", "")

                        if status == "completed":
                            # Training finished successfully
                            async with self:
                                self.batch_ml_state = {
                                    **self.batch_ml_state,
                                    project_name: False
                                }
                                self.batch_training_loading = False
                                self.batch_model_available = {
                                    **self.batch_model_available,
                                    project_name: True
                                }
                                self.batch_last_trained = {
                                    **self.batch_last_trained,
                                    project_name: status_result.get("completed_at", "")
                                }

                            # Fetch metrics from MLflow
                            await self.check_batch_model_available(project_name)

                            yield rx.toast.success(
                                "Training Complete",
                                description=f"Model for {project_name} trained successfully",
                                duration=5000
                            )
                            return

                        elif status == "failed":
                            # Training failed
                            exit_code = status_result.get("exit_code", -1)
                            async with self:
                                self.batch_ml_state = {
                                    **self.batch_ml_state,
                                    project_name: False
                                }
                                self.batch_training_loading = False
                                self.batch_training_error = f"Training failed (exit code: {exit_code})"

                            yield rx.toast.error(
                                "Training Failed",
                                description=f"Exit code: {exit_code}",
                                duration=5000
                            )
                            return

                        elif status != "running":
                            # Unknown status or idle - stop polling
                            async with self:
                                self.batch_ml_state = {
                                    **self.batch_ml_state,
                                    project_name: False
                                }
                                self.batch_training_loading = False
                            return

                    except Exception as poll_error:
                        print(f"Error polling batch status: {poll_error}")
                        # Continue polling on transient errors

                # Timeout - training took too long
                async with self:
                    self.batch_ml_state = {
                        **self.batch_ml_state,
                        project_name: False
                    }
                    self.batch_training_loading = False
                    self.batch_training_error = "Training timeout"

                yield rx.toast.error(
                    "Training Timeout",
                    description="Training took too long",
                    duration=5000
                )

            except Exception as e:
                print(f"Error starting batch training: {e}")
                async with self:
                    self.batch_ml_state = {
                        **self.batch_ml_state,
                        project_name: False
                    }
                    self.batch_training_loading = False
                    self.batch_training_error = str(e)
                yield rx.toast.error(
                    "Failed to Start Training",
                    description=str(e)[:100],
                    duration=5000
                )

    @rx.event(background=True)
    async def check_batch_model_available(self, project_name: str):
        """Check if a trained batch model is available in MLflow."""
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/model_available",
                json={
                    "project_name": project_name,
                    "model_name": "XGBClassifier"
                },
                timeout=30.0
            )
            result = response.json()

            async with self:
                self.batch_model_available = {
                    **self.batch_model_available,
                    project_name: result.get("available", False)
                }
                if result.get("available"):
                    self.batch_last_trained = {
                        **self.batch_last_trained,
                        project_name: result.get("trained_at", "")
                    }
                    # Store metrics from model_available response
                    metrics = result.get("metrics", {})
                    if metrics:
                        self.batch_training_metrics = {
                            **self.batch_training_metrics,
                            project_name: metrics
                        }
        except Exception as e:
            print(f"Error checking batch model availability: {e}")
            async with self:
                self.batch_model_available = {
                    **self.batch_model_available,
                    project_name: False
                }

    # Incremental model name mapping for MLflow
    _incremental_model_names: dict = {
        "Transaction Fraud Detection": "ARFClassifier",
        "Estimated Time of Arrival": "ARFRegressor",
        "E-Commerce Customer Interactions": "DBSTREAM",
    }

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
                self.incremental_model_available = {
                    **self.incremental_model_available,
                    project_name: result.get("available", False)
                }
                if result.get("available"):
                    self.incremental_model_last_trained = {
                        **self.incremental_model_last_trained,
                        project_name: result.get("trained_at", "")
                    }
                # Store experiment URL (available even if no model trained yet)
                experiment_url = result.get("experiment_url", "")
                if experiment_url:
                    self.mlflow_experiment_url = {
                        **self.mlflow_experiment_url,
                        project_name: experiment_url
                    }
        except Exception as e:
            print(f"Error checking incremental model availability: {e}")
            async with self:
                self.incremental_model_available = {
                    **self.incremental_model_available,
                    project_name: False
                }

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
                self.incremental_model_available = {
                    **self.incremental_model_available,
                    project_name: model_avail.get("available", False)
                }
                if model_avail.get("available"):
                    self.incremental_model_last_trained = {
                        **self.incremental_model_last_trained,
                        project_name: model_avail.get("trained_at", "")
                    }
                experiment_url = model_avail.get("experiment_url", "")
                if experiment_url:
                    self.mlflow_experiment_url = {
                        **self.mlflow_experiment_url,
                        project_name: experiment_url
                    }

                # MLflow metrics
                metrics = data.get("mlflow_metrics", {})
                if metrics:
                    self.mlflow_metrics = {
                        **self.mlflow_metrics,
                        project_name: metrics
                    }

                # Initial sample for form fields
                sample = data.get("initial_sample", {})
                if sample:
                    self.incremental_ml_sample = {
                        **self.incremental_ml_sample,
                        project_name: sample
                    }
                    # Update form data with sample values
                    self._update_form_from_sample(project_name, sample)

                # Dropdown options
                options = data.get("dropdown_options", {})
                if options:
                    self.dropdown_options = {
                        **self.dropdown_options,
                        project_name: options
                    }

                # Training status
                training = data.get("training_status", {})
                if training.get("is_training"):
                    self.activated_model = model_key
                    self.ml_training_enabled = True

        except Exception as e:
            print(f"Error in init_page for {project_name}: {e}")
            async with self:
                self.incremental_model_available = {
                    **self.incremental_model_available,
                    project_name: False
                }

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

    @rx.event(background=True)
    async def predict_batch_tfd(self):
        """Make batch ML prediction for Transaction Fraud Detection using XGBClassifier."""
        project_name = "Transaction Fraud Detection"
        current_form = self.form_data.get(project_name, {})

        # Combine date and time
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"

        # Prepare request payload
        payload = {
            "project_name": project_name,
            "model_name": "XGBClassifier",
            "transaction_id": current_form.get("transaction_id", ""),
            "user_id": current_form.get("user_id", ""),
            "timestamp": timestamp,
            "amount": float(current_form.get("amount", 0)),
            "currency": current_form.get("currency", ""),
            "merchant_id": current_form.get("merchant_id", ""),
            "product_category": current_form.get("product_category", ""),
            "transaction_type": current_form.get("transaction_type", ""),
            "payment_method": current_form.get("payment_method", ""),
            "location": {
                "lat": float(current_form.get("lat", 0)),
                "lon": float(current_form.get("lon", 0))
            },
            "ip_address": current_form.get("ip_address", ""),
            "device_info": {
                "os": current_form.get("os", ""),
                "browser": current_form.get("browser", "")
            },
            "user_agent": current_form.get("user_agent", ""),
            "account_age_days": int(current_form.get("account_age_days", 0)),
            "cvv_provided": bool(current_form.get("cvv_provided", False)),
            "billing_address_match": bool(current_form.get("billing_address_match", False))
        }

        try:
            print(f"Making batch prediction with payload: {payload}")
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/predict",
                json=payload,
                timeout=30.0
            )
            result = response.json()
            print(f"Batch prediction result: {result}")

            async with self:
                self.batch_prediction_results = {
                    **self.batch_prediction_results,
                    project_name: {
                        "prediction": result.get("prediction"),
                        "fraud_probability": result.get("fraud_probability"),
                        "show": True
                    }
                }
        except Exception as e:
            print(f"Error making batch prediction: {e}")
            async with self:
                self.batch_prediction_results = {
                    **self.batch_prediction_results,
                    project_name: {
                        "prediction": None,
                        "fraud_probability": 0.0,
                        "show": False
                    }
                }
            yield rx.toast.error(
                "Prediction failed",
                description=str(e),
                duration=5000
            )

    @rx.event(background=True)
    async def get_batch_mlflow_metrics(self, project_name: str):
        """Fetch batch ML MLflow metrics for a project."""
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": "XGBClassifier"
                },
                timeout=60.0
            )
            async with self:
                self.batch_mlflow_metrics = {
                    **self.batch_mlflow_metrics,
                    project_name: response.json()
                }
        except Exception as e:
            print(f"Error fetching batch MLflow metrics: {e}")
            async with self:
                self.batch_mlflow_metrics = {
                    **self.batch_mlflow_metrics,
                    project_name: {}
                }

    @rx.event(background=True)
    async def refresh_batch_mlflow_metrics(self, project_name: str):
        """Force refresh batch ML MLflow metrics bypassing cache."""
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": "XGBClassifier",
                    "force_refresh": True
                },
                timeout=60.0
            )
            async with self:
                self.batch_mlflow_metrics = {
                    **self.batch_mlflow_metrics,
                    project_name: response.json()
                }
            yield rx.toast.success(
                "Metrics refreshed",
                description=f"Latest batch ML metrics loaded for {project_name}",
                duration=2000
            )
        except Exception as e:
            print(f"Error refreshing batch MLflow metrics: {e}")
            yield rx.toast.error(
                "Refresh failed",
                description=str(e),
                duration=3000
            )

    # ==========================================================================
    # DELTA LAKE SQL TAB EVENTS
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
        default_query = "SELECT * FROM data ORDER BY timestamp DESC LIMIT 100"
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
        print(f"[DEBUG] set_sql_engine - project_name: {project_name}, engine: {engine}")
        self.sql_engine = {
            **self.sql_engine,
            project_name: engine
        }
        print(f"[DEBUG] set_sql_engine - sql_engine dict after update: {self.sql_engine}")

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
        """Toggle sorting for a column. Click once for asc, again for desc, third time to clear."""
        project_name = self.page_name
        current_col = self.sql_sort_column.get(project_name, "")
        current_dir = self.sql_sort_direction.get(project_name, "asc")

        if current_col != column:
            # New column - start with ascending
            new_col = column
            new_dir = "asc"
        elif current_dir == "asc":
            # Same column, was ascending - switch to descending
            new_col = column
            new_dir = "desc"
        else:
            # Same column, was descending - clear sort
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
        # Read state values inside async context to get latest values
        async with self:
            project_name = self.page_name
            query = self.sql_query_input.get(project_name, "")
            engine = self.sql_engine.get(project_name, "polars")
            # Debug: log state values
            print(f"[DEBUG] execute_sql_query - project_name: {project_name}")
            print(f"[DEBUG] execute_sql_query - sql_engine dict: {self.sql_engine}")
            print(f"[DEBUG] execute_sql_query - selected engine: {engine}")

        if not query.strip():
            yield rx.toast.warning(
                "Empty query",
                description="Please enter a SQL query to execute",
                duration=3000
            )
            return

        # Set loading state
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
                timeout=65.0  # Slightly more than server timeout
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
            # Try to extract error from response if available
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