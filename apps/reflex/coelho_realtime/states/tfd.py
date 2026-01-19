"""
Transaction Fraud Detection (TFD) state module.

This module contains:
- TFDState class with TFD-specific state variables and methods
- TFD form handlers
- TFD prediction methods
- TFD computed variables (incremental ML and batch ML)
"""
import reflex as rx
import datetime as dt
import plotly.graph_objects as go
from .shared import (
    SharedState,
    RIVER_BASE_URL,
    SKLEARN_BASE_URL,
    safe_str,
    safe_int_str,
    safe_float_str,
    safe_bool,
    get_str,
)
from ..utils import httpx_client_post


class TFDState(SharedState):
    """Transaction Fraud Detection state.

    Inherits from SharedState to access common state variables (form_data,
    prediction_results, mlflow_metrics, etc.) while adding TFD-specific
    computed vars and event handlers.
    """

    # ==========================================================================
    # TFD BATCH ML STATE
    # ==========================================================================
    batch_ml_model_name: dict = {
        "Transaction Fraud Detection": "XGBoost Classifier (Scikit-Learn)",
    }
    # YellowBrick visualization state
    yellowbrick_metric_type: str = "Classification"
    yellowbrick_metric_name: str = "Select visualization..."
    yellowbrick_image_base64: str = ""
    yellowbrick_loading: bool = False
    yellowbrick_error: str = ""
    _yellowbrick_cancel_requested: bool = False
    # Detailed metrics options for YellowBrick
    yellowbrick_metrics_options: dict[str, list[str]] = {
        "Classification": [
            "Select visualization...",
            "ClassificationReport",
            "ConfusionMatrix",
            "ROCAUC",
            "PrecisionRecallCurve",
            "ClassPredictionError"
        ],
        "Feature Analysis": [
            "Select visualization...",
            "Rank1D",
            "Rank2D",
            "PCA",
            "Manifold",
            "ParallelCoordinates",
            "RadViz",
            "JointPlot",
        ],
        "Target": [
            "Select visualization...",
            "ClassBalance",  # ESSENTIAL: Class distribution & imbalance detection
            "FeatureCorrelation",  # HIGH: Mutual info correlation (non-linear)
            "FeatureCorrelation_Pearson",  # Linear correlation
            "BalancedBinningReference",  # For regression binning (skip for TFD)
        ],
        "Model Selection": [
            "Select visualization...",
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

    # ==========================================================================
    # TFD FORM FIELD TYPE MAPPINGS (for automatic conversion)
    # ==========================================================================
    _tfd_float_fields = {"amount", "lat", "lon"}
    _tfd_int_fields = {"account_age_days"}
    _tfd_bool_fields = {"cvv_provided", "billing_address_match"}

    @staticmethod
    def _safe_bool(value) -> bool:
        """Safely convert value to bool, handling string 'false'/'true'."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    # ==========================================================================
    # TFD INCREMENTAL ML COMPUTED VARIABLES
    # ==========================================================================
    @rx.var
    def tfd_form_data(self) -> dict:
        """Get Transaction Fraud Detection form data."""
        return self.form_data.get("Transaction Fraud Detection", {})

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

    # ==========================================================================
    # TFD BATCH ML COMPUTED VARIABLES
    # ==========================================================================
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
        return self.yellowbrick_metrics_options.get(self.yellowbrick_metric_type, ["Select visualization..."])

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
        """Get all TFD batch ML metrics with appropriate formatting.

        - Percentage format (0-100%): classification metrics in [0, 1] range
        - Decimal format: loss metrics and other unbounded values
        """
        raw_metrics = self.batch_mlflow_metrics.get("Transaction Fraud Detection", {})
        if not isinstance(raw_metrics, dict):
            return {}
        # Metrics that should NOT be formatted as percentages (loss/calibration metrics)
        non_percentage_metrics = {
            "log_loss", "brier_score_loss", "d2_log_loss_score", "d2_brier_score",
            "preprocessing_time_seconds"
        }
        result = {}
        for key, value in raw_metrics.items():
            if key.startswith("metrics."):
                metric_name = key.replace("metrics.", "")
                if isinstance(value, (int, float)):
                    if metric_name in non_percentage_metrics:
                        # Format as decimal for loss/calibration metrics
                        result[metric_name] = f"{value:.4f}"
                    else:
                        # Format as percentage for classification metrics
                        result[metric_name] = f"{value * 100:.2f}%"
                else:
                    result[metric_name] = str(value) if value is not None else "N/A"
        return result

    @rx.var
    def tfd_batch_metric_names(self) -> list[str]:
        """Get list of batch ML metric names for TFD."""
        return list(self.tfd_batch_metrics.keys())

    # ==========================================================================
    # TFD INCREMENTAL ML EVENT HANDLERS
    # ==========================================================================
    @rx.event
    def update_tfd(self, field: str, value):
        """Update a TFD form field with automatic type conversion."""
        try:
            if field in self._tfd_float_fields:
                value = float(value) if value else 0.0
            elif field in self._tfd_int_fields:
                value = int(value) if value else 0
            elif field in self._tfd_bool_fields:
                value = self._safe_bool(value)
            # str fields need no conversion
        except (ValueError, TypeError):
            return  # Ignore invalid conversions
        if "Transaction Fraud Detection" not in self.form_data:
            self.form_data["Transaction Fraud Detection"] = {}
        self.form_data["Transaction Fraud Detection"][field] = value

    @rx.event(background=True)
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
            "cvv_provided": self._safe_bool(current_form.get("cvv_provided", False)),
            "billing_address_match": self._safe_bool(current_form.get("billing_address_match", False))
        }
        # Make prediction
        try:
            print(f"Making prediction with payload: {payload}")
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/predict",
                json=payload,
                timeout=30.0
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

    # ==========================================================================
    # TFD BATCH ML EVENT HANDLERS
    # ==========================================================================
    @rx.event
    def set_yellowbrick_metric_type(self, metric_type: str):
        """Set YellowBrick metric type and reset metric name."""
        self.yellowbrick_metric_type = metric_type
        self.yellowbrick_metric_name = "Select visualization..."
        self.yellowbrick_image_base64 = ""
        self.yellowbrick_error = ""

    @rx.event
    def clear_yellowbrick_visualization(self):
        """Clear YellowBrick visualization state (called on tab change)."""
        self.yellowbrick_image_base64 = ""
        self.yellowbrick_error = ""
        self.yellowbrick_metric_name = "Select visualization..."

    @rx.event
    def set_yellowbrick_metric_name(self, metric_name: str):
        """Set YellowBrick metric name."""
        self.yellowbrick_metric_name = metric_name
        if metric_name and metric_name != "Select visualization...":
            return TFDState.fetch_yellowbrick_metric("Transaction Fraud Detection")

    @rx.event
    def set_yellowbrick_visualization(self, category: str, metric_name: str):
        """Unified handler for all YellowBrick visualization categories.

        Args:
            category: The YellowBrick category (e.g., "Feature Analysis", "Target", etc.)
            metric_name: The visualization name to display
        """
        self.yellowbrick_metric_type = category
        self.yellowbrick_metric_name = metric_name
        if metric_name and metric_name != "Select visualization...":
            return TFDState.fetch_yellowbrick_metric("Transaction Fraud Detection")

    @rx.event(background=True)
    async def fetch_yellowbrick_metric(self, project_name: str):
        """Fetch YellowBrick visualization from FastAPI using selected MLflow run."""
        metric_type = self.yellowbrick_metric_type
        metric_name = self.yellowbrick_metric_name
        # Use selected run_id from SharedState (or None for best)
        run_id = self.selected_batch_run.get(project_name) or None

        if not metric_name or metric_name == "Select visualization...":
            async with self:
                self.yellowbrick_image_base64 = ""
                self.yellowbrick_error = ""
            return

        async with self:
            self.yellowbrick_loading = True
            self.yellowbrick_error = ""
            self.yellowbrick_image_base64 = ""  # Clear old image while loading new one
            self._yellowbrick_cancel_requested = False

        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/yellowbrick_metric",
                json={
                    "project_name": project_name,
                    "metric_type": metric_type,
                    "metric_name": metric_name,
                    "run_id": run_id,  # Use selected run's data
                },
                timeout=300.0  # 5 minutes for slow visualizations like Manifold
            )
            # Check if cancelled before updating UI
            if self._yellowbrick_cancel_requested:
                return
            result = response.json()
            async with self:
                self.yellowbrick_image_base64 = result.get("image_base64", "")
                self.yellowbrick_loading = False
                self.yellowbrick_error = result.get("error", "")
        except Exception as e:
            if self._yellowbrick_cancel_requested:
                return
            print(f"Error fetching YellowBrick metric: {e}")
            async with self:
                self.yellowbrick_loading = False
                self.yellowbrick_error = str(e)
                self.yellowbrick_image_base64 = ""

    @rx.event
    def cancel_yellowbrick_loading(self):
        """Cancel the current YellowBrick visualization loading."""
        self._yellowbrick_cancel_requested = True
        self.yellowbrick_loading = False
        self.yellowbrick_error = ""
        self.yellowbrick_image_base64 = ""
        self.yellowbrick_metric_name = "Select visualization..."
        yield rx.toast.info(
            "Visualization cancelled",
            description="Loading has been stopped.",
            duration=2000
        )

    @rx.event(background=True)
    async def check_batch_model_available(self, project_name: str):
        """Check if a batch (Scikit-Learn) model is available for prediction."""
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/model_available",
                json={
                    "project_name": project_name,
                    "model_name": "CatBoostClassifier"
                },
                timeout=30.0
            )
            result = response.json()
            async with self:
                self.batch_model_available[project_name] = result.get("available", False)
                if result.get("available"):
                    self.batch_last_trained[project_name] = result.get("trained_at", "")
        except Exception as e:
            print(f"Error checking batch model availability: {e}")
            async with self:
                self.batch_model_available[project_name] = False

    @rx.event(background=True)
    async def predict_batch_tfd(self):
        """Make batch prediction for TFD using Scikit-Learn model from selected run."""
        project_name = "Transaction Fraud Detection"
        current_form = self.form_data.get(project_name, {})
        # Use selected run_id from SharedState (or None for best)
        run_id = self.selected_batch_run.get(project_name) or None
        # Combine date and time
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"
        # Prepare request payload from current form state
        payload = {
            "project_name": project_name,
            "model_name": "CatBoostClassifier",
            "run_id": run_id,  # Use selected run's model
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
            "cvv_provided": self._safe_bool(current_form.get("cvv_provided", False)),
            "billing_address_match": self._safe_bool(current_form.get("billing_address_match", False))
        }
        # Make prediction
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
                        "model_source": result.get("model_source", "mlflow"),
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
                        "model_source": "mlflow",
                        "show": False
                    }
                }

    @rx.event(background=True)
    async def get_batch_mlflow_metrics(self, project_name: str):
        """Fetch batch MLflow metrics for TFD."""
        try:
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": "CatBoostClassifier"
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
