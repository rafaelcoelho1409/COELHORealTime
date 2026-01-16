"""
Transaction Fraud Detection (TFD) components module.

This module contains all TFD-specific UI components:
- transaction_fraud_detection_form() - Incremental ML form
- transaction_fraud_detection_batch_form() - Batch ML form
- transaction_fraud_detection_metrics() - Metrics dashboard
- TFD-specific helper components

All components use TFDState for TFD-specific state.
"""
import reflex as rx
from ..states import TFDState, SharedState, METRIC_INFO
from .shared import metric_info_dialog, ml_training_switch, batch_ml_training_box


# =============================================================================
# TFD Card Helper Functions
# =============================================================================
def metric_card(label: str, value_var, metric_key: str = None, project_key: str = "tfd") -> rx.Component:
    """Create a compact styled metric card with optional info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    label,
                    size="1",
                    weight="medium",
                    color="gray"
                ),
                metric_info_dialog(metric_key, project_key) if metric_key else rx.fragment(),
                spacing="1",
                align="center",
                justify="center"
            ),
            rx.text(
                value_var,
                size="4",
                weight="bold",
                align="center"
            ),
            spacing="1",
            align_items="center",
            justify="center",
            height="100%"
        ),
        variant="surface",
        size="1"
    )


def mlflow_run_info_badge(project_name: str) -> rx.Component:
    """Display MLflow experiment run info (run_id, status, start_time) for a project."""
    run_info = TFDState.mlflow_run_info[project_name]
    return rx.hstack(
        # MLflow source badge
        rx.badge(
            rx.hstack(
                rx.icon("flask-conical", size=12),
                rx.text("MLflow", size="1", weight="bold"),
                spacing="1",
                align="center"
            ),
            color_scheme="purple",
            variant="soft"
        ),
        # Status badge with conditional styling
        rx.cond(
            run_info["is_live"],
            rx.badge(
                rx.hstack(
                    rx.box(
                        width="8px",
                        height="8px",
                        border_radius="50%",
                        background_color="green",
                        class_name="animate-pulse"
                    ),
                    rx.text("LIVE", size="1"),
                    spacing="1",
                    align="center"
                ),
                color_scheme="green",
                variant="surface"
            ),
            rx.badge(
                run_info["status"],
                color_scheme=rx.cond(
                    run_info["status"] == "FINISHED",
                    "blue",
                    "gray"
                ),
                variant="surface"
            )
        ),
        # Run ID
        rx.cond(
            run_info["run_id"] != "",
            rx.hstack(
                rx.text("Run:", size="1", color="gray"),
                rx.code(run_info["run_id"], size="1"),
                spacing="1",
                align="center"
            ),
            rx.fragment()
        ),
        # Start time
        rx.cond(
            run_info["start_time"] != "",
            rx.hstack(
                rx.text("Started:", size="1", color="gray"),
                rx.text(run_info["start_time"], size="1"),
                spacing="1",
                align="center"
            ),
            rx.fragment()
        ),
        spacing="3",
        align="center",
        padding="2"
    )


def kpi_card_with_info(plotly_key: str, metric_key: str) -> rx.Component:
    """Create a KPI card with Plotly chart and info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.spacer(),
                metric_info_dialog(metric_key, "tfd"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=TFDState.tfd_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1"
    )


def gauge_card_with_info(plotly_key: str, metric_key: str) -> rx.Component:
    """Create a gauge card with Plotly chart and info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.spacer(),
                metric_info_dialog(metric_key, "tfd"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=TFDState.tfd_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1",
        width="50%"
    )


def heatmap_card_with_info(plotly_key: str, metric_key: str) -> rx.Component:
    """Create a heatmap card with Plotly chart and info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.spacer(),
                metric_info_dialog(metric_key, "tfd"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=TFDState.tfd_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1",
        width="50%"
    )


# =============================================================================
# TFD Metrics Dashboard
# =============================================================================
def transaction_fraud_detection_metrics() -> rx.Component:
    """Display MLflow classification metrics for TFD with Plotly dashboard layout."""
    return rx.vstack(
        # Run info badge
        mlflow_run_info_badge("Transaction Fraud Detection"),
        # ROW 1: KPI Indicators (primary metrics with delta from baseline)
        rx.grid(
            kpi_card_with_info("kpi_fbeta", "fbeta"),
            kpi_card_with_info("kpi_rocauc", "rocauc"),
            kpi_card_with_info("kpi_precision", "precision"),
            kpi_card_with_info("kpi_recall", "recall"),
            kpi_card_with_info("kpi_rolling_rocauc", "rolling_rocauc"),
            columns="5",
            spacing="2",
            width="100%"
        ),
        # ROW 2: Additional metrics (text cards with info buttons)
        rx.grid(
            metric_card("F1", TFDState.tfd_metrics["f1"], "f1"),
            metric_card("Accuracy", TFDState.tfd_metrics["accuracy"], "accuracy"),
            metric_card("Geo Mean", TFDState.tfd_metrics["geometric_mean"], "geometric_mean"),
            metric_card("Cohen k", TFDState.tfd_metrics["cohen_kappa"], "cohen_kappa"),
            metric_card("Jaccard", TFDState.tfd_metrics["jaccard"], "jaccard"),
            metric_card("LogLoss", TFDState.tfd_metrics["logloss"], "logloss"),
            columns="6",
            spacing="2",
            width="100%"
        ),
        # ROW 3: Gauges (secondary metrics)
        rx.hstack(
            gauge_card_with_info("gauge_mcc", "mcc"),
            gauge_card_with_info("gauge_balanced_accuracy", "balanced_accuracy"),
            spacing="2",
            width="100%"
        ),
        # ROW 4: Confusion Matrix + Classification Report (side by side)
        rx.hstack(
            heatmap_card_with_info("confusion_matrix", "confusion_matrix"),
            heatmap_card_with_info("classification_report", "classification_report"),
            spacing="2",
            width="100%"
        ),
        spacing="3",
        width="100%"
    )


def transaction_fraud_detection_batch_metrics() -> rx.Component:
    """Display batch ML metrics for TFD."""
    return rx.grid(
        metric_card("F1", TFDState.tfd_batch_metrics.get("F1", "N/A"), "f1"),
        metric_card("Accuracy", TFDState.tfd_batch_metrics.get("Accuracy", "N/A"), "accuracy"),
        metric_card("Recall", TFDState.tfd_batch_metrics.get("Recall", "N/A"), "recall"),
        metric_card("Precision", TFDState.tfd_batch_metrics.get("Precision", "N/A"), "precision"),
        metric_card("ROC AUC", TFDState.tfd_batch_metrics.get("ROCAUC", "N/A"), "rocauc"),
        metric_card("Geo Mean", TFDState.tfd_batch_metrics.get("GeometricMean", "N/A"), "geometric_mean"),
        columns="6",
        spacing="2",
        width="100%"
    )


# =============================================================================
# TFD Incremental ML Form
# =============================================================================
def transaction_fraud_detection_form(model_key: str = None, project_name: str = None) -> rx.Component:
    """Incremental ML form for Transaction Fraud Detection using ARFClassifier."""
    # Build form card with 3-column layout for compact display
    form_card = rx.card(
        rx.vstack(
            # Form Legend
            rx.hstack(
                rx.icon("credit-card", size=20, color=rx.color("accent", 10)),
                rx.heading("Transaction Details", size="4", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            # Predict button at TOP for visibility
            rx.button(
                "Predict",
                on_click=TFDState.predict_transaction_fraud_detection,
                size="2",
                width="100%",
                disabled=~TFDState.incremental_model_available["Transaction Fraud Detection"]
            ),
            # Randomize button below Predict
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size=14),
                    rx.text("Randomize All Fields", size="2"),
                    spacing="1",
                    align_items="center"
                ),
                on_click=TFDState.randomize_tfd_form,
                variant="soft",
                color_scheme="blue",
                size="2",
                width="100%"
            ),
            rx.divider(),
            # Form fields in 3-column grid
            rx.grid(
                # Amount
                rx.vstack(
                    rx.text("Amount", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("amount", ""),
                        on_change=lambda v: TFDState.update_tfd("amount", v),
                        step=0.01,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Account Age
                rx.vstack(
                    rx.text("Account Age", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("account_age_days", ""),
                        on_change=lambda v: TFDState.update_tfd("account_age_days", v),
                        min=0,
                        step=1,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Currency
                rx.vstack(
                    rx.text("Currency", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["currency"],
                        value=TFDState.tfd_form_data.get("currency", ""),
                        on_change=lambda v: TFDState.update_tfd("currency", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Date
                rx.vstack(
                    rx.text("Date", size="1", color="gray"),
                    rx.input(
                        type="date",
                        value=TFDState.tfd_form_data.get("timestamp_date", ""),
                        on_change=lambda v: TFDState.update_tfd("timestamp_date", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Time
                rx.vstack(
                    rx.text("Time", size="1", color="gray"),
                    rx.input(
                        type="time",
                        value=TFDState.tfd_form_data.get("timestamp_time", ""),
                        on_change=lambda v: TFDState.update_tfd("timestamp_time", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Merchant ID
                rx.vstack(
                    rx.text("Merchant ID", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["merchant_id"],
                        value=TFDState.tfd_form_data.get("merchant_id", ""),
                        on_change=lambda v: TFDState.update_tfd("merchant_id", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Product Category
                rx.vstack(
                    rx.text("Category", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["product_category"],
                        value=TFDState.tfd_form_data.get("product_category", ""),
                        on_change=lambda v: TFDState.update_tfd("product_category", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Transaction Type
                rx.vstack(
                    rx.text("Type", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["transaction_type"],
                        value=TFDState.tfd_form_data.get("transaction_type", ""),
                        on_change=lambda v: TFDState.update_tfd("transaction_type", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Payment Method
                rx.vstack(
                    rx.text("Payment", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["payment_method"],
                        value=TFDState.tfd_form_data.get("payment_method", ""),
                        on_change=lambda v: TFDState.update_tfd("payment_method", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Latitude
                rx.vstack(
                    rx.text("Latitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("lat", ""),
                        on_change=lambda v: TFDState.update_tfd("lat", v),
                        min=-90.0,
                        max=90.0,
                        step=0.0001,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Longitude
                rx.vstack(
                    rx.text("Longitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("lon", ""),
                        on_change=lambda v: TFDState.update_tfd("lon", v),
                        min=-180.0,
                        max=180.0,
                        step=0.0001,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Browser
                rx.vstack(
                    rx.text("Browser", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["browser"],
                        value=TFDState.tfd_form_data.get("browser", ""),
                        on_change=lambda v: TFDState.update_tfd("browser", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # OS
                rx.vstack(
                    rx.text("OS", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["os"],
                        value=TFDState.tfd_form_data.get("os", ""),
                        on_change=lambda v: TFDState.update_tfd("os", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # CVV Provided (with label for alignment)
                rx.vstack(
                    rx.text("CVV", size="1", color="gray"),
                    rx.checkbox(
                        "Provided",
                        checked=TFDState.tfd_form_data.get("cvv_provided", False),
                        on_change=lambda v: TFDState.update_tfd("cvv_provided", v),
                        size="1"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Billing Address Match (with label for alignment)
                rx.vstack(
                    rx.text("Billing", size="1", color="gray"),
                    rx.checkbox(
                        "Address Match",
                        checked=TFDState.tfd_form_data.get("billing_address_match", False),
                        on_change=lambda v: TFDState.update_tfd("billing_address_match", v),
                        size="1"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            # Display fields (read-only info stacked vertically)
            rx.vstack(
                rx.text(
                    f"Transaction ID: {TFDState.tfd_form_data.get('transaction_id', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"User ID: {TFDState.tfd_form_data.get('user_id', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"IP Address: {TFDState.tfd_form_data.get('ip_address', '')}",
                    size="1",
                    color="gray"
                ),
                spacing="1",
                align_items="start",
                width="100%"
            ),
            spacing="2",
            align_items="start",
            width="100%"
        ),
        width="100%"
    )

    # Build left column with optional ML training switch
    if model_key and project_name:
        left_column = rx.vstack(
            ml_training_switch(model_key, project_name),
            form_card,
            spacing="4",
            width="30%"
        )
    else:
        left_column = rx.vstack(
            form_card,
            spacing="4",
            width="30%"
        )

    # Right column - Tabs for Prediction and Metrics
    right_column = rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("target", size=14),
                        rx.text("Prediction"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="prediction"
                ),
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("chart-bar", size=14),
                        rx.text("Metrics"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="metrics"
                ),
            ),
            # Tab 1: Prediction
            rx.tabs.content(
                rx.vstack(
                    # Prediction section - header always visible
                    rx.hstack(
                        rx.icon("shield-alert", size=20, color=rx.color("accent", 10)),
                        rx.heading("Prediction Result", size="5", weight="bold"),
                        spacing="2",
                        align_items="center",
                        width="100%"
                    ),
                    # Model info row
                    rx.hstack(
                        rx.badge(
                            rx.hstack(
                                rx.icon("brain", size=12),
                                rx.text("RandomUnderSampler + ARFClassifier", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            color_scheme="blue",
                            variant="soft",
                            size="1"
                        ),
                        rx.badge("Imbalanced Learning", color_scheme="purple", variant="soft", size="1"),
                        spacing="2"
                    ),
                    # MLflow run info (LIVE/FINISHED status)
                    mlflow_run_info_badge("Transaction Fraud Detection"),
                    rx.cond(
                        TFDState.tfd_prediction_show,
                        # Show prediction results when available
                        rx.card(
                            rx.vstack(
                                # Plotly Gauge Chart
                                rx.plotly(data=TFDState.tfd_fraud_gauge, width="100%"),
                                # Prediction summary cards (compact)
                                rx.hstack(
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("triangle-alert", size=14, color=TFDState.tfd_prediction_color),
                                                rx.text("Classification", size="1", color="gray"),
                                                spacing="1",
                                                align_items="center"
                                            ),
                                            rx.text(
                                                TFDState.tfd_prediction_text,
                                                size="5",
                                                weight="bold",
                                                color=TFDState.tfd_prediction_color,
                                                align="center"
                                            ),
                                            spacing="1",
                                            align_items="center",
                                            width="100%"
                                        ),
                                        variant="surface",
                                        size="1",
                                        width="100%"
                                    ),
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("percent", size=14, color="red"),
                                                rx.text("Fraud", size="1", color="gray"),
                                                spacing="1",
                                                align_items="center"
                                            ),
                                            rx.text(
                                                f"{TFDState.tfd_fraud_probability * 100:.2f}%",
                                                size="5",
                                                weight="bold",
                                                align="center",
                                                color="red"
                                            ),
                                            spacing="1",
                                            align_items="center",
                                            width="100%"
                                        ),
                                        variant="surface",
                                        size="1",
                                        width="100%"
                                    ),
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("circle-check", size=14, color="green"),
                                                rx.text("Not Fraud", size="1", color="gray"),
                                                spacing="1",
                                                align_items="center"
                                            ),
                                            rx.text(
                                                f"{(1 - TFDState.tfd_fraud_probability) * 100:.2f}%",
                                                size="5",
                                                weight="bold",
                                                align="center",
                                                color="green"
                                            ),
                                            spacing="1",
                                            align_items="center",
                                            width="100%"
                                        ),
                                        variant="surface",
                                        size="1",
                                        width="100%"
                                    ),
                                    spacing="2",
                                    width="100%"
                                ),
                                spacing="4",
                                width="100%"
                            ),
                            variant="classic",
                            width="100%"
                        ),
                        # Show info or warning message when no prediction yet
                        rx.cond(
                            TFDState.incremental_model_available["Transaction Fraud Detection"],
                            rx.callout(
                                "Fill in the transaction details and click **Predict** to get the fraud probability.",
                                icon="info",
                                color="blue",
                                width="100%"
                            ),
                            rx.callout(
                                "No trained model available. Toggle **Real-time ML Training** to train first.",
                                icon="triangle-alert",
                                color="orange",
                                width="100%"
                            )
                        )
                    ),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="prediction"
            ),
            # Tab 2: Metrics
            rx.tabs.content(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Classification Metrics", size="5"),
                        rx.button(
                            rx.icon("refresh-cw", size=16),
                            on_click=TFDState.refresh_mlflow_metrics("Transaction Fraud Detection"),
                            size="1",
                            variant="ghost",
                            cursor="pointer",
                            title="Refresh metrics"
                        ),
                        align_items="center",
                        spacing="2"
                    ),
                    # Model info row
                    rx.hstack(
                        rx.badge(
                            rx.hstack(
                                rx.icon("brain", size=12),
                                rx.text("RandomUnderSampler + ARFClassifier", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            color_scheme="blue",
                            variant="soft",
                            size="1"
                        ),
                        rx.badge("Imbalanced Learning", color_scheme="purple", variant="soft", size="1"),
                        spacing="2"
                    ),
                    transaction_fraud_detection_metrics(),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="metrics"
            ),
            default_value="prediction",
            width="100%"
        ),
        # NOTE: on_mount removed - init_page already fetches mlflow_metrics on page mount
        align_items="start",
        spacing="4",
        width="70%"
    )

    return rx.hstack(
        left_column,
        right_column,
        spacing="6",
        align_items="start",
        width="100%"
    )


# =============================================================================
# TFD Batch ML Form
# =============================================================================
def transaction_fraud_detection_batch_form(model_key: str = None, project_name: str = None) -> rx.Component:
    """Batch ML form for Transaction Fraud Detection using XGBClassifier.

    Mirrors the Incremental ML form layout with batch_ml_training_box instead of ml_training_switch.
    """
    # Build form card with 3-column layout (same as incremental ML)
    form_card = rx.card(
        rx.vstack(
            # Form Legend
            rx.hstack(
                rx.icon("credit-card", size=20, color=rx.color("accent", 10)),
                rx.heading("Transaction Details", size="4", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            # Predict button at TOP for visibility
            rx.button(
                "Predict",
                on_click=TFDState.predict_batch_tfd,
                size="2",
                width="100%",
                disabled=~SharedState.batch_model_available["Transaction Fraud Detection"]
            ),
            # Randomize button below Predict
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size=14),
                    rx.text("Randomize All Fields", size="2"),
                    spacing="1",
                    align_items="center"
                ),
                on_click=TFDState.randomize_tfd_form,
                variant="soft",
                color_scheme="blue",
                size="2",
                width="100%"
            ),
            rx.divider(),
            # Form fields in 3-column grid (same as incremental ML)
            rx.grid(
                # Amount
                rx.vstack(
                    rx.text("Amount", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("amount", ""),
                        on_change=lambda v: TFDState.update_tfd("amount", v),
                        step=0.01,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Account Age
                rx.vstack(
                    rx.text("Account Age", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("account_age_days", ""),
                        on_change=lambda v: TFDState.update_tfd("account_age_days", v),
                        min=0,
                        step=1,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Currency
                rx.vstack(
                    rx.text("Currency", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["currency"],
                        value=TFDState.tfd_form_data.get("currency", ""),
                        on_change=lambda v: TFDState.update_tfd("currency", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Date
                rx.vstack(
                    rx.text("Date", size="1", color="gray"),
                    rx.input(
                        type="date",
                        value=TFDState.tfd_form_data.get("timestamp_date", ""),
                        on_change=lambda v: TFDState.update_tfd("timestamp_date", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Time
                rx.vstack(
                    rx.text("Time", size="1", color="gray"),
                    rx.input(
                        type="time",
                        value=TFDState.tfd_form_data.get("timestamp_time", ""),
                        on_change=lambda v: TFDState.update_tfd("timestamp_time", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Merchant ID
                rx.vstack(
                    rx.text("Merchant ID", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["merchant_id"],
                        value=TFDState.tfd_form_data.get("merchant_id", ""),
                        on_change=lambda v: TFDState.update_tfd("merchant_id", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Product Category
                rx.vstack(
                    rx.text("Category", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["product_category"],
                        value=TFDState.tfd_form_data.get("product_category", ""),
                        on_change=lambda v: TFDState.update_tfd("product_category", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Transaction Type
                rx.vstack(
                    rx.text("Type", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["transaction_type"],
                        value=TFDState.tfd_form_data.get("transaction_type", ""),
                        on_change=lambda v: TFDState.update_tfd("transaction_type", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Payment Method
                rx.vstack(
                    rx.text("Payment", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["payment_method"],
                        value=TFDState.tfd_form_data.get("payment_method", ""),
                        on_change=lambda v: TFDState.update_tfd("payment_method", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Latitude
                rx.vstack(
                    rx.text("Latitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("lat", ""),
                        on_change=lambda v: TFDState.update_tfd("lat", v),
                        min=-90.0,
                        max=90.0,
                        step=0.0001,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Longitude
                rx.vstack(
                    rx.text("Longitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("lon", ""),
                        on_change=lambda v: TFDState.update_tfd("lon", v),
                        min=-180.0,
                        max=180.0,
                        step=0.0001,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Browser
                rx.vstack(
                    rx.text("Browser", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["browser"],
                        value=TFDState.tfd_form_data.get("browser", ""),
                        on_change=lambda v: TFDState.update_tfd("browser", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # OS
                rx.vstack(
                    rx.text("OS", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["os"],
                        value=TFDState.tfd_form_data.get("os", ""),
                        on_change=lambda v: TFDState.update_tfd("os", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # CVV Provided
                rx.vstack(
                    rx.text("CVV", size="1", color="gray"),
                    rx.checkbox(
                        "Provided",
                        checked=TFDState.tfd_form_data.get("cvv_provided", False),
                        on_change=lambda v: TFDState.update_tfd("cvv_provided", v),
                        size="1"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Billing Address Match
                rx.vstack(
                    rx.text("Billing", size="1", color="gray"),
                    rx.checkbox(
                        "Address Match",
                        checked=TFDState.tfd_form_data.get("billing_address_match", False),
                        on_change=lambda v: TFDState.update_tfd("billing_address_match", v),
                        size="1"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            # Display fields (read-only info stacked vertically)
            rx.vstack(
                rx.text(
                    f"Transaction ID: {TFDState.tfd_form_data.get('transaction_id', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"User ID: {TFDState.tfd_form_data.get('user_id', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"IP Address: {TFDState.tfd_form_data.get('ip_address', '')}",
                    size="1",
                    color="gray"
                ),
                spacing="1",
                align_items="start",
                width="100%"
            ),
            spacing="2",
            align_items="start",
            width="100%"
        ),
        width="100%"
    )

    # Build left column with batch ML training box
    if model_key and project_name:
        left_column = rx.vstack(
            batch_ml_training_box(model_key, project_name),
            form_card,
            spacing="4",
            width="30%"
        )
    else:
        left_column = rx.vstack(
            batch_ml_training_box("XGBClassifier", "Transaction Fraud Detection"),
            form_card,
            spacing="4",
            width="30%"
        )

    # Right column - Tabs for Prediction and Metrics (same structure as incremental ML)
    right_column = rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("target", size=14),
                        rx.text("Prediction"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="prediction"
                ),
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("chart-bar", size=14),
                        rx.text("Metrics"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="metrics"
                ),
            ),
            # Tab 1: Prediction
            rx.tabs.content(
                rx.vstack(
                    # Prediction section - header always visible
                    rx.hstack(
                        rx.icon("shield-alert", size=20, color=rx.color("accent", 10)),
                        rx.heading("Prediction Result", size="5", weight="bold"),
                        spacing="2",
                        align_items="center",
                        width="100%"
                    ),
                    # Model info row
                    rx.hstack(
                        rx.badge(
                            rx.hstack(
                                rx.icon("brain", size=12),
                                rx.text("XGBoost Classifier (Scikit-Learn)", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            color_scheme="purple",
                            variant="soft",
                            size="1"
                        ),
                        rx.badge("Batch ML", color_scheme="blue", variant="soft", size="1"),
                        spacing="2"
                    ),
                    rx.cond(
                        TFDState.tfd_batch_prediction_show,
                        # Show prediction results when available
                        rx.card(
                            rx.vstack(
                                # Plotly Gauge Chart
                                rx.plotly(data=TFDState.tfd_batch_fraud_gauge, width="100%"),
                                # Prediction summary cards (compact)
                                rx.hstack(
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("triangle-alert", size=14, color=TFDState.tfd_batch_prediction_color),
                                                rx.text("Classification", size="1", color="gray"),
                                                spacing="1",
                                                align_items="center"
                                            ),
                                            rx.text(
                                                TFDState.tfd_batch_prediction_text,
                                                size="5",
                                                weight="bold",
                                                color=TFDState.tfd_batch_prediction_color,
                                                align="center"
                                            ),
                                            spacing="1",
                                            align_items="center",
                                            width="100%"
                                        ),
                                        variant="surface",
                                        size="1",
                                        width="100%"
                                    ),
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("percent", size=14, color="red"),
                                                rx.text("Fraud", size="1", color="gray"),
                                                spacing="1",
                                                align_items="center"
                                            ),
                                            rx.text(
                                                f"{TFDState.tfd_batch_fraud_probability * 100:.2f}%",
                                                size="5",
                                                weight="bold",
                                                align="center",
                                                color="red"
                                            ),
                                            spacing="1",
                                            align_items="center",
                                            width="100%"
                                        ),
                                        variant="surface",
                                        size="1",
                                        width="100%"
                                    ),
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("circle-check", size=14, color="green"),
                                                rx.text("Not Fraud", size="1", color="gray"),
                                                spacing="1",
                                                align_items="center"
                                            ),
                                            rx.text(
                                                f"{(1 - TFDState.tfd_batch_fraud_probability) * 100:.2f}%",
                                                size="5",
                                                weight="bold",
                                                align="center",
                                                color="green"
                                            ),
                                            spacing="1",
                                            align_items="center",
                                            width="100%"
                                        ),
                                        variant="surface",
                                        size="1",
                                        width="100%"
                                    ),
                                    spacing="2",
                                    width="100%"
                                ),
                                spacing="4",
                                width="100%"
                            ),
                            variant="classic",
                            width="100%"
                        ),
                        # Show info or warning message when no prediction yet
                        rx.cond(
                            SharedState.batch_model_available["Transaction Fraud Detection"],
                            rx.callout(
                                "Fill in the transaction details and click **Predict** to get the fraud probability.",
                                icon="info",
                                color="blue",
                                width="100%"
                            ),
                            rx.callout(
                                "No trained model available. Click **Train** to train the batch model first.",
                                icon="triangle-alert",
                                color="orange",
                                width="100%"
                            )
                        )
                    ),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="prediction"
            ),
            # Tab 2: Metrics
            rx.tabs.content(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Classification Metrics", size="5"),
                        rx.button(
                            rx.icon("refresh-cw", size=16),
                            on_click=SharedState.get_batch_mlflow_metrics("Transaction Fraud Detection"),
                            size="1",
                            variant="ghost",
                            cursor="pointer",
                            title="Refresh metrics"
                        ),
                        align_items="center",
                        spacing="2"
                    ),
                    # Model info row
                    rx.hstack(
                        rx.badge(
                            rx.hstack(
                                rx.icon("brain", size=12),
                                rx.text("XGBoost Classifier (Scikit-Learn)", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            color_scheme="purple",
                            variant="soft",
                            size="1"
                        ),
                        rx.badge("Batch ML", color_scheme="blue", variant="soft", size="1"),
                        spacing="2"
                    ),
                    transaction_fraud_detection_batch_metrics(),
                    # YellowBrick Visualizations section
                    rx.divider(),
                    rx.heading("YellowBrick Visualizations", size="5"),
                    rx.text("Select a metric type and visualization to display.", size="2", color="gray"),
                    # YellowBrick metric type selector
                    rx.hstack(
                        rx.vstack(
                            rx.text("Metric Type", size="1", color="gray"),
                            rx.select(
                                TFDState.yellowbrick_metric_types,
                                value=TFDState.yellowbrick_metric_type,
                                on_change=TFDState.set_yellowbrick_metric_type,
                                width="100%"
                            ),
                            spacing="1",
                            width="50%"
                        ),
                        rx.vstack(
                            rx.text("Visualization", size="1", color="gray"),
                            rx.select(
                                TFDState.yellowbrick_metric_options,
                                value=TFDState.yellowbrick_metric_name,
                                on_change=TFDState.set_yellowbrick_metric_name,
                                width="100%"
                            ),
                            spacing="1",
                            width="50%"
                        ),
                        spacing="3",
                        width="100%"
                    ),
                    # Loading spinner
                    rx.cond(
                        TFDState.yellowbrick_loading,
                        rx.hstack(
                            rx.spinner(size="3"),
                            rx.text("Loading visualization...", size="2", color="gray"),
                            spacing="2",
                            align_items="center",
                            justify="center",
                            width="100%",
                            padding="4em"
                        ),
                        rx.fragment()
                    ),
                    # Error display
                    rx.cond(
                        TFDState.yellowbrick_error != "",
                        rx.callout(
                            TFDState.yellowbrick_error,
                            icon="triangle-alert",
                            color="red",
                            width="100%"
                        ),
                        rx.fragment()
                    ),
                    # Image display
                    rx.cond(
                        TFDState.yellowbrick_image_base64 != "",
                        rx.image(
                            src=f"data:image/png;base64,{TFDState.yellowbrick_image_base64}",
                            width="100%",
                            alt="YellowBrick visualization"
                        ),
                        rx.fragment()
                    ),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="metrics"
            ),
            default_value="prediction",
            width="100%"
        ),
        align_items="start",
        spacing="4",
        width="70%"
    )

    return rx.hstack(
        left_column,
        right_column,
        spacing="6",
        align_items="start",
        width="100%"
    )
