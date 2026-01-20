"""TFD Batch ML Metrics page - sklearn overview + YellowBrick visualizations with tabs."""
import reflex as rx
from ....components import (
    coelho_realtime_navbar,
    page_sub_nav,
    batch_sub_nav,
    batch_ml_run_and_training_box,
)
from ....components.tfd import (
    transaction_fraud_detection_batch_metrics,
    yellowbrick_dynamic_info_button,
)
from ....states import TFDState, SharedState


PROJECT_NAME = "Transaction Fraud Detection"
BASE_ROUTE = "/tfd"
BATCH_ROUTE = "/tfd/batch"


def _form_card() -> rx.Component:
    """Build the transaction form card (same as prediction page)."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("credit-card", size=20, color=rx.color("accent", 10)),
                rx.heading("Transaction Details", size="4", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            rx.button(
                "Predict",
                on_click=TFDState.predict_batch_tfd,
                size="2",
                width="100%",
                disabled=~SharedState.batch_model_available["Transaction Fraud Detection"]
            ),
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
            rx.grid(
                rx.vstack(
                    rx.text("Amount", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("amount", ""),
                        on_change=lambda v: TFDState.update_tfd("amount", v),
                        step=0.01,
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Account Age", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("account_age_days", ""),
                        on_change=lambda v: TFDState.update_tfd("account_age_days", v),
                        min=0, step=1, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Currency", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["currency"],
                        value=TFDState.tfd_form_data.get("currency", ""),
                        on_change=lambda v: TFDState.update_tfd("currency", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Date", size="1", color="gray"),
                    rx.input(
                        type="date",
                        value=TFDState.tfd_form_data.get("timestamp_date", ""),
                        on_change=lambda v: TFDState.update_tfd("timestamp_date", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Time", size="1", color="gray"),
                    rx.input(
                        type="time",
                        value=TFDState.tfd_form_data.get("timestamp_time", ""),
                        on_change=lambda v: TFDState.update_tfd("timestamp_time", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Merchant ID", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["merchant_id"],
                        value=TFDState.tfd_form_data.get("merchant_id", ""),
                        on_change=lambda v: TFDState.update_tfd("merchant_id", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Category", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["product_category"],
                        value=TFDState.tfd_form_data.get("product_category", ""),
                        on_change=lambda v: TFDState.update_tfd("product_category", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Type", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["transaction_type"],
                        value=TFDState.tfd_form_data.get("transaction_type", ""),
                        on_change=lambda v: TFDState.update_tfd("transaction_type", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Payment", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["payment_method"],
                        value=TFDState.tfd_form_data.get("payment_method", ""),
                        on_change=lambda v: TFDState.update_tfd("payment_method", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Latitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("lat", ""),
                        on_change=lambda v: TFDState.update_tfd("lat", v),
                        min=-90.0, max=90.0, step=0.0001, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Longitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=TFDState.tfd_form_data.get("lon", ""),
                        on_change=lambda v: TFDState.update_tfd("lon", v),
                        min=-180.0, max=180.0, step=0.0001, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Browser", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["browser"],
                        value=TFDState.tfd_form_data.get("browser", ""),
                        on_change=lambda v: TFDState.update_tfd("browser", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("OS", size="1", color="gray"),
                    rx.select(
                        TFDState.tfd_options["os"],
                        value=TFDState.tfd_form_data.get("os", ""),
                        on_change=lambda v: TFDState.update_tfd("os", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("CVV", size="1", color="gray"),
                    rx.checkbox(
                        "Provided",
                        checked=TFDState.tfd_form_data.get("cvv_provided", False),
                        on_change=lambda v: TFDState.update_tfd("cvv_provided", v),
                        size="1"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Billing", size="1", color="gray"),
                    rx.checkbox(
                        "Address Match",
                        checked=TFDState.tfd_form_data.get("billing_address_match", False),
                        on_change=lambda v: TFDState.update_tfd("billing_address_match", v),
                        size="1"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            rx.vstack(
                rx.text(f"Transaction ID: {TFDState.tfd_form_data.get('transaction_id', '')}", size="1", color="gray"),
                rx.text(f"User ID: {TFDState.tfd_form_data.get('user_id', '')}", size="1", color="gray"),
                rx.text(f"IP Address: {TFDState.tfd_form_data.get('ip_address', '')}", size="1", color="gray"),
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


def _yellowbrick_content(category: str, description: str) -> rx.Component:
    """Build YellowBrick visualization content for a specific category."""
    return rx.cond(
        TFDState.tfd_batch_model_available,
        rx.vstack(
            rx.text(description, size="2", color="gray"),
            rx.hstack(
                rx.select(
                    TFDState.yellowbrick_metrics_options[category],
                    value=TFDState.yellowbrick_metric_name,
                    on_change=lambda v: TFDState.set_yellowbrick_visualization(category, v),
                    placeholder="Select visualization...",
                    width="100%"
                ),
                yellowbrick_dynamic_info_button(),
                spacing="2",
                align_items="center",
                width="100%"
            ),
            rx.cond(
                TFDState.yellowbrick_loading,
                rx.vstack(
                    rx.hstack(
                        rx.spinner(size="3"),
                        rx.text("Loading visualization...", size="2", color="gray"),
                        spacing="2",
                        align_items="center",
                    ),
                    rx.button(
                        rx.hstack(
                            rx.icon("square", size=12),
                            rx.text("Stop", size="1"),
                            spacing="1",
                            align_items="center"
                        ),
                        on_click=TFDState.cancel_yellowbrick_loading,
                        size="1",
                        color_scheme="red",
                        variant="soft"
                    ),
                    spacing="3",
                    align_items="center",
                    justify="center",
                    width="100%",
                    padding="4em"
                ),
                rx.fragment()
            ),
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
        ),
        rx.callout(
            "Train a model first to view visualizations. Go to the Prediction page and click Train.",
            icon="info",
            color="blue",
            width="100%"
        )
    )


def _sklearn_overview_tab() -> rx.Component:
    """Build the sklearn metrics overview tab content."""
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.icon("layout-dashboard", size=20, color=rx.color("accent", 10)),
                rx.heading("Classification Metrics Overview", size="5", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            rx.button(
                rx.icon("refresh-cw", size=16),
                on_click=SharedState.refresh_batch_mlflow_metrics("Transaction Fraud Detection"),
                size="1",
                variant="ghost",
                cursor="pointer",
                title="Refresh metrics"
            ),
            align_items="center",
            spacing="2",
            width="100%"
        ),
        transaction_fraud_detection_batch_metrics(),
        spacing="4",
        width="100%",
    )


def _performance_tab() -> rx.Component:
    """Build the YellowBrick Performance tab content (Classification visualizations)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("circle-check", size=20, color=rx.color("accent", 10)),
            rx.heading("Performance", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Classification", "Select a classification performance visualization."),
        spacing="4",
        width="100%",
    )


def _explainability_tab() -> rx.Component:
    """Build the Explainability tab content (SHAP - coming soon)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("lightbulb", size=20, color=rx.color("accent", 10)),
            rx.heading("Explainability", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        rx.callout(
            "SHAP (SHapley Additive exPlanations) visualizations coming soon. This will include feature importance, summary plots, and individual prediction explanations.",
            icon="info",
            color="blue",
            width="100%"
        ),
        spacing="4",
        width="100%",
    )


def _features_tab() -> rx.Component:
    """Build the YellowBrick Feature Analysis tab content."""
    return rx.vstack(
        rx.hstack(
            rx.icon("scatter-chart", size=20, color=rx.color("accent", 10)),
            rx.heading("Feature Analysis", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Feature Analysis", "Select a feature analysis visualization."),
        spacing="4",
        width="100%",
    )


def _target_tab() -> rx.Component:
    """Build the YellowBrick Target Analysis tab content."""
    return rx.vstack(
        rx.hstack(
            rx.icon("crosshair", size=20, color=rx.color("accent", 10)),
            rx.heading("Target Analysis", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Target", "Select a target analysis visualization."),
        spacing="4",
        width="100%",
    )


def _diagnostics_tab() -> rx.Component:
    """Build the YellowBrick Model Diagnostics tab content."""
    return rx.vstack(
        rx.hstack(
            rx.icon("settings-2", size=20, color=rx.color("accent", 10)),
            rx.heading("Model Diagnostics", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Model Selection", "Select a model diagnostics visualization."),
        spacing="4",
        width="100%",
    )


def _metrics_tabs() -> rx.Component:
    """Build the metrics tabs (sklearn overview + YellowBrick categories)."""
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger("Overview", value="overview", flex="1"),
            rx.tabs.trigger("Performance", value="performance", flex="1"),
            rx.tabs.trigger("Explainability", value="explainability", flex="1"),
            rx.tabs.trigger("Feature Analysis", value="features", flex="1"),
            rx.tabs.trigger("Target Analysis", value="target", flex="1"),
            rx.tabs.trigger("Model Diagnostics", value="diagnostics", flex="1"),
            size="2",
            width="100%",
        ),
        rx.tabs.content(
            rx.box(_sklearn_overview_tab(), padding_top="1em"),
            value="overview",
        ),
        rx.tabs.content(
            rx.box(_performance_tab(), padding_top="1em"),
            value="performance",
        ),
        rx.tabs.content(
            rx.box(_explainability_tab(), padding_top="1em"),
            value="explainability",
        ),
        rx.tabs.content(
            rx.box(_features_tab(), padding_top="1em"),
            value="features",
        ),
        rx.tabs.content(
            rx.box(_target_tab(), padding_top="1em"),
            value="target",
        ),
        rx.tabs.content(
            rx.box(_diagnostics_tab(), padding_top="1em"),
            value="diagnostics",
        ),
        default_value="overview",
        width="100%",
    )


def index() -> rx.Component:
    """TFD Batch ML Metrics page with sklearn overview and YellowBrick visualizations."""
    return rx.vstack(
        coelho_realtime_navbar(),
        rx.box(
            page_sub_nav(BASE_ROUTE, "batch"),
            padding_x="2em",
            padding_top="1em",
            width="100%"
        ),
        rx.box(
            rx.hstack(
                # Left column - Training controls + Form
                rx.vstack(
                    batch_ml_run_and_training_box("CatBoostClassifier", PROJECT_NAME),
                    _form_card(),
                    spacing="4",
                    width="30%"
                ),
                # Right column - Tabs + Metrics tabs
                rx.vstack(
                    batch_sub_nav(BATCH_ROUTE, "metrics"),
                    _metrics_tabs(),
                    align_items="start",
                    spacing="4",
                    width="70%"
                ),
                spacing="6",
                align_items="start",
                width="100%"
            ),
            padding="2em",
            width="100%"
        ),
        on_mount=[
            TFDState.init_tfd_form_if_empty,
            SharedState.init_batch_page(PROJECT_NAME),
            TFDState.clear_yellowbrick_visualization,
        ],
        on_unmount=[
            TFDState.clear_large_state_data,
            TFDState.cleanup_on_page_leave(PROJECT_NAME),
        ],
        spacing="0",
        width="100%"
    )
