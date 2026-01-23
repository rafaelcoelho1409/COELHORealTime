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
    """Build the YellowBrick Classification tab content (Classification visualizations)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("circle-check", size=20, color=rx.color("accent", 10)),
            rx.heading("Classification", size="5", weight="bold"),
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
            rx.tabs.trigger("Classification", value="performance", flex="1"),
            rx.tabs.trigger("Feature Analysis", value="features", flex="1"),
            rx.tabs.trigger("Target Analysis", value="target", flex="1"),
            rx.tabs.trigger("Model Diagnostics", value="diagnostics", flex="1"),
            rx.tabs.trigger("Explainability", value="explainability", flex="1"),
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
        rx.tabs.content(
            rx.box(_explainability_tab(), padding_top="1em"),
            value="explainability",
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
                # Left column - Training controls only (form is on Prediction tab)
                rx.vstack(
                    batch_ml_run_and_training_box("CatBoostClassifier", PROJECT_NAME),
                    spacing="4",
                    width="40%"
                ),
                # Right column - Tabs + Metrics tabs
                rx.vstack(
                    batch_sub_nav(BATCH_ROUTE, "metrics"),
                    _metrics_tabs(),
                    align_items="start",
                    spacing="4",
                    width="60%"
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
