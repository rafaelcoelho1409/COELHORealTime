"""ECCI Batch ML Metrics page - sklearn overview + YellowBrick visualizations (coming soon)."""
import reflex as rx
from ....components import (
    coelho_realtime_navbar,
    page_sub_nav,
    batch_sub_nav,
    batch_ml_run_and_training_box,
    ecci_map,
)
from ....states import ECCIState, SharedState


PROJECT_NAME = "E-Commerce Customer Interactions"
BASE_ROUTE = "/ecci"
BATCH_ROUTE = "/ecci/batch"


def _form_card() -> rx.Component:
    """Build the ECCI form card (same as prediction page)."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("shopping-cart", size=20, color=rx.color("accent", 10)),
                rx.heading("Customer Interaction", size="4", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            rx.button(
                "Predict",
                on_click=ECCIState.predict_batch_ecci,
                size="2",
                width="100%",
                disabled=~SharedState.batch_model_available["E-Commerce Customer Interactions"]
            ),
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size=14),
                    rx.text("Randomize All Fields", size="2"),
                    spacing="1",
                    align_items="center"
                ),
                on_click=ECCIState.randomize_ecci_form,
                variant="soft",
                color_scheme="blue",
                size="2",
                width="100%"
            ),
            rx.divider(),
            rx.grid(
                rx.vstack(
                    rx.text("Browser", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["browser"],
                        value=ECCIState.ecci_form_data.get("browser", ""),
                        on_change=lambda v: ECCIState.update_ecci("browser", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Device", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["device_type"],
                        value=ECCIState.ecci_form_data.get("device_type", ""),
                        on_change=lambda v: ECCIState.update_ecci("device_type", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("OS", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["os"],
                        value=ECCIState.ecci_form_data.get("os", ""),
                        on_change=lambda v: ECCIState.update_ecci("os", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Event Type", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["event_type"],
                        value=ECCIState.ecci_form_data.get("event_type", ""),
                        on_change=lambda v: ECCIState.update_ecci("event_type", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Category", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["product_category"],
                        value=ECCIState.ecci_form_data.get("product_category", ""),
                        on_change=lambda v: ECCIState.update_ecci("product_category", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Price", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("price", ""),
                        on_change=lambda v: ECCIState.update_ecci("price", v),
                        min=0.0, step=0.01, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Date", size="1", color="gray"),
                    rx.input(
                        type="date",
                        value=ECCIState.ecci_form_data.get("timestamp_date", ""),
                        on_change=lambda v: ECCIState.update_ecci("timestamp_date", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Time", size="1", color="gray"),
                    rx.input(
                        type="time",
                        value=ECCIState.ecci_form_data.get("timestamp_time", ""),
                        on_change=lambda v: ECCIState.update_ecci("timestamp_time", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Product ID", size="1", color="gray"),
                    rx.input(
                        value=ECCIState.ecci_form_data.get("product_id", ""),
                        on_change=lambda v: ECCIState.update_ecci("product_id", v),
                        placeholder="prod_1050",
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Latitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("lat", ""),
                        on_change=lambda v: ECCIState.update_ecci("lat", v),
                        min=29.5, max=30.1, step=0.001, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Longitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("lon", ""),
                        on_change=lambda v: ECCIState.update_ecci("lon", v),
                        min=-95.8, max=-95.0, step=0.001, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Coords", size="1", color="gray"),
                    rx.button(
                        rx.hstack(
                            rx.icon("shuffle", size=12),
                            rx.text("Random", size="1"),
                            spacing="1",
                            align_items="center"
                        ),
                        on_click=ECCIState.generate_random_ecci_coordinates,
                        variant="outline",
                        size="1",
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            rx.vstack(
                rx.text(f"User ID: {ECCIState.ecci_form_data.get('user_id', '')}", size="1", color="gray"),
                rx.text(f"Session ID: {ECCIState.ecci_form_data.get('session_id', '')}", size="1", color="gray"),
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


def _sklearn_overview_tab() -> rx.Component:
    """Build the sklearn metrics overview tab content (coming soon)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("layout-dashboard", size=20, color=rx.color("accent", 10)),
            rx.heading("Clustering Metrics Overview", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        rx.callout(
            "Clustering metrics dashboard (Silhouette Score, Inertia, Calinski-Harabasz Index) coming soon.",
            icon="info",
            color="blue",
            width="100%"
        ),
        spacing="4",
        width="100%",
    )


def _performance_tab() -> rx.Component:
    """Build the YellowBrick Performance tab content (Clustering visualizations - coming soon)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("circle-check", size=20, color=rx.color("accent", 10)),
            rx.heading("Performance", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        rx.callout(
            "Clustering performance visualizations (SilhouetteVisualizer, KElbowVisualizer, InterclusterDistance) coming soon.",
            icon="info",
            color="blue",
            width="100%"
        ),
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
    """Build the YellowBrick Feature Analysis tab content (coming soon)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("scatter-chart", size=20, color=rx.color("accent", 10)),
            rx.heading("Feature Analysis", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        rx.callout(
            "Feature analysis visualizations (Rank1D, Rank2D, PCA, RadViz, ParallelCoordinates) coming soon.",
            icon="info",
            color="blue",
            width="100%"
        ),
        spacing="4",
        width="100%",
    )


def _target_tab() -> rx.Component:
    """Build the YellowBrick Target Analysis tab content (coming soon)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("crosshair", size=20, color=rx.color("accent", 10)),
            rx.heading("Target Analysis", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        rx.callout(
            "Target analysis visualizations (ClassBalance, FeatureCorrelation) coming soon.",
            icon="info",
            color="blue",
            width="100%"
        ),
        spacing="4",
        width="100%",
    )


def _diagnostics_tab() -> rx.Component:
    """Build the YellowBrick Model Diagnostics tab content (coming soon)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("settings-2", size=20, color=rx.color("accent", 10)),
            rx.heading("Model Diagnostics", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        rx.callout(
            "Model diagnostics visualizations (FeatureImportances, LearningCurve, ValidationCurve) coming soon.",
            icon="info",
            color="blue",
            width="100%"
        ),
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
    """ECCI Batch ML Metrics page with sklearn overview and YellowBrick visualizations."""
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
                    batch_ml_run_and_training_box("KMeans", PROJECT_NAME),
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
            ECCIState.init_ecci_form_if_empty,
            SharedState.init_batch_page(PROJECT_NAME),
        ],
        on_unmount=ECCIState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
