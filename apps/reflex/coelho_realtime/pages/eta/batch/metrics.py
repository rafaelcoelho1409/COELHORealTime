"""ETA Batch ML Metrics page - sklearn overview + YellowBrick visualizations with tabs."""
import reflex as rx
from ....components import (
    coelho_realtime_navbar,
    page_sub_nav,
    batch_sub_nav,
    batch_ml_run_and_training_box,
)
from ....components.eta import estimated_time_of_arrival_batch_metrics
from ....states import ETAState, SharedState


PROJECT_NAME = "Estimated Time of Arrival"
BASE_ROUTE = "/eta"
BATCH_ROUTE = "/eta/batch"


def _form_card() -> rx.Component:
    """Build the ETA form card (same as prediction page)."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("clock", size=20, color=rx.color("accent", 10)),
                rx.heading("Trip Details", size="4", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            rx.button(
                "Predict",
                on_click=ETAState.predict_batch_eta,
                size="2",
                width="100%",
                disabled=~SharedState.batch_model_available["Estimated Time of Arrival"]
            ),
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size=14),
                    rx.text("Randomize All Fields", size="2"),
                    spacing="1",
                    align_items="center"
                ),
                on_click=ETAState.randomize_eta_form,
                variant="soft",
                color_scheme="blue",
                size="2",
                width="100%"
            ),
            rx.divider(),
            rx.grid(
                rx.vstack(
                    rx.text("Driver ID", size="1", color="gray"),
                    rx.select(
                        ETAState.eta_options["driver_id"],
                        value=ETAState.eta_form_data.get("driver_id", ""),
                        on_change=lambda v: ETAState.update_eta("driver_id", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Vehicle ID", size="1", color="gray"),
                    rx.select(
                        ETAState.eta_options["vehicle_id"],
                        value=ETAState.eta_form_data.get("vehicle_id", ""),
                        on_change=lambda v: ETAState.update_eta("vehicle_id", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Weather", size="1", color="gray"),
                    rx.select(
                        ETAState.eta_options["weather"],
                        value=ETAState.eta_form_data.get("weather", ""),
                        on_change=lambda v: ETAState.update_eta("weather", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Date", size="1", color="gray"),
                    rx.input(
                        type="date",
                        value=ETAState.eta_form_data.get("timestamp_date", ""),
                        on_change=lambda v: ETAState.update_eta("timestamp_date", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Time", size="1", color="gray"),
                    rx.input(
                        type="time",
                        value=ETAState.eta_form_data.get("timestamp_time", ""),
                        on_change=lambda v: ETAState.update_eta("timestamp_time", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Vehicle Type", size="1", color="gray"),
                    rx.select(
                        ETAState.eta_options["vehicle_type"],
                        value=ETAState.eta_form_data.get("vehicle_type", ""),
                        on_change=lambda v: ETAState.update_eta("vehicle_type", v),
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Origin Lat", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("origin_lat", ""),
                        on_change=lambda v: ETAState.update_eta("origin_lat", v),
                        min=29.5, max=30.1, step=0.0001, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Origin Lon", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("origin_lon", ""),
                        on_change=lambda v: ETAState.update_eta("origin_lon", v),
                        min=-95.8, max=-95.0, step=0.0001, width="100%"
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
                        on_click=ETAState.generate_random_eta_coordinates,
                        variant="outline",
                        size="1",
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Dest Lat", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("destination_lat", ""),
                        on_change=lambda v: ETAState.update_eta("destination_lat", v),
                        min=29.5, max=30.1, step=0.0001, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Dest Lon", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("destination_lon", ""),
                        on_change=lambda v: ETAState.update_eta("destination_lon", v),
                        min=-95.8, max=-95.0, step=0.0001, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                rx.vstack(
                    rx.text("Hour", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("hour_of_day", ""),
                        on_change=lambda v: ETAState.update_eta("hour_of_day", v),
                        min=0, max=23, step=1, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            rx.vstack(
                rx.text(f"Trip ID: {ETAState.eta_form_data.get('trip_id', '')}", size="1", color="gray"),
                rx.text(f"Estimated Distance: {ETAState.eta_estimated_distance_km} km", size="1", color="gray"),
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
        SharedState.batch_model_available["Estimated Time of Arrival"],
        rx.vstack(
            rx.text(description, size="2", color="gray"),
            rx.hstack(
                rx.select(
                    ETAState.yellowbrick_metrics_options[category],
                    value=ETAState.yellowbrick_metric_name,
                    on_change=lambda v: ETAState.set_yellowbrick_visualization(category, v),
                    placeholder="Select visualization...",
                    width="100%"
                ),
                spacing="2",
                align_items="center",
                width="100%"
            ),
            rx.cond(
                ETAState.yellowbrick_loading,
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
                        on_click=ETAState.cancel_yellowbrick_loading,
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
                ETAState.yellowbrick_error != "",
                rx.callout(
                    ETAState.yellowbrick_error,
                    icon="triangle-alert",
                    color="red",
                    width="100%"
                ),
                rx.fragment()
            ),
            rx.cond(
                ETAState.yellowbrick_image_base64 != "",
                rx.image(
                    src=f"data:image/png;base64,{ETAState.yellowbrick_image_base64}",
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
                rx.heading("Regression Metrics Overview", size="5", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            rx.button(
                rx.icon("refresh-cw", size=16),
                on_click=SharedState.refresh_batch_mlflow_metrics("Estimated Time of Arrival"),
                size="1",
                variant="ghost",
                cursor="pointer",
                title="Refresh metrics"
            ),
            align_items="center",
            spacing="2",
            width="100%"
        ),
        estimated_time_of_arrival_batch_metrics(),
        spacing="4",
        width="100%",
    )


def _performance_tab() -> rx.Component:
    """Build the YellowBrick Performance tab content (Regression visualizations)."""
    return rx.vstack(
        rx.hstack(
            rx.icon("trending-up", size=20, color=rx.color("accent", 10)),
            rx.heading("Performance", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Regression", "Select a regression performance visualization."),
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
    """ETA Batch ML Metrics page with sklearn overview and YellowBrick visualizations."""
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
                    batch_ml_run_and_training_box("CatBoostRegressor", PROJECT_NAME),
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
            ETAState.randomize_eta_form,
            SharedState.init_batch_page(PROJECT_NAME),
            ETAState.clear_yellowbrick_visualization,
        ],
        on_unmount=ETAState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
