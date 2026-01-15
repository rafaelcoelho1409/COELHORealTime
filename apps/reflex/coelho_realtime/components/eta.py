"""
Estimated Time of Arrival (ETA) components module.

This module contains:
- estimated_time_of_arrival_form() - Incremental ML form
- eta_map() - Folium map component
- estimated_time_of_arrival_metrics() - Metrics dashboard
- ETA-specific helper components
"""
import reflex as rx
from ..states import ETAState, METRIC_INFO
from .shared import metric_info_dialog, ml_training_switch


# =============================================================================
# ETA Card Helper Functions
# =============================================================================
def metric_card(label: str, value_var, metric_key: str = None, project_key: str = "eta") -> rx.Component:
    """Create a compact styled metric card with optional info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(label, size="1", weight="medium", color="gray"),
                metric_info_dialog(metric_key, project_key) if metric_key else rx.fragment(),
                spacing="1",
                align="center",
                justify="center"
            ),
            rx.text(value_var, size="4", weight="bold", align="center"),
            spacing="1",
            align_items="center",
            justify="center",
            height="100%"
        ),
        variant="surface",
        size="1"
    )


def mlflow_run_info_badge(project_name: str) -> rx.Component:
    """Display MLflow experiment run info for a project."""
    run_info = ETAState.mlflow_run_info[project_name]
    return rx.hstack(
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
                color_scheme=rx.cond(run_info["status"] == "FINISHED", "blue", "gray"),
                variant="surface"
            )
        ),
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
                metric_info_dialog(metric_key, "eta"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=ETAState.eta_dashboard_figures[plotly_key], width="100%"),
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
                metric_info_dialog(metric_key, "eta"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=ETAState.eta_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1",
        width="50%"
    )


# =============================================================================
# ETA Map Component
# =============================================================================
def eta_map() -> rx.Component:
    """Display Folium map for ETA origin/destination."""
    return rx.html(ETAState.eta_folium_map_html)


# =============================================================================
# ETA Metrics Dashboard
# =============================================================================
def estimated_time_of_arrival_metrics() -> rx.Component:
    """Display MLflow regression metrics for ETA with Plotly dashboard layout."""
    return rx.vstack(
        mlflow_run_info_badge("Estimated Time of Arrival"),
        # ROW 1: KPI Indicators
        rx.grid(
            kpi_card_with_info("kpi_mae", "mae"),
            kpi_card_with_info("kpi_rmse", "rmse"),
            kpi_card_with_info("kpi_r2", "r2"),
            kpi_card_with_info("kpi_rolling_mae", "rolling_mae"),
            columns="4",
            spacing="2",
            width="100%"
        ),
        # ROW 2: Additional metrics
        rx.grid(
            metric_card("MSE", ETAState.eta_metrics["mse"], "mse"),
            metric_card("RMSLE", ETAState.eta_metrics["rmsle"], "rmsle"),
            metric_card("SMAPE", ETAState.eta_metrics["smape"], "smape"),
            metric_card("Time Rolling MAE", ETAState.eta_metrics["time_rolling_mae"], "time_rolling_mae"),
            columns="4",
            spacing="2",
            width="100%"
        ),
        # ROW 3: Gauges
        rx.hstack(
            gauge_card_with_info("gauge_r2", "r2"),
            gauge_card_with_info("gauge_mape", "mape"),
            spacing="2",
            width="100%"
        ),
        spacing="3",
        width="100%"
    )


# =============================================================================
# ETA Incremental ML Form
# =============================================================================
def estimated_time_of_arrival_form(model_key: str = None, project_name: str = None) -> rx.Component:
    """Incremental ML form for Estimated Time of Arrival using ARFRegressor."""
    form_card = rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("clock", size=20, color=rx.color("accent", 10)),
                rx.heading("Trip Details", size="4", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            rx.button(
                "Predict",
                on_click=ETAState.predict_eta,
                size="2",
                width="100%",
                disabled=~ETAState.incremental_model_available["Estimated Time of Arrival"]
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
            # Form fields in 3-column grid
            rx.grid(
                # Driver ID
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
                # Vehicle ID
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
                # Weather
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
                # Date
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
                # Time
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
                # Vehicle Type
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
                # Origin Lat
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
                # Origin Lon
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
                # Random Coordinates button
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
                # Dest Lat
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
                # Dest Lon
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
                # Hour of Day
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
                # Driver Rating
                rx.vstack(
                    rx.text("Rating", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("driver_rating", ""),
                        on_change=lambda v: ETAState.update_eta("driver_rating", v),
                        min=3.5, max=5.0, step=0.1, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                # Temperature
                rx.vstack(
                    rx.text("Temp C", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("temperature_celsius", ""),
                        on_change=lambda v: ETAState.update_eta("temperature_celsius", v),
                        min=-50.0, max=50.0, step=0.1, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                # Debug Traffic Factor
                rx.vstack(
                    rx.text("Traffic Factor", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("debug_traffic_factor", ""),
                        on_change=lambda v: ETAState.update_eta("debug_traffic_factor", v),
                        min=0.3, max=1.9, step=0.1, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                # Debug Weather Factor
                rx.vstack(
                    rx.text("Weather Factor", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("debug_weather_factor", ""),
                        on_change=lambda v: ETAState.update_eta("debug_weather_factor", v),
                        min=1.0, max=2.0, step=0.1, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                # Debug Driver Factor
                rx.vstack(
                    rx.text("Driver Factor", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("debug_driver_factor", ""),
                        on_change=lambda v: ETAState.update_eta("debug_driver_factor", v),
                        min=0.85, max=1.15, step=0.01, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                # Debug Incident Delay
                rx.vstack(
                    rx.text("Incident (s)", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ETAState.eta_form_data.get("debug_incident_delay_seconds", ""),
                        on_change=lambda v: ETAState.update_eta("debug_incident_delay_seconds", v),
                        min=0, max=1800, step=1, width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            # Display fields
            rx.vstack(
                rx.text(f"Trip ID: {ETAState.eta_form_data.get('trip_id', '')}", size="1", color="gray"),
                rx.text(f"Estimated Distance: {ETAState.eta_estimated_distance_km} km", size="1", color="gray"),
                rx.text(f"Initial Estimated Travel Time: {ETAState.eta_initial_estimated_travel_time_seconds} s", size="1", color="gray"),
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
                    rx.hstack(rx.icon("target", size=14), rx.text("Prediction"), spacing="2", align_items="center"),
                    value="prediction"
                ),
                rx.tabs.trigger(
                    rx.hstack(rx.icon("chart-bar", size=14), rx.text("Metrics"), spacing="2", align_items="center"),
                    value="metrics"
                ),
            ),
            # Tab 1: Prediction
            rx.tabs.content(
                rx.vstack(
                    rx.hstack(
                        # Left: Map
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("map-pin", size=16, color=rx.color("accent", 10)),
                                    rx.text("Origin and Destination", size="3", weight="bold"),
                                    spacing="2",
                                    align_items="center"
                                ),
                                eta_map(),
                                rx.text(f"Estimated Distance: {ETAState.eta_estimated_distance_km} km", size="2", color="gray"),
                                rx.text(f"Initial Estimated Travel Time: {ETAState.eta_initial_estimated_travel_time_seconds} s", size="2", color="gray"),
                                spacing="2",
                                width="100%",
                                height="100%"
                            ),
                            variant="surface",
                            width="50%",
                            height="400px"
                        ),
                        # Right: ETA Prediction
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("clock", size=16, color=rx.color("accent", 10)),
                                    rx.text("ETA - Prediction", size="3", weight="bold"),
                                    spacing="2",
                                    align_items="center"
                                ),
                                mlflow_run_info_badge("Estimated Time of Arrival"),
                                rx.cond(
                                    ETAState.eta_prediction_show,
                                    rx.box(
                                        rx.plotly(data=ETAState.eta_prediction_figure, width="100%"),
                                        width="100%",
                                        flex="1",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center"
                                    ),
                                    rx.box(
                                        rx.cond(
                                            ETAState.incremental_model_available["Estimated Time of Arrival"],
                                            rx.callout(
                                                "Click **Predict** to get the estimated time of arrival.",
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
                                        ),
                                        width="100%",
                                        flex="1",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center"
                                    )
                                ),
                                spacing="2",
                                width="100%",
                                height="100%"
                            ),
                            variant="surface",
                            width="50%",
                            height="380px"
                        ),
                        spacing="3",
                        width="100%",
                        align_items="stretch"
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
                        rx.heading("Regression Metrics", size="5"),
                        rx.button(
                            rx.icon("refresh-cw", size=16),
                            on_click=ETAState.refresh_mlflow_metrics("Estimated Time of Arrival"),
                            size="1",
                            variant="ghost",
                            cursor="pointer",
                            title="Refresh metrics"
                        ),
                        align_items="center",
                        spacing="2"
                    ),
                    estimated_time_of_arrival_metrics(),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="metrics"
            ),
            default_value="prediction",
            width="100%"
        ),
        on_mount=ETAState.get_mlflow_metrics("Estimated Time of Arrival"),
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
