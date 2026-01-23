"""ETA Batch ML Prediction page - Form + Prediction results + Training controls."""
import reflex as rx
from ....components import (
    coelho_realtime_navbar,
    page_sub_nav,
    batch_sub_nav,
    batch_ml_run_and_training_box,
    eta_map,
)
from ....states import ETAState, SharedState


PROJECT_NAME = "Estimated Time of Arrival"
BASE_ROUTE = "/eta"
BATCH_ROUTE = "/eta/batch"


def _form_card() -> rx.Component:
    """Build the ETA form card."""
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
                disabled=~ETAState.batch_model_available["Estimated Time of Arrival"]
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


def _prediction_result() -> rx.Component:
    """Build the prediction result display with Origin/Destination and ETA boxes side by side."""
    return rx.hstack(
        # Left: Origin and Destination (Map)
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("map-pin", size=16, color=rx.color("accent", 10)),
                    rx.text("Origin and Destination", size="3", weight="bold"),
                    spacing="2",
                    align_items="center"
                ),
                rx.box(
                    eta_map(),
                    width="100%",
                    height="280px",
                    overflow="hidden",
                ),
                rx.vstack(
                    rx.text(f"Estimated Distance: {ETAState.eta_estimated_distance_km} km", size="2", color="gray"),
                    rx.text(f"Initial Estimated Travel Time: {ETAState.eta_initial_estimated_travel_time_seconds} s", size="2", color="gray"),
                    spacing="1",
                    width="100%",
                    padding_top="12px",
                ),
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
                rx.cond(
                    ETAState.eta_batch_prediction_show,
                    rx.box(
                        rx.plotly(data=ETAState.eta_batch_prediction_figure, width="100%"),
                        width="100%",
                        flex="1",
                        display="flex",
                        align_items="center",
                        justify_content="center"
                    ),
                    rx.box(
                        rx.cond(
                            ETAState.batch_model_available["Estimated Time of Arrival"],
                            rx.callout(
                                "Click **Predict** to get the estimated time of arrival.",
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
            height="400px"
        ),
        spacing="3",
        width="100%",
        align_items="stretch"
    )


def index() -> rx.Component:
    """ETA Batch ML Prediction page."""
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
                    width="40%"
                ),
                # Right column - Tabs + Prediction result
                rx.vstack(
                    batch_sub_nav(BATCH_ROUTE, "prediction"),
                    _prediction_result(),
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
            ETAState.init_eta_form_if_empty,
            SharedState.init_batch_page(PROJECT_NAME),
        ],
        on_unmount=ETAState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
