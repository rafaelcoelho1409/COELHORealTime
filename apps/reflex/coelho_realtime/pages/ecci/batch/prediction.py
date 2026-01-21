"""ECCI Batch ML Prediction page - Form + Prediction results + Training controls."""
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
    """Build the ECCI form card."""
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
                        min=0.0,
                        step=0.01,
                        width="100%"
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
                # Quantity
                rx.vstack(
                    rx.text("Quantity", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("quantity", ""),
                        on_change=lambda v: ECCIState.update_ecci("quantity", v),
                        min=1,
                        step=1,
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                # Time on Page
                rx.vstack(
                    rx.text("Time (s)", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("time_on_page_seconds", ""),
                        on_change=lambda v: ECCIState.update_ecci("time_on_page_seconds", v),
                        min=0,
                        step=1,
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                # Session Event Sequence
                rx.vstack(
                    rx.text("Sequence", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("session_event_sequence", ""),
                        on_change=lambda v: ECCIState.update_ecci("session_event_sequence", v),
                        min=1,
                        step=1,
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                # Referrer URL
                rx.vstack(
                    rx.text("Referrer", size="1", color="gray"),
                    rx.input(
                        value=ECCIState.ecci_form_data.get("referrer_url", ""),
                        on_change=lambda v: ECCIState.update_ecci("referrer_url", v),
                        placeholder="google.com",
                        width="100%"
                    ),
                    spacing="1", align_items="start", width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            # Display fields (read-only info)
            rx.vstack(
                rx.text(f"Customer ID: {ECCIState.ecci_form_data.get('customer_id', '')}", size="1", color="gray"),
                rx.text(f"Event ID: {ECCIState.ecci_form_data.get('event_id', '')}", size="1", color="gray"),
                rx.text(f"Page URL: {ECCIState.ecci_form_data.get('page_url', '')}", size="1", color="gray"),
                rx.text(f"Search Query: {ECCIState.ecci_form_data.get('search_query', '')}", size="1", color="gray"),
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


def _prediction_result() -> rx.Component:
    """Build the prediction result display - matches Incremental ML design."""
    return rx.vstack(
        # Customer Location Map (always visible)
        ecci_map(),
        # Prediction cards in 2-column layout
        rx.cond(
            ECCIState.ecci_batch_prediction_show,
            rx.vstack(
                rx.hstack(
                    # Left: Predicted Cluster (50%)
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("users", size=16, color="purple"),
                                rx.heading("Predicted Cluster", size="3", weight="bold"),
                                spacing="2",
                                align_items="center"
                            ),
                            rx.plotly(data=ECCIState.ecci_batch_prediction_figure, width="100%"),
                            spacing="2",
                            align_items="center",
                            width="100%"
                        ),
                        width="50%"
                    ),
                    # Right: Cluster Behavior (50%)
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("bar-chart-3", size=16, color="blue"),
                                rx.heading("Cluster Behavior", size="3", weight="bold"),
                                spacing="2",
                                align_items="center"
                            ),
                            rx.text("Select a feature to see its distribution in the predicted cluster:", size="2", color="gray"),
                            rx.select(
                                ECCIState.ecci_feature_options,
                                value=ECCIState.ecci_selected_feature,
                                on_change=ECCIState.set_ecci_selected_feature,
                                width="100%"
                            ),
                            rx.plotly(data=ECCIState.ecci_selected_cluster_feature_figure, width="100%"),
                            spacing="2",
                            align_items="start",
                            width="100%"
                        ),
                        width="50%"
                    ),
                    spacing="4",
                    width="100%"
                ),
                # Cluster interpretation info
                rx.callout(
                    rx.vstack(
                        rx.text("**Cluster Interpretation:**", weight="bold"),
                        rx.text(
                            "The predicted cluster represents a group of customers with similar interaction patterns. "
                            "Use the 'Cluster Behavior' chart to understand the characteristics of this cluster."
                        ),
                        spacing="1",
                        align_items="start"
                    ),
                    icon="info",
                    color="purple",
                    width="100%"
                ),
                spacing="4",
                width="100%"
            ),
            # No prediction yet - show appropriate callout
            rx.cond(
                SharedState.batch_model_available["E-Commerce Customer Interactions"],
                rx.callout(
                    "Fill in the customer interaction details and click **Predict** to get the cluster assignment.",
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
        width="100%"
    )


def index() -> rx.Component:
    """ECCI Batch ML Prediction page."""
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
                # Right column - Tabs + Prediction result
                rx.vstack(
                    batch_sub_nav(BATCH_ROUTE, "prediction"),
                    _prediction_result(),
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
