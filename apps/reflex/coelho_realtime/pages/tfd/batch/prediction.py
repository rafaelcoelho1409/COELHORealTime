"""TFD Batch ML Prediction page - Form + Prediction results + Training controls."""
import reflex as rx
from ....components import (
    coelho_realtime_navbar,
    page_sub_nav,
    batch_sub_nav,
    batch_ml_run_and_training_box,
)
from ....states import TFDState, SharedState


PROJECT_NAME = "Transaction Fraud Detection"
BASE_ROUTE = "/tfd"
BATCH_ROUTE = "/tfd/batch"


def _form_card() -> rx.Component:
    """Build the transaction form card."""
    return rx.card(
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
            # Display fields (read-only info)
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


def _prediction_result() -> rx.Component:
    """Build the prediction result display."""
    return rx.vstack(
        # Prediction section header
        rx.hstack(
            rx.icon("shield-alert", size=20, color=rx.color("accent", 10)),
            rx.heading("Prediction Result", size="5", weight="bold"),
            spacing="2",
            align_items="center",
            width="100%"
        ),
        rx.cond(
            TFDState.tfd_batch_prediction_show,
            # Show prediction results when available
            rx.card(
                rx.vstack(
                    # Plotly Gauge Chart
                    rx.plotly(data=TFDState.tfd_batch_fraud_gauge, width="100%"),
                    # Prediction summary cards
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
        width="100%"
    )


def index() -> rx.Component:
    """TFD Batch ML Prediction page."""
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
            TFDState.init_tfd_form_if_empty,
            SharedState.init_batch_page(PROJECT_NAME),
        ],
        on_unmount=TFDState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
