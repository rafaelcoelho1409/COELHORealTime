import reflex as rx
from .state import State


## COMPONENTS
def ml_training_switch(model_key: str, project_name: str) -> rx.Component:
    """
    A switch component to control real-time ML training.
    When enabled, starts Kafka consumer to process live data.
    When disabled or on page leave, stops the consumer.
    """
    return rx.card(
        rx.hstack(
            rx.vstack(
                rx.hstack(
                    rx.icon("activity", size=18, color=rx.cond(
                        State.ml_training_enabled,
                        rx.color("green", 9),
                        rx.color("gray", 9)
                    )),
                    rx.text(
                        "Real-time ML Training",
                        size="3",
                        weight="medium"
                    ),
                    spacing="2",
                    align_items="center"
                ),
                rx.text(
                    rx.cond(
                        State.ml_training_enabled,
                        "Processing live Kafka stream data",
                        "Toggle to start processing live data"
                    ),
                    size="1",
                    color="gray"
                ),
                spacing="1",
                align_items="start"
            ),
            rx.switch(
                checked=State.ml_training_enabled,
                on_change=lambda checked: State.toggle_ml_training(
                    checked,
                    model_key,
                    project_name
                ),
                size="3"
            ),
            justify="between",
            align_items="center",
            width="100%"
        ),
        variant="surface",
        size="2",
        width="100%"
    )


def navbar_link(text: str, url: str) -> rx.Component:
    return rx.link(
        rx.text(
            text,
            size = "3",
            weight = "medium"
        ),
        href = url,
        underline = "hover",
        color_scheme = "gray",
        high_contrast = True
    )


def coelho_realtime_navbar() -> rx.Component:
    return rx.box(
        rx.box(
            rx.hstack(
                # Left section - Logo and Title
                rx.hstack(
                    rx.image(
                        src = "/coelho_realtime_logo.png",
                        width = "5em",
                        height = "auto",
                        border_radius = "12px"
                    ),
                    rx.vstack(
                        rx.heading(
                            "COELHO RealTime",
                            size = "6",
                            weight = "bold",
                            color = rx.color("accent", 11)
                        ),
                        rx.text(
                            State.page_name,
                            size = "2",
                            weight = "medium",
                            color = rx.color("gray", 11)
                        ),
                        spacing = "1",
                        align_items = "start"
                    ),
                    spacing = "4",
                    align_items = "center"
                ),
                # Right section - Navigation
                rx.hstack(
                    navbar_link("Home", "/"),
                    rx.menu.root(
                        rx.menu.trigger(
                            rx.button(
                                rx.hstack(
                                    rx.text("Applications", size = "3", weight = "medium"),
                                    rx.icon("chevron-down", size = 16),
                                    spacing = "2",
                                    align_items = "center"
                                ),
                                variant = "soft",
                                size = "2",
                                color_scheme = "gray"
                            )
                        ),
                        rx.menu.content(
                            rx.menu.item(
                                navbar_link("Transaction Fraud Detection", "/transaction-fraud-detection")
                            ),
                            rx.menu.item(
                                navbar_link("Estimated Time of Arrival", "/estimated-time-of-arrival")
                            ),
                            rx.menu.item(
                                navbar_link("E-Commerce Customer Interactions", "/e-commerce-customer-interactions")
                            ),
                            size = "2"
                        )
                    ),
                    spacing = "4",
                    align_items = "center"
                ),
                justify = "between",
                align_items = "center",
                width = "100%"
            ),
            max_width = "1400px",
            width = "100%",
            padding_x = "2em",
            padding_y = "1.2em"
        ),
        bg = rx.color("accent", 2),
        border_bottom = f"1px solid {rx.color('gray', 6)}",
        width = "100%",
        position = "sticky",
        top = "0",
        z_index = "1000",
        backdrop_filter = "blur(10px)"
    )

def page_tabs() -> rx.Component:
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger(
                "Incremental ML",
                value = "incremental_ml"
            ),
            rx.tabs.trigger(
                "Batch ML",
                value = "batch_ml"
            ),
        ),
        default_value = "incremental_ml",
        on_change = State.update_sample(State.project_name),
        width = "100%",
    )



## TRANSACTION FRAUD DETECTION
def transaction_fraud_detection_form() -> rx.Component:
    return rx.hstack(
        # Left column - Form
        rx.card(
            rx.form(
                rx.vstack(
                    # Row 1: Amount and Account Age
                    rx.hstack(
                        rx.vstack(
                            rx.text("Amount", size = "1", color = "gray"),
                            rx.input(
                                name = "amount",
                                type = "number",
                                value = State.tfd_form_data.get("amount", ""),
                                step = 0.01,
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        rx.vstack(
                            rx.text("Account Age (days)", size = "1", color = "gray"),
                            rx.input(
                                name = "account_age_days",
                                type = "number",
                                value = State.tfd_form_data.get("account_age_days", ""),
                                min = 0,
                                step = 1,
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        spacing = "3",
                        width = "100%"
                    ),
                    # Row 2: Date and Time
                    rx.hstack(
                        rx.vstack(
                            rx.text("Date", size = "1", color = "gray"),
                            rx.input(
                                name = "timestamp_date",
                                type = "date",
                                value = State.tfd_form_data.get("timestamp_date", ""),
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        rx.vstack(
                            rx.text("Time", size = "1", color = "gray"),
                            rx.input(
                                name = "timestamp_time",
                                type = "time",
                                value = State.tfd_form_data.get("timestamp_time", ""),
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        spacing = "3",
                        width = "100%"
                    ),
                    # Row 3: Currency
                    rx.vstack(
                        rx.text("Currency", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_currency_options,
                            name = "currency",
                            value = State.tfd_form_data.get("currency", ""),
                            required = True,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    # Row 4: Merchant ID and Product Category
                    rx.hstack(
                        rx.vstack(
                            rx.text("Merchant ID", size = "1", color = "gray"),
                            rx.select(
                                State.tfd_merchant_id_options,
                                name = "merchant_id",
                                value = State.tfd_form_data.get("merchant_id", ""),
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        rx.vstack(
                            rx.text("Product Category", size = "1", color = "gray"),
                            rx.select(
                                State.tfd_product_category_options,
                                name = "product_category",
                                value = State.tfd_form_data.get("product_category", ""),
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        spacing = "3",
                        width = "100%"
                    ),
                    # Row 5: Transaction Type and Payment Method
                    rx.hstack(
                        rx.vstack(
                            rx.text("Transaction Type", size = "1", color = "gray"),
                            rx.select(
                                State.tfd_transaction_type_options,
                                name = "transaction_type",
                                value = State.tfd_form_data.get("transaction_type", ""),
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        rx.vstack(
                            rx.text("Payment Method", size = "1", color = "gray"),
                            rx.select(
                                State.tfd_payment_method_options,
                                name = "payment_method",
                                value = State.tfd_form_data.get("payment_method", ""),
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        spacing = "3",
                        width = "100%"
                    ),
                    # Row 6: Latitude and Longitude
                    rx.hstack(
                        rx.vstack(
                            rx.text("Latitude", size = "1", color = "gray"),
                            rx.input(
                                name = "lat",
                                type = "number",
                                value = State.tfd_form_data.get("lat", ""),
                                min = -90.0,
                                max = 90.0,
                                step = 0.0001,
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        rx.vstack(
                            rx.text("Longitude", size = "1", color = "gray"),
                            rx.input(
                                name = "lon",
                                type = "number",
                                value = State.tfd_form_data.get("lon", ""),
                                min = -180.0,
                                max = 180.0,
                                step = 0.0001,
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        spacing = "3",
                        width = "100%"
                    ),
                    # Row 7: Browser and OS
                    rx.hstack(
                        rx.vstack(
                            rx.text("Browser", size = "1", color = "gray"),
                            rx.select(
                                State.tfd_browser_options,
                                name = "browser",
                                value = State.tfd_form_data.get("browser", ""),
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        rx.vstack(
                            rx.text("OS", size = "1", color = "gray"),
                            rx.select(
                                State.tfd_os_options,
                                name = "os",
                                value = State.tfd_form_data.get("os", ""),
                                required = True,
                                width = "100%"
                            ),
                            spacing = "1",
                            align_items = "start",
                            width = "100%"
                        ),
                        spacing = "3",
                        width = "100%"
                    ),
                    # Row 8: CVV Provided and Billing Address Match
                    rx.hstack(
                        rx.checkbox(
                            "CVV Provided",
                            name = "cvv_provided",
                            checked = State.tfd_form_data.get("cvv_provided", False),
                            size = "1"
                        ),
                        rx.checkbox(
                            "Billing Address Match",
                            name = "billing_address_match",
                            checked = State.tfd_form_data.get("billing_address_match", False),
                            size = "1"
                        ),
                        spacing = "4",
                        width = "100%"
                    ),
                    # Display fields
                    rx.vstack(
                        rx.text(
                            f"Transaction ID: {State.tfd_form_data.get('transaction_id', '')}",
                            size = "1",
                            color = "gray"
                        ),
                        rx.text(
                            f"User ID: {State.tfd_form_data.get('user_id', '')}",
                            size = "1",
                            color = "gray"
                        ),
                        rx.text(
                            f"IP Address: {State.tfd_form_data.get('ip_address', '')}",
                            size = "1",
                            color = "gray"
                        ),
                        rx.text(
                            f"User Agent: {State.tfd_form_data.get('user_agent', '')}",
                            size = "1",
                            color = "gray"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    # Submit button
                    rx.button(
                        "Predict",
                        type = "submit",
                        size = "3",
                        width = "100%"
                    ),
                    spacing = "3",
                    align_items = "start",
                    width = "100%"
                ),
                on_submit = State.predict_transaction_fraud_detection,
                reset_on_submit = False,
                width = "100%"
            ),
            width = "30%"
        ),
        # Right column - Metrics and Results
        rx.vstack(
            rx.markdown(
                f"**Incremental ML model:** {State.incremental_ml_model_name["Transaction Fraud Detection"]}",
                size = "2",
                color = "gray"
            ),
            rx.heading("Classification Metrics", size = "6"),
            transaction_fraud_detection_metrics(),
            rx.divider(),
            rx.cond(
                State.tfd_prediction_show,
                rx.vstack(
                    rx.heading("Prediction", size = "6"),
                    rx.card(
                        rx.vstack(
                            # Prediction summary cards
                            rx.hstack(
                                rx.card(
                                    rx.vstack(
                                        rx.text("Prediction Result", size = "2", color = "gray", align = "center"),
                                        rx.text(
                                            State.tfd_prediction_text,
                                            size = "8",
                                            weight = "bold",
                                            color = State.tfd_prediction_color,
                                            align = "center"
                                        ),
                                        spacing = "2",
                                        align_items = "center",
                                        width = "100%"
                                    ),
                                    variant = "surface",
                                    size = "3"
                                ),
                                rx.card(
                                    rx.vstack(
                                        rx.text("Fraud Probability", size = "2", color = "gray", align = "center"),
                                        rx.text(
                                            f"{State.tfd_fraud_probability * 100:.2f}%",
                                            size = "8",
                                            weight = "bold",
                                            align = "center",
                                            color = rx.cond(
                                                State.tfd_fraud_probability > 0.5,
                                                "red",
                                                "green"
                                            )
                                        ),
                                        spacing = "2",
                                        align_items = "center",
                                        width = "100%"
                                    ),
                                    variant = "surface",
                                    size = "3"
                                ),
                                rx.card(
                                    rx.vstack(
                                        rx.text("Not Fraud Probability", size = "2", color = "gray", align = "center"),
                                        rx.text(
                                            f"{(1 - State.tfd_fraud_probability) * 100:.2f}%",
                                            size = "8",
                                            weight = "bold",
                                            align = "center",
                                            color = rx.cond(
                                                State.tfd_fraud_probability > 0.5,
                                                "green",
                                                "red"
                                            )
                                        ),
                                        spacing = "2",
                                        align_items = "center",
                                        width = "100%"
                                    ),
                                    variant = "surface",
                                    size = "3"
                                ),
                                spacing = "3",
                                width = "100%"
                            ),
                            # Visual probability bar
                            rx.vstack(
                                rx.text("Fraud Risk Assessment", size = "3", weight = "medium"),
                                rx.box(
                                    rx.box(
                                        width = f"{State.tfd_fraud_probability * 100}%",
                                        height = "40px",
                                        background = rx.cond(
                                            State.tfd_fraud_probability > 0.7,
                                            "linear-gradient(90deg, #FF0000, #FF6666)",
                                            rx.cond(
                                                State.tfd_fraud_probability > 0.3,
                                                "linear-gradient(90deg, #FFA500, #FFD700)",
                                                "linear-gradient(90deg, #00FF00, #66FF66)"
                                            )
                                        ),
                                        border_radius = "8px",
                                        transition = "width 0.5s ease-in-out"
                                    ),
                                    width = "100%",
                                    height = "40px",
                                    background = rx.color("gray", 3),
                                    border_radius = "8px",
                                    border = f"1px solid {rx.color('gray', 6)}"
                                ),
                                spacing = "2",
                                width = "100%"
                            ),
                            spacing = "4",
                            width = "100%"
                        ),
                        variant = "classic"
                    ),
                    spacing = "4",
                    align_items = "start",
                    width = "100%"
                )
            ),
            on_mount = State.get_mlflow_metrics("Transaction Fraud Detection"),
            align_items = "start",
            spacing = "4",
            width = "70%"
        ),
        spacing = "6",
        align_items = "start",
        width = "100%"
    )

def metric_card(label: str, value_var) -> rx.Component:
    """Create a styled metric card."""
    return rx.card(
        rx.vstack(
            rx.text(
                label,
                size = "2",
                weight = "medium",
                color = "gray",
                align = "center"
            ),
            rx.text(
                value_var,
                size = "7",
                weight = "bold",
                align = "center"
            ),
            spacing = "2",
            align_items = "center",
            justify = "center",
            height = "100%"
        ),
        variant = "surface",
        size = "2"
    )


def transaction_fraud_detection_metrics() -> rx.Component:
    """Display MLflow classification metrics for Transaction Fraud Detection as individual cards."""
    return rx.grid(
        metric_card("F1", State.tfd_metric_f1),
        metric_card("Accuracy", State.tfd_metric_accuracy),
        metric_card("Recall", State.tfd_metric_recall),
        metric_card("Precision", State.tfd_metric_precision),
        metric_card("ROCAUC", State.tfd_metric_rocauc),
        metric_card("Geometric Mean", State.tfd_metric_geometric_mean),
        columns = "3",
        spacing = "3",
        width = "100%"
    )