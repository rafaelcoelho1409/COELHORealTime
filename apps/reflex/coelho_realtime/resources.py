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
                    rx.icon("activity", size = 18, color = rx.cond(
                        State.ml_training_enabled,
                        rx.color("green", 9),
                        rx.color("gray", 9)
                    )),
                    rx.text(
                        "Real-time ML Training",
                        size = "3",
                        weight = "medium"
                    ),
                    spacing = "2",
                    align_items = "center"
                ),
                rx.text(
                    rx.cond(
                        State.ml_training_enabled,
                        "Processing live Kafka stream data",
                        "Toggle to start processing live data"
                    ),
                    size = "1",
                    color = "gray"
                ),
                spacing = "1",
                align_items = "start"
            ),
            rx.switch(
                checked = State.ml_training_enabled,
                on_change = lambda checked: State.toggle_ml_training(
                    checked,
                    model_key,
                    project_name
                ),
                size = "3"
            ),
            justify = "between",
            align_items = "center",
            width = "100%"
        ),
        variant = "surface",
        size = "2",
        width = "100%"
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
                    rx.menu.root(
                        rx.menu.trigger(
                            rx.button(
                                rx.hstack(
                                    rx.text("Services", size = "3", weight = "medium"),
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
                                rx.link(
                                    rx.hstack(
                                        rx.icon("zap", size = 16, color = rx.color("green", 9)),
                                        rx.text("FastAPI", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "http://localhost:8001/docs",
                                    is_external = True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.icon("flask-conical", size = 16, color = rx.color("blue", 9)),
                                        rx.text("MLflow", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "http://localhost:5001",
                                    is_external = True
                                )
                            ),
                            rx.menu.separator(),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.icon("activity", size = 16, color = rx.color("orange", 9)),
                                        rx.text("Prometheus", size = "3", weight = "medium", color = "gray"),
                                        rx.badge("Soon", color_scheme = "gray", size = "1"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "#",
                                    is_external = True
                                ),
                                disabled = True
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.icon("bar-chart-3", size = 16, color = rx.color("yellow", 9)),
                                        rx.text("Grafana", size = "3", weight = "medium", color = "gray"),
                                        rx.badge("Soon", color_scheme = "gray", size = "1"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "#",
                                    is_external = True
                                ),
                                disabled = True
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
        # Left column - Form (no form wrapper, using on_change handlers)
        rx.card(
            rx.vstack(
                # Form Legend
                rx.hstack(
                    rx.icon("credit-card", size = 20, color = rx.color("accent", 10)),
                    rx.heading("Transaction Details", size = "4", weight = "bold"),
                    spacing = "2",
                    align_items = "center"
                ),
                rx.text(
                    "Enter transaction data to predict fraud probability using the real-time ML model.",
                    size = "2",
                    color = "gray"
                ),
                # Randomize button
                rx.button(
                    rx.hstack(
                        rx.icon("shuffle", size = 14),
                        rx.text("Randomize All Fields", size = "2"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    on_click = State.randomize_tfd_form,
                    variant = "soft",
                    color_scheme = "blue",
                    size = "2",
                    width = "100%"
                ),
                rx.divider(),
                # Row 1: Amount and Account Age
                rx.hstack(
                    rx.vstack(
                        rx.text("Amount", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.tfd_form_data.get("amount", ""),
                            on_change = lambda v: State.update_tfd("amount", v),
                            step = 0.01,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Account Age (days)", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.tfd_form_data.get("account_age_days", ""),
                            on_change = lambda v: State.update_tfd("account_age_days", v),
                            min = 0,
                            step = 1,
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
                            type = "date",
                            value = State.tfd_form_data.get("timestamp_date", ""),
                            on_change = lambda v: State.update_tfd("timestamp_date", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Time", size = "1", color = "gray"),
                        rx.input(
                            type = "time",
                            value = State.tfd_form_data.get("timestamp_time", ""),
                            on_change = lambda v: State.update_tfd("timestamp_time", v),
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
                        State.tfd_options["currency"],
                        value = State.tfd_form_data.get("currency", ""),
                        on_change = lambda v: State.update_tfd("currency", v),
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
                            State.tfd_options["merchant_id"],
                            value = State.tfd_form_data.get("merchant_id", ""),
                            on_change = lambda v: State.update_tfd("merchant_id", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Product Category", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["product_category"],
                            value = State.tfd_form_data.get("product_category", ""),
                            on_change = lambda v: State.update_tfd("product_category", v),
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
                            State.tfd_options["transaction_type"],
                            value = State.tfd_form_data.get("transaction_type", ""),
                            on_change = lambda v: State.update_tfd("transaction_type", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Payment Method", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["payment_method"],
                            value = State.tfd_form_data.get("payment_method", ""),
                            on_change = lambda v: State.update_tfd("payment_method", v),
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
                            type = "number",
                            value = State.tfd_form_data.get("lat", ""),
                            on_change = lambda v: State.update_tfd("lat", v),
                            min = -90.0,
                            max = 90.0,
                            step = 0.0001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Longitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.tfd_form_data.get("lon", ""),
                            on_change = lambda v: State.update_tfd("lon", v),
                            min = -180.0,
                            max = 180.0,
                            step = 0.0001,
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
                            State.tfd_options["browser"],
                            value = State.tfd_form_data.get("browser", ""),
                            on_change = lambda v: State.update_tfd("browser", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("OS", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["os"],
                            value = State.tfd_form_data.get("os", ""),
                            on_change = lambda v: State.update_tfd("os", v),
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
                        checked = State.tfd_form_data.get("cvv_provided", False),
                        on_change = lambda v: State.update_tfd("cvv_provided", v),
                        size = "1"
                    ),
                    rx.checkbox(
                        "Billing Address Match",
                        checked = State.tfd_form_data.get("billing_address_match", False),
                        on_change = lambda v: State.update_tfd("billing_address_match", v),
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
                # Predict button (regular button with on_click)
                rx.button(
                    "Predict",
                    on_click = State.predict_transaction_fraud_detection,
                    size = "3",
                    width = "100%"
                ),
                spacing = "3",
                align_items = "start",
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
            rx.hstack(
                rx.heading("Classification Metrics", size = "6"),
                rx.button(
                    rx.icon("refresh-cw", size = 16),
                    on_click = State.refresh_mlflow_metrics("Transaction Fraud Detection"),
                    size = "1",
                    variant = "ghost",
                    cursor = "pointer",
                    title = "Refresh metrics"
                ),
                align_items = "center",
                spacing = "2"
            ),
            transaction_fraud_detection_metrics(),
            rx.divider(),
            # Prediction section - header always visible
            rx.hstack(
                rx.icon("shield-alert", size = 20, color = rx.color("accent", 10)),
                rx.heading("Prediction Result", size = "5", weight = "bold"),
                spacing = "2",
                align_items = "center",
                width = "100%"
            ),
            rx.cond(
                State.tfd_prediction_show,
                # Show prediction results when available
                rx.card(
                    rx.vstack(
                        # Plotly Gauge Chart
                        rx.plotly(data = State.tfd_fraud_gauge, width = "100%"),
                        # Prediction summary cards
                        rx.hstack(
                            rx.card(
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon("alert-triangle", size = 16, color = State.tfd_prediction_color),
                                        rx.text("Classification", size = "2", color = "gray"),
                                        spacing = "1",
                                        align_items = "center"
                                    ),
                                    rx.text(
                                        State.tfd_prediction_text,
                                        size = "7",
                                        weight = "bold",
                                        color = State.tfd_prediction_color,
                                        align = "center"
                                    ),
                                    spacing = "2",
                                    align_items = "center",
                                    width = "100%"
                                ),
                                variant = "surface",
                                size = "3",
                                width = "100%"
                            ),
                            rx.card(
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon("percent", size = 16, color = "red"),
                                        rx.text("Fraud", size = "2", color = "gray"),
                                        spacing = "1",
                                        align_items = "center"
                                    ),
                                    rx.text(
                                        f"{State.tfd_fraud_probability * 100:.2f}%",
                                        size = "7",
                                        weight = "bold",
                                        align = "center",
                                        color = "red"
                                    ),
                                    spacing = "2",
                                    align_items = "center",
                                    width = "100%"
                                ),
                                variant = "surface",
                                size = "3",
                                width = "100%"
                            ),
                            rx.card(
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon("check-circle", size = 16, color = "green"),
                                        rx.text("Not Fraud", size = "2", color = "gray"),
                                        spacing = "1",
                                        align_items = "center"
                                    ),
                                    rx.text(
                                        f"{(1 - State.tfd_fraud_probability) * 100:.2f}%",
                                        size = "7",
                                        weight = "bold",
                                        align = "center",
                                        color = "green"
                                    ),
                                    spacing = "2",
                                    align_items = "center",
                                    width = "100%"
                                ),
                                variant = "surface",
                                size = "3",
                                width = "100%"
                            ),
                            spacing = "3",
                            width = "100%"
                        ),
                        spacing = "4",
                        width = "100%"
                    ),
                    variant = "classic",
                    width = "100%"
                ),
                # Show info message when no prediction yet
                rx.callout(
                    rx.text("Fill in the transaction details and click the ", rx.text("Predict", weight = "bold"), " button to get the fraud probability."),
                    icon = "info",
                    color = "blue",
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
                color = "gray"
            ),
            rx.text(
                value_var,
                size = "6",
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
        metric_card("F1 Score", State.tfd_metrics["f1"]),
        metric_card("Accuracy", State.tfd_metrics["accuracy"]),
        metric_card("Recall", State.tfd_metrics["recall"]),
        metric_card("Precision", State.tfd_metrics["precision"]),
        metric_card("ROC AUC", State.tfd_metrics["rocauc"]),
        metric_card("Geo Mean", State.tfd_metrics["geometric_mean"]),
        columns = "3",
        spacing = "3",
        width = "100%"
    )


## ESTIMATED TIME OF ARRIVAL
def estimated_time_of_arrival_form() -> rx.Component:
    return rx.hstack(
        # Left column - Form
        rx.card(
            rx.vstack(
                # Form Legend
                rx.hstack(
                    rx.icon("clock", size = 20, color = rx.color("accent", 10)),
                    rx.heading("Trip Details", size = "4", weight = "bold"),
                    spacing = "2",
                    align_items = "center"
                ),
                rx.text(
                    "Enter trip data to predict estimated time of arrival using the real-time ML model.",
                    size = "2",
                    color = "gray"
                ),
                # Randomize button
                rx.button(
                    rx.hstack(
                        rx.icon("shuffle", size = 14),
                        rx.text("Randomize All Fields", size = "2"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    on_click = State.randomize_eta_form,
                    variant = "soft",
                    color_scheme = "blue",
                    size = "2",
                    width = "100%"
                ),
                rx.divider(),
                # Row 1: Driver ID and Vehicle ID
                rx.hstack(
                    rx.vstack(
                        rx.text("Driver ID", size = "1", color = "gray"),
                        rx.select(
                            State.eta_options["driver_id"],
                            value = State.eta_form_data.get("driver_id", ""),
                            on_change = lambda v: State.update_eta("driver_id", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Vehicle ID", size = "1", color = "gray"),
                        rx.select(
                            State.eta_options["vehicle_id"],
                            value = State.eta_form_data.get("vehicle_id", ""),
                            on_change = lambda v: State.update_eta("vehicle_id", v),
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
                            type = "date",
                            value = State.eta_form_data.get("timestamp_date", ""),
                            on_change = lambda v: State.update_eta("timestamp_date", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Time", size = "1", color = "gray"),
                        rx.input(
                            type = "time",
                            value = State.eta_form_data.get("timestamp_time", ""),
                            on_change = lambda v: State.update_eta("timestamp_time", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 3: Origin Latitude and Longitude
                rx.hstack(
                    rx.vstack(
                        rx.text("Origin Latitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("origin_lat", ""),
                            on_change = lambda v: State.update_eta("origin_lat", v),
                            min = 29.5,
                            max = 30.1,
                            step = 0.0001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Origin Longitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("origin_lon", ""),
                            on_change = lambda v: State.update_eta("origin_lon", v),
                            min = -95.8,
                            max = -95.0,
                            step = 0.0001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 4: Destination Latitude and Longitude
                rx.hstack(
                    rx.vstack(
                        rx.text("Destination Latitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("destination_lat", ""),
                            on_change = lambda v: State.update_eta("destination_lat", v),
                            min = 29.5,
                            max = 30.1,
                            step = 0.0001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Destination Longitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("destination_lon", ""),
                            on_change = lambda v: State.update_eta("destination_lon", v),
                            min = -95.8,
                            max = -95.0,
                            step = 0.0001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Random coordinates button
                rx.button(
                    rx.hstack(
                        rx.icon("shuffle", size = 14),
                        rx.text("Random Coordinates", size = "1"),
                        spacing = "1",
                        align_items = "center"
                    ),
                    on_click = State.generate_random_eta_coordinates,
                    variant = "outline",
                    size = "1",
                    width = "100%"
                ),
                # Row 5: Weather and Vehicle Type
                rx.hstack(
                    rx.vstack(
                        rx.text("Weather", size = "1", color = "gray"),
                        rx.select(
                            State.eta_options["weather"],
                            value = State.eta_form_data.get("weather", ""),
                            on_change = lambda v: State.update_eta("weather", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Vehicle Type", size = "1", color = "gray"),
                        rx.select(
                            State.eta_options["vehicle_type"],
                            value = State.eta_form_data.get("vehicle_type", ""),
                            on_change = lambda v: State.update_eta("vehicle_type", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 6: Hour of Day and Driver Rating
                rx.hstack(
                    rx.vstack(
                        rx.text("Hour of Day", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("hour_of_day", ""),
                            on_change = lambda v: State.update_eta("hour_of_day", v),
                            min = 0,
                            max = 23,
                            step = 1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Driver Rating", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("driver_rating", ""),
                            on_change = lambda v: State.update_eta("driver_rating", v),
                            min = 3.5,
                            max = 5.0,
                            step = 0.1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 7: Debug Traffic Factor and Debug Weather Factor
                rx.hstack(
                    rx.vstack(
                        rx.text("Debug Traffic Factor", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("debug_traffic_factor", ""),
                            on_change = lambda v: State.update_eta("debug_traffic_factor", v),
                            min = 0.3,
                            max = 1.9,
                            step = 0.1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Debug Weather Factor", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("debug_weather_factor", ""),
                            on_change = lambda v: State.update_eta("debug_weather_factor", v),
                            min = 1.0,
                            max = 2.0,
                            step = 0.1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 8: Debug Incident Delay and Debug Driver Factor
                rx.hstack(
                    rx.vstack(
                        rx.text("Debug Incident Delay (s)", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("debug_incident_delay_seconds", ""),
                            on_change = lambda v: State.update_eta("debug_incident_delay_seconds", v),
                            min = 0,
                            max = 1800,
                            step = 1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Debug Driver Factor", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("debug_driver_factor", ""),
                            on_change = lambda v: State.update_eta("debug_driver_factor", v),
                            min = 0.85,
                            max = 1.15,
                            step = 0.01,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 9: Temperature and Initial Estimated Travel Time
                rx.hstack(
                    rx.vstack(
                        rx.text("Temperature (°C)", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_form_data.get("temperature_celsius", ""),
                            on_change = lambda v: State.update_eta("temperature_celsius", v),
                            min = -50.0,
                            max = 50.0,
                            step = 0.1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Initial Est. Travel Time (s)", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.eta_initial_estimated_travel_time_seconds,
                            disabled = True,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Display fields
                rx.vstack(
                    rx.text(
                        f"Trip ID: {State.eta_form_data.get('trip_id', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    rx.text(
                        f"Estimated Distance: {State.eta_estimated_distance_km} km",
                        size = "1",
                        color = "gray"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Predict button
                rx.button(
                    "Predict",
                    on_click = State.predict_eta,
                    size = "3",
                    width = "100%"
                ),
                spacing = "3",
                align_items = "start",
                width = "100%"
            ),
            width = "30%"
        ),
        # Right column - Metrics and Results
        rx.vstack(
            rx.markdown(
                f"**Incremental ML model:** {State.incremental_ml_model_name['Estimated Time of Arrival']}",
                size = "2",
                color = "gray"
            ),
            rx.hstack(
                rx.heading("Regression Metrics", size = "6"),
                rx.button(
                    rx.icon("refresh-cw", size = 16),
                    on_click = State.refresh_mlflow_metrics("Estimated Time of Arrival"),
                    size = "1",
                    variant = "ghost",
                    cursor = "pointer",
                    title = "Refresh metrics"
                ),
                align_items = "center",
                spacing = "2"
            ),
            estimated_time_of_arrival_metrics(),
            rx.divider(),
            # Prediction section - always show both cards
            rx.hstack(
                # Left: Map - always visible with current coordinates
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("map-pin", size = 16, color = rx.color("accent", 10)),
                            rx.text("Origin and Destination", size = "3", weight = "bold"),
                            spacing = "2",
                            align_items = "center"
                        ),
                        rx.plotly(data = State.eta_map_figure, width = "100%"),
                        rx.text(
                            f"Estimated Distance: {State.eta_estimated_distance_km} km",
                            size = "2",
                            color = "gray"
                        ),
                        spacing = "2",
                        width = "100%",
                        height = "100%"
                    ),
                    variant = "surface",
                    width = "50%",
                    height = "380px"
                ),
                # Right: ETA Prediction - shows info or results
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("clock", size = 16, color = rx.color("accent", 10)),
                            rx.text("ETA - Prediction", size = "3", weight = "bold"),
                            spacing = "2",
                            align_items = "center"
                        ),
                        rx.cond(
                            State.eta_prediction_show,
                            # Show prediction results when available
                            rx.box(
                                rx.plotly(data = State.eta_prediction_figure, width = "100%"),
                                width = "100%",
                                flex = "1",
                                display = "flex",
                                align_items = "center",
                                justify_content = "center"
                            ),
                            # Show info message when no prediction yet
                            rx.box(
                                rx.callout(
                                    rx.text("Click the ", rx.text("Predict", weight = "bold"), " button to get the estimated time of arrival."),
                                    icon = "info",
                                    color = "blue",
                                    width = "100%"
                                ),
                                width = "100%",
                                flex = "1",
                                display = "flex",
                                align_items = "center",
                                justify_content = "center"
                            )
                        ),
                        spacing = "2",
                        width = "100%",
                        height = "100%"
                    ),
                    variant = "surface",
                    width = "50%",
                    height = "380px"
                ),
                spacing = "3",
                width = "100%",
                align_items = "stretch"
            ),
            on_mount = State.get_mlflow_metrics("Estimated Time of Arrival"),
            align_items = "start",
            spacing = "4",
            width = "70%"
        ),
        spacing = "6",
        align_items = "start",
        width = "100%"
    )


def estimated_time_of_arrival_metrics() -> rx.Component:
    """Display MLflow regression metrics for Estimated Time of Arrival as individual cards."""
    return rx.grid(
        metric_card("MAE", State.eta_metrics["mae"]),
        metric_card("MAPE", State.eta_metrics["mape"]),
        metric_card("MSE", State.eta_metrics["mse"]),
        metric_card("R²", State.eta_metrics["r2"]),
        metric_card("RMSE", State.eta_metrics["rmse"]),
        metric_card("RMSLE", State.eta_metrics["rmsle"]),
        metric_card("SMAPE", State.eta_metrics["smape"]),
        columns = "4",
        spacing = "3",
        width = "100%"
    )


## E-COMMERCE CUSTOMER INTERACTIONS
def e_commerce_customer_interactions_form() -> rx.Component:
    return rx.hstack(
        # Left column - Form
        rx.card(
            rx.vstack(
                # Form Legend
                rx.hstack(
                    rx.icon("shopping-cart", size = 20, color = rx.color("accent", 10)),
                    rx.heading("Customer Interaction", size = "4", weight = "bold"),
                    spacing = "2",
                    align_items = "center"
                ),
                rx.text(
                    "Enter customer interaction data to predict cluster assignment using the real-time ML model.",
                    size = "2",
                    color = "gray"
                ),
                # Randomize button
                rx.button(
                    rx.hstack(
                        rx.icon("shuffle", size = 14),
                        rx.text("Randomize All Fields", size = "2"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    on_click = State.randomize_ecci_form,
                    variant = "soft",
                    color_scheme = "blue",
                    size = "2",
                    width = "100%"
                ),
                rx.divider(),
                # Row 1: Browser, Device Type, OS
                rx.hstack(
                    rx.vstack(
                        rx.text("Browser", size = "1", color = "gray"),
                        rx.select(
                            State.ecci_options["browser"],
                            value = State.ecci_form_data.get("browser", ""),
                            on_change = lambda v: State.update_ecci("browser", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Device Type", size = "1", color = "gray"),
                        rx.select(
                            State.ecci_options["device_type"],
                            value = State.ecci_form_data.get("device_type", ""),
                            on_change = lambda v: State.update_ecci("device_type", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("OS", size = "1", color = "gray"),
                        rx.select(
                            State.ecci_options["os"],
                            value = State.ecci_form_data.get("os", ""),
                            on_change = lambda v: State.update_ecci("os", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 2: Latitude and Longitude
                rx.hstack(
                    rx.vstack(
                        rx.text("Latitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.ecci_form_data.get("lat", ""),
                            on_change = lambda v: State.update_ecci("lat", v),
                            min = 29.5,
                            max = 30.1,
                            step = 0.001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Longitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.ecci_form_data.get("lon", ""),
                            on_change = lambda v: State.update_ecci("lon", v),
                            min = -95.8,
                            max = -95.0,
                            step = 0.001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Random coordinates button
                rx.button(
                    rx.hstack(
                        rx.icon("shuffle", size = 14),
                        rx.text("Random Coordinates", size = "1"),
                        spacing = "1",
                        align_items = "center"
                    ),
                    on_click = State.generate_random_ecci_coordinates,
                    variant = "outline",
                    size = "1",
                    width = "100%"
                ),
                # Row 3: Event Type and Price
                rx.hstack(
                    rx.vstack(
                        rx.text("Event Type", size = "1", color = "gray"),
                        rx.select(
                            State.ecci_options["event_type"],
                            value = State.ecci_form_data.get("event_type", ""),
                            on_change = lambda v: State.update_ecci("event_type", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Price", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.ecci_form_data.get("price", ""),
                            on_change = lambda v: State.update_ecci("price", v),
                            min = 0.0,
                            step = 0.01,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 4: Product Category and Product ID
                rx.hstack(
                    rx.vstack(
                        rx.text("Product Category", size = "1", color = "gray"),
                        rx.select(
                            State.ecci_options["product_category"],
                            value = State.ecci_form_data.get("product_category", ""),
                            on_change = lambda v: State.update_ecci("product_category", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Product ID", size = "1", color = "gray"),
                        rx.input(
                            value = State.ecci_form_data.get("product_id", ""),
                            on_change = lambda v: State.update_ecci("product_id", v),
                            placeholder = "e.g., prod_1050",
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 5: Referrer URL and Session Event Sequence
                rx.hstack(
                    rx.vstack(
                        rx.text("Referrer URL", size = "1", color = "gray"),
                        rx.input(
                            value = State.ecci_form_data.get("referrer_url", ""),
                            on_change = lambda v: State.update_ecci("referrer_url", v),
                            placeholder = "e.g., google.com, direct",
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Session Event Sequence", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.ecci_form_data.get("session_event_sequence", ""),
                            on_change = lambda v: State.update_ecci("session_event_sequence", v),
                            min = 1,
                            step = 1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 6: Quantity and Time on Page
                rx.hstack(
                    rx.vstack(
                        rx.text("Quantity", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.ecci_form_data.get("quantity", ""),
                            on_change = lambda v: State.update_ecci("quantity", v),
                            min = 1,
                            step = 1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Time on Page (seconds)", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.ecci_form_data.get("time_on_page_seconds", ""),
                            on_change = lambda v: State.update_ecci("time_on_page_seconds", v),
                            min = 0,
                            step = 1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 7: Date and Time
                rx.hstack(
                    rx.vstack(
                        rx.text("Date", size = "1", color = "gray"),
                        rx.input(
                            type = "date",
                            value = State.ecci_form_data.get("timestamp_date", ""),
                            on_change = lambda v: State.update_ecci("timestamp_date", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Time", size = "1", color = "gray"),
                        rx.input(
                            type = "time",
                            value = State.ecci_form_data.get("timestamp_time", ""),
                            on_change = lambda v: State.update_ecci("timestamp_time", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Display fields (read-only)
                rx.vstack(
                    rx.text(
                        f"Customer ID: {State.ecci_form_data.get('customer_id', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    rx.text(
                        f"Event ID: {State.ecci_form_data.get('event_id', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    rx.text(
                        f"Page URL: {State.ecci_form_data.get('page_url', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    rx.text(
                        f"Search Query: {State.ecci_form_data.get('search_query', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    rx.text(
                        f"Session ID: {State.ecci_form_data.get('session_id', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Predict button
                rx.button(
                    "Predict Cluster",
                    on_click = State.predict_ecci,
                    size = "3",
                    width = "100%"
                ),
                spacing = "3",
                align_items = "start",
                width = "100%"
            ),
            width = "30%"
        ),
        # Right column - Tabs for Prediction and Analytics
        rx.vstack(
            rx.markdown(
                f"**Incremental ML model:** {State.incremental_ml_model_name['E-Commerce Customer Interactions']}",
                size = "2",
                color = "gray"
            ),
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("Cluster Prediction", value = "prediction"),
                    rx.tabs.trigger("Cluster Analytics", value = "analytics"),
                ),
                # Tab 1: Cluster Prediction
                rx.tabs.content(
                    rx.vstack(
                        # Algorithm info
                        rx.callout(
                            rx.vstack(
                                rx.text(
                                    "DBSTREAM is a density-based clustering algorithm for data streams.",
                                    size = "2"
                                ),
                                rx.text(
                                    "It identifies clusters dynamically as data points arrive, without predefined cluster counts.",
                                    size = "2",
                                    color = "gray"
                                ),
                                spacing = "1",
                                align_items = "start"
                            ),
                            icon = "info",
                            color = "blue",
                            width = "100%"
                        ),
                        # Customer Location map - always visible
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("map-pin", size = 16, color = rx.color("accent", 10)),
                                    rx.text("Customer Location", size = "3", weight = "bold"),
                                    spacing = "2",
                                    align_items = "center"
                                ),
                                rx.plotly(data = State.ecci_location_figure, width = "100%"),
                                spacing = "2",
                                width = "100%"
                            ),
                            variant = "surface",
                            width = "100%"
                        ),
                        # Prediction boxes - always visible
                        rx.hstack(
                            # Left: Cluster prediction
                            rx.card(
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon("target", size = 16, color = rx.color("accent", 10)),
                                        rx.text("Predicted Cluster", size = "3", weight = "bold"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    rx.cond(
                                        State.ecci_prediction_show,
                                        rx.box(
                                            rx.plotly(data = State.ecci_prediction_figure, width = "100%"),
                                            width = "100%",
                                            flex = "1",
                                            display = "flex",
                                            align_items = "center",
                                            justify_content = "center"
                                        ),
                                        rx.box(
                                            rx.callout(
                                                rx.text("Click ", rx.text("Predict Cluster", weight = "bold"), " to identify the segment."),
                                                icon = "info",
                                                color = "blue",
                                                width = "100%"
                                            ),
                                            width = "100%",
                                            flex = "1",
                                            display = "flex",
                                            align_items = "center",
                                            justify_content = "center"
                                        )
                                    ),
                                    spacing = "2",
                                    width = "100%",
                                    height = "100%"
                                ),
                                variant = "surface",
                                width = "50%",
                                height = "320px"
                            ),
                            # Right: Feature distribution for predicted cluster
                            rx.card(
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon("bar-chart-2", size = 16, color = rx.color("accent", 10)),
                                        rx.text("Cluster Behavior", size = "3", weight = "bold"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    rx.cond(
                                        State.ecci_prediction_show,
                                        rx.vstack(
                                            rx.select(
                                                State.ecci_feature_options,
                                                value = State.ecci_selected_feature,
                                                on_change = State.set_ecci_selected_feature,
                                                size = "1",
                                                width = "100%"
                                            ),
                                            rx.plotly(data = State.ecci_selected_cluster_feature_figure, width = "100%"),
                                            spacing = "2",
                                            width = "100%",
                                            flex = "1"
                                        ),
                                        rx.box(
                                            rx.callout(
                                                rx.text("Cluster behavior shown after prediction."),
                                                icon = "info",
                                                color = "blue",
                                                width = "100%"
                                            ),
                                            width = "100%",
                                            flex = "1",
                                            display = "flex",
                                            align_items = "center",
                                            justify_content = "center"
                                        )
                                    ),
                                    spacing = "2",
                                    width = "100%",
                                    height = "100%"
                                ),
                                variant = "surface",
                                width = "50%",
                                height = "320px"
                            ),
                            spacing = "3",
                            width = "100%",
                            align_items = "stretch"
                        ),
                        # Cluster interpretation - only after prediction
                        rx.cond(
                            State.ecci_prediction_show,
                            rx.card(
                                rx.hstack(
                                    rx.icon("info", size = 16, color = rx.color("blue", 9)),
                                    rx.text(
                                        f"This customer interaction was assigned to Cluster {State.ecci_predicted_cluster}. Clusters represent groups of similar customer behaviors based on their browsing patterns, device usage, and purchase activities.",
                                        size = "2"
                                    ),
                                    spacing = "2",
                                    align_items = "start"
                                ),
                                variant = "surface",
                                width = "100%"
                            ),
                            rx.box()
                        ),
                        spacing = "4",
                        width = "100%",
                        padding_top = "1em"
                    ),
                    value = "prediction"
                ),
                # Tab 2: Cluster Analytics
                rx.tabs.content(
                    rx.vstack(
                        # Samples per cluster chart
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("pie-chart", size = 16, color = rx.color("accent", 10)),
                                    rx.text("Samples per Cluster", size = "3", weight = "bold"),
                                    rx.spacer(),
                                    rx.button(
                                        rx.hstack(
                                            rx.icon("refresh-cw", size = 14),
                                            rx.text("Refresh", size = "1"),
                                            spacing = "1"
                                        ),
                                        on_click = State.fetch_ecci_cluster_counts,
                                        variant = "outline",
                                        size = "1"
                                    ),
                                    spacing = "2",
                                    align_items = "center",
                                    width = "100%"
                                ),
                                rx.plotly(data = State.ecci_cluster_counts_figure, width = "100%"),
                                spacing = "2",
                                width = "100%"
                            ),
                            variant = "surface",
                            width = "100%"
                        ),
                        # Feature distribution across all clusters
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("bar-chart-2", size = 16, color = rx.color("accent", 10)),
                                    rx.text("Feature Distribution", size = "3", weight = "bold"),
                                    rx.spacer(),
                                    rx.select(
                                        State.ecci_feature_options,
                                        value = State.ecci_selected_feature,
                                        on_change = State.set_ecci_selected_feature,
                                        size = "1",
                                        width = "200px"
                                    ),
                                    spacing = "2",
                                    align_items = "center",
                                    width = "100%"
                                ),
                                rx.plotly(data = State.ecci_all_clusters_feature_figure, width = "100%"),
                                spacing = "2",
                                width = "100%"
                            ),
                            variant = "surface",
                            width = "100%"
                        ),
                        # Info callout
                        rx.callout(
                            rx.text(
                                "This tab shows aggregated statistics across all clusters. Use the feature selector to explore how different attributes are distributed across customer segments."
                            ),
                            icon = "info",
                            color = "blue",
                            width = "100%"
                        ),
                        spacing = "4",
                        width = "100%",
                        padding_top = "1em"
                    ),
                    value = "analytics"
                ),
                default_value = "prediction",
                width = "100%"
            ),
            align_items = "start",
            spacing = "4",
            width = "70%"
        ),
        spacing = "6",
        align_items = "start",
        width = "100%"
    )