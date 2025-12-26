import reflex as rx
from ..resources import (
    coelho_realtime_navbar,
    page_tabs,
    transaction_fraud_detection_form,
    transaction_fraud_detection_batch_form,
    ml_training_switch
)
from ..state import State


PROJECT_NAME = "Transaction Fraud Detection"
MODEL_KEY = "transaction_fraud_detection_river.py"


def index() -> rx.Component:
    return rx.vstack(
        coelho_realtime_navbar(),
        rx.box(
            page_tabs(),
            padding_x = "2em",
            width = "100%"
        ),
        rx.box(
            rx.cond(
                State.is_batch_ml_tab,
                # Batch ML tab content
                rx.vstack(
                    transaction_fraud_detection_batch_form(),
                    spacing = "4",
                    width = "100%"
                ),
                # Incremental ML tab content
                rx.vstack(
                    # ML Training switch - user controls when to start/stop
                    ml_training_switch(MODEL_KEY, PROJECT_NAME),
                    # Main form content
                    transaction_fraud_detection_form(),
                    spacing = "4",
                    width = "100%"
                ),
            ),
            padding = "2em",
            width = "100%"
        ),
        # On mount: set page context, load sample data, and check batch model availability
        on_mount = [
            State.set_current_page_model(MODEL_KEY),
            State.update_sample(PROJECT_NAME),
            State.check_batch_model_available(PROJECT_NAME),
        ],
        # On unmount: cleanup when leaving the page
        on_unmount = State.cleanup_on_page_leave(PROJECT_NAME),
        spacing = "0",
        width = "100%"
    )
