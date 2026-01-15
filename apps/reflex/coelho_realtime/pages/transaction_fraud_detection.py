import reflex as rx
from ..components import (
    coelho_realtime_navbar,
    page_tabs,
    transaction_fraud_detection_form,
    transaction_fraud_detection_batch_form,
    delta_lake_sql_tab,
    ml_training_switch,
)
from ..states import TFDState


PROJECT_NAME = "Transaction Fraud Detection"
MODEL_KEY = "transaction_fraud_detection_river.py"


def index() -> rx.Component:
    return rx.vstack(
        coelho_realtime_navbar(),
        rx.box(
            page_tabs(),
            padding_x="2em",
            width="100%"
        ),
        rx.box(
            rx.cond(
                TFDState.is_delta_lake_sql_tab,
                # Delta Lake SQL tab content
                delta_lake_sql_tab(),
                rx.cond(
                    TFDState.is_batch_ml_tab,
                    # Batch ML tab content
                    rx.vstack(
                        transaction_fraud_detection_batch_form(),
                        spacing="4",
                        width="100%"
                    ),
                    # Incremental ML tab content - form includes ML training switch
                    transaction_fraud_detection_form(MODEL_KEY, PROJECT_NAME),
                ),
            ),
            padding="2em",
            width="100%"
        ),
        # On mount: combined page initialization (single HTTP call)
        on_mount=[
            TFDState.init_page(MODEL_KEY, PROJECT_NAME),
            TFDState.check_batch_model_available(PROJECT_NAME),
        ],
        # On unmount: cleanup when leaving the page
        on_unmount=TFDState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
