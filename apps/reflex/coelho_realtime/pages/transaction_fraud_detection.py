import reflex as rx
from ..components import (
    coelho_realtime_navbar,
    page_tabs,
    transaction_fraud_detection_form,
    transaction_fraud_detection_batch_form,
    delta_lake_sql_tab,
)
from ..states import TFDState, SharedState


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
                    transaction_fraud_detection_batch_form("XGBClassifier", PROJECT_NAME),
                    # Incremental ML tab content - form includes ML training switch
                    transaction_fraud_detection_form(MODEL_KEY, PROJECT_NAME),
                ),
            ),
            padding="2em",
            width="100%"
        ),
        # On mount: page initialization + form randomization (instant local)
        on_mount=[
            TFDState.randomize_tfd_form,  # Populate form with random values (local, instant)
            TFDState.init_page(MODEL_KEY, PROJECT_NAME),  # Fetch MLflow metrics
            SharedState.check_batch_model_available(PROJECT_NAME),  # Check batch model
        ],
        # On unmount: cleanup when leaving the page
        on_unmount=TFDState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
