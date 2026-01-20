"""TFD Incremental ML page - River-based real-time fraud detection."""
import reflex as rx
from ...components import (
    coelho_realtime_navbar,
    page_sub_nav,
    transaction_fraud_detection_form,
)
from ...states import TFDState


PROJECT_NAME = "Transaction Fraud Detection"
MODEL_KEY = "transaction_fraud_detection_river.py"
BASE_ROUTE = "/tfd"


def index() -> rx.Component:
    return rx.vstack(
        coelho_realtime_navbar(),
        rx.box(
            page_sub_nav(BASE_ROUTE, "incremental"),
            padding_x="2em",
            padding_top="1em",
            width="100%"
        ),
        rx.box(
            transaction_fraud_detection_form(MODEL_KEY, PROJECT_NAME),
            padding="2em",
            width="100%"
        ),
        on_mount=[
            TFDState.randomize_tfd_form,
            TFDState.init_page(MODEL_KEY, PROJECT_NAME),
        ],
        on_unmount=TFDState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
