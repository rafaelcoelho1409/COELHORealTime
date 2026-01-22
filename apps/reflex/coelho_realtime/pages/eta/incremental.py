"""ETA Incremental ML page - River-based real-time ETA prediction."""
import reflex as rx
from ...components import (
    coelho_realtime_navbar,
    page_sub_nav,
    estimated_time_of_arrival_form,
)
from ...states import ETAState


PROJECT_NAME = "Estimated Time of Arrival"
MODEL_KEY = "ml_training/river/eta.py"
BASE_ROUTE = "/eta"


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
            estimated_time_of_arrival_form(MODEL_KEY, PROJECT_NAME),
            padding="2em",
            width="100%"
        ),
        on_mount=[
            ETAState.randomize_eta_form,
            ETAState.init_page(MODEL_KEY, PROJECT_NAME),
        ],
        on_unmount=ETAState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
