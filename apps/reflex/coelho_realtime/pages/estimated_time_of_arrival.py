import reflex as rx
from ..resources import (
    coelho_realtime_navbar,
    page_tabs,
    ml_training_switch,
    estimated_time_of_arrival_form,
    delta_lake_sql_tab
)
from ..state import State


PROJECT_NAME = "Estimated Time of Arrival"
MODEL_KEY = "estimated_time_of_arrival_river.py"


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
                State.is_delta_lake_sql_tab,
                # Delta Lake SQL tab content
                delta_lake_sql_tab(),
                # Incremental ML tab content (ETA only has incremental ML)
                estimated_time_of_arrival_form(MODEL_KEY, PROJECT_NAME),
            ),
            padding = "2em",
            width = "100%"
        ),
        # On mount: combined page initialization (single HTTP call)
        on_mount = State.init_page(MODEL_KEY, PROJECT_NAME),
        # On unmount: cleanup when leaving the page
        on_unmount = State.cleanup_on_page_leave(PROJECT_NAME),
        spacing = "0",
        width = "100%"
    )