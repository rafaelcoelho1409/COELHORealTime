import reflex as rx
from ..resources import (
    coelho_realtime_navbar,
    page_tabs,
    ml_training_switch,
    estimated_time_of_arrival_form
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
            rx.vstack(
                # ML Training switch - user controls when to start/stop
                ml_training_switch(MODEL_KEY, PROJECT_NAME),
                estimated_time_of_arrival_form(),
                spacing = "4",
                align_items = "start",
                width = "100%"
            ),
            padding = "2em",
            width = "100%"
        ),
        # On mount: set page context and load sample data (don't auto-start ML)
        on_mount = [
            State.set_current_page_model(MODEL_KEY),
            State.update_sample(PROJECT_NAME),
        ],
        # On unmount: cleanup when leaving the page
        on_unmount = State.cleanup_on_page_leave(PROJECT_NAME),
        spacing = "0",
        width = "100%"
    )