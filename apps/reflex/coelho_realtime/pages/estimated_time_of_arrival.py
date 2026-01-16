import reflex as rx
from ..components import (
    coelho_realtime_navbar,
    page_tabs,
    estimated_time_of_arrival_form,
    delta_lake_sql_tab,
)
from ..states import ETAState


PROJECT_NAME = "Estimated Time of Arrival"
MODEL_KEY = "estimated_time_of_arrival_river.py"


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
                ETAState.is_delta_lake_sql_tab,
                # Delta Lake SQL tab content
                delta_lake_sql_tab(),
                # Incremental ML tab content (ETA only has incremental ML)
                estimated_time_of_arrival_form(MODEL_KEY, PROJECT_NAME),
            ),
            padding="2em",
            width="100%"
        ),
        # On mount: page initialization + form randomization (instant local)
        on_mount=[
            ETAState.randomize_eta_form,  # Populate form with random values (local, instant)
            ETAState.init_page(MODEL_KEY, PROJECT_NAME),  # Fetch MLflow metrics
        ],
        # On unmount: cleanup when leaving the page
        on_unmount=ETAState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )