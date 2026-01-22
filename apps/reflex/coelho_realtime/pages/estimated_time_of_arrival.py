import reflex as rx
from ..components import (
    coelho_realtime_navbar,
    page_tabs,
    estimated_time_of_arrival_form,
    estimated_time_of_arrival_batch_form,
    delta_lake_sql_tab,
)
from ..states import ETAState, SharedState


PROJECT_NAME = "Estimated Time of Arrival"
MODEL_KEY = "ml_training/river/eta.py"


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
                rx.cond(
                    ETAState.is_batch_ml_tab,
                    # Batch ML tab content
                    estimated_time_of_arrival_batch_form("XGBRegressor", PROJECT_NAME),
                    # Incremental ML tab content
                    estimated_time_of_arrival_form(MODEL_KEY, PROJECT_NAME),
                ),
            ),
            padding="2em",
            width="100%"
        ),
        # On mount: page initialization + form randomization (instant local)
        on_mount=[
            ETAState.randomize_eta_form,  # Populate form with random values (local, instant)
            ETAState.init_page(MODEL_KEY, PROJECT_NAME),  # Fetch MLflow metrics (Incremental ML)
            SharedState.init_batch_page(PROJECT_NAME),  # Check batch model + fetch metrics (Batch ML)
        ],
        # On unmount: cleanup when leaving the page
        on_unmount=ETAState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )