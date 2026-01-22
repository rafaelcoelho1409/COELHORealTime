"""ETA Delta Lake SQL page - DuckDB SQL query interface."""
import reflex as rx
from ...components import (
    coelho_realtime_navbar,
    page_sub_nav,
    delta_lake_sql_tab,
)
from ...states import ETAState, SharedState


PROJECT_NAME = "Estimated Time of Arrival"
BASE_ROUTE = "/eta"


def index() -> rx.Component:
    return rx.vstack(
        coelho_realtime_navbar(),
        rx.box(
            page_sub_nav(BASE_ROUTE, "sql"),
            padding_x="2em",
            padding_top="1em",
            width="100%"
        ),
        rx.box(
            delta_lake_sql_tab(),
            padding="2em",
            width="100%"
        ),
        on_mount=[
            SharedState.init_sql_page(PROJECT_NAME),
        ],
        on_unmount=ETAState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
