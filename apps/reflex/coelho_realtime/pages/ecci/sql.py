"""ECCI Delta Lake SQL page - DuckDB/Polars SQL query interface."""
import reflex as rx
from ...components import (
    coelho_realtime_navbar,
    page_sub_nav,
    delta_lake_sql_tab,
)
from ...states import ECCIState, SharedState


PROJECT_NAME = "E-Commerce Customer Interactions"
BASE_ROUTE = "/ecci"


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
        on_unmount=ECCIState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
