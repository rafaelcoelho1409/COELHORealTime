"""ECCI Incremental ML page - River-based real-time customer clustering."""
import reflex as rx
from ...components import (
    coelho_realtime_navbar,
    page_sub_nav,
    e_commerce_customer_interactions_form,
)
from ...states import ECCIState


PROJECT_NAME = "E-Commerce Customer Interactions"
MODEL_KEY = "e_commerce_customer_interactions_river.py"
BASE_ROUTE = "/ecci"


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
            e_commerce_customer_interactions_form(MODEL_KEY, PROJECT_NAME),
            padding="2em",
            width="100%"
        ),
        on_mount=[
            ECCIState.randomize_ecci_form,
            ECCIState.init_page(MODEL_KEY, PROJECT_NAME),
            ECCIState.fetch_ecci_cluster_counts,
            ECCIState.fetch_ecci_cluster_feature_counts,
        ],
        on_unmount=ECCIState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )
