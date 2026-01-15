import reflex as rx
from ..components import (
    coelho_realtime_navbar,
    page_tabs,
    e_commerce_customer_interactions_form,
    delta_lake_sql_tab,
)
from ..states import ECCIState


PROJECT_NAME = "E-Commerce Customer Interactions"
MODEL_KEY = "e_commerce_customer_interactions_river.py"


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
                ECCIState.is_delta_lake_sql_tab,
                # Delta Lake SQL tab content
                delta_lake_sql_tab(),
                # Incremental ML tab content (ECCI only has incremental ML)
                e_commerce_customer_interactions_form(MODEL_KEY, PROJECT_NAME),
            ),
            padding="2em",
            width="100%"
        ),
        # On mount: combined page initialization + ECCI-specific cluster analytics
        on_mount=[
            ECCIState.init_page(MODEL_KEY, PROJECT_NAME),
            ECCIState.fetch_ecci_cluster_counts,
            ECCIState.fetch_ecci_cluster_feature_counts,
        ],
        # On unmount: cleanup when leaving the page
        on_unmount=ECCIState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )