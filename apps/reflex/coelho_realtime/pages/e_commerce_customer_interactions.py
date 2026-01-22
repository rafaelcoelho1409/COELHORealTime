import reflex as rx
from ..components import (
    coelho_realtime_navbar,
    page_tabs,
    e_commerce_customer_interactions_form,
    e_commerce_customer_interactions_batch_form,
    delta_lake_sql_tab,
)
from ..states import ECCIState, SharedState


PROJECT_NAME = "E-Commerce Customer Interactions"
MODEL_KEY = "ml_training/river/ecci.py"


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
                rx.cond(
                    ECCIState.is_batch_ml_tab,
                    # Batch ML tab content
                    e_commerce_customer_interactions_batch_form("KMeans", PROJECT_NAME),
                    # Incremental ML tab content
                    e_commerce_customer_interactions_form(MODEL_KEY, PROJECT_NAME),
                ),
            ),
            padding="2em",
            width="100%"
        ),
        # On mount: page initialization + form randomization (instant local)
        on_mount=[
            ECCIState.randomize_ecci_form,  # Populate form with random values (local, instant)
            ECCIState.init_page(MODEL_KEY, PROJECT_NAME),  # Fetch MLflow metrics (Incremental ML)
            ECCIState.fetch_ecci_cluster_counts,
            ECCIState.fetch_ecci_cluster_feature_counts,
            SharedState.init_batch_page(PROJECT_NAME),  # Check batch model + fetch metrics (Batch ML)
        ],
        # On unmount: cleanup when leaving the page
        on_unmount=ECCIState.cleanup_on_page_leave(PROJECT_NAME),
        spacing="0",
        width="100%"
    )