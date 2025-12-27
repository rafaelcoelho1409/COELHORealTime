import reflex as rx
from ..resources import (
    coelho_realtime_navbar,
    page_tabs,
    ml_training_switch,
    e_commerce_customer_interactions_form
)
from ..state import State


PROJECT_NAME = "E-Commerce Customer Interactions"
MODEL_KEY = "e_commerce_customer_interactions_river.py"


def index() -> rx.Component:
    return rx.vstack(
        coelho_realtime_navbar(),
        rx.box(
            page_tabs(),
            padding_x = "2em",
            width = "100%"
        ),
        rx.box(
            # Form includes ML training switch in left column
            e_commerce_customer_interactions_form(MODEL_KEY, PROJECT_NAME),
            padding = "2em",
            width = "100%"
        ),
        # On mount: set page context, load sample data, and fetch cluster analytics
        on_mount = [
            State.set_current_page_model(MODEL_KEY),
            State.update_sample(PROJECT_NAME),
            State.fetch_ecci_cluster_counts,
            State.fetch_ecci_cluster_feature_counts,
        ],
        # On unmount: cleanup when leaving the page
        on_unmount = State.cleanup_on_page_leave(PROJECT_NAME),
        spacing = "0",
        width = "100%"
    )