import reflex as rx
from ..components import coelho_realtime_navbar


def index() -> rx.Component:
    return rx.vstack(
        coelho_realtime_navbar(),
        rx.image(
            src = "/coelho_realtime_logo.png",
            width = "50%",
            height = "50%",
            border_radius = "10%",
        ),
        align_items = "center"
    )