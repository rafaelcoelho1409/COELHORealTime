import reflex as rx
from ..resources import coelho_realtime_navbar
from ..state import State


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
    #return rx.container(
    #    rx.heading("Welcome to COELHO RealTime", size = "9"),
    #    rx.text("Multi-page Reflex application, hello Rafael"),
    #    rx.link("Counter Example", href = "/counter"),
    #)