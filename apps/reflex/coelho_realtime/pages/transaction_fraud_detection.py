import reflex as rx
from ..resources import coelho_realtime_navbar
from ..state import State


def index() -> rx.Component:
    return rx.vstack(
        coelho_realtime_navbar(),
        align_items = "center"
    )