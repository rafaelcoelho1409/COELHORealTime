import reflex as rx

def index() -> rx.Component:
    return rx.container(
        rx.heading("Welcome to COELHO RealTime", size = "9"),
        rx.text("Multi-page Reflex application, hello Rafael"),
        rx.link("Counter Example", href = "/counter"),
    )