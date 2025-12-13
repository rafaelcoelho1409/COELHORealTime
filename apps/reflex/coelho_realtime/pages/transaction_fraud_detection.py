import reflex as rx

def index() -> rx.Component:
    return rx.container(
        rx.heading(
            "Transaction Fraud Detection", 
            size = "7"),
        rx.link(
            "Back to Home", 
            href = "/"),
        padding = "2em",
    )