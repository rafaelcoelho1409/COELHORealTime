import reflex as rx

def navbar_link(text: str, url: str) -> rx.Component:
    return rx.link(
        rx.text(
            text, 
            size = "4", 
            weight = "medium"), 
            href = url)


def coelho_realtime_navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.image(
                    src = "/coelho_realtime_logo.png",
                    width = "5em",
                    height = "auto",
                    border_radius = "10%"
                ),
                rx.heading(
                    "COELHO RealTime",
                    size = "7",
                    weight = "bold"
                ),
                align_items = "center"
            ),
            rx.hstack(
                navbar_link("Home", "/"),
                rx.menu.root(
                    rx.menu.trigger(
                        rx.button(
                            rx.text("Applications", size = "4", weight = "medium"),
                            rx.icon("chevron-down"),
                            weight = "medium",
                            variant = "ghost",
                            size = "3"
                        )
                    ),
                    rx.menu.content(
                        rx.menu.item(
                            navbar_link("Transaction Fraud Detection", "/transaction-fraud-detection")
                        ),
                        rx.menu.item(
                            navbar_link("Estimated Time of Arrival", "/estimated-time-of-arrival")
                        ),
                        rx.menu.item(
                            navbar_link("E-Commerce Customer Interactions", "/e-commerce-customer-interactions")
                        ),
                        #rx.menu.item(
                        #    navbar_link("Sales Forecasting", "/sales_forecasting")
                        #)
                    )
                ),
                justify = "end",
                spacing = "5"
            ),
            justify = "between",
            align_items = "center"
        ),
        bg = rx.color("accent", 3),
        padding = "1em",
        width = "100%"
    )