import reflex as rx

class CounterState(rx.State):
    count: int = 0
    def increment(self):
        self.count += 1
    def decrement(self):
        self.count -= 1


def counter() -> rx.Component:
    return rx.container(
        rx.heading(
            "Counter Example", 
            size = "7"),
        rx.hstack(
            rx.button(
                "Decrement",
                color_scheme = "ruby",
                on_click = CounterState.decrement,
            ),
            rx.heading(
                CounterState.count,
                font_size = "2em"
            ),
            rx.button(
                "Increment",
                color_scheme = "grass",
                on_click = CounterState.increment,
            ),
            spacing = "4"
        ),
        rx.link(
            "Back to Home", 
            href = "/"),
        padding = "2em",
    )