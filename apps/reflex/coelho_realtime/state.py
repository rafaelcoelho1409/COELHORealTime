import reflex as rx

class State(rx.State):
    page_name: str = "Home"

    @rx.event
    def change_page_name(self, page_name: str):
        self.page_name = page_name