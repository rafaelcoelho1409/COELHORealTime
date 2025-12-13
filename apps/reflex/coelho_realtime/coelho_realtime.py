import reflex as rx
from .pages import home, counter

app = rx.App()

# Register pages
app.add_page(
    home.index, 
    route = "/", 
    title = "Home")
app.add_page(
    counter.counter, 
    route = "/counter", 
    title = "Counter")