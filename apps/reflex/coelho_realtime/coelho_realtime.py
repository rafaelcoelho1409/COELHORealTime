import reflex as rx
from .pages import (
    home, 
    transaction_fraud_detection, 
    counter
)

app = rx.App()

# Register pages
app.add_page(
    home.index, 
    route = "/", 
    title = "Home - COELHO RealTime")
app.add_page(
    transaction_fraud_detection.index,
    route = "/transaction-fraud-detection",
    title = "Transaction Fraud Detection - COELHO RealTime"
)
app.add_page(
    counter.counter, 
    route = "/counter", 
    title = "Counter")