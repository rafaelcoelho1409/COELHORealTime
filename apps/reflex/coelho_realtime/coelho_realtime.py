import reflex as rx
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from .pages import (
    home,
    transaction_fraud_detection,
    estimated_time_of_arrival,
    e_commerce_customer_interactions,
)

# Create a FastAPI app with Prometheus instrumentation
# This will be passed to Reflex as the api_transformer
api = FastAPI()
Instrumentator().instrument(api).expose(api)

# Pass the instrumented FastAPI as api_transformer
# Reflex will mount its internal API to this app
app = rx.App(api_transformer=api)

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
    estimated_time_of_arrival.index,
    route = "/estimated-time-of-arrival",
    title = "Estimated Time of Arrival - COELHO RealTime"
)
app.add_page(
    e_commerce_customer_interactions.index,
    route = "/e-commerce-customer-interactions",
    title = "E-Commerce Customer Interactions - COELHO RealTime"
)