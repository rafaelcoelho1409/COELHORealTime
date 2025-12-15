import reflex as rx
from .pages import (
    home, 
    transaction_fraud_detection,
    estimated_time_of_arrival,
    e_commerce_customer_interactions,
    #sales_forecasting,
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
    estimated_time_of_arrival.index,
    route = "/estimated-time-of-arrival",
    title = "Estimated Time of Arrival - COELHO RealTime"
)
app.add_page(
    e_commerce_customer_interactions.index,
    route = "/e-commerce-customer-interactions",
    title = "E-Commerce Customer Interactions - COELHO RealTime"
)
#app.add_page(
#    sales_forecasting.index,
#    route = "/sales-forecasting",
#    title = "Sales Forecasting - COELHO RealTime"
#)