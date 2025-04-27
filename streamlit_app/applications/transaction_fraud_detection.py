import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.metric_cards import style_metric_cards
import requests
import pandas as pd
from faker import Faker
import uuid
import datetime as dt
import plotly.express as px
import mlflow
from functions import (
    timestamp_to_api_response
)

fake = Faker()

tabs = st.tabs([
    "Incremental ML",
])


with tabs[0]: # Incremental ML
    layout_grid = grid([0.3, 0.7])
    layout_grid_1 = layout_grid.container()
    layout_grid_2 = layout_grid.container()
    sample = requests.get("http://fastapi:8000/initial_transaction_data").json()
    with layout_grid_1.form("Predict"):
        form_cols1 = st.columns(2)
        amount = form_cols1[0].number_input(
            "Amount", 
            value = sample["amount"], 
            step = 0.01)
        account_age_days = form_cols1[1].number_input(
            "Account Age (days)", 
            value = sample["account_age_days"],
            min_value = 0,
            step = 1)
        form_cols2 = st.columns(2)
        sample["timestamp"] = dt.datetime.strptime(
                sample["timestamp"],
                "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%dT%H:%M:%S")
        timestamp_date = form_cols2[0].date_input(
            "Date",    
            value = dt.datetime.strptime(
                sample["timestamp"],
                "%Y-%m-%dT%H:%M:%S").date())
        timestamp_time = form_cols2[1].time_input(
            "Time",    
            value = dt.datetime.strptime(
                sample["timestamp"],
                "%Y-%m-%dT%H:%M:%S").time(),
                step = dt.timedelta(minutes = 1))
        timestamp = timestamp_to_api_response(
            timestamp_date, 
            timestamp_time)
        currency_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "currency"}).json()["unique_values"]
        currency = st.selectbox(
            "Currency", 
            currency_options, 
            index = currency_options.index(sample["currency"]))
        form_cols3 = st.columns(2)
        merchant_id_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "merchant_id"}).json()["unique_values"]
        merchant_id = form_cols3[0].selectbox(
            "Merchant ID", 
            merchant_id_options, 
            index = merchant_id_options.index(sample["merchant_id"]))
        product_category_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "product_category"}).json()["unique_values"]
        product_category = form_cols3[1].selectbox(
            "Product Category", 
            product_category_options, 
            index = product_category_options.index(sample["product_category"]))
        form_cols4 = st.columns(2)
        transaction_type_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "transaction_type"}).json()["unique_values"]
        transaction_type = form_cols4[0].selectbox(
            "Transaction Type", 
            transaction_type_options, 
            index = transaction_type_options.index(sample["transaction_type"]))
        payment_method_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "payment_method"}).json()["unique_values"]
        payment_method = form_cols4[1].selectbox(
            "Payment Method", 
            payment_method_options, 
            index = payment_method_options.index(sample["payment_method"]))
        form_cols5 = st.columns(2)
        lat = form_cols5[0].number_input(
            "Latitude", 
            value = sample["location"]["lat"],
            min_value = -90.0,
            max_value = 90.0,
            step = 0.0001)
        lon = form_cols5[1].number_input(
            "Longitude", 
            value = sample["location"]["lon"],
            min_value = -180.0,
            max_value = 180.0,
            step = 0.0001)
        form_cols6 = st.columns(2)
        device_info_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "device_info"}).json()["unique_values"]
        device_info_options = pd.DataFrame([eval(x) for x in device_info_options])
        browser_options = device_info_options["browser"].unique().tolist()
        os_options = device_info_options["os"].unique().tolist()
        browser = form_cols6[0].selectbox(
            "Browser",
            browser_options,
            index = browser_options.index(sample["device_info"]["browser"]))    
        os = form_cols6[1].selectbox(
            "OS",
            os_options,
            index = os_options.index(sample["device_info"]["os"]))
        form_cols7 = st.columns(2)
        cvv_provided = form_cols7[0].checkbox(
            "CVV Provided", 
            value = sample["cvv_provided"])
        billing_address_match = form_cols7[1].checkbox(
            "Billing Address Match", 
            value = sample["billing_address_match"])
        transaction_id = sample['transaction_id']
        user_id = sample['user_id']
        ip_address = sample['ip_address']
        user_agent = sample['user_agent']
        st.caption(f"**Transaction ID:** {transaction_id}")
        st.caption(f"**User ID:** {user_id}")
        st.caption(f"**IP Address:** {ip_address}")
        st.caption(f"**User Agent:** {user_agent}")
        predict_button = st.form_submit_button(
            "Predict",
            use_container_width = True)
    layout_grid_2.header("Classification Metrics")
    mlflow_metrics = requests.post(
        "http://fastapi:8000/mlflow_metrics",
        json = {"project_name": "Transaction Fraud Detection"}).json()
    metrics_cols = layout_grid_2.columns(3)
    metrics_cols_dict = {
        0: ["F1", "Accuracy"],
        1: ["Recall", "Precision"],
        2: ["ROCAUC", "GeometricMean"]
    }
    for i, metric_list in zip(metrics_cols_dict.keys(), metrics_cols_dict.values()):
        for metric in metric_list:
            metrics_cols[i].metric(
                metric,
                f"{mlflow_metrics[f'metrics.{metric}']*100:.2f}%")
    style_metric_cards(
        background_color = "#000000"
    )
    layout_grid_2.divider()
    if predict_button:
        layout_grid_2.header("Prediction")
        x = {
            'transaction_id':        transaction_id,
            'user_id':               user_id,
            'timestamp':             timestamp + ".000000+00:00",
            'amount':                amount,
            'currency':              currency,
            'merchant_id':           merchant_id,
            'product_category':      product_category,
            'transaction_type':      transaction_type,
            'payment_method':        payment_method,
            'location':              {'lat': lat, 'lon': lon},
            'ip_address':            ip_address,
            'device_info':           {'os': os, 'browser': browser}, # Nested structure for device details
            'user_agent':            user_agent,
            'account_age_days':      account_age_days,
            'cvv_provided':          cvv_provided, # Boolean flag
            'billing_address_match': billing_address_match, # Boolean flag
        }
        y_pred = requests.post(
            "http://fastapi:8000/predict",
            json = x).json()
        fraud_prob_df = pd.DataFrame({
            "Fraud": [y_pred["fraud_probability"]],
            "Not Fraud": [1 - y_pred["fraud_probability"]]
        })
        fraud_prob_fig = px.pie(
            fraud_prob_df,
            values = fraud_prob_df.iloc[0],
            names = fraud_prob_df.columns,
            title = f"Fraud Probability: {y_pred['fraud_probability']:.2%} - {"Fraud" if y_pred['prediction'] == 1 else "Not Fraud"}",
            color_discrete_sequence = ['#FF0000', '#0000FF'],
            hole = 0.2
        )
        fraud_prob_fig.update_traces(
            textposition = 'inside', 
            textinfo = 'percent+label')
        layout_grid_2.plotly_chart(
            fraud_prob_fig,
            use_container_width = True)