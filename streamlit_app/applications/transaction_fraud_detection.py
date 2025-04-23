import streamlit as st
from streamlit_extras.grid import grid
import requests
import pandas as pd
from faker import Faker
import uuid
import datetime as dt

fake = Faker()

tabs = st.tabs([
    "Incremental ML"
])


with tabs[0]: # Incremental ML
    layout_grid = grid([0.3, 0.7])
    layout_grid_1 = layout_grid.container()
    layout_grid_2 = layout_grid.container()
    sample = requests.get("http://fastapi:8000/sample").json()
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
        timestamp_date = form_cols2[0].date_input(
            "Date",    
            value = dt.datetime.strptime(
                sample["timestamp"],
                "%Y-%m-%dT%H:%M:%S.%f%z").date())
        timestamp_time = form_cols2[1].time_input(
            "Time",    
            value = dt.datetime.strptime(
                sample["timestamp"],
                "%Y-%m-%dT%H:%M:%S.%f%z").time(),
                step = dt.timedelta(minutes = 1))   
        currency_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "currency"}).json()["unique_values"]
        currency = st.selectbox(
            "Currency", 
            currency_options, 
            index = currency_options.index(sample["currency"]))
        merchant_id_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "merchant_id"}).json()["unique_values"]
        merchant_id = st.selectbox(
            "Merchant ID", 
            merchant_id_options, 
            index = merchant_id_options.index(sample["merchant_id"]))
        product_category_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "product_category"}).json()["unique_values"]
        product_category = st.selectbox(
            "Product Category", 
            product_category_options, 
            index = product_category_options.index(sample["product_category"]))
        transaction_type_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "transaction_type"}).json()["unique_values"]
        transaction_type = st.selectbox(
            "Transaction Type", 
            transaction_type_options, 
            index = transaction_type_options.index(sample["transaction_type"]))
        payment_method_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "payment_method"}).json()["unique_values"]
        payment_method = st.selectbox(
            "Payment Method", 
            payment_method_options, 
            index = payment_method_options.index(sample["payment_method"]))
        form_cols3 = st.columns(2)
        lat = form_cols3[0].number_input(
            "Latitude", 
            value = sample["location"]["lat"],
            min_value = -90.0,
            max_value = 90.0,
            step = 0.0001)
        long = form_cols3[1].number_input(
            "Longitude", 
            value = sample["location"]["lon"],
            min_value = -180.0,
            max_value = 180.0,
            step = 0.0001)
        device_info_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {"column_name": "device_info"}).json()["unique_values"]
        device_info_options = pd.DataFrame([eval(x) for x in device_info_options])
        browser_options = device_info_options["browser"].unique().tolist()
        os_options = device_info_options["os"].unique().tolist()
        browser = st.selectbox(
            "Browser",
            browser_options,
            index = browser_options.index(sample["device_info"]["browser"]))    
        os = st.selectbox(
            "OS",
            os_options,
            index = os_options.index(sample["device_info"]["os"]))
        form_cols4 = st.columns(2)
        cvv_provided = form_cols4[0].checkbox(
            "CVV Provided", 
            value = sample["cvv_provided"])
        billing_address_match = form_cols4[1].checkbox(
            "Billing Address Match", 
            value = sample["billing_address_match"])
        st.caption(f"**Transaction ID:** {str(uuid.uuid4())}")
        st.caption(f"**User ID:** {fake.uuid4()}")
        st.caption(f"**IP Address:** {fake.ipv4()}")
        st.caption(f"**User Agent:** {fake.user_agent()}")
        predict_button = st.form_submit_button(
            "Predict",
            use_container_width = True)
    layout_grid_2.write(sample)
