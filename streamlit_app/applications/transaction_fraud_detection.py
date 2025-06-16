import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.metric_cards import style_metric_cards
import requests
import pandas as pd
from faker import Faker
import datetime as dt
import plotly.express as px
import re
from functions import (
    timestamp_to_api_response,
    switch_active_model
)

fake = Faker()
PROJECT_NAME = "Transaction Fraud Detection"
FASTAPI_URL = "http://fastapi:8000"


tabs_ = st.segmented_control(
    "",
    options = ["Incremental ML", "Batch ML"],
    selection_mode = "single",
)
#tabs = st.tabs([
#    "Incremental ML",
#    "Batch ML"
#])


#with tabs[0]: # Incremental ML
if tabs_ == "Incremental ML":
    st.caption("**Incremental ML model:** Adaptive Random Forest Classifier (River)")
    MODEL_KEY = f"{PROJECT_NAME.replace(' ', '_').replace('-', '_').lower()}_river.py"
    if 'activated_model' not in st.session_state or st.session_state.activated_model != MODEL_KEY:
        # If no model is marked as active, or a different one is, try to activate this page's model.
        # This also handles the initial load of the page.
        switch_active_model(MODEL_KEY)
        # You might want a small delay or a button to prevent rapid switching if users click around fast
        # time.sleep(1) # Optional: brief pause
    layout_grid = grid([0.3, 0.7])
    layout_grid_1 = layout_grid.container()
    layout_grid_2 = layout_grid.container()
    sample = requests.post(
        "http://fastapi:8000/initial_sample",
        json = {
            "project_name": PROJECT_NAME
        }).json()
    with layout_grid_1.form("Incremental ML"):
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
            json = {
                "column_name": "currency",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        currency = st.selectbox(
            "Currency", 
            currency_options, 
            index = currency_options.index(sample["currency"]))
        form_cols3 = st.columns(2)
        merchant_id_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "merchant_id",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        merchant_id = form_cols3[0].selectbox(
            "Merchant ID", 
            merchant_id_options, 
            index = merchant_id_options.index(sample["merchant_id"]))
        product_category_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "product_category",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        product_category = form_cols3[1].selectbox(
            "Product Category", 
            product_category_options, 
            index = product_category_options.index(sample["product_category"]))
        form_cols4 = st.columns(2)
        transaction_type_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "transaction_type",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        transaction_type = form_cols4[0].selectbox(
            "Transaction Type", 
            transaction_type_options, 
            index = transaction_type_options.index(sample["transaction_type"]))
        payment_method_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "payment_method",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
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
            json = {
                "column_name": "device_info",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
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
        json = {
            "project_name": PROJECT_NAME,
            "model_name": "ARFClassifier"
        }).json()
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
            json = {
                "project_name": PROJECT_NAME,
                "model_name": "ARFClassifier"} | x).json()
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
#with tabs[1]: # Batch ML
elif tabs_ == "Batch ML":
    st.caption("**Batch ML model:** XGBoost Classifier (Scikit-Learn)")
    MODEL_KEY = f"{PROJECT_NAME.replace(' ', '_').replace('-', '_').lower()}_sklearn.py"
    if 'activated_model' not in st.session_state or st.session_state.activated_model != MODEL_KEY:
        # If no model is marked as active, or a different one is, try to activate this page's model.
        # This also handles the initial load of the page.
        switch_active_model(MODEL_KEY)
        # You might want a small delay or a button to prevent rapid switching if users click around fast
        # time.sleep(1) # Optional: brief pause
    layout_grid = grid([0.3, 0.7])
    layout_grid_1 = layout_grid.container()
    layout_grid_2 = layout_grid.container()
    sample = requests.post(
        "http://fastapi:8000/initial_sample",
        json = {
            "project_name": PROJECT_NAME
        }).json()
    with layout_grid_1.form("Batch ML"):
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
            json = {
                "column_name": "currency",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        currency = st.selectbox(
            "Currency", 
            currency_options, 
            index = currency_options.index(sample["currency"]))
        form_cols3 = st.columns(2)
        merchant_id_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "merchant_id",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        merchant_id = form_cols3[0].selectbox(
            "Merchant ID", 
            merchant_id_options, 
            index = merchant_id_options.index(sample["merchant_id"]))
        product_category_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "product_category",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        product_category = form_cols3[1].selectbox(
            "Product Category", 
            product_category_options, 
            index = product_category_options.index(sample["product_category"]))
        form_cols4 = st.columns(2)
        transaction_type_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "transaction_type",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        transaction_type = form_cols4[0].selectbox(
            "Transaction Type", 
            transaction_type_options, 
            index = transaction_type_options.index(sample["transaction_type"]))
        payment_method_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "payment_method",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
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
            json = {
                "column_name": "device_info",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
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
    layout_grid_2_tabs = layout_grid_2.tabs([
        "Predictions", 
        "Detailed Metrics"])
    mlflow_metrics = requests.post(
        "http://fastapi:8000/mlflow_metrics",
        json = {
            "project_name": PROJECT_NAME,
            "model_name": "XGBClassifier"
        }).json()
    metric_names = [
        x.replace("metrics.", "") 
        for x 
        in mlflow_metrics.keys() 
        if x.startswith("metrics.")]
    group_size = 5 # Define how many metrics per row
    with layout_grid_2_tabs[0]:
        st.header("Classification Metrics")
        metrics_cols = st.columns(group_size)
        # Group metric names (base names)
        metrics_cols_dict = {}
        for i in range(0, len(metric_names), group_size):
            chunk = metric_names[i:i + group_size]
            group_key = i // group_size
            metrics_cols_dict[group_key] = chunk
        # Display metrics horizontally
        for group_key in sorted(metrics_cols_dict.keys()):
            metric_list_in_group = metrics_cols_dict[group_key]
            if not metric_list_in_group: # Should not happen if base_metric_names is not empty
                continue
            for i, metric_base_name in enumerate(metric_list_in_group):
                full_metric_name_in_mlflow = f"metrics.{metric_base_name}"
                metric_value = mlflow_metrics.get(full_metric_name_in_mlflow)
                if metric_value is not None:
                    display_value = f"{metric_value*100:.2f}%"
                    metrics_cols[i].metric(
                        label = " ".join([x.capitalize() for x in metric_base_name.split("_")]),
                        value = display_value
                    )
                else:
                    pass
        # Apply styling to all metric cards generated
        style_metric_cards(
            background_color = "#000000" # Example, adjust color as needed
        )
        st.divider()
        if predict_button:
            st.header("Prediction")
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
                json = {
                    "project_name": PROJECT_NAME,
                    "model_name": "XGBClassifier"} | x).json()
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
                #CORRECT COLOR LATER, NOT FRAUD IS DISPLAYED AS RED (NOT GOOD)
                hole = 0.2
            )
            fraud_prob_fig.update_traces(
                textposition = 'inside', 
                textinfo = 'percent+label')
            st.plotly_chart(
                fraud_prob_fig,
                use_container_width = True)
    with layout_grid_2_tabs[1]:
        st.header("Classification Metrics")
        metrics_cols = st.columns(group_size)
        # Group metric names (base names)
        metrics_cols_dict = {}
        for i in range(0, len(metric_names), group_size):
            chunk = metric_names[i:i + group_size]
            group_key = i // group_size
            metrics_cols_dict[group_key] = chunk
        # Display metrics horizontally
        for group_key in sorted(metrics_cols_dict.keys()):
            metric_list_in_group = metrics_cols_dict[group_key]
            if not metric_list_in_group: # Should not happen if base_metric_names is not empty
                continue
            for i, metric_base_name in enumerate(metric_list_in_group):
                full_metric_name_in_mlflow = f"metrics.{metric_base_name}"
                metric_value = mlflow_metrics.get(full_metric_name_in_mlflow)
                if metric_value is not None:
                    display_value = f"{metric_value*100:.2f}%"
                    metrics_cols[i].metric(
                        label = " ".join([x.capitalize() for x in metric_base_name.split("_")]),
                        value = display_value
                    )
                else:
                    pass
        # Apply styling to all metric cards generated
        style_metric_cards(
            background_color = "#000000" # Example, adjust color as needed
        )
        st.divider()
        st.header("Detailed Metrics")
        yb_tabs = st.tabs([
            "Classification",
            "Feature Analysis",
            "Target"
        ])
        with yb_tabs[0]: #Classification
            yellowbrick_metrics_dict = {
            x: re.sub(r'([a-z])([A-Z])', r'\1 \2', x)
            for x in [
                "ClassificationReport",
                "ConfusionMatrix",
                "ROCAUC",
                "PrecisionRecallCurve",
                "ClassPredictionError"
            ]}
            yellowbrick_metrics_dict = {
                y: x 
                for x, y in yellowbrick_metrics_dict.items()
            }
            yellowbrick_metrics_index_dict = {
                x: i
                for i, x in enumerate(yellowbrick_metrics_dict.keys())}
            yb_subtabs = st.tabs(yellowbrick_metrics_dict.keys())
            for i in range(len(yellowbrick_metrics_dict.keys())):
                with yb_subtabs[i]:
                    yb_image = requests.get(
                        "http://fastapi:8000/yellowbrick/transaction_fraud_detection/classification/" + list(yellowbrick_metrics_dict.values())[i],
                        stream = True)
                    st.image(
                        yb_image.content,
                        use_container_width = True)
        with yb_tabs[1]: #Feature Analysis
            ...
        with yb_tabs[2]: #Target
            yellowbrick_metrics_dict = {
                x: re.sub(r'([a-z])([A-Z])', r'\1 \2', x)
                for x in [
                    "BalancedBinningReference",
                    "ClassBalance"
            ]}
            yellowbrick_metrics_dict = {
                y: x 
                for x, y in yellowbrick_metrics_dict.items()
            }
            yellowbrick_metrics_index_dict = {
                x: i
                for i, x in enumerate(yellowbrick_metrics_dict.keys())}
            yb_subtabs = st.tabs(yellowbrick_metrics_dict.keys())
            for i in range(len(yellowbrick_metrics_dict.keys())):
                with yb_subtabs[i]:
                    yb_image = requests.get(
                        "http://fastapi:8000/yellowbrick/transaction_fraud_detection/target/" + list(yellowbrick_metrics_dict.values())[i],
                        stream = True)
                    st.image(
                        yb_image.content,
                        use_container_width = True)