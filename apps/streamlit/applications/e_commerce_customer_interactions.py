import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.metric_cards import style_metric_cards
import requests
import pandas as pd
from faker import Faker
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import uuid
import os
from functions import (
    timestamp_to_api_response,
    convert_cluster_feature_dict_to_dataframe,
    process_device_info_to_dataframes,
    create_location_heatmaps,
    switch_active_model
)


FASTAPI_HOST = os.environ["FASTAPI_HOST"]


fake = Faker()
PROJECT_NAME = "E-Commerce Customer Interactions"
MODEL_KEY = f"{PROJECT_NAME.replace(' ', '_').replace('-', '_').lower()}_river.py"
FASTAPI_URL = f"http://{FASTAPI_HOST}:8000"

if 'activated_model' not in st.session_state or st.session_state.activated_model != MODEL_KEY:
    # If no model is marked as active, or a different one is, try to activate this page's model.
    # This also handles the initial load of the page.
    switch_active_model(MODEL_KEY)
    # You might want a small delay or a button to prevent rapid switching if users click around fast
    # time.sleep(1) # Optional: brief pause


tabs = st.tabs([
    "Incremental ML",
])


with tabs[0]: # Incremental ML
    st.caption("**Incremental ML model:** DBSTREAM (River)")
    layout_grid = grid([0.3, 0.7])
    layout_grid_1 = layout_grid.container()
    layout_grid_2 = layout_grid.container()
    sample = requests.post(
        f"http://{FASTAPI_HOST}:8000/initial_sample",
        json = {
            "project_name": PROJECT_NAME
        }).json()
    with layout_grid_1.form("Predict"):
        form_cols1 = st.columns(3)
        device_info_options = requests.post(
            f"http://{FASTAPI_HOST}:8000/unique_values",
            json = {
                "column_name": "device_info",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        device_info_options = pd.DataFrame([eval(x) for x in device_info_options])
        browser_options = device_info_options["browser"].unique().tolist()
        device_type_options = device_info_options["device_type"].unique().tolist()
        os_options = device_info_options["os"].unique().tolist()
        browser = form_cols1[0].selectbox(
            "Browser",
            browser_options,
            index = browser_options.index(str(sample["device_info"]["browser"])))  
        device_type = form_cols1[1].selectbox(
            "Device Type",
            device_type_options,
            index = device_type_options.index(str(sample["device_info"]["device_type"])))  
        os = form_cols1[2].selectbox(
            "OS",
            os_options,
            index = os_options.index(str(sample["device_info"]["os"])))
        form_cols2 = st.columns(2)
        lat = form_cols2[0].number_input(
            "Latitude", 
            value = sample["location"]["lat"],
            min_value = 29.5,
            max_value = 30.1,
            step = 0.001)
        lon = form_cols2[1].number_input(
            "Longitude", 
            value = sample["location"]["lon"],
            min_value = -95.8,
            max_value = -95.0,
            step = 0.001)
        form_cols3 = st.columns(2)
        event_type_options = requests.post(
            f"http://{FASTAPI_HOST}:8000/unique_values",
            json = {
                "column_name": "event_type",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        event_type = form_cols3[0].selectbox(
            "Event Type", 
            event_type_options, 
            index = event_type_options.index(str(sample["event_type"])))
        price = form_cols3[1].number_input(
            "Price", 
            value = sample["price"],
            min_value = 0.0,
            step = 0.01
        )
        form_cols4 = st.columns(2)
        product_category_options = requests.post(
            f"http://{FASTAPI_HOST}:8000/unique_values",
            json = {
                "column_name": "product_category",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        product_category = form_cols4[0].selectbox(
            "Product Category", 
            product_category_options, 
            index = product_category_options.index(str(sample["product_category"])))
        product_id_options = requests.post(
            f"http://{FASTAPI_HOST}:8000/unique_values",
            json = {
                "column_name": "product_id",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        product_id = form_cols4[1].selectbox(
            "Product ID", 
            product_id_options, 
            index = product_id_options.index(str(sample["product_id"])))
        form_cols5 = st.columns(2)
        referrer_url_options = requests.post(
            f"http://{FASTAPI_HOST}:8000/unique_values",
            json = {
                "column_name": "referrer_url",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        referrer_url = form_cols5[0].selectbox(
            "Referrer URL", 
            referrer_url_options, 
            index = referrer_url_options.index(str(sample["referrer_url"]))
            )
        session_event_sequence_options = requests.post(
            f"http://{FASTAPI_HOST}:8000/unique_values",
            json = {
                "column_name": "session_event_sequence",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        session_event_sequence_options = [int(x) for x in session_event_sequence_options]
        session_event_sequence = form_cols5[1].number_input(
            "Session Event Sequence", 
            value = sample["session_event_sequence"],
            min_value = int(min(session_event_sequence_options)), 
            max_value = int(max(session_event_sequence_options)), 
            step = 1
        )
        form_cols6 = st.columns(2)
        quantity_options = requests.post(
            f"http://{FASTAPI_HOST}:8000/unique_values",
            json = {
                "column_name": "quantity",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        quantity_options = [int(float(x)) if x != "nan" else 1 for x in quantity_options]   
        quantity = form_cols6[0].number_input(
            "Quantity", 
            value = sample["quantity"] if sample["quantity"] is not None else int(float(min(quantity_options))),
            min_value = int(float(min(quantity_options))),
            max_value = int(float(max(quantity_options))),
            step = 1
        )
        time_on_page_seconds_options = requests.post(
            f"http://{FASTAPI_HOST}:8000/unique_values",
            json = {
                "column_name": "time_on_page_seconds",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        time_on_page_seconds_options = [int(float(x)) for x in time_on_page_seconds_options]
        time_on_page_seconds = form_cols6[1].number_input(
            "Time on page (seconds)", 
            value = sample["time_on_page_seconds"],
            min_value = int(float(min(time_on_page_seconds_options))),
            max_value = int(float(max(time_on_page_seconds_options))),
            step = 1
        )
        form_cols7 = st.columns(2)
        sample["timestamp"] = dt.datetime.strptime(
                sample["timestamp"],
                "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%Y-%m-%dT%H:%M:%S")
        timestamp_date = form_cols7[0].date_input(
            "Date",    
            value = dt.datetime.strptime(
                sample["timestamp"],
                "%Y-%m-%dT%H:%M:%S").date())
        timestamp_time = form_cols7[1].time_input(
            "Time",    
            value = dt.datetime.strptime(
                sample["timestamp"],
                "%Y-%m-%dT%H:%M:%S").time(),
                step = dt.timedelta(minutes = 1))
        timestamp = timestamp_to_api_response(
            timestamp_date, 
            timestamp_time)
        customer_id = sample['customer_id']
        event_id = sample['event_id']
        page_url = sample['page_url']
        search_query = sample['search_query']
        session_id = sample['session_id']
        st.caption(f"**Customer ID:** {customer_id}")
        st.caption(f"**Event ID:** {event_id}")
        st.caption(f"**Page URL:** {page_url}")
        st.caption(f"**Search Query:** {search_query}")
        st.caption(f"**Session ID:** {session_id}")
        predict_button = st.form_submit_button(
            "Predict",
            use_container_width = True)
    cluster_counts = requests.get(
        f"http://{FASTAPI_HOST}:8000/cluster_counts").json()
    device_info_columns = [
            'device_type',
            'browser',
            'os'
        ]
    feature_per_cluster_options = [
        x for x in sample.keys()
        if x not in [
            'event_id',
            'customer_id',
            'session_id',
            'timestamp',
            'price',
            'page_url',
            'search_query',
            'location',
            'product_id',
            'device_info',
        ]] + device_info_columns
    subtabs = layout_grid_2.tabs([
        "Prediction",
        "Clusters features",
        "Location Heatmap",
    ])
    with subtabs[0]:
        x = {
          "customer_id": sample['customer_id'],
          "device_info": {
            "device_type": device_type,
            "browser": browser,
            "os": os
          },
          "event_id": sample["event_id"],
          "event_type": event_type,
          "location": {
            "lat": lat,
            "lon": lon
          },
          "page_url": sample["page_url"],
          "price": price,
          "product_category": product_category,
          "product_id": product_id,
          "quantity": quantity,
          "referrer_url": referrer_url,
          "search_query": sample["search_query"],
          "session_event_sequence": session_event_sequence,
          "session_id": sample["session_id"],
          "time_on_page_seconds": time_on_page_seconds,
          "timestamp": timestamp + ".000000+00:00"
        }
        st.header("Prediction")
        layout_grid_2_cols_2 = st.columns(2)
        if predict_button:
            y_pred = requests.post(
                f"http://{FASTAPI_HOST}:8000/predict",
                json = {"project_name": PROJECT_NAME} | x).json()[
                    "cluster"
                ]
            st.session_state["predicted_cluster"] = y_pred
        if "predicted_cluster" in st.session_state and st.session_state["predicted_cluster"] is not None:
            pred_fig = go.Figure()
            pred_fig.add_trace(
                go.Indicator(
                    mode = "number",
                    value = st.session_state["predicted_cluster"],
                    title = {'text': "Cluster"},
                    domain = {'row': 0, 'column': 0}
                ))
            pred_fig.update_layout(
                grid = {
                    'rows': 1, 
                    'columns': 1, 
                    #'pattern': "independent"
                    },
                width = int(600 * 0.9),  # Set desired width in pixels
                height = int(400 * 0.9),
                )
            layout_grid_2_cols_2[0].plotly_chart(
                pred_fig,
                use_container_width = True
            )
            feature_per_cluster2 = layout_grid_2_cols_2[1].selectbox(
                "Feature per cluster",
                feature_per_cluster_options,
                index = (
                    feature_per_cluster_options.index(
                        st.session_state["feature_per_cluster2"])
                        if "feature_per_cluster2" in st.session_state
                        else 0
                ),
                #key = "feature_per_cluster2"
            )
            st.session_state["feature_per_cluster2"] = feature_per_cluster2
            cluster_feature_counts2 = requests.post(
                f"http://{FASTAPI_HOST}:8000/cluster_feature_counts",
                json = {
                    "column_name": "device_info" if feature_per_cluster2 in device_info_columns else feature_per_cluster2,
                }
            ).json()
            if feature_per_cluster2 in device_info_columns:
                cluster_feature_counts_df2 = process_device_info_to_dataframes(
                    cluster_feature_counts2
                )[feature_per_cluster2]
            else:
                cluster_feature_counts_df2 = convert_cluster_feature_dict_to_dataframe(
                    cluster_feature_counts2, # Pass the dictionary directly
                    feature_name = feature_per_cluster2
                )
            #layout_grid_2_cols_2[1]
            #st.write(cluster_feature_counts_df2[f"Cluster {st.session_state["predicted_cluster"]}"])
            cluster_behavior_df2 = cluster_feature_counts_df2[f"Cluster {st.session_state["predicted_cluster"]}"]
            cluster_behavior_fig2 = px.bar(
                cluster_behavior_df2,
                x = cluster_behavior_df2.index,
                y = cluster_behavior_df2.values,
                title = f"Cluster {st.session_state["predicted_cluster"]} - {
                    st.session_state['feature_per_cluster2'] if 'feature_per_cluster2' in st.session_state else feature_per_cluster2}",
                color_discrete_sequence = px.colors.sequential.RdBu
            )
            layout_grid_2_cols_2[1].plotly_chart(
                cluster_behavior_fig2,
                use_container_width = True
            )
        else:
            st.info("Click 'Predict' to analyze the prediction.")
    with subtabs[1]:
        layout_grid_2_cols = st.columns(2)
        samples_per_cluster_fig = px.bar(
            pd.DataFrame(cluster_counts.items(), columns = ["Cluster", "Quantity"]),
            x = "Cluster",
            y = "Quantity",
            title = "Samples per cluster",
            color_discrete_sequence = px.colors.sequential.RdBu
        )
        layout_grid_2_cols[0].plotly_chart(
            samples_per_cluster_fig,
            use_container_width = True)
        feature_per_cluster = layout_grid_2_cols[1].selectbox(
            "Feature per cluster",
            feature_per_cluster_options,
            key = str(uuid.uuid4()))
        cluster_feature_counts = requests.post(
            f"http://{FASTAPI_HOST}:8000/cluster_feature_counts",
            json = {
                "column_name": "device_info" if feature_per_cluster in device_info_columns else feature_per_cluster,
            }
        ).json()
        if feature_per_cluster in device_info_columns:
            cluster_feature_counts_df = process_device_info_to_dataframes(
                cluster_feature_counts
            )[feature_per_cluster]
        else:
            cluster_feature_counts_df = convert_cluster_feature_dict_to_dataframe(
                cluster_feature_counts, # Pass the dictionary directly
                feature_name = feature_per_cluster 
            )
        layout_grid_2_cols[1].write(cluster_feature_counts_df)
    with subtabs[2]:
        turn_heatmap_on = st.toggle(
            "Turn heatmap on",
            value = False
        )
        #Location heatmap
        if turn_heatmap_on:
            cluster_heatmap_filter = st.selectbox(
                "Cluster", 
                list(cluster_counts.keys()),
                index = list(cluster_counts.keys()).index(
                    str(st.session_state['predicted_cluster'])) 
                    if "predicted_cluster" in st.session_state and st.session_state["predicted_cluster"] is not None 
                    else 0
            )
            sample_location_data = requests.post(
                f"http://{FASTAPI_HOST}:8000/cluster_feature_counts",
                json = {
                    "column_name": "location",
                }
            ).json()
            location_heatmaps = create_location_heatmaps(sample_location_data)
            st.plotly_chart(
                location_heatmaps[f"Cluster {cluster_heatmap_filter}"], 
                use_container_width = True)
