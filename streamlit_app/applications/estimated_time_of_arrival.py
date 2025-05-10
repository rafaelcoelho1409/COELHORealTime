import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.metric_cards import style_metric_cards
import requests
import pandas as pd
from faker import Faker
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from functions import (
    timestamp_to_api_response,
    switch_active_model
)

pio.renderers.default = "notebook_connected"
fake = Faker()
PROJECT_NAME = "Estimated Time of Arrival"
MODEL_KEY = f"{PROJECT_NAME.replace(' ', '_').replace('-', '_').lower()}.py"
FASTAPI_URL = "http://fastapi:8000"

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
    st.caption("**Incremental ML model:** Adaptive Random Forest Regressor (River)")
    layout_grid = grid([0.3, 0.7])
    layout_grid_1 = layout_grid.container()
    layout_grid_2 = layout_grid.container()
    sample = requests.post(
        "http://fastapi:8000/initial_sample",
        json = {
            "project_name": PROJECT_NAME
        }).json()
    with layout_grid_1.form("Predict"):
        form_cols1 = st.columns(2)
        driver_id_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "driver_id",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        driver_id = form_cols1[0].selectbox(
            "Driver ID", 
            driver_id_options, 
            index = driver_id_options.index(sample["driver_id"]))
        vehicle_id_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "vehicle_id",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        vehicle_id = form_cols1[1].selectbox(
            "Vehicle ID", 
            vehicle_id_options, 
            index = vehicle_id_options.index(sample["vehicle_id"]))
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
        form_cols3 = st.columns(2)
        origin_lat = form_cols3[0].number_input(
            "Origin Latitude", 
            value = sample["origin"]["lat"],
            min_value = 29.5,
            max_value = 30.1,
            step = 0.0001)
        origin_lon = form_cols3[1].number_input(
            "Origin Longitude", 
            value = sample["origin"]["lon"],
            min_value = -95.8,
            max_value = -95.0,
            step = 0.0001)
        form_cols4 = st.columns(2)
        destination_lat = form_cols4[0].number_input(
            "Destination Latitude", 
            value = sample["destination"]["lat"],
            min_value = 29.5,
            max_value = 30.1,
            step = 0.0001)
        destination_lon = form_cols4[1].number_input(
            "Destination Longitude", 
            value = sample["destination"]["lon"],
            min_value = -95.8,
            max_value = -95.0,
            step = 0.0001)
        forms_cols5 = st.columns(2)
        weather_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "weather",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        weather = forms_cols5[0].selectbox(
            "Weather", 
            weather_options, 
            index = weather_options.index(sample["weather"]))
        vehicle_type_options = requests.post(
            "http://fastapi:8000/unique_values",
            json = {
                "column_name": "vehicle_type",
                "project_name": PROJECT_NAME
            }).json()["unique_values"]
        vehicle_type = forms_cols5[1].selectbox(
            "Vehicle Type", 
            vehicle_type_options, 
            index = vehicle_type_options.index(sample["vehicle_type"]))
        form_cols6 = st.columns(2)
        hour_of_day = form_cols6[0].number_input(
            "Hour of Day", 
            value = sample["hour_of_day"],
            min_value = 0,
            max_value = 23,
            step = 1)
        driver_rating = form_cols6[1].number_input(
            "Driver Rating", 
            value = sample["driver_rating"],
            min_value = 3.5,
            max_value = 5.0,
            step = 0.1
        )
        form_cols7 = st.columns(2)
        debug_traffic_factor = form_cols7[0].number_input(
            "Debug Traffic Factor", 
            value = sample["debug_traffic_factor"],
            min_value = 0.8 - 0.5,
            max_value = 1.4 + 0.5,
            step = 0.1
        )
        debug_weather_factor = form_cols7[1].number_input(
            "Debug Weather Factor", 
            value = sample["debug_weather_factor"],
            min_value = 1.0,
            max_value = 2.0,
            step = 0.1
        )
        form_cols8 = st.columns(2)
        debug_incident_delay_seconds = form_cols8[0].number_input(
            "Debug Incident Delay (seconds)", 
            value = int(sample["debug_incident_delay_seconds"]),
            min_value = 0, 
            max_value = 1800,   
            step = 1
        )
        debug_driver_factor = form_cols8[1].number_input(
            "Debug Driver Factor", 
            value = sample["debug_driver_factor"],
            min_value = 1.0 - (5.0 - 4.5) * 0.05 - 0.1, #5.0: max driver rating
            max_value = 1.0 - (3.5 - 4.5) * 0.05 + 0.1, #3.5: min driver rating
            step = 0.1
        )
        form_cols9 = st.columns(2)
        temperature_celsius = form_cols9[0].number_input(
            "Temperature (Celsius)", 
            value = sample["temperature_celsius"],
            min_value = -50.0,
            max_value = 50.0,
            step = 0.1
        )
        initial_estimated_travel_time_seconds = form_cols9[1].number_input(
            "Initial Estimated Travel Time (seconds)", 
            value = int(sample["initial_estimated_travel_time_seconds"]),
            min_value = 60, 
            max_value = int((sample["estimated_distance_km"] / 40) * 3600 * 1.1),  
            #AVG_SPEED_KMH = 40; 1.1 comes from random.uniform(0.9, 1.1)
            step = 1)
        trip_id = sample["trip_id"]
        st.caption(f"**Trip ID:** {trip_id}")
        predict_button = st.form_submit_button(
            "Predict",
            use_container_width = True)
    layout_grid_2.header("Regression Metrics")
    mlflow_metrics = requests.post(
        "http://fastapi:8000/mlflow_metrics",
        json = {
            "project_name": PROJECT_NAME
        }).json()
    metrics_cols = layout_grid_2.columns(4)
    metrics_cols_dict = {
        0: ["MAE", "MAPE"],
        1: ["MSE", "R2"],
        2: ["RMSE", "RMSLE"],
        3: ["SMAPE"]
    }
    for i, metric_list in zip(metrics_cols_dict.keys(), metrics_cols_dict.values()):
        for metric in metric_list:
            metrics_cols[i].metric(
                metric,
                f"{mlflow_metrics[f'metrics.{metric}']:.2f}")
    style_metric_cards(
        background_color = "#000000"
    )
    layout_grid_2.divider()
    if predict_button:
        x = {
            'trip_id':                               trip_id,
            'driver_id':                             driver_id,
            'vehicle_id':                            vehicle_id,
            'timestamp':                             timestamp + ".000000+00:00",
            'origin':                                {"lat": origin_lat, "lon": origin_lon},
            'destination':                           {"lat": destination_lat, "lon": destination_lon},
            'estimated_distance_km':                 sample["estimated_distance_km"],
            'weather':                               weather,
            'temperature_celsius':                   temperature_celsius,
            'day_of_week':                           sample['day_of_week'],
            'hour_of_day':                           hour_of_day,
            'driver_rating':                         driver_rating,
            'vehicle_type':                          vehicle_type,
            'initial_estimated_travel_time_seconds': initial_estimated_travel_time_seconds,
            'debug_traffic_factor':                  debug_traffic_factor,
            'debug_weather_factor':                  debug_weather_factor,
            'debug_incident_delay_seconds':          debug_incident_delay_seconds,
            'debug_driver_factor':                   debug_driver_factor
        }
        map_and_pred_grid = layout_grid_2.columns(2)
        map_and_pred_grid[0].header("Origin and Destination")
        locations_df = pd.DataFrame({
            'lat': [origin_lat, destination_lat],
            'lon': [origin_lon, destination_lon],
            'label': ['Origin', 'Destination'],
            'color': ['blue', 'red'], # Assign colors for markers
            'size': [20, 20] # Assign sizes for markers
        })
        # Create a list of traces
        traces = []
        # 1. Add trace for the markers (Origin and Destination points)
        traces.append(
            go.Scattermapbox(
                lat = locations_df['lat'],
                lon = locations_df['lon'],
                mode = 'markers+text', # Show markers and text labels
                marker = go.scattermapbox.Marker(
                    size = locations_df['size'],
                    color = locations_df['color'],
                    # opacity=0.8
                ),
                text = locations_df['label'], # Text next to markers
                textposition = 'bottom right', # Adjust text position as needed
                name = "Locations" # Added name for legend
            )
        )
        # 2. Add trace for the dotted line connecting Origin and Destination
        traces.append(
            go.Scattermapbox(
                lat = [origin_lat, destination_lat], # Latitudes for the line
                lon = [origin_lon, destination_lon], # Longitudes for the line
                mode = 'lines',
                line = go.scattermapbox.Line(
                    width = 2,
                    color = 'black', # Or any color you prefer for the line
                    #dash = 'dot'    # Style of the line: 'solid', 'dot', 'dash', 'longdash', 'dashdot', or 'longdashdot'
                ),
                name = "Route (approx.)", # Added name for legend
                hoverinfo = 'none' # Optional: disable hover for the line itself
            )
        )
        # Create the figure with both traces
        fig_mapbox = go.Figure(data = traces)
        fig_mapbox.update_layout(
            title = 'Origin and Destination Map',
            autosize = True,
            showlegend = False, # Set to True if you want to show legend for "Locations" and "Route"
            hovermode = 'closest',
            mapbox = dict(
                # accesstoken = mapbox_access_token, # Token needed for styles other than open-street-map
                bearing = 0,
                center = dict(
                    lat = (origin_lat + destination_lat) / 2,
                    lon = (origin_lon + destination_lon) / 2
                ),
                pitch = 0,
                zoom = 9 if abs(origin_lat - destination_lat) > 0.01 or abs(origin_lon - destination_lon) > 0.01 else 11, # Adjust zoom
                style = 'open-street-map' # This style works without a token
            ),
            margin = {"r": 0, "t": 30, "l": 0, "b": 0},
            height = 300 # Retaining your height setting
        )
        map_and_pred_grid[0].plotly_chart(
            fig_mapbox,
            use_container_width = True
        )
        map_and_pred_grid[0].caption(
            f"Estimated Distance: {sample['estimated_distance_km']:.2f} km")
        map_and_pred_grid[1].header("ETA - Prediction")
        y_pred = requests.post(
            "http://fastapi:8000/predict",
            json = {"project_name": PROJECT_NAME} | x).json()[
                "Estimated Time of Arrival"
            ]
        pred_fig = go.Figure()
        pred_fig.add_trace(
            go.Indicator(
                mode = "number",
                value = y_pred,
                title = {'text': "Seconds"},
                domain = {'row': 0, 'column': 0}
            ))
        pred_fig.add_trace(
            go.Indicator(
                mode = "number",
                value = round(y_pred / 60, 2),
                title = {'text': "Minutes"},
                domain = {'row': 1, 'column': 0}
            ))
        pred_fig.update_layout(
            grid = {'rows': 2, 'columns': 1, 'pattern': "independent"},
            )
        map_and_pred_grid[1].plotly_chart(
            pred_fig,
            use_container_width = True
        )