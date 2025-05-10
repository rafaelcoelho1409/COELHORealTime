import streamlit as st
import base64
import datetime as dt
import requests
import pandas as pd
import ast
from collections import Counter, defaultdict
import plotly.graph_objects as go
import os


###>>>---LOCAL FUNCTIONS---<<<###
def image_border_radius(image_path, border_radius, width, height, page_object = None, is_html = False):
    if is_html == False:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        # Create HTML string with the image
        img_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="border-radius: {border_radius}px; width: {width}%; height: {height}%">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)
    else:
        # Create HTML string with the image
        img_html = f'<img src="{image_path}" style="border-radius: {border_radius}px; width: 300px;">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)

def timestamp_to_api_response(timestamp_date, timestamp_time):
    timestamp_string = "T".join([str(timestamp_date), str(timestamp_time)])
    input_format = "%Y-%m-%dT%H:%M:%S"
    target_format = "%Y-%m-%dT%H:%M:%S"
    timestamp_input = dt.datetime.strptime(timestamp_string, input_format)
    timestamp_target = timestamp_input.replace(tzinfo = dt.timezone.utc)
    timestamp = timestamp_target.strftime(target_format)
    return timestamp

@st.cache_resource
def convert_cluster_feature_dict_to_dataframe(data_dict: dict, feature_name: str = "Feature Values") -> pd.DataFrame:
    """
    Converts a Python dictionary representing feature value counts per cluster into a Pandas DataFrame.

    Args:
        data_dict: A Python dictionary where keys are cluster IDs (can be str or int)
                   and values are dictionaries of {feature_value: count}.
                   Example: {"0":{"null":12},"1":{"Toys & Games":583,...}}
        feature_name: The name to give to the DataFrame's index (representing the feature values).

    Returns:
        A Pandas DataFrame with feature values as the index, cluster IDs as columns,
        and counts as cell values. Missing values are filled with 0.
    """
    if not isinstance(data_dict, dict):
        print("Error: Input must be a dictionary.")
        return pd.DataFrame() # Return an empty DataFrame on error
    # Rename keys to be more descriptive column names (e.g., "Cluster 0", "Cluster 1")
    # This step is optional but improves readability.
    processed_data = {}
    for cluster_id, feature_counts in data_dict.items():
        processed_data[f"Cluster {cluster_id}"] = feature_counts
    # Create DataFrame from the processed dictionary
    # Pandas will automatically align the inner dictionary keys (feature values) as the index.
    df = pd.DataFrame(processed_data)
    # Fill NaN values with 0, as NaN means the feature value didn't appear in that cluster
    df = df.fillna(0).astype(int)
    # Set the name for the index
    df.index.name = feature_name
    return df

@st.cache_resource
def process_device_info_to_dataframes(device_info_data: dict) -> dict:
    """
    Processes device_info data, where feature values are stringified dictionaries,
    into separate DataFrames for device_type, browser, and os, aggregated by cluster.

    Args:
        device_info_data: A Python dictionary where keys are cluster IDs (e.g., "0", "1")
                          and values are dictionaries of {stringified_device_dict: count}.
                          Example: {"0": {"{'device_type': 'Desktop', ...}": 5, ...}, ...}

    Returns:
        A dictionary of Pandas DataFrames:
        {
            'device_type': DataFrame,
            'browser': DataFrame,
            'os': DataFrame
        }
        Each DataFrame has sub-feature values as index, 'Cluster X' as columns, and counts as values.
    """
    if not isinstance(device_info_data, dict):
        print("Error: Input to process_device_info_to_dataframes must be a dictionary.")
        return {}
    # Initialize dictionaries to hold aggregated counts for each sub-feature per cluster
    # Structure: {sub_feature_name: {cluster_id: Counter_for_sub_feature_values}}
    aggregated_counts = {
        'device_type': defaultdict(Counter),
        'browser': defaultdict(Counter),
        'os': defaultdict(Counter)
    }
    # Iterate through each cluster and its device_info counts
    for cluster_id, string_dict_counts in device_info_data.items():
        if not isinstance(string_dict_counts, dict):
            print(f"Warning: Expected a dictionary for cluster {cluster_id}, got {type(string_dict_counts)}. Skipping.")
            continue
        for stringified_key, count in string_dict_counts.items():
            try:
                # Safely convert the stringified dictionary key back to a Python dict
                device_dict = ast.literal_eval(stringified_key)
                if not isinstance(device_dict, dict):
                    print(f"Warning: ast.literal_eval did not return a dict for key '{stringified_key}'. Skipping.")
                    continue
                # Extract sub-features and update their counts for the current cluster
                if 'device_type' in device_dict:
                    aggregated_counts['device_type'][str(cluster_id)][device_dict['device_type']] += count
                if 'browser' in device_dict:
                    aggregated_counts['browser'][str(cluster_id)][device_dict['browser']] += count
                if 'os' in device_dict:
                    aggregated_counts['os'][str(cluster_id)][device_dict['os']] += count
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse stringified dict key '{stringified_key}' for cluster {cluster_id}. Error: {e}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred while processing key '{stringified_key}' for cluster {cluster_id}. Error: {e}. Skipping.")
    # Convert the aggregated counts into DataFrames
    result_dataframes = {}
    for sub_feature, cluster_counts_map in aggregated_counts.items():
        # Prepare data for DataFrame creation: { "Cluster X": {"value1": count1, ...}, ...}
        df_input_data = {}
        for cluster_id, value_counts in cluster_counts_map.items():
             df_input_data[f"Cluster {cluster_id}"] = dict(value_counts)
        df = pd.DataFrame(df_input_data)
        df = df.fillna(0).astype(int)
        df.index.name = sub_feature
        result_dataframes[sub_feature] = df
    return result_dataframes

@st.cache_resource
def create_location_heatmaps(location_data_per_cluster: dict, mapbox_access_token: str = None) -> dict:
    if not isinstance(location_data_per_cluster, dict):
        print("Error: Input to create_location_heatmaps must be a dictionary.")
        return {}
    if mapbox_access_token is None:
        mapbox_access_token = os.getenv("MAPBOX_ACCESS_TOKEN")
    if not mapbox_access_token:
        print("Warning: Mapbox access token not provided or found in environment variables. Maps may not render.")
        # You might choose to return empty figures or raise an error here
        # For now, it will proceed, but map tiles won't load.
    heatmap_figures = {}
    for cluster_id, loc_counts in location_data_per_cluster.items():
        lats = []
        lons = []
        weights = [] # Counts of each location, to be used as 'z' for density
        if not isinstance(loc_counts, dict):
            print(f"Warning: Expected a dictionary for location counts in cluster {cluster_id}, got {type(loc_counts)}. Skipping this cluster.")
            continue
        for str_loc_dict, count in loc_counts.items():
            try:
                loc_dict = ast.literal_eval(str_loc_dict)
                if isinstance(loc_dict, dict) and 'lat' in loc_dict and 'lon' in loc_dict:
                    # Round latitude and longitude to 3 decimal places
                    lat_rounded = round(float(loc_dict['lat']), 3)
                    lon_rounded = round(float(loc_dict['lon']), 3)
                    lats.append(lat_rounded)
                    lons.append(lon_rounded)
                    weights.append(count)
                else:
                    print(f"Warning: Invalid location format or missing lat/lon in '{str_loc_dict}' for cluster {cluster_id}. Skipping.")
            except (ValueError, SyntaxError, TypeError) as e: # Added TypeError for float conversion
                print(f"Warning: Could not parse or process stringified location dict '{str_loc_dict}' for cluster {cluster_id}. Error: {e}. Skipping.")
            except Exception as e:
                 print(f"An unexpected error occurred while processing location key '{str_loc_dict}' for cluster {cluster_id}. Error: {e}. Skipping.")
        if not lats: # No valid location data for this cluster
            print(f"No valid location data to plot for Cluster {cluster_id}.")
            # Optionally create an empty figure or skip
            fig = go.Figure()
            fig.update_layout(
                title_text = f"Location Heatmap - Cluster {cluster_id} (No Data)",
                #height=500,
                #width=700
            )
            heatmap_figures[f"Cluster {cluster_id}"] = fig
            continue
        fig = go.Figure(go.Densitymapbox(
            lat = lats,
            lon = lons,
            z = weights,
            radius = 10, # Adjust radius for desired appearance
            opacity = 0.9,
            colorscale = "Viridis" # Or any other colorscale like "Hot", "Jet", etc.
        ))
        fig.update_layout(
            mapbox_style = (
                "open-street-map" 
                if not mapbox_access_token 
                else "mapbox://styles/mapbox/streets-v11"), # Use a basic style if no token
            mapbox_accesstoken = mapbox_access_token,
            mapbox_center_lon = pd.Series(lons).median() if lons else 0, # Center map on median longitude
            mapbox_center_lat = pd.Series(lats).median() if lats else 0, # Center map on median latitude
            mapbox_zoom = 8,#3 if lons else 1, # Adjust zoom level
            margin = {"r":0,"t":50,"l":0,"b":0},
            title_text = f"Location Heatmap - Cluster {cluster_id}",
            #height = 600, # Adjust as needed
            #width = 800   # Adjust as needed
        )
        heatmap_figures[f"Cluster {cluster_id}"] = fig
    return heatmap_figures


def switch_active_model(key_to_activate, FASTAPI_URL = "http://fastapi:8000"):
    try:
        response = requests.post(f"{FASTAPI_URL}/switch_model/{key_to_activate}")
        response.raise_for_status()
        st.toast(
            f"Request to switch to model '{key_to_activate}' sent: {response.json().get('message')}", 
            icon = "✅")
        # Store in session state which model we *think* is active
        st.session_state.activated_model = key_to_activate
    except requests.exceptions.RequestException as e:
        st.error(f"Error switching model: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"FastAPI response: {e.response.text}")
        # If switch failed, reflect that no model is reliably active from this page's perspective
        if 'activated_model' in st.session_state and st.session_state.activated_model == key_to_activate:
            del st.session_state.activated_model


###>>>---STREAMLIT FUNCTIONS---<<<###


###>>>---CACHE FUNCTION---<<<###
