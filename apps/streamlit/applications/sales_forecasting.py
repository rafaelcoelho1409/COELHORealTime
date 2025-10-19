import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.metric_cards import style_metric_cards
import requests
import pandas as pd
from faker import Faker
import datetime as dt
import plotly.express as px
import os
from functions import (
    timestamp_to_api_response,
    switch_active_model
)


FASTAPI_HOST = os.environ["FASTAPI_HOST"]


fake = Faker()
PROJECT_NAME = "Sales Forecasting"
MODEL_KEY = f"{PROJECT_NAME.replace(' ', '_').replace('-', '_').lower()}.py"
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
    st.caption("**Incremental ML model:** SNARIMAX (River)")
    layout_grid = grid([0.3, 0.7])
    layout_grid_1 = layout_grid.container()
    layout_grid_2 = layout_grid.container()
    sample = requests.post(
        f"http://{FASTAPI_HOST}:8000/sample",
        json = {
            "project_name": PROJECT_NAME
        }).json()
    st.write(sample)
    #Implement the logic later
