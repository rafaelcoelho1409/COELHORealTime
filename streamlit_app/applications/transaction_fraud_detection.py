import streamlit as st
import requests

st.write(requests.get("http://fastapi:8000/hello").json())