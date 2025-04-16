import streamlit as st
from streamlit_extras.grid import grid


st.set_page_config(
    page_title = "COELHO RealTime", 
    page_icon = ":material/home:",
    layout = "wide")

pages_dict = {
    #"Home": "applications/home.py",
    "Transaction Fraud Detection": "applications/transaction_fraud_detection.py",
}
pages = {
    "Home": st.Page("applications/home.py", title = "Home", icon = ":material/home:")} | {
    name: st.Page(path, title = name, icon = ":material/edit:")
    for name, path 
    in pages_dict.items()}


pg = st.navigation({
    "COELHO RealTime by Rafael Coelho": [
        pages["Home"]],
    "Applications": [
        pages["Transaction Fraud Detection"],
    ],
})

with open("style.css") as css:
    st.html(f"<style>{css.read()}</style>")

with st.container(key = "app_title"):
    st.title(("$$\\textbf{" + pg.title + "}$$").replace("&", "\&"))


pg.run()