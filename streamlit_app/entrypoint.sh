#!/bin/sh

# Create and activate the virtual environment
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start applications in the background *within the activated environment*
streamlit run app.py