#!/bin/sh

# Create and activate the virtual environment
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Start both applications in the background *within the activated environment*
#While in test stage, please run "docker compose exec fastapi /bin/bash"
#After this, run in the shell:
#source .venv/bin/activate
#python3 consumer.py
#Only after everything ready, uncomment the following command
(python3 consumer.py > consumer.log 2>&1) &
(uvicorn app:app --host 0.0.0.0 --port 8000 --reload) &

wait