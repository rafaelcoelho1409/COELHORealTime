import json
from kafka import KafkaConsumer
from river import (
    compose, 
    linear_model, 
    preprocessing, 
    metrics, 
    anomaly
)
import datetime
import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

@app.get("/hello")
def read_root():
    return {"Hello": "World"}