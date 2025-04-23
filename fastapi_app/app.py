import json
from kafka import KafkaConsumer
from river import (
    compose, 
    linear_model, 
    preprocessing, 
    metrics, 
    optim,
    tree,
    ensemble,
    imblearn,
    drift,
    forest
)
import pandas as pd
import pickle
import os
from fastapi import (
    FastAPI,
    HTTPException
)
from pydantic import BaseModel
from typing import Optional
import requests

# Configuration
KAFKA_TOPIC = 'transactions'
KAFKA_BROKERS = 'kafka-producer:29092'  # Adjust as needed
MODEL_PATH = 'predictor.pkl'
DATA_PATH = 'river_data.parquet'

data = pd.read_parquet(DATA_PATH)

####---Functions----####
#Data processing functions
def extract_device_info(x):
    x_ = x['device_info']
    return {
        'os': x_['os'],
        'browser': x_['browser'],
    }

def load_or_create_model(model_type):
    """Load existing model or create a new one"""
    # Create a new model pipeline
    pipe1 = compose.Select(
        "amount",
        "account_age_days",
        "cvv_provided",
        "billing_address_match"
    )
    pipe2 = compose.Select(
        "currency",
        "merchant_id",
        "payment_method",
        "product_category",
        "transaction_type",
        "user_agent"
    )
    pipe2 |= preprocessing.OrdinalEncoder()
    pipe3 = compose.Select(
        "device_info"
    )
    pipe3 |= compose.FuncTransformer(
        extract_device_info,
    )
    pipe3 |= preprocessing.OrdinalEncoder()
    global pipe
    pipe = pipe1 + pipe2 + pipe3
    global LOAD_MODEL_MESSAGE
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            predictor = pickle.load(f)
            LOAD_MODEL_MESSAGE = "Model loaded from disk"
            print(LOAD_MODEL_MESSAGE)
    else:
        LOAD_MODEL_MESSAGE = "Creating new model"
        print(LOAD_MODEL_MESSAGE)
        if model_type == "LogisticRegression":
            predictor = linear_model.LogisticRegression(
                loss = optim.losses.CrossEntropyLoss(
                    class_weight = {0: 1, 1: 10}),
                optimizer = optim.SGD(0.01)
            )
        elif model_type == "ADWINBoostingClassifier":
            base_estimator = tree.HoeffdingAdaptiveTreeClassifier(
                splitter = tree.splitter.HistogramSplitter(),
                drift_detector = drift.ADWIN(),
                max_depth = 20,
                nominal_attributes = [
                    "currency",
                    "merchant_id",
                    "payment_method",
                    "product_category",
                    "transaction_type",
                    "user_agent",
                    "device_info_os",
                    "device_info_browser"
                ],
                leaf_prediction = 'mc',#'nba',
                grace_period = 200,
                delta = 1e-7
            )
            boosting_classifier = ensemble.ADWINBoostingClassifier(
                model = base_estimator,
                n_models = 15,
            )
            predictor = imblearn.RandomOverSampler(
                classifier = boosting_classifier,
                desired_dist = {1: 0.5, 0: 0.5},
                seed = 42
            )
        elif model_type == "AdaptiveRandomForestClassifier":
            predictor = forest.ARFClassifier(
                n_models = 10,                  # More models = better accuracy but higher latency
                drift_detector = drift.ADWIN(),  # Auto-detects concept drift
                warning_detector = drift.ADWIN(),
                metric = metrics.ROCAUC(),       # Optimizes for imbalanced data
                max_features = "sqrt",           # Better for high-dimensional data
                lambda_value = 6,               # Controls tree depth (higher = more complex)
                seed = 42
            )
    model = pipe | predictor
    return model

def create_consumer():
    """Create and return Kafka consumer"""
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers = KAFKA_BROKERS,
        auto_offset_reset = 'earliest',
        value_deserializer = lambda v: json.loads(v.decode('utf-8')),
        group_id = 'river_trainer'
    )

consumer = create_consumer()

###---FastAPI App---###
app = FastAPI()

class TransactionData(BaseModel):
    transaction_id: str
    user_id: str
    timestamp: str
    amount: float
    currency: str
    merchant_id: str
    product_category: str
    transaction_type: str
    payment_method: str
    location: dict
    ip_address: str
    device_info: dict
    user_agent: str
    account_age_days: int
    cvv_provided: bool
    billing_address_match: bool


@app.post("/predict")
async def predict_fraud(transaction: TransactionData):
    x = transaction.model_dump()
    model = load_or_create_model("AdaptiveRandomForestClassifier")
    try:
        y_pred_proba = model.predict_proba_one(x)
        fraud_probability = y_pred_proba.get(1, 0)
        binary_prediction = 1 if fraud_probability >= 0.5 else 0
        return {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": fraud_probability,
            "prediction": binary_prediction,
            "model_message": LOAD_MODEL_MESSAGE
        }
    except Exception as e:
        print(f"Error during prediction for transaction {transaction.transaction_id}: {e}")
        return {"error": f"Prediction failed: {e}"}, 500
    

@app.get("/sample")
async def get_sample():
    sample = data.sample(1).to_dict(orient = 'records')[0]
    return sample


class UniqueValuesResponse(BaseModel):
    column_name: str


@app.post("/unique_values")
async def get_unique_values(request: UniqueValuesResponse):
    column = request.column_name
    if column not in data.columns:
        raise HTTPException(
            status_code = 400, 
            detail = f"Column '{column}' not found in dataset.")
    unique_values = data[column].apply(str).unique().tolist()
    return {"unique_values": unique_values}

initial_transaction_data = TransactionData(
    **data.sample(1).to_dict(orient = 'records')[0]
)

@app.get("/initial_transaction_data")
async def get_initial_transaction_data():
    return initial_transaction_data

@app.put("/initial_transaction_data")
async def update_initial_transaction_data(transaction: TransactionData):
    global initial_transaction_data
    initial_transaction_data = transaction
    return initial_transaction_data
