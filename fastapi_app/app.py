from fastapi import (
    FastAPI,
    HTTPException
)
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
import pickle
import subprocess
import sys
import json
from functions import (
    process_sample,
    load_or_create_model,
    load_or_create_data,
    create_consumer,
    create_ordinal_encoders,
    CustomOrdinalEncoder # Assuming this is needed
)

# Initialize global variables as None or placeholder BEFORE startup
consumer = None
data = None
model = None
ordinal_encoder_1 = None
ordinal_encoder_2 = None
initial_transaction_data = None # Also move this initialization

class Healthcheck(BaseModel):
    model_load: str | None = None # "success", "failed", "not_attempted"
    model_message: str | None = None
    model_file: str | None = None
    model_type: str | None = None
    ordinal_encoder_1_load: str | None = None # "success", "failed", "not_attempted"
    ordinal_encoder_2_load: str | None = None # "success", "failed", "not_attempted"
    ordinal_encoders_load_message: str | None = None
    data_load: str | None = None # Added data load status
    data_message: str | None = None # Added data load message
    initial_data_sample_loaded: bool | None = None # Added sample status
    initial_data_sample_message: str | None = None


# Initialize healthcheck with default state
healthcheck = Healthcheck(
    model_load = "not_attempted",
    ordinal_encoder_1_load = "not_attempted",
    ordinal_encoder_2_load = "not_attempted",
    data_load = "not_attempted",
    initial_data_sample_loaded = False
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global consumer, data, model, ordinal_encoder_1, ordinal_encoder_2, initial_transaction_data, healthcheck
    # 1. Load data
    try:
        print("Loading data...")
        consumer = create_consumer()
        data = load_or_create_data(consumer)
        healthcheck.data_load = "success"
        healthcheck.data_message = "Data loaded successfully"
        # 2. Create initial transaction data sample (requires data)
        try:
            if data is not None and not data.empty:
                initial_transaction_data = TransactionData(
                    **data.sample(1).to_dict(orient = 'records')[0]
                )
                healthcheck.initial_data_sample_loaded = True
                healthcheck.initial_data_sample_message = "Initial transaction sample created."
            else:
                healthcheck.initial_data_sample_loaded = False
                healthcheck.initial_data_sample_message = "Data was not loaded or is empty, cannot create initial sample."
        except Exception as e:
            healthcheck.initial_data_sample_loaded = False
            healthcheck.initial_data_sample_message = f"Error loading data: {e}"
            print(f"Error creating initial transaction sample: {e}", file = sys.stderr)
    except Exception as e:
        healthcheck.data_load = "failed"
        healthcheck.data_message = f"Error loading data: {e}"
        print(f"Error loading data: {e}", file = sys.stderr)
    # 3. Load or create the model
    try:
        print("Loading model...")
        model = load_or_create_model("AdaptiveRandomForestClassifier") # Or use a config variable
        healthcheck.model_load = "success"
        healthcheck.model_message = "Model loaded successfully"
        print("Model loaded successfully.")
    except Exception as e:
        healthcheck.model_load = "failed"
        healthcheck.model_message = f"Error loading model: {e}"
        print(f"Error loading model: {e}", file = sys.stderr)
    # 4. Create or load the ordinal encoders
    try:
        print("Loading encoders...")
        ordinal_encoder_1, ordinal_encoder_2 = create_ordinal_encoders()
        healthcheck.ordinal_encoder_1_load = "success"
        healthcheck.ordinal_encoder_2_load = "success"
        print("Encoders loaded successfully.")
    except Exception as e:
        healthcheck.ordinal_encoder_1_load = "failed"
        healthcheck.ordinal_encoder_2_load = "failed"
        healthcheck.ordinal_encoders_load_message = f"Error loading encoders: {e}"
        print(f"Error loading encoders: {e}", file = sys.stderr)
    print("Application setup finished. Yielding control...")
    yield # <-- Application is now ready to serve requests
    # --- Shutdown Logic (Equivalent to @app.on_event("shutdown")) ---
    print("Starting application shutdown (lifespan)...")
    print("Application shutdown finished.")


###---FastAPI App---###
app = FastAPI(lifespan = lifespan)


# IMPORTANT OBSERVATION: you can put the PUT only after the GET
@app.get("/healthcheck")
async def get_healthcheck():
    # You can add more detailed checks here if needed, e.g., if model is None
    if healthcheck.model_load == "failed" or healthcheck.data_load == "failed":
        raise HTTPException(
            status_code = 503, 
            detail = "Service Unavailable: Core components failed to load.")
    return healthcheck

@app.put("/healthcheck", response_model = Healthcheck)
async def update_healthcheck(update_data: Healthcheck):
    global healthcheck # Declare that you intend to modify the global variable
    update_dict = update_data.model_dump(exclude_unset = True)
    for field, value in update_dict.items():
        setattr(healthcheck, field, value)
    return healthcheck


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


# IMPORTANT OBSERVATION: you can put the PUT only after the GET
@app.get("/sample")
async def get_sample():
    global data # Need to access the global data
    if data is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Data is not loaded.")
    try:
        sample = data.sample(1).to_dict(orient = 'records')[0]
        return sample
    except Exception as e:
        raise HTTPException(
            status_code = 500, 
            detail = f"Error sampling data: {e}")


class UniqueValuesResponse(BaseModel):
    column_name: str


@app.post("/unique_values")
async def get_unique_values(request: UniqueValuesResponse):
    global data # Need to access the global data
    if data is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Data is not loaded.")
    column = request.column_name
    if column not in data.columns:
        raise HTTPException(
            status_code = 400,
            detail = f"Column '{column}' not found in dataset.")
    try:
        # Use apply(str) for robustness against mixed types or non-string/numeric data
        unique_values = data[column].apply(str).unique().tolist()
        return {"unique_values": unique_values}
    except Exception as e:
        raise HTTPException(
            status_code = 500, 
            detail = f"Error getting unique values for column '{column}': {e}")


@app.get("/initial_transaction_data")
async def get_initial_transaction_data():
    global initial_transaction_data # Need to access the global variable
    if initial_transaction_data is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Initial transaction data sample is not loaded.")
    return initial_transaction_data

@app.put("/initial_transaction_data")
async def update_initial_transaction_data(transaction: TransactionData):
    global initial_transaction_data
    initial_transaction_data = transaction
    return initial_transaction_data


@app.post("/predict")
async def predict_fraud(transaction: TransactionData):
    global model, ordinal_encoder_1, ordinal_encoder_2
    if model is None or ordinal_encoder_1 is None or ordinal_encoder_2 is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Model or encoders are not loaded.")
    x = transaction.model_dump()
    try:
        ordinal_encoder_1, ordinal_encoder_2 = create_ordinal_encoders()
        processed_x, _, _ = process_sample(x, ordinal_encoder_1, ordinal_encoder_2) # Discard returned encoders if they are meant to be global state
        y_pred_proba = model.predict_proba_one(processed_x) # Use processed data
        fraud_probability = y_pred_proba.get(1, 0.0) # Use 0.0 as default
        binary_prediction = 1 if fraud_probability >= 0.5 else 0
        return {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": fraud_probability,
            "prediction": binary_prediction,
        }
    except Exception as e:
        print(f"Error during prediction for transaction {transaction.transaction_id}: {e}", file=sys.stderr)
        raise HTTPException(
            status_code = 500, 
            detail = f"Prediction failed for transaction {transaction.transaction_id}: {e}")
    

@app.get("/get_ordinal_encoder_1")
async def get_ordinal_encoder_1():
    global ordinal_encoder_1
    if ordinal_encoder_1 is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Ordinal encoder 1 is not loaded.")
    return ordinal_encoder_1.get_feature_mappings()


@app.get("/get_ordinal_encoder_2")
async def get_ordinal_encoder_2():
    global ordinal_encoder_2
    if ordinal_encoder_2 is None:
        raise HTTPException(
            status_code = 503, 
            detail = "Ordinal encoder 2 is not loaded.")
    return ordinal_encoder_2.get_feature_mappings()