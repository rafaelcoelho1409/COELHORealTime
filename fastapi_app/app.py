from fastapi import (
    FastAPI,
    HTTPException
)
from pydantic import BaseModel
import pandas as pd
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    create_ordinal_encoders,
    DATA_PATH
)

data = pd.read_parquet(DATA_PATH)
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
    ordinal_encoder_1, ordinal_encoder_2 = create_ordinal_encoders()
    x, ordinal_encoder_1, ordinal_encoder_2 = process_sample(x, ordinal_encoder_1, ordinal_encoder_2)
    model, LOAD_MODEL_MESSAGE = load_or_create_model("AdaptiveRandomForestClassifier")
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
