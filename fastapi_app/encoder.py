import argparse
import json
import sys
from river import compose
from functions import (
    create_ordinal_encoders,
    load_or_create_model,
    process_sample
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Process transaction data and make fraud predictions.')
    parser.add_argument(
        '--data', 
        type = str, 
        required = True,
        help = 'JSON string of the transaction data to process')
    args = parser.parse_args()
    data = json.loads(args.data)
    ordinal_encoder_1, ordinal_encoder_2 = create_ordinal_encoders()
    x, ordinal_encoder_1, ordinal_encoder_2 = process_sample(
        data, 
        ordinal_encoder_1, 
        ordinal_encoder_2)
    model = load_or_create_model(
        "AdaptiveRandomForestClassifier")
    #try:
    y_pred_proba = model.predict_proba_one(x)
    fraud_probability = y_pred_proba.get(1, 0)
    binary_prediction = 1 if fraud_probability >= 0.5 else 0
    json_output = {
        "transaction_id": data["transaction_id"],
        "fraud_probability": fraud_probability,
        "prediction": binary_prediction,
    }
    print(json.dumps(json_output))
    #sys.exit(0)
    #except Exception as e:
    #    print(f"Error during prediction: {e}")
    #    sys.exit(1)