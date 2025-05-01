from river import (
    metrics, 
    preprocessing,
    imblearn
)
import pickle
import os
import pandas as pd
import mlflow
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_data,
    load_or_create_ordinal_encoder,
)

DATA_PATH = "data/transaction_fraud_detection_data.parquet"
MODEL_FOLDER = "models/transaction_fraud_detection"
ORDINAL_ENCODER_PATH = "ordinal_encoders/transaction_fraud_detection"
FRAUD_PROBABILITY = 0.01

os.makedirs(MODEL_FOLDER, exist_ok = True)
os.makedirs(ORDINAL_ENCODER_PATH, exist_ok = True)
os.makedirs("data", exist_ok = True)


def main():
    # Initialize model and metrics
    #MODEL_TYPE = "LogisticRegression"
    #MODEL_TYPE = "ADWINBoostingClassifier"
    MODEL_TYPE = "AdaptiveRandomForestClassifier"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Transaction Fraud Detection")
    ordinal_encoder = load_or_create_ordinal_encoder(
        ORDINAL_ENCODER_PATH
    )
    scaler = preprocessing.StandardScaler()
    model = load_or_create_model(
        MODEL_TYPE,
        MODEL_FOLDER
        #from_scratch = True
    )
    # Create consumer
    consumer = create_consumer("Transaction Fraud Detection")
    print("Consumer started. Waiting for transactions...")
    data_df = load_or_create_data(
        consumer,
        "Transaction Fraud Detection")
    binary_classification_metrics = [
        'Accuracy',
        'Precision',          # Typically for the positive class (Fraud)
        'Recall',             # Typically for the positive class (Fraud)
        'F1',                 # Typically for the positive class (Fraud)
        'GeometricMean',
        'ROCAUC',             # Requires probabilities
    ]
    binary_classification_metrics_dict = {
        x: getattr(metrics, x)() for x in binary_classification_metrics
    }
    BATCH_SIZE_OFFSET = 100
    with mlflow.start_run(run_name = MODEL_TYPE):
        try:
            #fraud_count, normal_count = 0, 0
            for message in consumer:
                transaction = message.value
                # Create a new DataFrame from the received data
                new_row = pd.DataFrame([transaction])
                # Append the new row to the existing DataFrame
                data_df = pd.concat([data_df, new_row], ignore_index = True)
                # Process the transaction
                x = {
                    'transaction_id':        transaction['transaction_id'],
                    'user_id':               transaction['user_id'],
                    'timestamp':             transaction['timestamp'],
                    'amount':                transaction['amount'],
                    'currency':              transaction['currency'],
                    'merchant_id':           transaction['merchant_id'],
                    'product_category':      transaction['product_category'],
                    'transaction_type':      transaction['transaction_type'],
                    'payment_method':        transaction['payment_method'],
                    'location':              transaction['location'],
                    'ip_address':            transaction['ip_address'],
                    'device_info':           transaction['device_info'], # Nested structure for device details
                    'user_agent':            transaction['user_agent'],
                    'account_age_days':      transaction['account_age_days'],
                    'cvv_provided':          transaction['cvv_provided'], # Boolean flag
                    'billing_address_match': transaction['billing_address_match'], # Boolean flag
                }
                x, ordinal_encoder = process_sample(x, ordinal_encoder)
                y = transaction['is_fraud']
                # Update the model
                prediction = model.predict_one(x)
                model.learn_one(x, y)
                # Update metrics if provided
                try:
                    for metric in binary_classification_metrics:
                        binary_classification_metrics_dict[metric].update(y, prediction)
                except Exception as e:
                    print(f"Error updating metric {metric}: {str(e)}")
                # Periodically log progress
                if message.offset % BATCH_SIZE_OFFSET == 0:
                    print(f"Processed {message.offset} messages")
                    for metric in binary_classification_metrics:
                        try:
                            binary_classification_metrics_dict[metric].update(y, prediction)
                        except Exception as e:
                            print(f"Error updating metric {metric}: {str(e)}")
                        mlflow.log_metric(metric, binary_classification_metrics_dict[metric].get())
                        #print(f"{metric}: {binary_classification_metrics_dict[metric].get():.2%}")
                    with open(f"{ORDINAL_ENCODER_PATH}/ordinal_encoder.pkl", 'wb') as f:
                        pickle.dump(ordinal_encoder, f)
                    mlflow.log_artifact(f"{ORDINAL_ENCODER_PATH}/ordinal_encoder.pkl")
                    MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                    with open(MODEL_VERSION, 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(MODEL_VERSION)
                    #print(f"Last prediction: {'Fraud' if prediction == 1 else 'Legit'}")
                    data_df.to_parquet(DATA_PATH)
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
            with open(MODEL_VERSION, 'wb') as f:
                pickle.dump(model, f)
            with open(f"{ORDINAL_ENCODER_PATH}/ordinal_encoder.pkl", 'wb') as f:
                pickle.dump(ordinal_encoder, f)
            data_df.to_parquet(DATA_PATH)
            consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()