from river import (
    metrics, 
    drift,
)
import pickle
import os
import pandas as pd
import mlflow
import datetime as dt
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_data,
    load_or_create_ordinal_encoders,
    DATA_PATH
)


#create a folder for model versioning
os.makedirs("model_versions", exist_ok = True)
os.makedirs("ordinal_encoders", exist_ok = True)


def main():
    # Initialize model and metrics
    #MODEL_TYPE = "LogisticRegression"
    #MODEL_TYPE = "ADWINBoostingClassifier"
    MODEL_TYPE = "AdaptiveRandomForestClassifier"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Transaction Fraud Detection - River")
    ordinal_encoder_1, ordinal_encoder_2 = load_or_create_ordinal_encoders()
    model = load_or_create_model(
        MODEL_TYPE,
        #from_scratch = True
    )
    # Create consumer
    consumer = create_consumer()
    print("Consumer started. Waiting for transactions...")
    data_df = load_or_create_data(consumer)
    binary_classification_metrics = [
        'Accuracy',
        #'BalancedAccuracy',
        'Precision',          # Typically for the positive class (Fraud)
        'Recall',             # Typically for the positive class (Fraud)
        'F1',                 # Typically for the positive class (Fraud)
        #'FBeta',              # Typically for the positive class (Fraud), specify beta
        #'MCC',                # Matthews Correlation Coefficient
        'GeometricMean',
        'ROCAUC',             # Requires probabilities
        #'RollingROCAUC',      # Requires probabilities
        #'LogLoss',            # Requires probabilities
        #'CrossEntropy',       # Same as LogLoss, requires probabilities
    ]
    binary_classification_metrics_dict = {
        x: getattr(metrics, x)() for x in binary_classification_metrics
    }
    drift_detector = drift.ADWIN()
    BATCH_SIZE_OFFSET = 100
    WINDOW_SIZE = 1000
    with mlflow.start_run(run_name = MODEL_TYPE):
        try:
            fraud_count, normal_count = 0, 0
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
                x, ordinal_encoder_1, ordinal_encoder_2 = process_sample(x, ordinal_encoder_1, ordinal_encoder_2)
                y = transaction['is_fraud']
                if y == 1:
                    fraud_count += 1
                else:
                    normal_count += 1
                #--------------DELETE HERE IF MODEL PERFORMS POORLY--------------
                if MODEL_TYPE == "LogisticRegression":
                    y_pred_proba = model.predict_proba_one(x).get(1, 0)
                    # Update drift detector with prediction error
                    error = 1 - (1 if (y_pred_proba > 0.5) == y else 0)
                    drift_detector.update(error)
                    # Handle concept drift
                    if drift_detector.drift_detected:
                        print(f"{dt.datetime.now()} - Drift detected! Resetting model...")
                        model = load_or_create_model(MODEL_TYPE)
                        fraud_count, normal_count = 0, 0
                        drift_detector = drift.ADWIN()
                    #----------------------------------------------------------------
                    # Update weights every window_size samples
                    if (fraud_count + normal_count) % WINDOW_SIZE == 0:
                        ratio = max(1, normal_count / (fraud_count + 1))  # Prevent division by zero
                        model[-1].class_weight = {0: 1, 1: ratio}
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
                        with open("ordinal_encoders/ordinal_encoder_1.pkl", 'wb') as f:
                            pickle.dump(ordinal_encoder_1, f)
                        with open("ordinal_encoders/ordinal_encoder_2.pkl", 'wb') as f:
                            pickle.dump(ordinal_encoder_2, f)
                        mlflow.log_artifact("ordinal_encoders/ordinal_encoder_1.pkl")
                        mlflow.log_artifact("ordinal_encoders/ordinal_encoder_2.pkl")
                MODEL_VERSION = f"model_versions/{model.__class__.__name__}.pkl"
                if message.offset % (BATCH_SIZE_OFFSET * 100) == 0:
                    with open(MODEL_VERSION, 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(MODEL_VERSION)
                    #print(f"Last prediction: {'Fraud' if prediction == 1 else 'Legit'}")
                    data_df.to_parquet(DATA_PATH)
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            with open(MODEL_VERSION, 'wb') as f:
                pickle.dump(model, f)
            with open("ordinal_encoders/ordinal_encoder_1.pkl", 'wb') as f:
                pickle.dump(ordinal_encoder_1, f)
            with open("ordinal_encoders/ordinal_encoder_2.pkl", 'wb') as f:
                pickle.dump(ordinal_encoder_2, f)
            data_df.to_parquet(DATA_PATH)
            consumer.close()
            print("Consumer closed.")

if __name__ == "__main__":
    main()