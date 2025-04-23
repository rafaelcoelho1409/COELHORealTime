import json
from kafka import KafkaConsumer
from river import (
    compose, 
    linear_model, 
    preprocessing, 
    metrics, 
    optim,
    drift,
    ensemble,
    tree,
    imblearn,
    forest
)
import pickle
import os
import pandas as pd
import mlflow
import datetime as dt


# Configuration
KAFKA_TOPIC = 'transactions'
KAFKA_BROKERS = 'kafka-producer:29092'  # Adjust as needed
MODEL_PATH = 'predictor.pkl'
DATA_PATH = 'river_data.pkl'

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
    pipe = pipe1 + pipe2 + pipe3
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            predictor = pickle.load(f)
            print("Model loaded from disk")
    else:
        print("Creating new model")
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
    
def load_or_create_data():
    """Load existing model or create a new one"""
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        data_df = pd.DataFrame(
            columns = [
                'transaction_id',
                'user_id',
                'timestamp',
                'amount',
                'currency',
                'merchant_id',
                'product_category',
                'transaction_type',
                'payment_method',
                'location',
                'ip_address',
                'device_info', # Nested structure for device details
                'user_agent',
                'account_age_days',
                'cvv_provided', # Boolean flag
                'billing_address_match', # Boolean flag
                'is_fraud'
            ]
        )
        return data_df


def create_consumer():
    """Create and return Kafka consumer"""
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers = KAFKA_BROKERS,
        auto_offset_reset = 'earliest',
        value_deserializer = lambda v: json.loads(v.decode('utf-8')),
        group_id = 'river_trainer'
    )

def main():
    # Initialize model and metrics
    #MODEL_TYPE = "LogisticRegression"
    #MODEL_TYPE = "ADWINBoostingClassifier"
    MODEL_TYPE = "AdaptiveRandomForestClassifier"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Transaction Fraud Detection - River")
    model = load_or_create_model(MODEL_TYPE)
    data_df = load_or_create_data()
    metric = metrics.Accuracy()  # Or other relevant metric
    # Create consumer
    consumer = create_consumer()
    print("Consumer started. Waiting for transactions...")
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
    print(f"Offset: {BATCH_SIZE_OFFSET}")
    with mlflow.start_run(run_name = MODEL_TYPE):
        #try:
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
                    model = load_or_create_model()
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
                print(f"Error updating metric {metric}: {e}")
            # Periodically log progress
            if message.offset % BATCH_SIZE_OFFSET == 0:
                print(f"Processed {message.offset} messages")
                for metric in binary_classification_metrics:
                    try:
                        binary_classification_metrics_dict[metric].update(y, prediction)
                    except Exception as e:
                        print(f"Error updating metric {metric}: {e}")
                    print(f"{metric}: {binary_classification_metrics_dict[metric].get():.2%}")
                    mlflow.log_metric(metric, binary_classification_metrics_dict[metric].get())
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(model[-1], f)
                #print(f"Last prediction: {'Fraud' if prediction == 1 else 'Legit'}")
                data_df.to_pickle(DATA_PATH)
        #except Exception as e:
        #    print(f"Error processing message: {e}")
        #    print("Stopping consumer...")
        #finally:
        #    data_df.to_pickle(DATA_PATH)
        #    consumer.close()
        #    print("Consumer closed.")

if __name__ == "__main__":
    main()