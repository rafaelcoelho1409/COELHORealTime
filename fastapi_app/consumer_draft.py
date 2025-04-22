import json
from kafka import (
    KafkaConsumer,
    OffsetAndMetadata
)
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
import threading
import time
import sys
from fastapi import FastAPI
from pydantic import BaseModel


# Configuration
KAFKA_TOPIC = 'transactions'
KAFKA_BROKERS = 'kafka-producer:29092'  # Adjust as needed
MODEL_PATH = 'river_model.pkl'
DATA_PATH = 'river_data.pkl'

# Global variables for the shared model and lock
shared_model = None
model_lock = threading.Lock()
stop_training = False


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


#Data processing functions
def extract_device_info(x):
    x_ = x['device_info']
    return {
        'os': x_['os'],
        'browser': x_['browser'],
    }


def build_river_pipeline(model_type):
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model = pipe | predictor
    #Save the model to future use
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model(model_path):
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
    return None

def save_model(model, model_path):
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")
    
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
    
def run_kafka_training():
    global shared_model
    global stop_training
    #MODEL_TYPE = "LogisticRegression"
    #MODEL_TYPE = "ADWINBoostingClassifier"
    MODEL_TYPE = "AdaptiveRandomForestClassifier"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Transaction Fraud Detection - River")
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
    BATCH_SIZE_OFFSET = 100
    SAVE_INTERVAL = 1000
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers = KAFKA_BROKERS,
        auto_offset_reset = 'earliest',
        enable_auto_commit = False,
        value_deserializer = lambda v: json.loads(v.decode('utf-8')),
        group_id = 'river_trainer'
    )
    print("Kafka training consumer started. Waiting for transactions...")
    message_count = 0
    with mlflow.start_run(
        run_name = f"{MODEL_TYPE}_training_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"):
        # Wait briefly for the startup event to finish loading the initial model
        time.sleep(5)
        with model_lock:
            if shared_model:
                if MODEL_TYPE == "ADWINBoostingClassifier":
                    mlflow.log_param("model_type", MODEL_TYPE)
                    mlflow.log_param("n_models", shared_model[-1].classifier.n_models)
                    mlflow.log_param("base_estimator", type(shared_model[-1].classifier.model).__name__)
                    mlflow.log_param("hat_max_depth", shared_model[-1].classifier.model.max_depth)
                    mlflow.log_param("hat_leaf_prediction", shared_model[-1].classifier.model.leaf_prediction)
                    mlflow.log_param("hat_grace_period", shared_model[-1].classifier.model.grace_period)
                    mlflow.log_param("hat_delta", shared_model[-1].classifier.model.delta)
                    mlflow.log_param("sampler", type(shared_model[-1]).__name__)
                    mlflow.log_param("sampler_desired_dist", json.dumps(shared_model[-1].desired_dist))
            else:
                print("Warning: Shared model not initialized, skipping initial MLflow param logging.")
        #try:
        while not stop_training:
            messages = consumer.poll(
                timeout_ms = 1000, 
                max_records = 100)
            if not messages:
                continue
            for tp, records in messages.items():
                for message in records:
                    if stop_training:
                        break
                    message_count += 1
                    transaction = message.value
                    x = {
                        'amount': transaction.get('amount'),
                        'account_age_days': transaction.get('account_age_days'),
                        'cvv_provided': transaction.get('cvv_provided'),
                        'billing_address_match': transaction.get('billing_address_match'),
                        'currency': transaction.get('currency'),
                        'merchant_id': transaction.get('merchant_id'),
                        'payment_method': transaction.get('payment_method'),
                        'product_category': transaction.get('product_category'),
                        'transaction_type': transaction.get('transaction_type'),
                        'user_agent': transaction.get('user_agent'),
                        'device_info': transaction.get('device_info')
                    }
                    y = transaction.get('is_fraud')
                    if y is None:
                        print(f"Skipping message {message.offset} due to missing 'is_fraud' target.")
                        #consumer.commit({tp: message.offset + 1})
                        consumer.commit({tp: OffsetAndMetadata(message.offset + 1, message.leader_epoch, None)})
                        continue
                    with model_lock:
                        try:
                            y_pred_proba = shared_model.predict_proba_one(x)
                            prediction_proba_fraud = y_pred_proba.get(1, 0)
                            prediction = 1 if prediction_proba_fraud >= 0.5 else 0
                        except Exception as e:
                            print(f"Error during prediction for message {message.offset} in training thread: {e}")
                            prediction = 0
                            prediction_proba_fraud = 0.0
                        try:
                            shared_model.learn_one(x, y)
                        except Exception as e:
                            print(f"Error during model.learn_one for message {message.offset}: {e}")
                    try:
                        for metric_name in binary_classification_metrics:
                            binary_classification_metrics_dict[metric_name].update(y, prediction)
                        if 'ROCAUC' in binary_classification_metrics_dict:
                            binary_classification_metrics_dict['ROCAUC'].update(y, prediction_proba_fraud)
                    except Exception as e:
                        print(f"Error updating metrics for message {message.offset}: {e}")
                    if message_count % BATCH_SIZE_OFFSET == 0:
                        print(f"Processed {message_count} messages")
                        with model_lock: # Lock briefly to print/log metrics if they access model state
                            for metric_name, metric_obj in binary_classification_metrics_dict.items():
                                try:
                                    metric_value = metric_obj.get()
                                    if isinstance(metric_value, (int, float)):
                                        print(f"{metric_name}: {metric_value:.4f}")
                                        mlflow.log_metric(metric_name, metric_value, step = message_count)
                                    elif isinstance(metric_value, dict):
                                        print(f"{metric_name}: {metric_value}")
                                        mlflow.log_text(json.dumps(metric_value), f"{metric_name}_offset_{message_count}.json", step=message_count)
                                    else:
                                        print(f"{metric_name}: {metric_value}")
                                except Exception as e:
                                    print(f"Error getting or logging metric {metric_name}: {e}")
                            print(f"Last prediction (binary): {'Fraud' if prediction == 1 else 'Legit'}")
                    if message_count % SAVE_INTERVAL == 0:
                         with model_lock: # Lock for saving
                             save_model(shared_model, MODEL_PATH)
                    # Corrected commit format: Use the determined leader_epoch
                    try:
                        consumer.commit({tp: OffsetAndMetadata(message.offset + 1, message.leader_epoch, None)})
                    except Exception as commit_e:
                        print(f"Error committing offset for message {message.offset}: {commit_e}")
        #except Exception as e:
        #    print(f"An error occurred in the Kafka training loop: {e}", file = sys.stderr)
        #finally:
        #    print("Kafka training thread stopping.")
        #    with model_lock: # Lock for final save
        #        save_model(shared_model, MODEL_PATH)
        #    consumer.close()
        #    print("Kafka training consumer closed.")
        #    mlflow.end_run()

@app.on_event("startup")
async def startup_event():
    global shared_model
    global training_thread
    #MODEL_TYPE = "LogisticRegression"
    #MODEL_TYPE = "ADWINBoostingClassifier"
    MODEL_TYPE = "AdaptiveRandomForestClassifier"
    print("FastAPI startup: Loading or creating model...")
    # Load or create the initial model
    with model_lock:
        shared_model = load_model(MODEL_PATH)
        if shared_model is None:
            print(f"Model not found at {MODEL_PATH}. Building a new one.")
            shared_model = build_river_pipeline(MODEL_TYPE)
            save_model(shared_model, MODEL_PATH) # Save initial model
    print("FastAPI startup: Model loaded/created. Starting Kafka training thread...")
    training_thread = threading.Thread(target = run_kafka_training, daemon = True)
    training_thread.start()
    print("FastAPI startup: Kafka training thread started.")

@app.on_event("shutdown")
async def shutdown_event():
    global stop_training
    global training_thread
    print("FastAPI shutdown: Signaling training thread to stop...")
    stop_training = True
    if 'training_thread' in globals() and training_thread.is_alive():
        print("FastAPI shutdown: Waiting for training thread to finish...")
        training_thread.join(timeout = 30)
        if training_thread.is_alive():
            print("Warning: Training thread did not finish gracefully within timeout.")
        else:
            print("FastAPI shutdown: Training thread stopped.")
    else:
        print("FastAPI shutdown: Training thread was not active.")
    print("FastAPI shutdown: Complete.")


@app.post("/predict")
async def predict_fraud(transaction: TransactionData):
    x = transaction.model_dump()
    with model_lock:
        if shared_model is None:
            print("Prediction requested but model is None.")
            return {"error": "Model is not initialized yet."}, 503
        try:
            y_pred_proba = shared_model.predict_proba_one(x)
            fraud_probability = y_pred_proba.get(1, 0)
            binary_prediction = 1 if fraud_probability >= 0.5 else 0
            return {
                "transaction_id": transaction.transaction_id,
                "fraud_probability": fraud_probability,
                "prediction": binary_prediction
            }
        except Exception as e:
            print(f"Error during prediction for transaction {transaction.transaction_id}: {e}")
            return {"error": f"Prediction failed: {e}"}, 500