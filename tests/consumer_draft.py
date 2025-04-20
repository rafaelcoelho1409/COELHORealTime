import json
from kafka import KafkaConsumer
from river import (
    compose, 
    linear_model, 
    preprocessing, 
    metrics, 
)
import pickle
import os
import pandas as pd
import mlflow


# Configuration
KAFKA_TOPIC = 'transactions'
KAFKA_BROKERS = 'localhost:9092'  # Adjust as needed
MODEL_PATH = 'river_model.pkl'
DATA_PATH = 'river_data.pkl'

#Data processing functions
def extract_device_info(x):
    x_ = x['device_info']
    return {
        'os': x_['os'],
        'browser': x_['browser'],
    }


def load_or_create_model():
    """Load existing model or create a new one"""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            #open the model in Scikit-learn format
            model = pickle.load(f)
            return model
    else:
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
        model = pipe | linear_model.LogisticRegression()
        #Save the model to future use
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
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
    

def process_transaction(model, transaction, metric=None):
    """Process a single transaction and update the model"""
    # Prepare features and label
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
    # Update the model
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    # Update metrics if provided
    if metric:
        metric.update(y, y_pred)
    return y_pred

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
    #mlflow.set_tracking_uri("http://mlflow:5000")
    #mlflow.set_experiment("Transaction Fraud Detection - River (Test)")
    model = load_or_create_model()
    #data_df = load_or_create_data()
    #metric = metrics.Accuracy()  # Or other relevant metric
    # Create consumer
    consumer = create_consumer()
    print("Consumer started. Waiting for transactions...")
    binary_classification_metrics = [
        'Accuracy',
        #'BalancedAccuracy',
        #'ConfusionMatrix',
        #'Precision',          # Typically for the positive class (Fraud)
        #'Recall',             # Typically for the positive class (Fraud)
        #'F1',                 # Typically for the positive class (Fraud)
        #'FBeta',              # Typically for the positive class (Fraud), specify beta
        #'MCC',                # Matthews Correlation Coefficient
        #'GeometricMean',
        #'ROCAUC',             # Requires probabilities
        #'RollingROCAUC',      # Requires probabilities
        #'LogLoss',            # Requires probabilities
        #'CrossEntropy',       # Same as LogLoss, requires probabilities
    ]
    binary_classification_metrics_dict = {
        x: getattr(metrics, x)() for x in binary_classification_metrics
    }
    #with mlflow.start_run(run_name = "River_LogisticRegression"):
    #try:
    for message in consumer:
        transaction = message.value
        print(message.offset)
        # Create a new DataFrame from the received data
        #new_row = pd.DataFrame([transaction])
        # Append the new row to the existing DataFrame
        #data_df = pd.concat([data_df, new_row], ignore_index = True)
        # Process the transaction
        #prediction = process_transaction(model, transaction, metric)
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
        # Update the model
        prediction = model.predict_one(x)
        model.learn_one(x, y)
        # Update metrics if provided
        #metric.update(y, prediction)
        for metric in binary_classification_metrics:
            binary_classification_metrics_dict[metric].update(y, prediction)
        # Periodically log progress
        if message.offset % 100 == 0:
            #print(f"Processed {message.offset} messages")
            #print(f"Current accuracy: {metric.get():.2%}")
            #mlflow.log_metric("Accuracy", metric.get())
            for metric in binary_classification_metrics:
                print(f"{metric}: {binary_classification_metrics_dict[metric].get():.2%}")
            print(f"Last prediction: {'Fraud' if prediction == 1 else 'Legit'}")
            #print(f"Metric: {metric}")
            #data_df.to_pickle(DATA_PATH)
    #except:
    #    print("Stopping consumer...")
    #finally:
    #    data_df.to_pickle(DATA_PATH)
    #    consumer.close()
    #    print("Consumer closed.")

if __name__ == "__main__":
    main()