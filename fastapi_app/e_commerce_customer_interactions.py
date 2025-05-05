from river import (
    metrics
)
import pickle
import os
import pandas as pd
import mlflow
from collections import Counter
import sys
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_data,
    load_or_create_encoders,
)


DATA_PATH = "data/e_commerce_customer_interactions_data.parquet"
MODEL_FOLDER = "models/e_commerce_customer_interactions"
ENCODERS_PATH = "encoders/e_commerce_customer_interactions.pkl"
PROJECT_NAME = "E-Commerce Customer Interactions"
try:
    with open("data/cluster_counts.pkl", 'rb') as f:
        cluster_counts = pickle.load(f)
    print("Cluster counts loaded from disk.")
except FileNotFoundError as e:
    cluster_counts = Counter()
    print(f"Creating cluster counts: {e}", file = sys.stderr)

os.makedirs(MODEL_FOLDER, exist_ok = True)
os.makedirs("encoders", exist_ok = True)
os.makedirs("data", exist_ok = True)


def main():
    # Initialize model and metrics
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("E-Commerce Customer Interactions")
    encoders = load_or_create_encoders(
        PROJECT_NAME
    )
    model = load_or_create_model(
        PROJECT_NAME,
        MODEL_FOLDER
    )
    # Create consumer
    consumer = create_consumer(PROJECT_NAME)
    print("Consumer started. Waiting for transactions...")
    data_df = load_or_create_data(
        consumer,
        PROJECT_NAME)
    #clustering_metrics_dict = {
    #    x: getattr(metrics, x)() for x in clustering_metrics
    #}
    BATCH_SIZE_OFFSET = 1000
    with mlflow.start_run(run_name = model.__class__.__name__):
        try:
            for message in consumer:
                interaction_event = message.value
                # Create a new DataFrame from the received data
                new_row = pd.DataFrame([interaction_event])
                # Append the new row to the existing DataFrame
                data_df = pd.concat([data_df, new_row], ignore_index = True)
                # Process the transaction
                x = {
                     'customer_id':            interaction_event['customer_id'],
                     'device_info':            interaction_event['device_info'],
                     'event_id':               interaction_event['event_id'],
                     'event_type':             interaction_event['event_type'],
                     'location':               interaction_event['location'],
                     'page_url':               interaction_event['page_url'],
                     'price':                  interaction_event['price'],
                     'product_category':       interaction_event['product_category'],
                     'product_id':             interaction_event['product_id'],
                     'quantity':               interaction_event['quantity'],
                     'referrer_url':           interaction_event['referrer_url'],
                     'search_query':           interaction_event['search_query'],
                     'session_event_sequence': interaction_event['session_event_sequence'],
                     'session_id':             interaction_event['session_id'],
                     'time_on_page_seconds':   interaction_event['time_on_page_seconds'],
                     'timestamp':              interaction_event['timestamp']
                    }
                x, encoders = process_sample(
                    x, 
                    encoders,
                    PROJECT_NAME)
                # Update the model
                model.learn_one(x)
                prediction = model.predict_one(x)
                # Update metrics if provided
                try:
                    cluster_counts[prediction] += 1
                except Exception as e:
                    print(f"Error updating metric: {str(e)}")
                # Periodically log progress
                if message.offset % BATCH_SIZE_OFFSET == 0:
                    try:
                        cluster_counts[prediction] += 1
                    except Exception as e:
                        print(f"Error updating metric: {str(e)}")
                    with open(ENCODERS_PATH, 'wb') as f:
                        pickle.dump(encoders, f)
                    mlflow.log_artifact(ENCODERS_PATH)
                    MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                    with open(MODEL_VERSION, 'wb') as f:
                        pickle.dump(model, f)
                    # Save samples_per_clusters as artifact to be retrieved by Streamlit
                    with open("data/cluster_counts.pkl", 'wb') as f:
                        pickle.dump(cluster_counts, f)
                    mlflow.log_artifact("data/cluster_counts.pkl")
                if message.offset % (BATCH_SIZE_OFFSET * 10) == 0:
                    mlflow.log_artifact(MODEL_VERSION)
                    data_df.to_parquet(DATA_PATH)
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
            with open(MODEL_VERSION, 'wb') as f:
                pickle.dump(model, f)
            with open(ENCODERS_PATH, 'wb') as f:
                pickle.dump(encoders, f)
            with open("data/cluster_counts.pkl", 'wb') as f:
                pickle.dump(cluster_counts, f)
            data_df.to_parquet(DATA_PATH)
            consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()