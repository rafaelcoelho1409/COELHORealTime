from river import (
    metrics
)
import pickle
import json
import os
import pandas as pd
import mlflow
from collections import (
    Counter,
    defaultdict
)
import sys
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_data,
    load_or_create_encoders,
)


DATA_PATH = "data/e_commerce_customer_interactions.parquet"
MODEL_FOLDER = "models/e_commerce_customer_interactions"
ENCODERS_PATH = "encoders/river/e_commerce_customer_interactions.pkl"
CLUSTER_COUNTS_PATH = "data/cluster_counts.json"
CLUSTER_FEATURE_COUNTS_PATH = "data/cluster_feature_counts.json"
PROJECT_NAME = "E-Commerce Customer Interactions"

try:
    with open(CLUSTER_COUNTS_PATH, 'rb') as f:
        cluster_counts = Counter(json.load(f))
    print("Cluster counts loaded from disk.")
except FileNotFoundError:
    cluster_counts = Counter()
    print("Creating new cluster counts (FileNotFoundError).", file = sys.stderr)
except json.JSONDecodeError:
    cluster_counts = Counter()
    print("Creating new cluster counts (JSONDecodeError).", file = sys.stderr)

# Load or create cluster feature counts
try:
    with open(CLUSTER_FEATURE_COUNTS_PATH, 'r') as f:
        loaded_counts_str_keys = json.load(f)
        cluster_feature_counts = defaultdict(lambda: defaultdict(Counter))
        for cluster_id_str, features in loaded_counts_str_keys.items():
            try:
                # Assuming cluster IDs from model.predict_one() are integers.
                # JSON keys are strings, so convert back to int.
                cluster_id_int = int(cluster_id_str)
                for feature_name, value_counts in features.items():
                    cluster_feature_counts[cluster_id_int][feature_name] = Counter(value_counts)
            except ValueError:
                # This handles cases where a cluster_id_str might not be a valid integer.
                # For example, if the JSON file was manually edited with non-integer keys.
                print(f"Warning: Could not convert cluster ID '{cluster_id_str}' to int. Skipping this entry for feature counts.", file=sys.stderr)
    print("Cluster feature counts loaded from disk.")
except FileNotFoundError:
    cluster_feature_counts = defaultdict(lambda: defaultdict(Counter))
    print("Creating new cluster feature counts (FileNotFoundError).", file=sys.stderr)
except json.JSONDecodeError:
    cluster_feature_counts = defaultdict(lambda: defaultdict(Counter))
    print("Error decoding cluster feature counts JSON, creating new (JSONDecodeError).", file=sys.stderr)

os.makedirs(MODEL_FOLDER, exist_ok = True)
os.makedirs("encoders/river", exist_ok = True)
os.makedirs("data", exist_ok = True)


def main():
    # Initialize model and metrics
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(PROJECT_NAME)
    encoders = load_or_create_encoders(
        PROJECT_NAME,
        "river"
    )
    model = load_or_create_model(
        PROJECT_NAME,
        "DBSTREAM",
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
    BATCH_SIZE_OFFSET = 100
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
                # Update cluster_counts if provided
                try:
                    cluster_counts[prediction] += 1
                except Exception as e:
                    print(f"Error updating cluster_counts: {str(e)}")
                predicted_cluster_label = prediction # prediction is likely an int
                for feature_key, feature_value in interaction_event.items():
                    if feature_key not in [
                        'event_id',
                        'customer_id',
                        'session_id',
                        'timestamp',
                        'price',
                        'page_url',
                        'search_query'
                        ]:
                        # Ensure feature_value is hashable for Counter keys.
                        # Convert complex types (like dicts or lists) to their string representation.
                        # Simple types (str, int, float, bool, None) are fine.
                        if not isinstance(feature_value, (str, int, float, bool, type(None))):
                            processed_feature_value = str(feature_value)
                        else:
                            processed_feature_value = feature_value
                        try:
                            cluster_feature_counts[predicted_cluster_label][feature_key][processed_feature_value] += 1
                        except Exception as e:
                            print(f"Error updating cluster_feature_counts for feature '{feature_key}': {str(e)}", file = sys.stderr)
                # Periodically log progress
                if message.offset % BATCH_SIZE_OFFSET == 0:
                    try:
                        cluster_counts[prediction] += 1
                    except Exception as e:
                        print(f"Error updating metric: {str(e)}")
                    with open(ENCODERS_PATH, 'wb') as f:
                        pickle.dump(encoders, f)
                    #mlflow.log_artifact(ENCODERS_PATH)
                    # Save samples_per_clusters
                    with open(CLUSTER_COUNTS_PATH, 'w') as f:
                        json.dump(dict(cluster_counts), f, indent = 4)
                    plain_dict_feature_counts = {
                        k: {fk: dict(fv) for fk, fv in v.items()}
                        for k, v in cluster_feature_counts.items()
                    }
                    with open(CLUSTER_FEATURE_COUNTS_PATH, 'w') as f:
                        json.dump(plain_dict_feature_counts, f, indent = 4)
                if message.offset % (BATCH_SIZE_OFFSET * 10) == 0:
                    MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                    with open(MODEL_VERSION, 'wb') as f:
                        pickle.dump(model, f)
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
            with open(CLUSTER_COUNTS_PATH, 'w') as f:
                json.dump(dict(cluster_counts), f, indent = 4)
            plain_dict_feature_counts = {
                k: {fk: dict(fv) for fk, fv in v.items()}
                for k, v in cluster_feature_counts.items()
            }
            with open(CLUSTER_FEATURE_COUNTS_PATH, 'w') as f:
                json.dump(plain_dict_feature_counts, f, indent = 4)
            data_df.to_parquet(DATA_PATH)
            consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()