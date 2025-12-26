from river import (
    metrics
)
import pickle
import json
import os
import signal
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


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    print(f"\nReceived {signal_name}, initiating graceful shutdown...")
    _shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


MLFLOW_HOST = os.environ["MLFLOW_HOST"]


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
    mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
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
    # Batch sizes for different operations (tuned for performance)
    PROGRESS_LOG_INTERVAL = 100     # Log progress every N messages
    ARTIFACT_SAVE_INTERVAL = 1000   # Save model/encoders/cluster data to S3 every N messages
    DATA_SAVE_INTERVAL = 5000       # Save parquet data every N messages

    # Buffer for efficient DataFrame building (avoid pd.concat on every message)
    pending_rows = []
    BUFFER_FLUSH_SIZE = 500  # Flush buffer to DataFrame every N rows

    print(f"Starting MLflow run with model: {model.__class__.__name__}")
    with mlflow.start_run(run_name = model.__class__.__name__):
        print("MLflow run started, entering consumer loop...")
        try:
            # Use while loop to handle consumer timeout and check for shutdown
            while not _shutdown_requested:
                for message in consumer:
                    # Check for graceful shutdown request
                    if _shutdown_requested:
                        print("Shutdown requested, breaking out of consumer loop...")
                        break
                    interaction_event = message.value
                    # Buffer rows instead of concat on every message (major performance fix)
                    pending_rows.append(interaction_event)
                    # Flush buffer periodically to avoid memory buildup
                    if len(pending_rows) >= BUFFER_FLUSH_SIZE:
                        data_df = pd.concat([data_df, pd.DataFrame(pending_rows)], ignore_index=True)
                        pending_rows = []
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
                    if message.offset % PROGRESS_LOG_INTERVAL == 0:
                        print(f"Processed {message.offset} messages")
                    # Save artifacts less frequently (S3 uploads are expensive)
                    if message.offset % ARTIFACT_SAVE_INTERVAL == 0 and message.offset > 0:
                        MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                        with open(MODEL_VERSION, 'wb') as f:
                            pickle.dump(model, f)
                        with open(ENCODERS_PATH, 'wb') as f:
                            pickle.dump(encoders, f)
                        # Save cluster counts
                        with open(CLUSTER_COUNTS_PATH, 'w') as f:
                            json.dump(dict(cluster_counts), f, indent = 4)
                        plain_dict_feature_counts = {
                            k: {fk: dict(fv) for fk, fv in v.items()}
                            for k, v in cluster_feature_counts.items()
                        }
                        with open(CLUSTER_FEATURE_COUNTS_PATH, 'w') as f:
                            json.dump(plain_dict_feature_counts, f, indent = 4)
                        mlflow.log_artifact(MODEL_VERSION)
                        mlflow.log_artifact(ENCODERS_PATH)
                        mlflow.log_artifact(CLUSTER_COUNTS_PATH)
                        mlflow.log_artifact(CLUSTER_FEATURE_COUNTS_PATH)
                        print(f"Artifacts saved at offset {message.offset}")
                    # Save data even less frequently (parquet write is heavy)
                    if message.offset % DATA_SAVE_INTERVAL == 0 and message.offset > 0:
                        # Flush pending rows before saving
                        if pending_rows:
                            data_df = pd.concat([data_df, pd.DataFrame(pending_rows)], ignore_index=True)
                            pending_rows = []
                        data_df.to_parquet(DATA_PATH)
                        print(f"Data saved at offset {message.offset}")
                # Consumer timeout reached, loop continues if not shutdown
            print("Graceful shutdown completed.")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            # Flush any remaining buffered rows
            if pending_rows:
                data_df = pd.concat([data_df, pd.DataFrame(pending_rows)], ignore_index=True)
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
            if consumer is not None:
                consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()