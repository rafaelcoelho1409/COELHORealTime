from river import (
    metrics,
    utils,
)
import datetime as dt
import pickle
import json
import os
import signal
import tempfile
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
    load_or_create_encoders,
    load_kafka_offset_from_mlflow,
    get_best_mlflow_run,
    MLFLOW_MODEL_NAMES,
    ENCODER_ARTIFACT_NAMES,
    KAFKA_OFFSET_ARTIFACT,
    # Redis live model cache (real-time predictions during training)
    save_live_model_to_redis,
    clear_live_model_from_redis,
    REDIS_CACHE_INTERVAL,
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
PROJECT_NAME = "E-Commerce Customer Interactions"
MODEL_NAME = MLFLOW_MODEL_NAMES[PROJECT_NAME]
ENCODER_ARTIFACT_NAME = ENCODER_ARTIFACT_NAMES[PROJECT_NAME]
CLUSTER_COUNTS_ARTIFACT = "cluster_counts.json"
CLUSTER_FEATURE_COUNTS_ARTIFACT = "cluster_feature_counts.json"


def load_cluster_data_from_mlflow():
    """Load cluster counts and feature counts from the best MLflow run.
    For clustering, since there are no metrics, get_best_mlflow_run
    will return the latest run.
    """
    cluster_counts = Counter()
    cluster_feature_counts = defaultdict(lambda: defaultdict(Counter))
    try:
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
        run_id = get_best_mlflow_run(PROJECT_NAME, MODEL_NAME)
        if run_id is None:
            print("No MLflow run found, starting with empty cluster data.")
            return cluster_counts, cluster_feature_counts
        # Load cluster counts
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id = run_id,
                artifact_path = CLUSTER_COUNTS_ARTIFACT
            )
            with open(local_path, 'r') as f:
                cluster_counts = Counter(json.load(f))
            print("Cluster counts loaded from MLflow.")
        except Exception as e:
            print(f"Could not load cluster counts from MLflow: {e}")
        # Load cluster feature counts
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id = run_id,
                artifact_path = CLUSTER_FEATURE_COUNTS_ARTIFACT
            )
            with open(local_path, 'r') as f:
                loaded_counts_str_keys = json.load(f)
                for cluster_id_str, features in loaded_counts_str_keys.items():
                    try:
                        cluster_id_int = int(cluster_id_str)
                        for feature_name, value_counts in features.items():
                            cluster_feature_counts[cluster_id_int][feature_name] = Counter(value_counts)
                    except ValueError:
                        print(f"Warning: Could not convert cluster ID '{cluster_id_str}' to int.", file=sys.stderr)
            print("Cluster feature counts loaded from MLflow.")
        except Exception as e:
            print(f"Could not load cluster feature counts from MLflow: {e}")
    except Exception as e:
        print(f"Error loading cluster data from MLflow: {e}")
    return cluster_counts, cluster_feature_counts


def main():
    """River ML Training for E-Commerce Customer Interactions (Clustering).

    Continues training from the best model in MLflow.
    """
    # Initialize model and metrics
    print(f"Connecting to MLflow at http://{MLFLOW_HOST}:5000")
    mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
    mlflow.set_experiment(PROJECT_NAME)
    print(f"MLflow experiment '{PROJECT_NAME}' set successfully")
    # Load or create model and encoders from MLflow
    print("Loading model from MLflow (best historical model)")
    encoders = load_or_create_encoders(PROJECT_NAME, "river")
    model = load_or_create_model(PROJECT_NAME, MODEL_NAME)
    # Load cluster data from MLflow
    cluster_counts, cluster_feature_counts = load_cluster_data_from_mlflow()
    print("Encoders loaded")
    print(f"Model loaded: {model.__class__.__name__}")
    # Load last processed Kafka offset from MLflow
    last_offset = load_kafka_offset_from_mlflow(PROJECT_NAME)
    # Create consumer with starting offset
    consumer = create_consumer(PROJECT_NAME, start_offset=last_offset)
    print("Consumer started. Waiting for transactions...")
    # Track current offset for persistence
    current_offset = last_offset if last_offset is not None else -1
    # -----------------------------------------------------------------------------
    # CLUSTERING METRICS
    # -----------------------------------------------------------------------------
    clustering_metric_classes = {
        "Silhouette": metrics.Silhouette,
    }
    clustering_metric_args = {
        "Silhouette": {},
    }
    # -----------------------------------------------------------------------------
    # ROLLING METRICS (for concept drift detection)
    # -----------------------------------------------------------------------------
    rolling_metric_classes = {
        "RollingSilhouette": utils.Rolling,
    }
    rolling_metric_args = {
        "RollingSilhouette": {"obj": metrics.Silhouette(), "window_size": 1000},
    }
    # -----------------------------------------------------------------------------
    # TIME-BASED ROLLING METRICS
    # -----------------------------------------------------------------------------
    time_rolling_metric_classes = {
        "TimeRollingSilhouette": utils.TimeRolling,
    }
    time_rolling_metric_args = {
        "TimeRollingSilhouette": {"obj": metrics.Silhouette(), "period": dt.timedelta(minutes=5)},
    }
    # =============================================================================
    # INSTANTIATE ALL METRICS
    # =============================================================================
    clustering_metrics = {
        name: clustering_metric_classes[name](**clustering_metric_args[name])
        for name in clustering_metric_classes
    }
    rolling_metrics = {
        name: rolling_metric_classes[name](**rolling_metric_args[name])
        for name in rolling_metric_classes
    }
    time_rolling_metrics = {
        name: time_rolling_metric_classes[name](**time_rolling_metric_args[name])
        for name in time_rolling_metric_classes
    }
    # Initialize cluster history
    n_clusters_history = []
    n_micro_clusters_history = []
    # Batch sizes for different operations (tuned for performance)
    PROGRESS_LOG_INTERVAL = 100     # Log progress every N messages
    ARTIFACT_SAVE_INTERVAL = 1000   # Save model/encoders/cluster data to S3 every N messages
    # Get traceability info from best run (if continuing from existing model)
    best_run_id = get_best_mlflow_run(PROJECT_NAME, MODEL_NAME)
    baseline_metrics = {}
    if best_run_id:
        try:
            best_run = mlflow.get_run(best_run_id)
            baseline_metrics = best_run.data.metrics
            print(f"Loaded baseline metrics from best run: {best_run_id}")
        except Exception as e:
            print(f"Could not get baseline metrics from best run: {e}")
    # Use MODEL_NAME for run name (not model.__class__.__name__) to maintain
    # compatibility with MLflow queries that filter by run name.
    print(f"Starting MLflow run with model: {MODEL_NAME}")
    with mlflow.start_run(run_name = MODEL_NAME):
        # Log traceability tags for model lineage
        if best_run_id:
            mlflow.set_tag("training_mode", "continued")
            mlflow.set_tag("continued_from_run", best_run_id)
            # Log baseline metrics as tags (clustering + rolling metrics)
            all_metric_names = list(clustering_metrics.keys()) + list(rolling_metrics.keys()) + list(time_rolling_metrics.keys()) + ["n_clusters", "n_micro_clusters"]
            for metric_name in all_metric_names:
                if metric_name in baseline_metrics:
                    mlflow.set_tag(f"baseline_{metric_name}", f"{baseline_metrics[metric_name]:.4f}")
            # Also log cluster counts if available
            if cluster_counts:
                mlflow.set_tag("baseline_num_clusters", str(len(cluster_counts)))
                mlflow.set_tag("baseline_total_samples", str(sum(cluster_counts.values())))
            print(f"Traceability tags set: continued_from_run={best_run_id}")
        else:
            mlflow.set_tag("training_mode", "from_scratch")
            print("Starting training from scratch (no previous model in MLflow)")
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
                    timestamp = dt.datetime.strptime(
                        interaction_event['timestamp'],
                        "%Y-%m-%dT%H:%M:%S.%f%z")
                    # Update the model
                    model.learn_one(x)
                    prediction = model.predict_one(x)
                    # Track current offset for persistence
                    current_offset = message.offset
                    # Track cluster statistics
                    n_clusters_history.append(model.n_clusters)
                    n_micro_clusters_history.append(len(model.micro_clusters))
                    # Update cluster_counts if provided
                    try:
                        cluster_counts[prediction] += 1
                    except Exception as e:
                        print(f"Error updating cluster_counts: {str(e)}")
                    # Update metrics (only if at least 2 clusters exist)
                    if len(model.centers) >= 2:
                        for metric in clustering_metrics.values():
                            try:
                                metric.update(x, prediction, model.centers)
                            except Exception as e:
                                print(f"Error updating metric {metric}: {str(e)}")
                        for metric in rolling_metrics.values():
                            try:
                                metric.update(x, prediction, model.centers)
                            except Exception as e:
                                print(f"Error updating metric {metric}: {str(e)}")
                        for metric in time_rolling_metrics.values():
                            try:
                                metric.update(x, prediction, model.centers, t=timestamp)
                            except Exception as e:
                                print(f"Error updating metric {metric}: {str(e)}")
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
                        metrics_to_log = {}
                        for name, metric in clustering_metrics.items():
                            metrics_to_log[name] = metric.get()
                        for name, metric in rolling_metrics.items():
                            metrics_to_log[name] = metric.get()
                        for name, metric in time_rolling_metrics.items():
                            metrics_to_log[name] = metric.get()
                        # Add cluster statistics
                        metrics_to_log["n_clusters"] = model.n_clusters
                        metrics_to_log["n_micro_clusters"] = len(model.micro_clusters)
                        mlflow.log_metrics(metrics_to_log, step = message.offset)
                    # Save live model to Redis for real-time predictions
                    if message.offset % REDIS_CACHE_INTERVAL == 0:
                        save_live_model_to_redis(PROJECT_NAME, MODEL_NAME, model, encoders)
                    # Save artifacts to MLflow (using temp files)
                    if message.offset % ARTIFACT_SAVE_INTERVAL == 0 and message.offset > 0:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                            encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                            cluster_counts_path = os.path.join(tmpdir, CLUSTER_COUNTS_ARTIFACT)
                            cluster_feature_counts_path = os.path.join(tmpdir, CLUSTER_FEATURE_COUNTS_ARTIFACT)
                            offset_path = os.path.join(tmpdir, KAFKA_OFFSET_ARTIFACT)
                            with open(model_path, 'wb') as f:
                                pickle.dump(model, f)
                            with open(encoder_path, 'wb') as f:
                                pickle.dump(encoders, f)
                            with open(cluster_counts_path, 'w') as f:
                                json.dump(dict(cluster_counts), f, indent=4)
                            plain_dict_feature_counts = {
                                k: {fk: dict(fv) for fk, fv in v.items()}
                                for k, v in cluster_feature_counts.items()
                            }
                            with open(cluster_feature_counts_path, 'w') as f:
                                json.dump(plain_dict_feature_counts, f, indent=4)
                            with open(offset_path, 'w') as f:
                                json.dump({"last_offset": current_offset}, f)
                            mlflow.log_artifact(model_path)
                            mlflow.log_artifact(encoder_path)
                            mlflow.log_artifact(cluster_counts_path)
                            mlflow.log_artifact(cluster_feature_counts_path)
                            mlflow.log_artifact(offset_path)
                        print(f"Artifacts saved to MLflow at offset {message.offset}")
                # Consumer timeout reached, loop continues if not shutdown
            print("Graceful shutdown completed.")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            # Save final model, encoders, cluster data, and offset to MLflow on shutdown
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                cluster_counts_path = os.path.join(tmpdir, CLUSTER_COUNTS_ARTIFACT)
                cluster_feature_counts_path = os.path.join(tmpdir, CLUSTER_FEATURE_COUNTS_ARTIFACT)
                offset_path = os.path.join(tmpdir, KAFKA_OFFSET_ARTIFACT)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoders, f)
                with open(cluster_counts_path, 'w') as f:
                    json.dump(dict(cluster_counts), f, indent = 4)
                plain_dict_feature_counts = {
                    k: {fk: dict(fv) for fk, fv in v.items()}
                    for k, v in cluster_feature_counts.items()
                }
                with open(cluster_feature_counts_path, 'w') as f:
                    json.dump(plain_dict_feature_counts, f, indent = 4)
                with open(offset_path, 'w') as f:
                    json.dump({"last_offset": current_offset}, f)
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(encoder_path)
                mlflow.log_artifact(cluster_counts_path)
                mlflow.log_artifact(cluster_feature_counts_path)
                mlflow.log_artifact(offset_path)
            print(f"Final artifacts saved to MLflow (offset={current_offset})")
            # Clear live model from Redis (training stopped)
            clear_live_model_from_redis(PROJECT_NAME, MODEL_NAME)
            if consumer is not None:
                consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()