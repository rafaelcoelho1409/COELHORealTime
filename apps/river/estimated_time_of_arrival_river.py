from river import (
    metrics
)
import pickle
import json
import os
import signal
import tempfile
import mlflow
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


MLFLOW_HOST = os.environ["MLFLOW_HOST"]
PROJECT_NAME = "Estimated Time of Arrival"
MODEL_NAME = MLFLOW_MODEL_NAMES[PROJECT_NAME]
ENCODER_ARTIFACT_NAME = ENCODER_ARTIFACT_NAMES[PROJECT_NAME]

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


def main():
    """River ML Training for Estimated Time of Arrival.

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

    print("Encoders loaded")
    print(f"Model loaded: {model.__class__.__name__}")

    # Load last processed Kafka offset from MLflow
    last_offset = load_kafka_offset_from_mlflow(PROJECT_NAME)

    # Create consumer with starting offset
    consumer = create_consumer(PROJECT_NAME, start_offset=last_offset)
    print("Consumer started. Waiting for transactions...")
    # Track current offset for persistence
    current_offset = last_offset if last_offset is not None else -1
    regression_metrics = [
        'MAE',
        'MAPE',
        'MSE',
        'R2',
        'RMSE',
        'RMSLE',
        'SMAPE',
    ]
    regression_metrics_dict = {
        x: getattr(metrics, x)() for x in regression_metrics
    }
    # Batch sizes for different operations (tuned for performance)
    METRICS_LOG_INTERVAL = 100      # Log metrics to MLflow every N messages
    ARTIFACT_SAVE_INTERVAL = 1000   # Save model/encoders to S3 every N messages
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
    print(f"Starting MLflow run with model: {model.__class__.__name__}")
    with mlflow.start_run(run_name = model.__class__.__name__):
        # Log traceability tags for model lineage
        if best_run_id:
            mlflow.set_tag("training_mode", "continued")
            mlflow.set_tag("continued_from_run", best_run_id)
            # Log all baseline metrics as tags
            for metric_name in regression_metrics:
                if metric_name in baseline_metrics:
                    mlflow.set_tag(f"baseline_{metric_name}", f"{baseline_metrics[metric_name]:.4f}")
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
                    eta_event = message.value
                    # Process the transaction
                    x = {
                        'trip_id':                               eta_event['trip_id'],
                        'driver_id':                             eta_event['driver_id'],
                        'vehicle_id':                            eta_event['vehicle_id'],
                        'timestamp':                             eta_event['timestamp'],
                        'origin':                                eta_event['origin'],
                        'destination':                           eta_event['destination'],
                        'estimated_distance_km':                 eta_event['estimated_distance_km'],
                        'weather':                               eta_event['weather'],
                        'temperature_celsius':                   eta_event['temperature_celsius'],
                        'day_of_week':                           eta_event['day_of_week'],
                        'hour_of_day':                           eta_event['hour_of_day'],
                        'driver_rating':                         eta_event['driver_rating'],
                        'vehicle_type':                          eta_event['vehicle_type'],
                        'initial_estimated_travel_time_seconds': eta_event['initial_estimated_travel_time_seconds'],
                        'debug_traffic_factor':                  eta_event['debug_traffic_factor'],
                        'debug_weather_factor':                  eta_event['debug_weather_factor'],
                        'debug_incident_delay_seconds':          eta_event['debug_incident_delay_seconds'],
                        'debug_driver_factor':                   eta_event['debug_driver_factor']
                    }
                    x, encoders = process_sample(
                        x,
                        encoders,
                        PROJECT_NAME)
                    y = eta_event['simulated_actual_travel_time_seconds']
                    # Update the model
                    model.learn_one(x, y)
                    prediction = model.predict_one(x)
                    # Track current offset for persistence
                    current_offset = message.offset
                    # Update metrics (once per message, not twice)
                    for metric in regression_metrics:
                        try:
                            regression_metrics_dict[metric].update(y, prediction)
                        except Exception as e:
                            print(f"Error updating metric {metric}: {str(e)}")
                    # Log metrics to MLflow periodically (batched for efficiency)
                    if message.offset % METRICS_LOG_INTERVAL == 0:
                        print(f"Processed {message.offset} messages")
                        # Batch log all metrics in one call (reduces HTTP overhead)
                        metrics_to_log = {
                            metric: regression_metrics_dict[metric].get()
                            for metric in regression_metrics
                        }
                        mlflow.log_metrics(metrics_to_log, step=message.offset)
                    # Save live model to Redis for real-time predictions
                    if message.offset % REDIS_CACHE_INTERVAL == 0:
                        save_live_model_to_redis(PROJECT_NAME, MODEL_NAME, model, encoders)
                    # Save artifacts to MLflow (using temp files)
                    if message.offset % ARTIFACT_SAVE_INTERVAL == 0 and message.offset > 0:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                            encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                            offset_path = os.path.join(tmpdir, KAFKA_OFFSET_ARTIFACT)
                            with open(model_path, 'wb') as f:
                                pickle.dump(model, f)
                            with open(encoder_path, 'wb') as f:
                                pickle.dump(encoders, f)
                            with open(offset_path, 'w') as f:
                                json.dump({"last_offset": current_offset}, f)
                            mlflow.log_artifact(model_path)
                            mlflow.log_artifact(encoder_path)
                            mlflow.log_artifact(offset_path)
                        print(f"Artifacts saved to MLflow at offset {message.offset}")
                # Consumer timeout reached, loop continues if not shutdown
            print("Graceful shutdown completed.")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            # Save final model, encoders, and offset to MLflow on shutdown
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                offset_path = os.path.join(tmpdir, KAFKA_OFFSET_ARTIFACT)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoders, f)
                with open(offset_path, 'w') as f:
                    json.dump({"last_offset": current_offset}, f)
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(encoder_path)
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