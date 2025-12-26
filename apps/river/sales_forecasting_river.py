from river import (
    metrics,
    drift
)
import pickle
import os
import signal
import pandas as pd
import mlflow
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_data,
    load_or_create_encoders,
)


MLFLOW_HOST = os.environ["MLFLOW_HOST"]


DATA_PATH = "data/sales_forecasting.parquet"
MODEL_FOLDER = "models/sales_forecasting"
ENCODERS_PATH = "encoders/sales_forecasting.pkl"
PROJECT_NAME = "Sales Forecasting"

os.makedirs(MODEL_FOLDER, exist_ok = True)
os.makedirs("encoders", exist_ok = True)
os.makedirs("data", exist_ok = True)

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
    # Initialize model and metrics
    mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
    mlflow.set_experiment(PROJECT_NAME)
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
    regression_metrics = [
        'MAE',
        'MAPE',
        'MSE',
        'R2',
        'RMSE',
        #'RMSLE',
        'SMAPE',
    ]
    regression_metrics_dict = {
        x: getattr(metrics, x)() for x in regression_metrics
    }
    # Batch sizes for different operations (tuned for performance)
    METRICS_LOG_INTERVAL = 100      # Log metrics to MLflow every N messages
    ARTIFACT_SAVE_INTERVAL = 1000   # Save model/encoders to S3 every N messages
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
                    sale = message.value
                    # Buffer rows instead of concat on every message (major performance fix)
                    pending_rows.append(sale)
                    # Flush buffer periodically to avoid memory buildup
                    if len(pending_rows) >= BUFFER_FLUSH_SIZE:
                        data_df = pd.concat([data_df, pd.DataFrame(pending_rows)], ignore_index=True)
                        pending_rows = []
                    # Process the sale
                    x = {
                     'concept_drift_stage': sale['concept_drift_stage'],
                     'day_of_week':         sale['day_of_week'],
                     'event_id':            sale['event_id'],
                     'is_holiday':          sale['is_holiday'],
                     'is_promotion_active': sale['is_promotion_active'],
                     'month':               sale['month'],
                     'product_id':          sale['product_id'],
                     'promotion_id':        sale['promotion_id'],
                     'store_id':            sale['store_id'],
                     'timestamp':           sale['timestamp'],
                     'total_sales_amount':  sale['total_sales_amount'],
                     'unit_price':          sale['unit_price'],
                    }
                    x, encoders = process_sample(
                        x,
                        encoders,
                        PROJECT_NAME)
                    y = sale['quantity_sold']
                    # Update the model
                    model.learn_one(x = x, y = y)
                    prediction = model.forecast(horizon = 1, xs = [x])[0]
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
                    # Save artifacts less frequently (S3 uploads are expensive)
                    if message.offset % ARTIFACT_SAVE_INTERVAL == 0 and message.offset > 0:
                        MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                        with open(MODEL_VERSION, 'wb') as f:
                            pickle.dump(model, f)
                        with open(ENCODERS_PATH, 'wb') as f:
                            pickle.dump(encoders, f)
                        mlflow.log_artifact(MODEL_VERSION)
                        mlflow.log_artifact(ENCODERS_PATH)
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
            data_df.to_parquet(DATA_PATH)
            if consumer is not None:
                consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()