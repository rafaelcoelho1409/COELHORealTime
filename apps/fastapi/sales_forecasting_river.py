from river import (
    metrics,
    drift
)
import pickle
import os
import pandas as pd
import mlflow
from pprint import pprint
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
    BATCH_SIZE_OFFSET = 100
    with mlflow.start_run(run_name = model.__class__.__name__):
        try:
            for message in consumer:
                sale = message.value
                # Create a new DataFrame from the received data
                new_row = pd.DataFrame([sale])
                # Append the new row to the existing DataFrame
                data_df = pd.concat([data_df, new_row], ignore_index = True)
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
                # Update metrics if provided
                try:
                    for metric in regression_metrics:
                        regression_metrics_dict[metric].update(y, prediction)
                except Exception as e:
                    print(f"Error updating metric {metric}: {str(e)}")
                # Periodically log progress
                if message.offset % BATCH_SIZE_OFFSET == 0:
                    #print(f"Processed {message.offset} messages")
                    for metric in regression_metrics:
                        try:
                            regression_metrics_dict[metric].update(y, prediction)
                        except Exception as e:
                            print(f"Error updating metric {metric}: {str(e)}")
                        mlflow.log_metric(metric, regression_metrics_dict[metric].get())
                        #print(f"{metric}: {binary_classification_metrics_dict[metric].get():.2%}")
                    with open(ENCODERS_PATH, 'wb') as f:
                        pickle.dump(encoders, f)
                    mlflow.log_artifact(ENCODERS_PATH)
                if message.offset % (BATCH_SIZE_OFFSET * 10) == 0:
                    MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                    with open(MODEL_VERSION, 'wb') as f:
                        pickle.dump(model, f)
                    data_df.to_parquet(DATA_PATH)
        except:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
            with open(MODEL_VERSION, 'wb') as f:
                pickle.dump(model, f)
            with open(ENCODERS_PATH, 'wb') as f:
                pickle.dump(encoders, f)
            data_df.to_parquet(DATA_PATH)
            consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()