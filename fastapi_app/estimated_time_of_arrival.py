from river import (
    metrics
)
import pickle
import os
import pandas as pd
import mlflow
from functions import (
    process_sample,
    load_or_create_model,
    create_consumer,
    load_or_create_data,
    load_or_create_encoders,
)

DATA_PATH = "data/estimated_time_of_arrival_data.parquet"
MODEL_FOLDER = "models/estimated_time_of_arrival"
ENCODERS_PATH = "encoders/estimated_time_of_arrival.pkl"
PROJECT_NAME = "Estimated Time of Arrival"

os.makedirs(MODEL_FOLDER, exist_ok = True)
os.makedirs("encoders", exist_ok = True)
os.makedirs("data", exist_ok = True)

def main():
    # Initialize model and metrics
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Estimated Time of Arrival")
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
        'RMSLE',
        'SMAPE',
    ]
    regression_metrics_dict = {
        x: getattr(metrics, x)() for x in regression_metrics
    }
    BATCH_SIZE_OFFSET = 1000
    with mlflow.start_run(run_name = model.__class__.__name__):
        try:
            for message in consumer:
                eta_event = message.value
                # Create a new DataFrame from the received data
                new_row = pd.DataFrame([eta_event])
                # Append the new row to the existing DataFrame
                data_df = pd.concat([data_df, new_row], ignore_index = True)
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
                    MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
                    with open(MODEL_VERSION, 'wb') as f:
                        pickle.dump(model, f)
                if message.offset % (BATCH_SIZE_OFFSET * 10) == 0:
                    mlflow.log_artifact(MODEL_VERSION)
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