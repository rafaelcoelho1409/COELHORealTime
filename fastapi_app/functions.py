import pickle
import os
import sys
from typing import Any, Dict, Hashable
from kafka import KafkaConsumer
import json
import pandas as pd
import datetime as dt
from river import (
    base,
    compose, 
    metrics, 
    drift,
    forest,
    cluster,
    preprocessing,
    time_series,
    linear_model,
    optim,
    bandit,
    model_selection
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier



# Configuration
KAFKA_BROKERS = 'kafka:29092'  # Adjust as needed

###---Functions----####
#Data processing functions
class CustomOrdinalEncoder:
    """
    An incremental ordinal encoder that is picklable and processes dictionaries.
    Assigns a unique integer ID to each unique category encountered for each feature.
    """
    def __init__(self):
        # Dictionary to store mappings for each feature.
        # Keys are feature names (from input dictionary), values are dictionaries
        # mapping category value to integer ID for that feature.
        self._feature_mappings: Dict[Hashable, Dict[Any, int]] = {}
        # Dictionary to store the next available integer ID for each feature.
        # Keys are feature names, values are integers.
        self._feature_next_ids: Dict[Hashable, int] = {}
    def learn_one(self, x: Dict[Hashable, Any]):
        """
        Learns categories from a single sample dictionary.
        Iterates through the dictionary's items and learns each category value
        for its corresponding feature.
        Args:
            x: A dictionary representing a single sample.
               Keys are feature names, values are feature values.
               Assumes categorical features are present in this dictionary.
        """
        for feature_name, category_value in x.items():
            # Ensure the category value is hashable (dictionaries/lists are not)
            # You might need more sophisticated type checking or handling
            # if your input dictionaries contain complex unhashable types
            if not isinstance(category_value, Hashable):
                 print(f"Warning: Skipping unhashable value for feature '{feature_name}': {category_value}")
                 continue # Skip this feature for learning
            # If this is the first time we see this feature, initialize its mapping and counter
            if feature_name not in self._feature_mappings:
                self._feature_mappings[feature_name] = {}
                self._feature_next_ids[feature_name] = 0
            # Get the mapping and counter for this specific feature
            feature_map = self._feature_mappings[feature_name]
            feature_next_id = self._feature_next_ids[feature_name]
            # Check if the category value is already in the mapping for this feature
            if category_value not in feature_map:
                # If it's a new category for this feature, assign the next available ID
                feature_map[category_value] = feature_next_id
                # Increment the counter for the next new category for this feature
                self._feature_next_ids[feature_name] += 1
    def transform_one(self, x: Dict[Hashable, Any]) -> Dict[Hashable, int]:
        """
        Transforms categorical features in a single sample dictionary into integer IDs.
        Args:
            x: A dictionary representing a single sample.
               Keys are feature names, values are feature values.
        Returns:
            A new dictionary containing the transformed integer IDs for the
            categorical features that the encoder has seen. Features not
            seen by the encoder are excluded from the output dictionary.
        Raises:
            KeyError: If a feature is seen but a specific category value
                      within that feature has not been seen during learning.
                      You might want to add logic here to handle unseen categories
                      (e.g., return a default value like -1 or NaN for that feature).
        """
        transformed_sample: Dict[Hashable, int] = {}
        for feature_name, category_value in x.items():
            # Only attempt to transform features that the encoder has seen
            if feature_name in self._feature_mappings:
                feature_map = self._feature_mappings[feature_name]

                # Check if the category value for this feature has been seen
                if category_value in feature_map:
                    # Transform the category value using the feature's mapping
                    transformed_sample[feature_name] = feature_map[category_value]
                else:
                    # Handle unseen category values for a known feature
                    # By default, this will raise a KeyError as per the docstring.
                    # Example: return a placeholder value instead of raising error:
                    # transformed_sample[feature_name] = -1 # Or some other indicator
                    # print(f"Warning: Unseen category '{category_value}' for feature '{feature_name}' during transform.")
                    # Or raise the error explicitly:
                    raise KeyError(f"Unseen category '{category_value}' for feature '{feature_name}' during transform.")
            # Features not in self._feature_mappings are ignored in the output.
            # If you need to include them (e.g., original numerical features),
            # you would copy them over here. This encoder only outputs encoded features.
        return transformed_sample
    def get_feature_mappings(self) -> Dict[Hashable, Dict[Any, int]]:
        """Returns the current mappings for all features."""
        return self._feature_mappings
    def get_feature_next_ids(self) -> Dict[Hashable, int]:
        """Returns the next available IDs for all features."""
        return self._feature_next_ids
    def __repr__(self) -> str:
        """String representation of the encoder."""
        num_features = len(self._feature_mappings)
        feature_details = ", ".join([f"{name}: {len(mapping)} categories" for name, mapping in self._feature_mappings.items()])
        return f"CustomPicklableOrdinalEncoder(features={num_features} [{feature_details}])"
    
class DictImputer(base.Transformer):
    """
    Imputes missing values (None or missing keys) for specified features in a dictionary.

    Parameters
    ----------
    on
        List of feature names to impute.
    fill_value
        The value to use for imputation.
    """
    def __init__(self, on: list, fill_value):
        self.on = on
        self.fill_value = fill_value
    def transform_one(self, x: dict):
        x_transformed = x.copy()
        for feature in self.on:
            if x_transformed.get(feature) is None:
                x_transformed[feature] = self.fill_value
        return x_transformed

    

def extract_device_info(x):
    x_ = x['device_info']
    return {
        'os': x_['os'],
        'browser': x_['browser'],
    }

def extract_device_info_sklearn(data):
    data = data.copy()
    data_to_join = pd.json_normalize(data["device_info"])
    data = data.drop("device_info", axis = 1)
    data = data.join(data_to_join)
    return data

def extract_timestamp_info(x):
    x_ = dt.datetime.strptime(
        x['timestamp'],
        "%Y-%m-%dT%H:%M:%S.%f%z")
    return {
        'year': x_.year,
        'month': x_.month,
        'day': x_.day,
        'hour': x_.hour,
        'minute': x_.minute,
        'second': x_.second
    }

def extract_timestamp_info_sklearn(data):
    data = data.copy()
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        format = 'ISO8601')
    data["year"] = data["timestamp"].dt.year
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["second"] = data["timestamp"].dt.second
    data = data.drop("timestamp", axis = 1)
    return data

def extract_coordinates(x):
    x_ = x['location']
    return {
        'lat': x_['lat'],
        'lon': x_['lon'],
    }

def extract_coordinates_sklearn(data):
    data = data.copy()
    data_to_join = pd.json_normalize(data["location"])
    data = data.drop("location", axis = 1)
    data = data.join(data_to_join)
    return data

def load_or_create_encoders(project_name, library):
    encoders_folders_river = {
        "Transaction Fraud Detection": "encoders/river/transaction_fraud_detection.pkl",
        "Estimated Time of Arrival": "encoders/river/estimated_time_of_arrival.pkl",
        "E-Commerce Customer Interactions": "encoders/river/e_commerce_customer_interactions.pkl",
        #"Sales Forecasting": "encoders/river/sales_forecasting.pkl"
    }
    encoders_folders_sklearn = {
        "Transaction Fraud Detection": "encoders/sklearn/transaction_fraud_detection.pkl",
        "Estimated Time of Arrival": "encoders/sklearn/estimated_time_of_arrival.pkl",
        "E-Commerce Customer Interactions": "encoders/sklearn/e_commerce_customer_interactions.pkl",
        #"Sales Forecasting": "encoders/sklearn/sales_forecasting.pkl"
    }
    if library == "river":
        encoder_path = encoders_folders_river[project_name]
        try:
            with open(encoder_path, 'rb') as f:
                encoders = pickle.load(f)
            print("Ordinal encoder loaded from disk.")
        except FileNotFoundError as e:
            if project_name in ["Transaction Fraud Detection", "Estimated Time of Arrival"]:
                encoders = {
                    "ordinal_encoder": CustomOrdinalEncoder()
                }
            elif project_name in ["E-Commerce Customer Interactions"]:
                encoders = {
                    "standard_scaler": preprocessing.StandardScaler(),
                    "feature_hasher": preprocessing.FeatureHasher()
                }
            elif project_name in ["Sales Forecasting"]:
                encoders = {
                    "one_hot_encoder": preprocessing.OneHotEncoder(),
                    "standard_scaler": preprocessing.StandardScaler(),
                } #remove after developing the right model strategy
            print(f"Creating encoders: {e}", file = sys.stderr)
        except Exception as e:
            if project_name in ["Transaction Fraud Detection", "Estimated Time of Arrival"]:
                encoders = {
                    "ordinal_encoder": CustomOrdinalEncoder()
                }
            elif project_name in ["E-Commerce Customer Interactions"]:
                encoders = {
                    "standard_scaler": preprocessing.StandardScaler(),
                    "feature_hasher": preprocessing.FeatureHasher()
                }
            elif project_name in ["Sales Forecasting"]:
                encoders = {
                    "one_hot_encoder": preprocessing.OneHotEncoder(),
                    "standard_scaler": preprocessing.StandardScaler(),
                } #remove after developing the right model strategy
            print(f"Creating encoders: {e}", file = sys.stderr)
        return encoders
    elif library == "sklearn":
        #Sklearn encoders must be only loaded from disk
        encoder_path = encoders_folders_sklearn[project_name]
        try:
            with open(encoder_path, 'rb') as f:
                encoders = pickle.load(f)
            print("Scikit-Learn encoders loaded from disk.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Scikit-Learn encoders not found for project {project_name}.") from e
        except Exception as e:
            raise Exception(f"Error loading Scikit-Learn encoders for project {project_name}: {e}") from e
        return encoders



def process_sample(x, encoders, project_name, library = "river"):
    if project_name == "Transaction Fraud Detection":
        pipe1 = compose.Select(
            "amount",
            "account_age_days",
            "cvv_provided",
            "billing_address_match"
        )
        pipe1.learn_one(x)
        x1 = pipe1.transform_one(x)
        pipe2 = compose.Select(
            "currency",
            "merchant_id",
            "payment_method",
            "product_category",
            "transaction_type",
            #"user_agent"
        )
        pipe2.learn_one(x)
        x_pipe_2 = pipe2.transform_one(x)
        pipe3a = compose.Select(
            "device_info"
        )
        pipe3a.learn_one(x)
        x_pipe_3 = pipe3a.transform_one(x)
        pipe3b = compose.FuncTransformer(
            extract_device_info,
        )
        pipe3b.learn_one(x_pipe_3)
        x_pipe_3 = pipe3b.transform_one(x_pipe_3)
        pipe4a = compose.Select(
            "timestamp",
        )
        pipe4a.learn_one(x)
        x_pipe_4 = pipe4a.transform_one(x)
        pipe4b = compose.FuncTransformer(
            extract_timestamp_info,
        )
        pipe4b.learn_one(x_pipe_4)
        x_pipe_4 = pipe4b.transform_one(x_pipe_4)
        x_to_encode = x_pipe_2 | x_pipe_3 | x_pipe_4
        if library == "river":
            encoders["ordinal_encoder"].learn_one(x_to_encode)
            x2 = encoders["ordinal_encoder"].transform_one(x_to_encode)
            return x1 | x2, {
                "ordinal_encoder": encoders["ordinal_encoder"]
            }
        elif library == "sklearn":
            return x1 | x_to_encode
    elif project_name == "Estimated Time of Arrival":
        pipe1 = compose.Select(
            'estimated_distance_km',
            'temperature_celsius',
            'hour_of_day',
            'driver_rating',
            'initial_estimated_travel_time_seconds',
            'debug_traffic_factor',
            'debug_weather_factor',
            'debug_incident_delay_seconds',
            'debug_driver_factor'
        )
        pipe1.learn_one(x)
        x1 = pipe1.transform_one(x)
        pipe2 = compose.Select(
            'driver_id',
            'vehicle_id',
            'weather',
            'vehicle_type'
        )
        pipe2.learn_one(x)
        x_pipe_2 = pipe2.transform_one(x)
        pipe3a = compose.Select(
            "timestamp",
        )
        pipe3a.learn_one(x)
        x_pipe_3 = pipe3a.transform_one(x)
        pipe3b = compose.FuncTransformer(
            extract_timestamp_info,
        )
        pipe3b.learn_one(x_pipe_3)
        x_pipe_3 = pipe3b.transform_one(x_pipe_3)
        x_to_encode = x_pipe_2 | x_pipe_3
        encoders["ordinal_encoder"].learn_one(x_to_encode)
        x2 = encoders["ordinal_encoder"].transform_one(x_to_encode)
        return x1 | x2, {
            "ordinal_encoder": encoders["ordinal_encoder"]
        }
    elif project_name == "E-Commerce Customer Interactions":
        pipe1 = compose.Select(
            'price',
            'quantity',
            'session_event_sequence',
            'time_on_page_seconds'
        )
        pipe1.learn_one(x)
        x1 = pipe1.transform_one(x)
        pipe2 = compose.Select(
            'event_type',
            'product_category',
            'product_id',
            'referrer_url',
        )
        pipe2.learn_one(x)
        x_pipe_2 = pipe2.transform_one(x)
        pipe3a = compose.Select(
            "device_info"
        )
        pipe3a.learn_one(x)
        x_pipe_3 = pipe3a.transform_one(x)
        pipe3b = compose.FuncTransformer(
            extract_device_info,
        )
        pipe3b.learn_one(x_pipe_3)
        x_pipe_3 = pipe3b.transform_one(x_pipe_3)
        pipe4a = compose.Select(
            "timestamp",
        )
        pipe4a.learn_one(x)
        x_pipe_4 = pipe4a.transform_one(x)
        pipe4b = compose.FuncTransformer(
            extract_timestamp_info,
        )
        pipe4b.learn_one(x_pipe_4)
        x_pipe_4 = pipe4b.transform_one(x_pipe_4)
        pipe5a = compose.Select(
            "location",
        )
        pipe5a.learn_one(x)
        x_pipe_5 = pipe5a.transform_one(x)
        pipe5b = compose.FuncTransformer(
            extract_coordinates,
        )
        pipe5b.learn_one(x_pipe_5)
        x_pipe_5 = pipe5b.transform_one(x_pipe_5)
        x_to_prep = x1 | x_pipe_2 | x_pipe_3 | x_pipe_4 | x_pipe_5
        x_to_prep = DictImputer(
            fill_value = False, 
            on = list(x_to_prep.keys())).transform_one(
                x_to_prep)
        numerical_features = [
            'price',
            'session_event_sequence',
            'time_on_page_seconds',
            'quantity'
        ]
        categorical_features = [
            'event_type',
            'product_category',
            'product_id',
            'referrer_url',
            'os',
            'browser',
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second'
        ]
        num_pipe = compose.Select(*numerical_features)
        num_pipe.learn_one(x_to_prep)
        x_num = num_pipe.transform_one(x_to_prep)
        cat_pipe = compose.Select(*categorical_features)
        cat_pipe.learn_one(x_to_prep)
        x_cat = cat_pipe.transform_one(x_to_prep)
        encoders["standard_scaler"].learn_one(x_num)
        x_scaled = encoders["standard_scaler"].transform_one(x_num)
        encoders["feature_hasher"].learn_one(x_cat)
        x_hashed = encoders["feature_hasher"].transform_one(x_cat)
        return x_scaled | x_hashed, {
            "standard_scaler": encoders["standard_scaler"], 
            "feature_hasher": encoders["feature_hasher"]
        }
    elif project_name == "Sales Forecasting":
        pipe1 = compose.Select(
            'concept_drift_stage',
            'day_of_week',
            'is_holiday',
            'is_promotion_active',
            'month',
            #'total_sales_amount',
            'unit_price'
        )
        pipe1.learn_one(x)
        x1 = pipe1.transform_one(x)
        pipe2a = compose.Select(
            "timestamp",
        )
        pipe2a.learn_one(x)
        x_pipe_2 = pipe2a.transform_one(x)
        pipe2b = compose.FuncTransformer(
            extract_timestamp_info,
        )
        pipe2b.learn_one(x_pipe_2)
        x2 = pipe2b.transform_one(x_pipe_2)
        pipe3a = compose.Select(
            'product_id',
            'promotion_id',
            'store_id'
        )
        pipe3a.learn_one(x)
        x3 = pipe3a.transform_one(x)
        x_to_process = x1 | x2 | x3
        numerical_features = [
            'unit_price',
            #'total_sales_amount',
        ]
        categorical_features = [
            'is_promotion_active',
            'is_holiday',
            'day_of_week',
            'concept_drift_stage',
            'year',
            'month',
            'day',
            #'hour',
            #'minute',
            #'second',
            'product_id',
            'promotion_id',
            'store_id',
        ]
        pipe_num = compose.Select(*numerical_features)
        pipe_num.learn_one(x_to_process)
        x_num = pipe_num.transform_one(x_to_process)
        pipe_cat = compose.Select(*categorical_features)
        pipe_cat.learn_one(x_to_process)
        x_cat = pipe_cat.transform_one(x_to_process)
        encoders["standard_scaler"].learn_one(x_num)
        x_num = encoders["standard_scaler"].transform_one(x_num)
        encoders["one_hot_encoder"].learn_one(x_cat)
        x_cat = encoders["one_hot_encoder"].transform_one(x_cat)
        return x_num | x_cat, {
            "one_hot_encoder": encoders["one_hot_encoder"],
            "standard_scaler": encoders["standard_scaler"],
        }


def load_or_create_model(project_name, model_name, folder_path = None):
    """Load existing model or create a new one"""
    try:
        model_files = os.listdir(folder_path)
        model_files = [
            os.path.join(folder_path, entry) 
            for entry 
            in model_files 
            if os.path.isfile(os.path.join(folder_path, entry)) and entry.endswith(".pkl")]
        MODEL_PATH = [x for x in model_files if model_name in x][0]
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded from disk.")
    except:
        if project_name == "Transaction Fraud Detection":
            model = forest.ARFClassifier(
                n_models = 10,                  # More models = better accuracy but higher latency
                drift_detector = drift.ADWIN(),  # Auto-detects concept drift
                warning_detector = drift.ADWIN(),
                metric = metrics.ROCAUC(),       # Optimizes for imbalanced data
                max_features = "sqrt",           # Better for high-dimensional data
                lambda_value = 6,               # Controls tree depth (higher = more complex)
                seed = 42
            )
        elif project_name == "Estimated Time of Arrival":
            model = forest.ARFRegressor(
                n_models = 10,                  # More models = better accuracy but higher latency
                drift_detector = drift.ADWIN(),  # Auto-detects concept drift
                warning_detector = drift.ADWIN(),
                metric = metrics.RMSE(),       # Optimizes for imbalanced data
                max_features = "sqrt",           # Better for high-dimensional data
                lambda_value = 6,               # Controls tree depth (higher = more complex)
                seed = 42
            )
        elif project_name == "E-Commerce Customer Interactions":
            model = cluster.DBSTREAM(
                clustering_threshold = 1.0,
                fading_factor = 0.01,
                cleanup_interval = 2,
            )
        elif project_name == "Sales Forecasting":
            regressor_snarimax = linear_model.PARegressor(
                    C = 0.01, 
                    mode = 1)
            model = time_series.SNARIMAX(
                p = 2,          # Start with a slightly lower non-seasonal AR
                d = 1,          # For trend
                q = 1,          # Start with a slightly lower non-seasonal MA
                m = 7,          # Weekly seasonality
                sp = 1,         # Seasonal AR
                sd = 0,         # No seasonal differencing initially
                sq = 1,         # Seasonal MA
                regressor = regressor_snarimax # The pipeline defined above
            )
        print(f"Creating model: {project_name}", file = sys.stderr)
    return model


def create_consumer(project_name):
    """Create and return Kafka consumer"""
    consumer_name_dict = {
        "Transaction Fraud Detection": "transaction_fraud_detection",
        "Estimated Time of Arrival": "estimated_time_of_arrival",
        "E-Commerce Customer Interactions": "e_commerce_customer_interactions",
        "Sales Forecasting": "sales_forecasting"
    }
    KAFKA_TOPIC = consumer_name_dict[project_name]
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers = KAFKA_BROKERS,
        auto_offset_reset = 'earliest',
        value_deserializer = lambda v: json.loads(v.decode('utf-8')),
        group_id = f'{KAFKA_TOPIC}_group'
    )


def load_or_create_data(consumer, project_name):
    """Load existing model or create a new one"""
    data_name_dict = {
        "Transaction Fraud Detection": "transaction_fraud_detection.parquet",
        "Estimated Time of Arrival": "estimated_time_of_arrival.parquet",
        "E-Commerce Customer Interactions": "e_commerce_customer_interactions.parquet",
        #"Sales Forecasting": "sales_forecasting.parquet"
    }
    DATA_PATH = f"data/{data_name_dict[project_name]}"
    try:
        data_df = pd.read_parquet(DATA_PATH)
        print("Data loaded from disk.")
    except:
        for message in consumer:
            transaction = message.value
            break
        data_df = pd.DataFrame([transaction])
        print(f"Creating data: {project_name}", file = sys.stderr)  
    return data_df


def process_batch_data(data, project_name):
    data = data.copy()
    os.makedirs("encoders/sklearn", exist_ok = True)
    filename = "encoders/sklearn/" + project_name.lower().replace(' ', '_').replace("-", "_") + ".pkl"
    if project_name == "Transaction Fraud Detection":
        data = data.copy()
        data = extract_device_info_sklearn(data)
        data = extract_timestamp_info_sklearn(data)
        numerical_features = [
            "amount",
            "account_age_days",
            "cvv_provided",
            "billing_address_match"
        ]
        binary_features = [
            "cvv_provided",
            "billing_address_match"
        ]
        categorical_features = [
            "currency",
            "merchant_id",
            "payment_method",
            "product_category",
            "transaction_type",
            "browser",
            "os",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
        ]
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            handle_unknown = 'ignore',
            sparse_output = False)
        preprocessor = ColumnTransformer(
            transformers = [
                ("numerical", numerical_transformer, numerical_features),
                ("binary", "passthrough", binary_features),
                ("categorical", categorical_transformer, categorical_features),
            ]
        )
        preprocessor.set_output(transform = "pandas") # Output pandas DataFrame
        X = data.drop('is_fraud', axis = 1)
        y = data['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = 0.2,       # 20% for testing
            stratify = y,           # Crucial for imbalanced data
            random_state = 42
        )
        preprocessor.fit(X_train)
        preprocessor_dict = {
            "preprocessor": preprocessor
        }
        with open(filename, 'wb') as f:
            pickle.dump(preprocessor_dict, f)
        #Don't apply preprocessor directly on X (it characterizes data leakage)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
    return X_train, X_test, y_train, y_test


def process_sklearn_sample(x, project_name):
    filename = "encoders/sklearn/" + project_name.lower().replace(' ', '_').replace("-", "_") + ".pkl"
    if project_name == "Transaction Fraud Detection":
        x = pd.DataFrame(x.copy())
        x = extract_device_info_sklearn(x)
        x = extract_timestamp_info_sklearn(x)
        with open(filename, 'rb') as f:
            preprocessor_dict = pickle.load(f)
        preprocessor = preprocessor_dict["preprocessor"]
        x = preprocessor.transform(x)
    return x


def create_batch_model(project_name, **kwargs):
    if project_name == "Transaction Fraud Detection":
        neg_samples = sum(kwargs["y_train"] == 0)
        pos_samples = sum(kwargs["y_train"] == 1)
        if pos_samples > 0:
            calculated_scale_pos_weight = neg_samples / pos_samples
        else:
            calculated_scale_pos_weight = 1 # Default or handle as error if no positive samples
        model = XGBClassifier(
            objective = 'binary:logistic',  # For binary classification
            eval_metric = 'auc',            # Primary metric to monitor (aligns with River's ROCAUC)
                                            # Consider 'aucpr' (PR AUC) as well, often better for severe imbalance.
            enable_categorical = True,
            use_label_encoder = False,      # Suppresses a warning with newer XGBoost versions
            random_state = 42,              # For reproducibility
            n_jobs = -1,                    # Use all available CPU cores
            # --- Parameters to TUNE ---
            n_estimators = 200,             # Start: 100-500. Tune with early stopping.
            learning_rate = 0.05,           # Start: 0.01, 0.05, 0.1. Smaller values need more n_estimators.
            max_depth = 5,                  # Start: 3-7. Deeper trees can overfit.
            subsample = 0.8,                # Start: 0.6-0.9.
            colsample_bytree = 0.8,         # Start: 0.6-0.9.
            gamma = 0,                      # Start: 0-0.2. Higher values make the algorithm more conservative.
            # reg_alpha=0,                # Consider tuning if many features (e.g., 0, 0.01, 0.1, 1)
            # reg_lambda=1,               # Consider tuning (e.g., 0.1, 1, 10)
            # --- CRITICAL for Imbalance ---
            scale_pos_weight = calculated_scale_pos_weight # Use your calculated value here!
        )
    return model