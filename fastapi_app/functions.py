import pickle
import os
import sys
from typing import Any, Dict, Hashable
from river import (
    compose, 
    linear_model, 
    metrics, 
    optim,
    tree,
    ensemble,
    imblearn,
    drift,
    forest
)
from kafka import KafkaConsumer
import json
import pandas as pd
import datetime as dt



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
    

def extract_device_info(x):
    x_ = x['device_info']
    return {
        'os': x_['os'],
        'browser': x_['browser'],
    }

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

def load_or_create_ordinal_encoder(ordinal_encoders_folder):
    try:
        with open(f"{ordinal_encoders_folder}/ordinal_encoder.pkl", 'rb') as f:
            ordinal_encoder = pickle.load(f)
    except FileNotFoundError as e:
        ordinal_encoder = CustomOrdinalEncoder()
        print(f"Creating ordinal encoder: {e}", file = sys.stderr)
    except Exception as e:
        ordinal_encoder = CustomOrdinalEncoder()
        print(f"Error loading ordinal encoder: {e}", file = sys.stderr)
    return ordinal_encoder


def process_sample(x, ordinal_encoder):
    pipe1 = compose.Select(
        "amount",
        "account_age_days",
        "cvv_provided",
        "billing_address_match"
    )
    pipe1.learn_one(x)
    x1 = pipe1.transform_one(x)
    pipe2a = compose.Select(
        "currency",
        "merchant_id",
        "payment_method",
        "product_category",
        "transaction_type",
        #"user_agent"
    )
    pipe2a.learn_one(x)
    x_pipe_2 = pipe2a.transform_one(x)
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
    ordinal_encoder.learn_one(x_to_encode)
    x2 = ordinal_encoder.transform_one(x_to_encode)
    return x1 | x2, ordinal_encoder


def load_or_create_model(model_type, folder_path = None, from_scratch = False):
    """Load existing model or create a new one"""
    if from_scratch == False:
        #Take the most recent file
        try:
            model_files = os.listdir(folder_path)
            model_files = [
                os.path.join(folder_path, entry) 
                for entry 
                in model_files 
                if os.path.isfile(os.path.join(folder_path, entry))]
            #Get the most recent model
            MODEL_PATH = max(model_files, key = os.path.getmtime)
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        except:
            if model_type == "LogisticRegression":
                model = linear_model.LogisticRegression(
                    loss = optim.losses.CrossEntropyLoss(
                        class_weight = {0: 1, 1: 10}),
                    optimizer = optim.SGD(0.01)
                )
            elif model_type == "ADWINBoostingClassifier":
                base_estimator = tree.HoeffdingAdaptiveTreeClassifier(
                    splitter = tree.splitter.HistogramSplitter(),
                    drift_detector = drift.ADWIN(),
                    max_depth = 20,
                    nominal_attributes = [
                        "currency",
                        "merchant_id",
                        "payment_method",
                        "product_category",
                        "transaction_type",
                        #"user_agent",
                        "device_info_os",
                        "device_info_browser"
                    ],
                    leaf_prediction = 'mc',#'nba',
                    grace_period = 200,
                    delta = 1e-7
                )
                boosting_classifier = ensemble.ADWINBoostingClassifier(
                    model = base_estimator,
                    n_models = 15,
                )
                model = imblearn.RandomOverSampler(
                    classifier = boosting_classifier,
                    desired_dist = {1: 0.5, 0: 0.5},
                    seed = 42
                )
            elif model_type == "AdaptiveRandomForestClassifier":
                model = forest.ARFClassifier(
                    n_models = 10,                  # More models = better accuracy but higher latency
                    drift_detector = drift.ADWIN(),  # Auto-detects concept drift
                    warning_detector = drift.ADWIN(),
                    metric = metrics.ROCAUC(),       # Optimizes for imbalanced data
                    max_features = "sqrt",           # Better for high-dimensional data
                    lambda_value = 6,               # Controls tree depth (higher = more complex)
                    seed = 42
                )
    else:
        if model_type == "LogisticRegression":
            model = linear_model.LogisticRegression(
                loss = optim.losses.CrossEntropyLoss(
                    class_weight = {0: 1, 1: 10}),
                optimizer = optim.SGD(0.01)
            )
        elif model_type == "ADWINBoostingClassifier":
            base_estimator = tree.HoeffdingAdaptiveTreeClassifier(
                splitter = tree.splitter.HistogramSplitter(),
                drift_detector = drift.ADWIN(),
                max_depth = 20,
                nominal_attributes = [
                    "currency",
                    "merchant_id",
                    "payment_method",
                    "product_category",
                    "transaction_type",
                    #"user_agent",
                    "os",
                    "browser"
                ],
                leaf_prediction = 'mc',#'nba',
                grace_period = 200,
                delta = 1e-7
            )
            boosting_classifier = ensemble.ADWINBoostingClassifier(
                model = base_estimator,
                n_models = 15,
            )
            model = imblearn.RandomOverSampler(
                classifier = boosting_classifier,
                desired_dist = {1: 0.5, 0: 0.5},
                seed = 42
            )
        elif model_type == "AdaptiveRandomForestClassifier":
            model = forest.ARFClassifier(
                n_models = 10,                  # More models = better accuracy but higher latency
                drift_detector = drift.ADWIN(),  # Auto-detects concept drift
                warning_detector = drift.ADWIN(),
                metric = metrics.ROCAUC(),       # Optimizes for imbalanced data
                max_features = "sqrt",           # Better for high-dimensional data
                lambda_value = 6,               # Controls tree depth (higher = more complex)
                seed = 42
            )
    return model


def create_consumer(project_name):
    """Create and return Kafka consumer"""
    if project_name == "Transaction Fraud Detection":
        KAFKA_TOPIC = "transaction_fraud_detection"
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers = KAFKA_BROKERS,
        auto_offset_reset = 'earliest',
        value_deserializer = lambda v: json.loads(v.decode('utf-8')),
        group_id = f'{KAFKA_TOPIC}_group'
    )


def load_or_create_data(consumer, project_name):
    """Load existing model or create a new one"""
    if project_name == "Transaction Fraud Detection":
        DATA_PATH = "transaction_fraud_detection_data.parquet"
    try:
        data_df = pd.read_parquet(DATA_PATH)
    except:
        for message in consumer:
            transaction = message.value
            break
        data_df = pd.DataFrame([transaction])
    return data_df