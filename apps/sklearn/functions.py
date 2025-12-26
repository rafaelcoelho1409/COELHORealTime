"""
Scikit-Learn Batch ML Helper Functions

Functions for batch machine learning training and YellowBrick visualizations.
"""
import pickle
import os
import sys
import time
import io
import base64
from typing import Any, Dict
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import json
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)
from xgboost import XGBClassifier
from yellowbrick import (
    classifier,
    features,
    target,
    model_selection
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


KAFKA_HOST = os.environ.get("KAFKA_HOST", "localhost")
KAFKA_BROKERS = f'{KAFKA_HOST}:9092'


# =============================================================================
# Data Processing Functions (Sklearn)
# =============================================================================
def extract_device_info_sklearn(data):
    """Extract device info from nested dict to separate columns."""
    data = data.copy()
    data_to_join = pd.json_normalize(data["device_info"])
    data = data.drop("device_info", axis=1)
    data = data.join(data_to_join)
    return data


def extract_timestamp_info_sklearn(data):
    """Extract timestamp components to separate columns."""
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], format='ISO8601')
    data["year"] = data["timestamp"].dt.year
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["second"] = data["timestamp"].dt.second
    data = data.drop("timestamp", axis=1)
    return data


def extract_coordinates_sklearn(data):
    """Extract location coordinates from nested dict to separate columns."""
    data = data.copy()
    data_to_join = pd.json_normalize(data["location"])
    data = data.drop("location", axis=1)
    data = data.join(data_to_join)
    return data


def load_sklearn_encoders(project_name: str):
    """Load pre-trained sklearn encoders from disk."""
    encoders_folders = {
        "Transaction Fraud Detection": "encoders/sklearn/transaction_fraud_detection.pkl",
        "Estimated Time of Arrival": "encoders/sklearn/estimated_time_of_arrival.pkl",
        "E-Commerce Customer Interactions": "encoders/sklearn/e_commerce_customer_interactions.pkl",
    }
    encoder_path = encoders_folders.get(project_name)
    if not encoder_path:
        raise ValueError(f"Unknown project: {project_name}")

    try:
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        print(f"Scikit-Learn encoders loaded for {project_name}")
        return encoders
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Scikit-Learn encoders not found for project {project_name}.") from e
    except Exception as e:
        raise Exception(f"Error loading Scikit-Learn encoders for project {project_name}: {e}") from e


def create_consumer(project_name: str, max_retries: int = 5, retry_delay: float = 5.0):
    """Create Kafka consumer with manual partition assignment.

    Note: Uses manual partition assignment instead of group-based subscription
    due to Kafka 4.0 compatibility issues with kafka-python's consumer group protocol.
    """
    from kafka import TopicPartition

    consumer_name_dict = {
        "Transaction Fraud Detection": "transaction_fraud_detection",
        "Estimated Time of Arrival": "estimated_time_of_arrival",
        "E-Commerce Customer Interactions": "e_commerce_customer_interactions",
    }
    KAFKA_TOPIC = consumer_name_dict.get(project_name)
    if not KAFKA_TOPIC:
        raise ValueError(f"Unknown project: {project_name}")

    for attempt in range(max_retries):
        try:
            # Create consumer without topic subscription (manual assignment)
            consumer = KafkaConsumer(
                bootstrap_servers=KAFKA_BROKERS,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                consumer_timeout_ms=1000,
                api_version=(3, 7),  # Force API version for Kafka 4.0 compatibility
            )

            # Manually assign partition 0 of the topic
            tp = TopicPartition(KAFKA_TOPIC, 0)
            consumer.assign([tp])

            # Seek to beginning to read all messages
            consumer.seek_to_beginning(tp)

            print(f"Kafka consumer created for {project_name} (manual assignment)")
            return consumer
        except NoBrokersAvailable as e:
            if attempt < max_retries - 1:
                print(f"Kafka not available for {project_name}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect to Kafka for {project_name} after {max_retries} attempts.")
                return None
        except Exception as e:
            print(f"Error creating Kafka consumer for {project_name}: {e}")
            return None
    return None


def load_or_create_data(consumer, project_name: str) -> pd.DataFrame:
    """Load data from disk or Kafka."""
    data_name_dict = {
        "Transaction Fraud Detection": "transaction_fraud_detection.parquet",
        "Estimated Time of Arrival": "estimated_time_of_arrival.parquet",
        "E-Commerce Customer Interactions": "e_commerce_customer_interactions.parquet",
    }
    DATA_PATH = f"data/{data_name_dict.get(project_name, '')}"

    try:
        data_df = pd.read_parquet(DATA_PATH)
        print(f"Data loaded from disk for {project_name}")
        return data_df
    except FileNotFoundError:
        print(f"Data file not found at {DATA_PATH}, attempting Kafka...")
    except Exception as e:
        print(f"Error loading data from disk: {e}")

    if consumer is not None:
        try:
            transaction = None
            for message in consumer:
                transaction = message.value
                break
            if transaction is not None:
                data_df = pd.DataFrame([transaction])
                print(f"Created data from Kafka for {project_name}")
                return data_df
        except Exception as e:
            print(f"Error loading data from Kafka: {e}")

    print(f"Warning: No data available for {project_name}")
    return pd.DataFrame()


def process_batch_data(data: pd.DataFrame, project_name: str):
    """Process batch data and fit/save sklearn encoders."""
    data = data.copy()
    os.makedirs("encoders/sklearn", exist_ok=True)
    filename = "encoders/sklearn/" + project_name.lower().replace(' ', '_').replace("-", "_") + ".pkl"

    if project_name == "Transaction Fraud Detection":
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
            handle_unknown='ignore',
            sparse_output=False)
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_transformer, numerical_features),
                ("binary", "passthrough", binary_features),
                ("categorical", categorical_transformer, categorical_features),
            ]
        )
        preprocessor.set_output(transform="pandas")
        X = data.drop('is_fraud', axis=1)
        y = data['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        preprocessor.fit(X_train)
        preprocessor_dict = {"preprocessor": preprocessor}
        with open(filename, 'wb') as f:
            pickle.dump(preprocessor_dict, f)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test, y_train, y_test
    else:
        raise ValueError(f"Unsupported project for batch processing: {project_name}")


def process_sklearn_sample(x: dict, project_name: str) -> pd.DataFrame:
    """Process a single sample for sklearn prediction."""
    filename = "encoders/sklearn/" + project_name.lower().replace(' ', '_').replace("-", "_") + ".pkl"
    if project_name == "Transaction Fraud Detection":
        x = pd.DataFrame([x])
        x = extract_device_info_sklearn(x)
        x = extract_timestamp_info_sklearn(x)
        with open(filename, 'rb') as f:
            preprocessor_dict = pickle.load(f)
        preprocessor = preprocessor_dict["preprocessor"]
        x = preprocessor.transform(x)
        return x
    else:
        raise ValueError(f"Unsupported project for sample processing: {project_name}")


# =============================================================================
# Model Creation
# =============================================================================
def create_batch_model(project_name: str, **kwargs):
    """Create batch ML model (XGBClassifier) for the given project."""
    if project_name == "Transaction Fraud Detection":
        y_train = kwargs.get("y_train")
        neg_samples = sum(y_train == 0) if y_train is not None else 1
        pos_samples = sum(y_train == 1) if y_train is not None else 1
        calculated_scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            enable_categorical=True,
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0,
            scale_pos_weight=calculated_scale_pos_weight
        )
        return model
    raise ValueError(f"Unknown project: {project_name}")


# =============================================================================
# YellowBrick Classification Visualizers
# =============================================================================
def yellowbrick_classification_kwargs(
    project_name: str,
    metric_name: str,
    y_train: pd.Series,
    binary_classes: list
) -> dict:
    """Get kwargs for YellowBrick classification visualizers."""
    kwargs = {
        "ClassificationReport": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "classes": binary_classes,
            "support": True,
        },
        "ConfusionMatrix": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "classes": binary_classes,
        },
        "ROCAUC": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "classes": binary_classes,
        },
        "PrecisionRecallCurve": {
            "estimator": create_batch_model(project_name, y_train=y_train),
        },
        "ClassPredictionError": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "classes": binary_classes,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_classification_visualizers(
    yb_kwargs: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """Create and fit YellowBrick classification visualizer."""
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(classifier, visualizer_name)(**params)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        return visualizer
    return None


# =============================================================================
# YellowBrick Feature Analysis Visualizers
# =============================================================================
def yellowbrick_feature_analysis_kwargs(
    project_name: str,
    metric_name: str,
    classes: list,
    feature_names: list = None
) -> dict:
    """Get kwargs for YellowBrick feature analysis visualizers."""
    kwargs = {
        "ParallelCoordinates": {
            "classes": classes,
            "features": feature_names,
            "sample": 0.05,
            "shuffle": True,
            "n_jobs": 1,
        },
        "PCA": {
            "classes": classes,
            "scale": True,
            "n_jobs": 1,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_feature_analysis_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create and fit YellowBrick feature analysis visualizer."""
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(features, visualizer_name)(**params)
        if visualizer_name in ["ParallelCoordinates", "PCA", "Manifold"]:
            visualizer.fit_transform(X, y)
        else:
            visualizer.fit(X, y)
            visualizer.transform(X)
        return visualizer
    return None


# =============================================================================
# YellowBrick Target Visualizers
# =============================================================================
def yellowbrick_target_kwargs(
    project_name: str,
    metric_name: str,
    labels: list = None,
    feature_names: list = None
) -> dict:
    """Get kwargs for YellowBrick target visualizers."""
    kwargs = {
        "BalancedBinningReference": {},
        "ClassBalance": {
            "labels": labels,
        },
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_target_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create and fit YellowBrick target visualizer."""
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(target, visualizer_name)(**params)
        if visualizer_name in ["BalancedBinningReference", "ClassBalance"]:
            visualizer.fit(y)
        else:
            visualizer.fit(X, y)
        return visualizer
    return None


# =============================================================================
# YellowBrick Model Selection Visualizers
# =============================================================================
def yellowbrick_model_selection_kwargs(
    project_name: str,
    metric_name: str,
    y_train: pd.Series
) -> dict:
    """Get kwargs for YellowBrick model selection visualizers."""
    kwargs = {
        "ValidationCurve": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "param_name": "gamma",
            "param_range": np.logspace(-6, -1, 10),
            "logx": True,
            "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "scoring": "average_precision",
            "n_jobs": 1,
        },
        "LearningCurve": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "scoring": "average_precision",
            "train_sizes": np.linspace(0.3, 1.0, 8),
            "n_jobs": 1,
        },
        "CVScores": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            "scoring": "average_precision",
            "n_jobs": 1,
        },
        "FeatureImportances": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "n_jobs": 1,
        },
        "DroppingCurve": {
            "estimator": create_batch_model(project_name, y_train=y_train),
            "n_jobs": 1,
        }
    }
    return {metric_name: kwargs.get(metric_name, {})}


def yellowbrick_model_selection_visualizers(
    yb_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create and fit YellowBrick model selection visualizer."""
    for visualizer_name, params in yb_kwargs.items():
        visualizer = getattr(model_selection, visualizer_name)(**params)
        if visualizer_name in ["ValidationCurve", "RFECV"]:
            X_stratified, _, y_stratified, _ = train_test_split(
                X, y,
                train_size=min(50000, len(X)),
                shuffle=True,
                stratify=y,
                random_state=42
            )
            visualizer.fit(X_stratified, y_stratified)
        else:
            visualizer.fit(X, y)
        return visualizer
    return None


# =============================================================================
# Data Manager Class
# =============================================================================
class ModelDataManager:
    """Manages loaded data for batch ML models."""
    def __init__(self):
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.project_name: str | None = None

    def load_data(self, project_name: str):
        """Load and process data for a project."""
        if self.project_name == project_name and self.y_train is not None:
            print(f"Data for {project_name} is already loaded.")
            return

        print(f"Loading data for project: {project_name}")
        consumer = create_consumer(project_name)
        data_df = load_or_create_data(consumer, project_name)
        self.X_train, self.X_test, self.y_train, self.y_test = process_batch_data(
            data_df, project_name
        )
        self.X = pd.concat([self.X_train, self.X_test])
        self.y = pd.concat([self.y_train, self.y_test])
        self.project_name = project_name
        print("Data loaded successfully.")


# =============================================================================
# YellowBrick Image Generation
# =============================================================================
def generate_yellowbrick_image(visualizer) -> str:
    """Generate base64 encoded PNG image from YellowBrick visualizer."""
    buf = io.BytesIO()
    visualizer.fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(visualizer.fig)
    return image_base64
