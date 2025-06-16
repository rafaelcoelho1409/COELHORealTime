import os
import mlflow
import pandas as pd
import pickle
from sklearn import metrics
from yellowbrick import classifier
import matplotlib.pyplot as plt
from functions import (
    create_consumer,
    load_or_create_data,
    process_batch_data,
    create_batch_model,
    yellowbrick_classification_kwargs,
    yellowbrick_classification_visualizers,
    yellowbrick_feature_analysis_kwargs,
    yellowbrick_feature_analysis_visualizers,
    yellowbrick_target_kwargs,
    yellowbrick_target_visualizers
)

DATA_PATH = "data/transaction_fraud_detection.parquet"
MODEL_FOLDER = "models/transaction_fraud_detection"
ENCODERS_PATH = "encoders/sklearn/transaction_fraud_detection.pkl"
YELLOWBRICK_PATH = f"{MODEL_FOLDER}/yellowbrick"
PROJECT_NAME = "Transaction Fraud Detection"

os.makedirs(MODEL_FOLDER, exist_ok = True)
os.makedirs("encoders/sklearn", exist_ok = True)
os.makedirs("data", exist_ok = True)
for x in [
    "classification",
    "feature_analysis",
    "target"
]:
    os.makedirs(f"{YELLOWBRICK_PATH}/{x}", exist_ok = True)


def main():
    # Initialize model and metrics
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(PROJECT_NAME)
    # Create consumer
    consumer = create_consumer(PROJECT_NAME)
    print("Consumer started. Waiting for transactions...")
    data_df = load_or_create_data(
        consumer,
        PROJECT_NAME) #REPLACE FOR DELTA LAKE AND PYSPARK LATER
    X_train, X_test, y_train, y_test = process_batch_data(
        data_df, 
        PROJECT_NAME)
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    # Metrics typically using y_true, y_pred (predicted class labels)
    classification_metrics_ypred = [
        "accuracy_score", #(y_true, y_pred)
        "balanced_accuracy_score", #(y_true, y_pred)
        "precision_score", #(y_true, y_pred, pos_label=1, average='binary') # Specify for binary
        "recall_score", #(y_true, y_pred, pos_label=1, average='binary')   # Specify for binary
        "f1_score", #(y_true, y_pred, pos_label=1, average='binary')       # Specify for binary
        #"fbeta_score", #(y_true, y_pred, beta=0.5, pos_label=1, average='binary') # Example with beta
        "matthews_corrcoef", #(y_true, y_pred)
        "cohen_kappa_score", #(y_true, y_pred)
        "jaccard_score", #(y_true, y_pred, pos_label=1, average='binary')  # Specify for binary
        "hinge_loss", #(y_true, y_pred_decision_function) # Needs decision_function output
        "zero_one_loss", #(y_true, y_pred)
    ]
    classification_metrics_ypred_dict = {
        x: getattr(metrics, x) for x in classification_metrics_ypred
    }
    ## Metrics typically using y_true, y_score (predicted probabilities or decision function values)
    #classification_metrics_yscore = [
    #    "roc_auc_score", #(y_true, y_score)
    #    "average_precision_score", #(y_true, y_score) # Also known as PR AUC
    #    "log_loss", #(y_true, y_score_proba) # Needs probability estimates
    #    "brier_score_loss", #(y_true, y_score_proba) # Needs probability estimates
    #    # roc_curve(y_true, y_score) -> returns fpr, tpr, thresholds
    #    # precision_recall_curve(y_true, y_score) -> returns precision, recall, thresholds
    #    # det_curve(y_true, y_score) -> returns fpr, fnr, thresholds
    #]
    MODEL_NAME = "XGBClassifier"
    with mlflow.start_run(run_name = MODEL_NAME):
        try:
            binary_classes = list(set(y_train.unique().tolist() + y_test.unique().tolist()))
            binary_classes.sort()
            ##>>>--- Classification Visualizers ---<<<#
            #yb_classification_kwargs = yellowbrick_classification_kwargs(
            #    PROJECT_NAME,
            #    y_train,
            #    binary_classes
            #)
            #yellowbrick_classification_visualizers(
            #    yb_classification_kwargs,
            #    X_train,
            #    X_test,
            #    y_train,
            #    y_test,
            #    YELLOWBRICK_PATH
            #)
            ##>>>--- Feature Analysis ---<<<#
            #yb_feature_analysis_kwargs = yellowbrick_feature_analysis_kwargs(
            #    binary_classes
            #)
            #yellowbrick_feature_analysis_visualizers(
            #    yb_feature_analysis_kwargs,
            #    X,
            #    y,
            #    YELLOWBRICK_PATH
            #)
            ##>>>--- Target ---<<<#
            labels = list(set(y_train.unique().tolist() + y_test.unique().tolist()))
            features = X_train.columns.tolist()
            yb_target_kwargs = yellowbrick_target_kwargs(
                labels,
                features
            )
            yellowbrick_target_visualizers(
                yb_target_kwargs,
                X,
                y,
                YELLOWBRICK_PATH
            )
            #>>>--- Model Selection ---<<<#
        except Exception as e:
            print(f"Error creating visualizers: {str(e)}")
            #----------------------------------#
        try:
            model = create_batch_model(
                PROJECT_NAME,
                y_train = y_train)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            for metric in classification_metrics_ypred:
                print(metric)
                try:
                    classification_metrics_ypred_dict[metric] = classification_metrics_ypred_dict[metric](y_test, y_pred)
                    mlflow.log_metric(metric, classification_metrics_ypred_dict[metric])
                except Exception as e:
                    print(f"Error updating metric {metric}: {str(e)}")
            MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
            with open(MODEL_VERSION, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print("Stopping consumer...")
        finally:
            MODEL_VERSION = f"{MODEL_FOLDER}/{model.__class__.__name__}.pkl"
            with open(MODEL_VERSION, 'wb') as f:
                pickle.dump(model, f)
            consumer.close()
            mlflow.end_run()
            print("Consumer closed.")

if __name__ == "__main__":
    main()