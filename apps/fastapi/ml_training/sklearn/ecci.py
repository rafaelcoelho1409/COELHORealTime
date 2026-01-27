"""
E-Commerce Customer Interactions - Batch ML Training Script (Clustering)

Trains a KMeans model for event clustering using batch learning.
Reads data from Delta Lake on MinIO via DuckDB, trains the model,
logs to MLflow, and saves artifacts.

Uses event-level approach with DENSE_RANK encoding (like TFD/ETA).
Each row is an individual event, not aggregated by customer.
Clustering requires StandardScaler for proper distance calculations.

Usage:
    python e_commerce_customer_interactions_sklearn.py
    python e_commerce_customer_interactions_sklearn.py --sample-frac 0.3
    python e_commerce_customer_interactions_sklearn.py --max-rows 100000
    python e_commerce_customer_interactions_sklearn.py --n-clusters 5

Environment variables:
    MLFLOW_HOST: MLflow server hostname (required)
"""
import pickle
import os
import signal
import sys
import time
import tempfile
import click
import mlflow
import requests
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import duckdb
from utils.batch import (
    load_ecci_event_data_duckdb,
    get_ecci_label_encodings,
    ECCI_ALL_FEATURES,
    DELTA_PATHS,
)


MLFLOW_HOST = os.environ["MLFLOW_HOST"]
FASTAPI_STATUS_URL = "http://localhost:8000/api/v1/batch/training-status"

# =============================================================================
# GRACEFUL SHUTDOWN (Signal Handling - matches River pattern)
# =============================================================================
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


def check_shutdown(stage: str = "unknown"):
    """Check if shutdown was requested and exit gracefully if so."""
    if _shutdown_requested:
        print(f"Shutdown requested during {stage}, exiting gracefully...")
        update_status(f"Training stopped during {stage}", progress=0, stage="stopped")
        sys.exit(0)


def update_status(message: str, progress: int = None, stage: str = None, metrics: dict = None, total_rows: int = None, kmeans_log: dict = None):
    """Post training status update to sklearn service."""
    try:
        payload = {"message": message}
        if progress is not None:
            payload["progress"] = progress
        if stage is not None:
            payload["stage"] = stage
        if metrics is not None:
            payload["metrics"] = metrics
        if total_rows is not None:
            payload["total_rows"] = total_rows
        if kmeans_log is not None:
            payload["kmeans_log"] = kmeans_log
        requests.post(FASTAPI_STATUS_URL, json=payload, timeout=2)
    except Exception:
        pass  # Don't fail training if status update fails


PROJECT_NAME = "E-Commerce Customer Interactions"
MODEL_NAME = "KMeans"
ENCODER_ARTIFACT_NAME = "sklearn_encoders.pkl"


def find_optimal_k(X: np.ndarray, k_range: tuple = (2, 10), total_rows: int = 0) -> dict:
    """Find optimal K using multiple metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)."""
    results = {'k': [], 'inertia': [], 'silhouette': [], 'calinski': [], 'davies': []}
    k_min, k_max = k_range
    total_k = k_max - k_min + 1

    for i, k in enumerate(range(k_min, k_max + 1)):
        # Check for shutdown before each K iteration
        check_shutdown(f"k_search_k{k}")

        model = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        labels = model.fit_predict(X)
        results['k'].append(k)
        results['inertia'].append(model.inertia_)
        results['silhouette'].append(metrics.silhouette_score(X, labels))
        results['calinski'].append(metrics.calinski_harabasz_score(X, labels))
        results['davies'].append(metrics.davies_bouldin_score(X, labels))

        # Log to console
        print(f"K={k}: Silhouette={results['silhouette'][-1]:.3f}, "
              f"CH={results['calinski'][-1]:.0f}, DB={results['davies'][-1]:.3f}")

        # Calculate progress within K-search (30-60% of total training)
        k_progress = 30 + int((i + 1) / total_k * 30)

        # Send K-search log update to FastAPI
        kmeans_log = {
            "current_k": k,
            "k_range": f"{k_min}-{k_max}",
            "silhouette": round(results['silhouette'][-1], 3),
            "calinski_harabasz": int(results['calinski'][-1]),
            "davies_bouldin": round(results['davies'][-1], 3),
            "best_k_so_far": results['k'][np.argmax(results['silhouette'])],
            "best_silhouette_so_far": round(max(results['silhouette']), 3),
            "progress": f"{i + 1}/{total_k}",
        }
        update_status(
            f"K-search: K={k}, Silhouette={results['silhouette'][-1]:.3f}",
            progress=k_progress,
            stage="k_search",
            total_rows=total_rows,
            kmeans_log=kmeans_log,
        )

    return results


def select_optimal_k(results: dict) -> int:
    """Select optimal K based on Silhouette score (higher is better)."""
    idx = np.argmax(results['silhouette'])
    return results['k'][idx]


@click.command()
@click.option(
    "--sample-frac",
    type=float,
    default=None,
    help="Fraction of data to sample (0.0-1.0). E.g., 0.3 = 30% of data.",
)
@click.option(
    "--max-rows",
    type=int,
    default=None,
    help="Maximum number of event rows to load from Delta Lake.",
)
@click.option(
    "--n-clusters",
    type=int,
    default=None,
    help="Number of clusters. If not specified, uses silhouette optimization to find optimal K.",
)
@click.option(
    "--k-range-min",
    type=int,
    default=2,
    help="Minimum K for optimization range (default: 2).",
)
@click.option(
    "--k-range-max",
    type=int,
    default=10,
    help="Maximum K for optimization range (default: 10).",
)
def main(sample_frac: float | None, max_rows: int | None, n_clusters: int | None,
         k_range_min: int, k_range_max: int):
    """Batch ML Training for E-Commerce Customer Interactions (Clustering).

    Trains KMeans on Delta Lake data and logs to MLflow.
    Uses event-level approach with DENSE_RANK encoding (like TFD/ETA).
    Clustering requires StandardScaler for proper distance calculations.
    """
    start_time = time.time()
    print(f"Starting batch ML training for {PROJECT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Preprocessing: DuckDB SQL (event-level, DENSE_RANK encoding)")
    print(f"MLflow host: {MLFLOW_HOST}")
    if sample_frac:
        print(f"Data sampling: {sample_frac * 100}%")
    if max_rows:
        print(f"Max rows: {max_rows}")
    if n_clusters:
        print(f"Number of clusters: {n_clusters}")
    else:
        print(f"K optimization range: [{k_range_min}, {k_range_max}]")
    update_status("Initializing training...", progress=5, stage="init")
    try:
        # Configure MLflow
        print(f"Connecting to MLflow at http://{MLFLOW_HOST}:5000")
        mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")
        mlflow.set_experiment(PROJECT_NAME)
        print(f"MLflow experiment '{PROJECT_NAME}' set successfully")
        update_status("Loading data from Delta Lake...", progress=10, stage="loading_data")
        # Load and process data using DuckDB SQL (event-level with DENSE_RANK encoding)
        load_start = time.time()
        print("\n=== Loading and Preprocessing Data ===")
        print("Event-level data with DENSE_RANK encoding (like TFD/ETA)...")
        df, feature_names, search_queries = load_ecci_event_data_duckdb(
            sample_frac=sample_frac,
            max_rows=max_rows,
            include_search_queries=True,
        )
        load_time = time.time() - load_start
        print(f"\nData loading/preprocessing completed in {load_time:.2f} seconds")
        print(f"Events: {len(df)}")
        print(f"Features: {len(feature_names)}")
        total_rows = len(df)
        update_status(
            f"Data loaded: {len(df):,} events, {len(feature_names)} features",
            progress=25,
            stage="data_loaded",
            total_rows=total_rows
        )
        # Check for shutdown after data loading
        check_shutdown("data_loading")
        # Prepare features for clustering
        print("\n=== Preparing Features ===")
        X = df[feature_names].copy()
        # Handle infinite/NaN values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Scale features (required for distance-based clustering)
        print("Scaling features with StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Data shape after scaling: {X_scaled.shape}")
        # Check for shutdown after preprocessing
        check_shutdown("preprocessing")
        update_status("Finding optimal K or training model...", progress=30, stage="training", total_rows=total_rows)
        # Find optimal K or use provided value
        if n_clusters is None:
            print(f"\n=== Finding Optimal K (range: {k_range_min}-{k_range_max}) ===")
            k_results = find_optimal_k(X_scaled, k_range=(k_range_min, k_range_max), total_rows=total_rows)
            optimal_k = select_optimal_k(k_results)
            print(f"\nOptimal K based on Silhouette score: {optimal_k}")
        else:
            optimal_k = n_clusters
            k_results = None
            print(f"\nUsing specified K: {optimal_k}")
        # Check for shutdown after K optimization
        check_shutdown("k_optimization")
        # Train final model
        print(f"\n=== Training Final Model (K={optimal_k}) ===")
        model = KMeans(
            n_clusters=optimal_k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-4,
            random_state=42,
            algorithm='lloyd',
        )
        cluster_labels = model.fit_predict(X_scaled)
        print(f"Model training complete. Inertia: {model.inertia_:.2f}")
        # Check for shutdown after training
        check_shutdown("training")
        update_status("Evaluating model performance...", progress=70, stage="evaluating", total_rows=total_rows)
        # Evaluate model
        print("\n=== Evaluating Model ===")

        # =============================================================================
        # METRICS - SKLEARN CLUSTERING
        # =============================================================================
        # -----------------------------------------------------------------------------
        # PRIMARY METRICS - Core clustering metrics for customer segmentation
        # These measure cluster quality (cohesion, separation, validity)
        # -----------------------------------------------------------------------------
        clustering_metric_functions = {
            # Silhouette: [-1, 1], higher is better (measures cluster cohesion & separation)
            "silhouette_score": metrics.silhouette_score,
            # Calinski-Harabasz: [0, inf), higher is better (ratio of between/within variance)
            "calinski_harabasz_score": metrics.calinski_harabasz_score,
            # Davies-Bouldin: [0, inf), lower is better (avg similarity of each cluster with its most similar)
            "davies_bouldin_score": metrics.davies_bouldin_score,
        }
        clustering_metric_args = {
            "silhouette_score": {"metric": "euclidean"},
            "calinski_harabasz_score": {},
            "davies_bouldin_score": {},
        }
        # COMPUTE ALL METRICS
        metrics_to_log = {}
        for name, func in clustering_metric_functions.items():
            metrics_to_log[name] = float(func(X_scaled, cluster_labels, **clustering_metric_args[name]))
        # Model-specific metrics
        metrics_to_log["inertia"] = float(model.inertia_)
        metrics_to_log["n_clusters"] = optimal_k
        print("\nMetrics:")
        for name, value in metrics_to_log.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
        # Check for shutdown after evaluation (before MLflow logging)
        check_shutdown("evaluation")
        # Send metrics preview to status
        update_status(
            "Logging to MLflow...",
            progress=85,
            stage="logging_mlflow",
            metrics={
                "silhouette": metrics_to_log["silhouette_score"],
                "calinski_harabasz": metrics_to_log["calinski_harabasz_score"],
                "davies_bouldin": metrics_to_log["davies_bouldin_score"],
                "inertia": metrics_to_log["inertia"],
            },
            total_rows=total_rows
        )
        # Log to MLflow
        print("\nLogging to MLflow...")
        with mlflow.start_run(run_name=MODEL_NAME):
            # Log tags
            mlflow.set_tag("training_mode", "batch")
            mlflow.set_tag("preprocessing", "DuckDB SQL")
            mlflow.set_tag("task_type", "clustering")
            mlflow.set_tag("model_type", "KMeans")
            # Log data parameters
            mlflow.log_param("n_events", len(df))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("preprocessing_method", "DuckDB SQL (event-level, DENSE_RANK)")
            mlflow.log_param("scaler", "StandardScaler")
            if sample_frac:
                mlflow.log_param("sample_frac", sample_frac)
            if max_rows:
                mlflow.log_param("max_rows", max_rows)
            # Log model parameters
            mlflow.log_param("n_clusters", optimal_k)
            mlflow.log_param("init", "k-means++")
            mlflow.log_param("n_init", 10)
            mlflow.log_param("max_iter", 300)
            mlflow.log_param("algorithm", "lloyd")
            mlflow.log_param("random_state", 42)
            if k_results is not None:
                mlflow.log_param("k_optimization_range", f"{k_range_min}-{k_range_max}")
            # Log timing metrics
            mlflow.log_metric("preprocessing_time_seconds", load_time)
            # Log clustering metrics
            for metric_name, metric_value in metrics_to_log.items():
                mlflow.log_metric(metric_name, metric_value)
            # Log model, scaler, and data as artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save model
                model_path = os.path.join(tmpdir, f"{MODEL_NAME}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_path)
                # Save scaler and label encodings (needed for inference)
                print("Getting label encodings for categorical features...")
                label_encodings = get_ecci_label_encodings()
                encoder_path = os.path.join(tmpdir, ENCODER_ARTIFACT_NAME)
                preprocessor_dict = {
                    "scaler": scaler,
                    "feature_names": feature_names,
                    "label_encodings": label_encodings,  # value -> int mappings
                }
                with open(encoder_path, 'wb') as f:
                    pickle.dump(preprocessor_dict, f)
                mlflow.log_artifact(encoder_path)
                print(f"  Saved label encodings for {len(label_encodings)} categorical features")
                # Save training data for YellowBrick visualization reproducibility
                # Uses snappy compression (fast decompression, industry default)
                print("Saving training data artifacts...")
                # Save event features (X) with cluster_label
                df_with_labels = df.copy()
                df_with_labels['cluster_label'] = cluster_labels
                df_with_labels.to_parquet(os.path.join(tmpdir, "X_events.parquet"), compression="snappy")
                # Save scaled features
                X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
                X_scaled_df['cluster_label'] = cluster_labels
                X_scaled_df.to_parquet(os.path.join(tmpdir, "X_scaled.parquet"), compression="snappy")
                mlflow.log_artifact(os.path.join(tmpdir, "X_events.parquet"), artifact_path="training_data")
                mlflow.log_artifact(os.path.join(tmpdir, "X_scaled.parquet"), artifact_path="training_data")
                print(f"  Saved: X_events={df_with_labels.shape}, X_scaled={X_scaled_df.shape}")
                # Save cluster centers
                centers_df = pd.DataFrame(model.cluster_centers_, columns=feature_names)
                centers_df.index.name = 'cluster_id'
                centers_df.to_parquet(os.path.join(tmpdir, "cluster_centers.parquet"), compression="snappy")
                mlflow.log_artifact(os.path.join(tmpdir, "cluster_centers.parquet"), artifact_path="training_data")
                print(f"  Saved: cluster_centers={centers_df.shape}")
                # Save search queries for YellowBrick text analysis
                # Uses pre-loaded search_queries DataFrame from initial data load (full dataset, no limits)
                print("Saving search queries for text analysis...")
                if search_queries is not None and len(search_queries) > 0:
                    # Save as parquet for efficient storage and DuckDB querying
                    search_queries_path = os.path.join(tmpdir, "search_queries.parquet")
                    search_queries.to_parquet(search_queries_path, compression="snappy", index=False)
                    mlflow.log_artifact(search_queries_path, artifact_path="training_data")
                    print(f"  Saved: search_queries={len(search_queries)} unique queries (full dataset, parquet)")
                else:
                    print("  Warning: No search queries available for text analysis")
            # Log model using MLflow's sklearn flavor
            mlflow.sklearn.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run ID: {run_id}")
        training_time = time.time() - start_time
        print(f"\nTraining completed successfully in {training_time:.2f} seconds")
        update_status(
            f"Training complete! Time: {training_time:.1f}s",
            progress=100,
            stage="complete",
            metrics={
                "silhouette": metrics_to_log["silhouette_score"],
                "calinski_harabasz": metrics_to_log["calinski_harabasz_score"],
                "davies_bouldin": metrics_to_log["davies_bouldin_score"],
                "inertia": metrics_to_log["inertia"],
                "n_clusters": optimal_k,
            },
            total_rows=total_rows
        )
    except Exception as e:
        update_status(f"Training failed: {str(e)}", progress=0, stage="error")
        print(f"Error during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
