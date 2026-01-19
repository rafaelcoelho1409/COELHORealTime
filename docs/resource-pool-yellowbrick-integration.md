# MLflow Training Data + YellowBrick Integration

## Overview

YellowBrick visualizations use the EXACT same training data stored in MLflow artifacts,
guaranteeing 100% reproducibility. Training data is saved alongside the model in each
MLflow run.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Training Flow                                      │
│                                                                              │
│  Delta Lake → DuckDB SQL (DENSE_RANK encoding) → All-numeric DataFrame       │
│                                      ↓                                       │
│                          Train CatBoost Model                                │
│                                      ↓                                       │
│                          MLflow.log_model() + log_artifacts():               │
│                            - model/                                          │
│                            - training_data/X_train.parquet                   │
│                            - training_data/X_test.parquet                    │
│                            - training_data/y_train.parquet                   │
│                            - training_data/y_test.parquet                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         YellowBrick Flow                                     │
│                                                                              │
│  User requests visualization → /yellowbrick_metric endpoint                  │
│                                      ↓                                       │
│                          Check Redis cache (HIT → return PNG)                │
│                                      ↓ (MISS)                                │
│                          get_best_mlflow_run() → best run_id                 │
│                          load_training_data_from_mlflow() → X, y             │
│                                      ↓                                       │
│                          Generate visualization → Cache in Redis → Return    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Benefits

| Aspect | Description |
|--------|-------------|
| **100% Reproducibility** | Visualizations use EXACT same data as best model |
| **No Data Mismatch** | Data tied to specific MLflow run, not random reload |
| **Persistent** | Training data survives pod restarts |
| **Consistent with Predictions** | Same best model selection logic |

## Components

### 1. Training Data Artifacts (MLflow)
- **Storage**: MinIO (S3-compatible) via MLflow artifact store
- **Format**: Parquet with snappy compression (~2-3 MB per run)
- **Contents**: X_train, X_test, y_train, y_test
- **Selection**: Best run by fbeta_score (fraud detection)

### 2. Redis Visualization Cache
- **Purpose**: Cache generated PNG visualizations
- **TTL**: 1 hour (configurable)
- **Invalidation**: On new training completion

### 3. YellowBrick Visualizers
- **Feature Analysis**: Rank1D, Rank2D, PCA, Manifold, ParallelCoordinates, RadViz, JointPlot
- **Classification**: ConfusionMatrix, ROCAUC, PrecisionRecallCurve, etc.
- **Target**: ClassBalance, BalancedBinningReference
- **Model Selection**: LearningCurve, ValidationCurve

## Data Flow Details

### DuckDB SQL Preprocessing
All categorical features are label-encoded using `DENSE_RANK() - 1`:
```sql
SELECT
    amount, account_age_days, ...  -- Numerical (unchanged)
    DENSE_RANK() OVER (ORDER BY currency) - 1 AS currency,  -- Categorical (encoded)
    ...
FROM delta_scan('s3://lakehouse/delta/transaction_fraud_detection')
```

This produces all-numeric data compatible with:
- CatBoost (via `cat_features` indices)
- YellowBrick (requires numeric data)
- All sklearn tools

### MLflow Training Data Loading
```python
def load_training_data_from_mlflow(project_name: str):
    """Load training data from best MLflow run's artifacts."""
    run_id = get_best_mlflow_run(project_name, model_name)

    X_train = pd.read_parquet(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="training_data/X_train.parquet"
    ))
    # ... same for X_test, y_train, y_test

    return X_train, X_test, y_train, y_test, feature_names
```

## API Endpoints

### YellowBrick
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/yellowbrick_metric` | POST | Generate visualization (with Redis caching) |
| `/yellowbrick/visualizers` | GET | List available visualizers |
| `/yellowbrick/cache/status` | GET | Cache connection status |
| `/yellowbrick/cache/clear` | POST | Clear visualization cache |

## Error Handling

| Scenario | Response |
|----------|----------|
| No training data in MLflow | 400: "No training data found. Train a model first." |
| Invalid visualizer name | 400: "Unknown metric type/name" |
| Visualization timeout | 504: "Visualization generation timed out" |

## Storage Estimates

| Data | Size (per run) |
|------|----------------|
| X_train (80K × 17) | ~1-2 MB |
| X_test (20K × 17) | ~300-500 KB |
| y_train, y_test | ~50-100 KB |
| **Total** | **~2-3 MB** |

Minimal compared to CatBoost models (~50-100 MB).

## MLflow Experiment Cleanup

To clear all experiments for a fresh start:

```bash
# Port-forward MLflow if needed
kubectl port-forward svc/coelho-realtime-mlflow 5000:5000

# Option 1: Delete via MLflow CLI
mlflow experiments delete --experiment-id <id>

# Option 2: Delete via MinIO (artifacts)
mc rm --recursive minio/mlflow/

# Option 3: Delete MLflow database (PostgreSQL)
kubectl exec -it <postgres-pod> -- psql -U postgres -d mlflow -c "TRUNCATE experiments, runs CASCADE;"
```

## Migration from ResourcePool

ResourcePool has been removed. The new architecture:
- ❌ ~~ResourcePool~~ (removed)
- ✅ MLflow training data artifacts
- ✅ Redis visualization cache (unchanged)
