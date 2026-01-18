# Resource Pool + YellowBrick Integration

## Overview

This document describes the integration between ResourcePool and YellowBrick visualizations in the sklearn FastAPI service.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Training Flow                                      │
│                                                                              │
│  User clicks "Train" → process_batch_data_duckdb() → ResourcePool.store()   │
│                                      ↓                                       │
│                              Model trained → MLflow.log_model()              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YellowBrick Flow                                     │
│                                                                              │
│  User requests visualization → /yellowbrick_metric endpoint                  │
│                                      ↓                                       │
│                          ResourcePool.get() → data (X, y)                    │
│                          ModelCache.get_model() → best model from MLflow     │
│                                      ↓                                       │
│                          Generate visualization → Return PNG                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. ResourcePool (`resource_pool.py`)
- **Purpose**: Store training data from latest session
- **Stores**: X_train, X_test, y_train, y_test, feature_names, cat_feature_indices
- **Does NOT store**: Model (loaded from MLflow on-demand)
- **Pattern**: Thread-safe singleton, one session at a time

### 2. ModelCache (`app.py`)
- **Purpose**: Cache best model from MLflow
- **Selection**: Best model based on fbeta_score (for fraud detection)
- **TTL**: 5 minutes, auto-reloads when expired
- **Used by**: Predictions AND YellowBrick visualizations

### 3. YellowBrick Functions (`functions.py`)
- Classification visualizers: ConfusionMatrix, ROCAUC, PrecisionRecallCurve, etc.
- Feature analysis visualizers: PCA, ParallelCoordinates, Manifold, etc.
- Target visualizers: ClassBalance, BalancedBinningReference
- Model selection visualizers: LearningCurve, ValidationCurve, FeatureImportances

## Implementation Status

### Completed
- [x] Create `resource_pool.py` with SessionResources dataclass
- [x] Modify `functions.py` to store data in pool after `process_batch_data_duckdb()`
- [x] Add `/resource_pool/status` and `/resource_pool/clear` endpoints to `app.py`

### In Progress
- [ ] Update `_sync_generate_yellowbrick_plot()` to use ResourcePool instead of ModelDataManager
- [ ] Update YellowBrick classification visualizers to use best model from ModelCache
- [ ] Add proper error handling when ResourcePool has no session

### Future
- [ ] Add YellowBrick dropdown selector in Reflex UI
- [ ] Add loading/spinner state for visualization generation
- [ ] Add cancellation support for slow visualizations (Manifold, RFECV)
- [ ] Cache generated visualizations in MLflow artifacts

## Implementation Details

### Current Code (to be replaced)
```python
# app.py - _sync_generate_yellowbrick_plot()
def _sync_generate_yellowbrick_plot(..., dm: ModelDataManager):
    dm.load_data(project_name)  # Reloads from DuckDB every time
    # ... generate visualization
```

### New Code (using ResourcePool)
```python
# app.py - _sync_generate_yellowbrick_plot()
def _sync_generate_yellowbrick_plot(...):
    session = resource_pool.get()
    if session is None:
        raise ValueError("No training session available. Run training first.")

    # Use data from ResourcePool
    X_train, X_test = session.X_train, session.X_test
    y_train, y_test = session.y_train, session.y_test
    X, y = session.X, session.y  # Combined data

    # For classification visualizers that need a model:
    # Use ModelCache to get best model from MLflow
    model, run_id = model_cache.get_model(project_name)

    # ... generate visualization
```

### YellowBrick Visualizer Categories

| Category | Needs Model? | Data Used |
|----------|--------------|-----------|
| Classification (ConfusionMatrix, ROCAUC, etc.) | Yes | X_train, X_test, y_train, y_test |
| Feature Analysis (PCA, Manifold, etc.) | No | X, y (combined) |
| Target (ClassBalance, etc.) | No | y only |
| Model Selection (LearningCurve, etc.) | Yes (creates new) | X, y (combined) |

### Classification Visualizers - Model Handling

Current behavior: Creates a NEW untrained model for each visualization
```python
"ConfusionMatrix": {
    "estimator": create_batch_model(project_name, y_train=y_train),  # New model
}
```

Better behavior: Use the BEST trained model from MLflow
```python
"ConfusionMatrix": {
    "estimator": model_cache.get_model(project_name)[0],  # Best model
}
```

This ensures visualizations reflect the actual production model performance.

## API Endpoints

### Resource Pool
- `GET /resource_pool/status` - Get data availability and best model info
- `POST /resource_pool/clear` - Clear stored data (free memory)

### YellowBrick
- `POST /yellowbrick_metric` - Generate visualization
  - Body: `{"project_name": "...", "metric_type": "...", "metric_name": "..."}`
  - Returns: PNG image

## Error Handling

| Scenario | Response |
|----------|----------|
| No training session in ResourcePool | 400: "No training session available. Run training first." |
| No model in MLflow | 404: "No trained model found. Train a model first." |
| Visualization timeout | 504: "Visualization generation timed out" |
| Invalid visualizer name | 400: "Unknown metric type/name" |
