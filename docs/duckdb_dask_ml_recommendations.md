# DuckDB + Dask ML Training Recommendations

> Generated: January 2026
> Context: Optimizing ML training for 1M+ rows with DuckDB data loading and parallel training strategies

---

## 1. DuckDB for ML Training Data Loading

Your architecture already has DuckDB integration via `delta_scan()`. For ML training, DuckDB offers significant advantages:

### Why DuckDB is Faster

| Aspect | Polars `scan_delta()` | DuckDB `delta_scan()` |
|--------|----------------------|----------------------|
| **Query pushdown** | Limited SQL support | Full SQL (aggregations, CTEs, window functions) |
| **Memory model** | Lazy → eager on `.collect()` | Streaming execution, constant memory |
| **JOIN performance** | Good | Excellent (hash joins optimized) |
| **Predicate pushdown** | Yes | Yes + partition pruning |
| **Arrow integration** | Native | Native (zero-copy) |

### Recommended Data Loading for Sklearn

```python
# apps/sklearn/functions.py - New function

import duckdb

def load_training_data_duckdb(project_name: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Load training data via DuckDB for faster query execution.

    Advantages over Polars for large datasets:
    - Full SQL pushdown (filter, aggregate BEFORE loading)
    - Streaming execution (constant memory overhead)
    - Better partition pruning on Delta Lake
    """
    delta_path = DELTA_PATHS.get(project_name)
    if not delta_path:
        raise ValueError(f"Unknown project: {project_name}")

    conn = duckdb.connect()
    conn.execute("INSTALL delta; LOAD delta;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(f"SET s3_endpoint='{MINIO_ENDPOINT}';")
    conn.execute(f"SET s3_access_key_id='{AWS_ACCESS_KEY}';")
    conn.execute(f"SET s3_secret_access_key='{AWS_SECRET_KEY}';")
    conn.execute("SET s3_use_ssl=false;")
    conn.execute("SET s3_url_style='path';")

    # Use SQL to filter/sample BEFORE loading into memory
    query = f"""
    SELECT * FROM delta_scan('{delta_path}')
    {"USING SAMPLE " + str(sample_frac * 100) + " PERCENT (bernoulli)" if sample_frac < 1.0 else ""}
    """

    # Zero-copy to Arrow, then to pandas
    return conn.execute(query).fetch_arrow_table().to_pandas()
```

### Key Optimization: Push Computation to DuckDB

```python
# Instead of loading all data and then filtering in Python:
data = load_all_data()  # BAD: loads 1M+ rows
data = data[data['amount'] > 100]  # Filter in memory

# Push filtering to DuckDB:
query = """
SELECT * FROM delta_scan('{delta_path}')
WHERE amount > 100
  AND timestamp >= '2025-01-01'
"""
data = conn.execute(query).fetch_arrow_table().to_pandas()  # Only filtered rows loaded
```

---

## 2. Parallel Training Strategies for 1M+ Rows

### Option A: XGBoost Native Parallelism (Already in Use)

Your current setup uses `n_jobs=-1` which enables multi-threaded training. XGBoost is already highly optimized:

```python
# Current: apps/sklearn/functions.py:300-314
model = XGBClassifier(
    n_jobs=-1,  # Uses all CPU cores
    tree_method='hist',  # Add this: histogram-based (faster, less memory)
    ...
)
```

**Quick win**: Add `tree_method='hist'` - uses ~10x less memory than default.

### Option B: Dask-ML for Distributed Sklearn

For preprocessing pipelines that don't fit in memory:

```python
# Dask-ML approach
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler, OneHotEncoder
from dask_ml.model_selection import train_test_split
from dask_ml.xgboost import XGBClassifier as DaskXGBClassifier

def train_with_dask(delta_path: str):
    # Load data as Dask DataFrame (lazy, partitioned)
    # DuckDB can export to partitioned parquet, then Dask reads it
    ddf = dd.read_parquet(delta_path)  # Or from DuckDB export

    # Preprocessing in parallel across partitions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ddf[numerical_cols])

    # Train-test split (maintains partitions)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, ddf['is_fraud'],
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    # XGBoost with Dask (distributed training)
    model = DaskXGBClassifier(
        n_estimators=200,
        tree_method='hist',  # Required for Dask
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
```

### Option C: Incremental/Out-of-Core Learning

For models supporting `partial_fit()` (not XGBoost, but works with SGDClassifier, MiniBatchKMeans):

```python
from sklearn.linear_model import SGDClassifier

def train_incremental(delta_path: str, chunk_size: int = 50_000):
    model = SGDClassifier(loss='log_loss', warm_start=True)

    conn = duckdb.connect()
    # ... setup S3 credentials ...

    offset = 0
    while True:
        query = f"""
        SELECT * FROM delta_scan('{delta_path}')
        LIMIT {chunk_size} OFFSET {offset}
        """
        chunk = conn.execute(query).fetch_arrow_table().to_pandas()
        if len(chunk) == 0:
            break

        X_chunk = preprocess(chunk)
        y_chunk = chunk['is_fraud']
        model.partial_fit(X_chunk, y_chunk, classes=[0, 1])

        offset += chunk_size

    return model
```

---

## 3. Best Strategy Recommendation

Given your architecture, here's the recommended approach:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZED BATCH ML PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. DATA LOADING (DuckDB)                                               │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │ • delta_scan() for direct Delta Lake access                  │    │
│     │ • Push filters/aggregations to SQL (reduce data loaded)      │    │
│     │ • SAMPLE clause for initial experiments                      │    │
│     │ • Zero-copy Arrow → pandas                                   │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  2. PREPROCESSING (Vectorized)                                          │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │ • pandas with PyArrow backend (memory efficient)             │    │
│     │ • Use float32 instead of float64 (halves memory)             │    │
│     │ • Categorical dtypes for string columns                      │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  3. TRAINING (XGBoost Optimized)                                        │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │ • tree_method='hist' (10x less memory)                       │    │
│     │ • n_jobs=-1 (all CPU cores)                                  │    │
│     │ • early_stopping_rounds=10 (avoid overtraining)              │    │
│     │ • device='cuda' if GPU available                             │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Concrete Implementation

```python
# apps/sklearn/functions.py - Updated load_or_create_data

import duckdb
import pyarrow as pa

def load_or_create_data_optimized(
    consumer,
    project_name: str,
    use_duckdb: bool = True,  # New flag
    sample_frac: float = 1.0
) -> pd.DataFrame:
    """
    Load data optimized for large datasets (1M+ rows).

    Strategy:
    1. DuckDB for query pushdown and streaming
    2. Arrow for zero-copy transfer
    3. Memory-efficient dtypes
    """
    delta_path = DELTA_PATHS.get(project_name, "")

    if use_duckdb:
        try:
            conn = _get_duckdb_connection()

            # Build optimized query with pushdown
            query = f"""
            SELECT * FROM delta_scan('{delta_path}')
            {"USING SAMPLE " + str(sample_frac * 100) + " PERCENT (bernoulli)" if sample_frac < 1.0 else ""}
            """

            # Execute and convert via Arrow (zero-copy)
            arrow_table = conn.execute(query).fetch_arrow_table()

            # Convert to pandas with memory-efficient dtypes
            data_df = arrow_table.to_pandas(
                types_mapper=pd.ArrowDtype,  # Use PyArrow-backed dtypes
                self_destruct=True  # Free Arrow memory immediately
            )

            print(f"DuckDB loaded {len(data_df):,} rows for {project_name}")
            return data_df

        except Exception as e:
            print(f"DuckDB failed: {e}, falling back to Polars")

    # Fallback to existing Polars implementation
    return load_or_create_data(consumer, project_name)


def create_batch_model_optimized(project_name: str, **kwargs):
    """Create memory-optimized XGBoost model."""
    if project_name == "Transaction Fraud Detection":
        y_train = kwargs.get("y_train")
        neg_samples = sum(y_train == 0) if y_train is not None else 1
        pos_samples = sum(y_train == 1) if y_train is not None else 1
        scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            # === Memory optimizations ===
            tree_method='hist',      # Histogram-based (10x less memory)
            max_bin=256,             # Reduce histogram bins
            # === Training efficiency ===
            n_jobs=-1,               # All CPU cores
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            # === Early stopping (add eval_set in fit()) ===
            early_stopping_rounds=10,
        )
        return model
    raise ValueError(f"Unknown project: {project_name}")
```

---

## 4. Dask vs No-Dask Decision Matrix

| Scenario | Use Dask? | Recommendation |
|----------|-----------|----------------|
| **1M rows, 16GB RAM, XGBoost** | No | DuckDB + `tree_method='hist'` is sufficient |
| **10M+ rows, preprocessing bottleneck** | Yes | Dask for parallel preprocessing |
| **Sklearn models with `partial_fit`** | Maybe | DuckDB chunked loading + incremental fit |
| **GPU available** | No | XGBoost `device='cuda'` is faster than Dask |
| **Need distributed cluster** | Yes | Dask with `dask.distributed` |

---

## 5. Summary

For the current architecture with **1M+ rows**:

1. **Start with DuckDB optimizations** - push filtering to SQL, use Arrow
2. **Add `tree_method='hist'`** to XGBoost - immediate memory reduction
3. **Consider Dask only if** preprocessing pipeline becomes the bottleneck (not XGBoost itself)

### Quick Wins (No Architecture Changes)

| Change | Location | Impact |
|--------|----------|--------|
| Add `tree_method='hist'` | `functions.py:300` | 10x less memory |
| Add `max_bin=256` | `functions.py:300` | Further memory reduction |
| Use DuckDB for data loading | `functions.py:175` | Faster queries, SQL pushdown |
| Add `early_stopping_rounds=10` | `functions.py:300` | Faster training, less overfitting |
