# Batch ML Model Recommendations

> Research completed: January 2026
> Purpose: Define the optimal scikit-learn-flavored models for TFD, ETA, and ECCI projects with full visualization library compatibility

---

## Executive Summary

| Project | Problem Type | Recommended Model | Primary Metric | Visualization Compatibility |
|---------|-------------|-------------------|----------------|----------------------------|
| **TFD** | Binary Classification (Imbalanced) | CatBoostClassifier | FBeta (β=2) | ✅ All libraries |
| **ETA** | Regression | LightGBMRegressor | MAE | ✅ All libraries |
| **ECCI** | Clustering | HDBSCAN (sklearn 1.3+) | Silhouette | ⚠️ SHAP via KernelExplainer |

---

## Table of Contents

1. [TFD - Transaction Fraud Detection](#tfd---transaction-fraud-detection)
2. [ETA - Estimated Time of Arrival](#eta---estimated-time-of-arrival)
3. [ECCI - E-Commerce Customer Interactions](#ecci---e-commerce-customer-interactions)
4. [Visualization Library Compatibility Matrix](#visualization-library-compatibility-matrix)
5. [Hyperparameter Tuning Strategy](#hyperparameter-tuning-strategy)
6. [Dependencies](#dependencies)
7. [Sources](#sources)

---

## TFD - Transaction Fraud Detection

### Problem Characteristics

| Aspect | Value |
|--------|-------|
| Task | Binary Classification |
| Imbalance Ratio | ~1-5% fraud (highly imbalanced) |
| Features | 18 (4 numerical + 10 categorical + 6 temporal) |
| Primary Concern | Catching fraud (Recall) > Avoiding false positives (Precision) |
| Current Model | XGBClassifier |

### Recommended Model: **CatBoostClassifier**

#### Why CatBoost Over XGBoost?

| Factor | CatBoost | XGBoost | Winner |
|--------|----------|---------|--------|
| **F1 Score (Fraud)** | 0.9161 | 0.8926 | CatBoost |
| **Categorical Features** | Native handling (no encoding) | Requires encoding | CatBoost |
| **Imbalance Handling** | `auto_class_weights`, `scale_pos_weight` | `scale_pos_weight` only | CatBoost |
| **Ordered Boosting** | Reduces overfitting on small/imbalanced data | N/A | CatBoost |
| **Default Performance** | Works "out of the box" | Needs tuning | CatBoost |
| **SHAP Support** | ✅ TreeExplainer | ✅ TreeExplainer | Tie |
| **dtreeviz Support** | ❌ Not supported | ✅ Supported | XGBoost |

**Recommendation**: Use **CatBoost** for production accuracy, but keep XGBoost as secondary for dtreeviz visualizations.

#### Optimal Hyperparameters

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    # Core parameters
    iterations=500,                    # Number of boosting rounds
    learning_rate=0.05,                # Lower for better generalization
    depth=6,                           # Tree depth (4-8 typical)

    # Imbalanced data handling
    auto_class_weights='Balanced',     # Automatic class weighting
    # OR use explicit: scale_pos_weight=<n_negative/n_positive>

    # Regularization
    l2_leaf_reg=3.0,                   # L2 regularization (default: 3)
    random_strength=1.0,               # Random noise for scoring
    bagging_temperature=1.0,           # Controls Bayesian bootstrapping

    # Categorical features (auto-detected, but explicit is better)
    cat_features=['currency', 'transaction_type', 'payment_method',
                  'product_category', 'merchant_id', 'browser', 'os'],

    # Training settings
    early_stopping_rounds=50,          # Stop if no improvement
    boosting_type='Ordered',           # Better for imbalanced/small data

    # Evaluation
    eval_metric='F1',                  # Or 'AUC', 'Logloss'
    custom_metric=['AUC', 'Precision', 'Recall'],

    # GPU (optional, for faster training)
    task_type='CPU',                   # 'GPU' if available

    # Reproducibility
    random_seed=42,
    verbose=100
)
```

#### Alternative: XGBoost (for dtreeviz compatibility)

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    # Core parameters
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,

    # Imbalanced data handling
    scale_pos_weight='auto',           # Auto-calculated: sum(negative) / sum(positive)

    # Regularization
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,                    # L1 regularization
    reg_lambda=1.0,                    # L2 regularization

    # Training
    objective='binary:logistic',
    eval_metric='auc',
    early_stopping_rounds=50,

    # Reproducibility
    random_state=42,
    n_jobs=-1
)
```

### Fraud Detection Metrics Priority

```python
# Primary metric: FBeta with β=2 (prioritizes Recall 2x over Precision)
from sklearn.metrics import fbeta_score

def fraud_fbeta(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2.0)

# Secondary metrics (in order of importance)
metrics = ['roc_auc', 'precision', 'recall', 'f1', 'balanced_accuracy', 'mcc']
```

---

## ETA - Estimated Time of Arrival

### Problem Characteristics

| Aspect | Value |
|--------|-------|
| Task | Regression (continuous) |
| Target | Travel time in seconds |
| Features | 13 (9 numerical + 4 categorical + 6 temporal) |
| Primary Concern | Accurate prediction (low MAE) |
| Current Model | River ARFRegressor (incremental) |

### Recommended Model: **LightGBMRegressor**

#### Why LightGBM Over XGBoost?

| Factor | LightGBM | XGBoost | Winner |
|--------|----------|---------|--------|
| **Training Speed** | Fastest (leaf-wise growth) | Fast (level-wise) | LightGBM |
| **Memory Usage** | Lower (histogram-based) | Higher | LightGBM |
| **Large Datasets** | Excellent scalability | Good | LightGBM |
| **Latency (P95/P99)** | 25-30% lower | Baseline | LightGBM |
| **SHAP Support** | ✅ TreeExplainer | ✅ TreeExplainer | Tie |
| **dtreeviz Support** | ✅ Supported | ✅ Supported | Tie |

#### Optimal Hyperparameters

```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    # Core parameters
    n_estimators=2000,                 # High with early stopping
    learning_rate=0.05,                # Lower for better generalization
    max_depth=8,                       # -1 for unlimited
    num_leaves=31,                     # Default: 31 (should be < 2^max_depth)

    # Speed/efficiency
    boosting_type='gbdt',              # 'dart' for harder problems

    # Regularization (prevent overfitting)
    min_child_samples=20,              # Min data in leaf
    min_child_weight=0.001,            # Min sum of weights
    subsample=0.8,                     # Row subsampling
    colsample_bytree=0.8,              # Feature subsampling
    reg_alpha=0.01,                    # L1 regularization
    reg_lambda=0.1,                    # L2 regularization

    # Categorical features (LightGBM native support)
    categorical_feature=['weather', 'vehicle_type', 'driver_id', 'vehicle_id'],

    # Training
    objective='regression',            # 'regression_l1' for MAE objective
    metric='mae',

    # Early stopping (set in fit())
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

# Training with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)
```

#### Alternative: XGBRegressor

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1.0,
    objective='reg:squarederror',      # 'reg:absoluteerror' for MAE
    eval_metric='mae',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)
```

### ETA Metrics Priority

```python
# Primary metric: MAE (same units as target, most interpretable)
from sklearn.metrics import mean_absolute_error

# Secondary metrics
metrics = ['rmse', 'mape', 'r2']

# Industry standard: MAE in minutes
def mae_minutes(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / 60
```

---

## ECCI - E-Commerce Customer Interactions

### Problem Characteristics

| Aspect | Value |
|--------|-------|
| Task | Clustering (unsupervised segmentation) |
| Features | 16 (4 numerical + 8 categorical + 4 temporal) |
| Primary Concern | Meaningful customer segments |
| Current Model | River DBSTREAM (incremental) |

### Recommended Model: **HDBSCAN** (Hierarchical DBSCAN)

#### Why HDBSCAN Over K-Means?

| Factor | HDBSCAN | K-Means | Winner |
|--------|---------|---------|--------|
| **No K required** | Auto-detects clusters | Requires k | HDBSCAN |
| **Varying densities** | Handles well | Assumes equal variance | HDBSCAN |
| **Outlier detection** | Built-in noise labels | Forces all to clusters | HDBSCAN |
| **Arbitrary shapes** | Any shape | Spherical only | HDBSCAN |
| **Robustness** | Minimal tuning needed | Sensitive to init | HDBSCAN |
| **SHAP Support** | ⚠️ KernelExplainer only | ⚠️ KernelExplainer only | Tie |
| **sklearn native** | ✅ (1.3+) | ✅ | Tie |

**Note**: For batch processing, HDBSCAN is superior. For streaming/incremental, DBSTREAM (current) remains the best choice.

#### Optimal Hyperparameters

```python
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Preprocessing is critical for clustering
preprocessor = StandardScaler()

# HDBSCAN model
model = HDBSCAN(
    # Core parameters
    min_cluster_size=50,               # Minimum cluster membership
    min_samples=10,                    # Density threshold (more = denser)

    # Cluster selection
    cluster_selection_epsilon=0.0,     # Distance threshold (0 = auto)
    cluster_selection_method='eom',    # 'eom' (recommended) or 'leaf'

    # Performance
    metric='euclidean',                # Distance metric
    algorithm='auto',                  # 'best', 'auto', 'brute'
    leaf_size=40,                      # For tree algorithms

    # Output
    store_centers='centroid',          # Store cluster centers for analysis
    allow_single_cluster=False,        # Require multiple clusters

    n_jobs=-1
)

# Full pipeline
clustering_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('hdbscan', model)
])
```

#### Alternative: K-Means (for simpler interpretation)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# K-Means requires knowing k in advance
# Use Elbow method or Silhouette analysis to find optimal k

model = KMeans(
    n_clusters=5,                      # Determined via Elbow/Silhouette
    init='k-means++',                  # Smart initialization
    n_init=10,                         # Number of initializations
    max_iter=300,
    tol=1e-4,
    random_state=42,
    algorithm='lloyd'                  # 'lloyd' or 'elkan'
)

# Pipeline with scaling
kmeans_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', model)
])
```

### Clustering Metrics

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Primary metric: Silhouette Score (-1 to 1, higher is better)
def evaluate_clustering(X, labels):
    return {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),  # Lower is better
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'noise_ratio': (labels == -1).sum() / len(labels)   # HDBSCAN only
    }
```

---

## Visualization Library Compatibility Matrix

### Full Compatibility Table

| Library | CatBoost (TFD) | LightGBM (ETA) | XGBoost (Alt) | HDBSCAN (ECCI) |
|---------|----------------|----------------|---------------|----------------|
| **YellowBrick** | ✅ via `wrap()` | ✅ via `wrap()` | ✅ Native | ✅ SilhouetteVisualizer |
| **SHAP TreeExplainer** | ✅ Native | ✅ Native | ✅ Native | ❌ |
| **SHAP KernelExplainer** | ✅ | ✅ | ✅ | ⚠️ Slow |
| **SHAP Interactions** | ⚠️ Limited | ⚠️ Limited | ✅ Full | ❌ |
| **SHAP-IQ** | ✅ TreeSHAP-IQ | ✅ TreeSHAP-IQ | ✅ TreeSHAP-IQ | ❌ |
| **dtreeviz** | ❌ Not supported | ✅ Native | ✅ Native | ❌ N/A |
| **PDPBox** | ✅ | ✅ | ✅ | ❌ N/A |
| **Plotly** | ✅ | ✅ | ✅ | ✅ |

### YellowBrick Wrapper Usage

```python
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.classifier import ClassificationReport

# For CatBoost/LightGBM (third-party estimators)
from catboost import CatBoostClassifier

model = CatBoostClassifier(...)
model.fit(X_train, y_train)

# Wrap for YellowBrick compatibility
wrapped_model = wrap(model)

# Now use with any visualizer
viz = ClassificationReport(wrapped_model, classes=['Legit', 'Fraud'])
viz.score(X_test, y_test)
viz.show()
```

### SHAP Usage Examples

```python
import shap

# CatBoost / LightGBM / XGBoost - TreeExplainer (fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# For clustering or any model - KernelExplainer (slow but universal)
def model_predict(X):
    return model.predict_proba(X)[:, 1]  # For classification

explainer = shap.KernelExplainer(model_predict, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:50])  # Limit samples for speed
```

---

## Hyperparameter Tuning Strategy

### Recommended: Optuna with Cross-Validation

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = LGBMRegressor(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

    return -scores.mean()  # Minimize MAE

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

best_params = study.best_params
```

### Parameter Search Spaces by Model

| Parameter | CatBoost | LightGBM | XGBoost |
|-----------|----------|----------|---------|
| **iterations/n_estimators** | 100-1000 | 100-2000 | 100-500 |
| **learning_rate** | 0.01-0.3 | 0.01-0.3 | 0.01-0.3 |
| **depth/max_depth** | 4-10 | 3-12 | 3-10 |
| **num_leaves** | N/A | 15-127 | N/A |
| **l2_leaf_reg/reg_lambda** | 1-10 | 0-10 | 0-10 |
| **subsample** | 0.5-1.0 | 0.5-1.0 | 0.5-1.0 |

---

## Dependencies

### pyproject.toml Additions

```toml
[project.dependencies]
# Current (keep)
yellowbrick = "^1.5"
scikit-learn = "^1.5"
xgboost = "^2.0"
matplotlib = "^3.8"
plotly = "^5.18"

# NEW: Primary models
catboost = "^1.2"           # TFD - Best for fraud detection
lightgbm = "^4.3"           # ETA - Best for regression

# NEW: Explainability
shap = "^0.50"              # SHAP values & plots
shapiq = "^1.3"             # Feature interactions

# NEW: Tree visualization
dtreeviz = {version = "^2.3", extras = ["xgboost", "lightgbm"]}

# NEW: Partial dependence
pdpbox = "^0.3"

# NEW: Hyperparameter tuning
optuna = "^3.5"

# Optional: Enhanced debugging
eli5 = "^0.13"
```

### Installation Command

```bash
pip install catboost lightgbm shap shapiq "dtreeviz[xgboost,lightgbm]" pdpbox optuna
```

---

## Implementation Roadmap

### Phase 1: Model Migration (sklearn service)

1. **TFD**: Replace XGBClassifier → CatBoostClassifier
   - Keep XGBClassifier as secondary for dtreeviz
   - Add `auto_class_weights='Balanced'` for imbalance

2. **ETA**: Add batch training with LightGBMRegressor
   - Keep River ARFRegressor for incremental
   - Match hyperparameters where possible

3. **ECCI**: Add HDBSCAN for batch analysis
   - Keep DBSTREAM for incremental
   - Add cluster interpretation pipeline

### Phase 2: Visualization Endpoints (FastAPI)

1. Add `/shap/*` endpoints for all three projects
2. Add `/dtreeviz/*` endpoints for ETA (LightGBM only)
3. Add `/pdp/*` endpoints for TFD and ETA
4. Add `/clustering/*` endpoints for ECCI

### Phase 3: Reflex Integration

1. New tabs for SHAP visualizations
2. Interactive Plotly charts from SHAP JSON
3. Cluster exploration dashboard for ECCI

---

## Model Summary

| Project | Production Model | Backup Model | Key Config |
|---------|-----------------|--------------|------------|
| **TFD** | CatBoostClassifier | XGBClassifier | `auto_class_weights='Balanced'` |
| **ETA** | LGBMRegressor | XGBRegressor | `objective='regression'`, `metric='mae'` |
| **ECCI** | HDBSCAN | KMeans | `min_cluster_size=50`, `min_samples=10` |

---

## Sources

### Research Papers & Benchmarks
- [Benchmarking state-of-the-art gradient boosting algorithms](https://arxiv.org/pdf/2305.17094)
- [Comparative Study: CatBoost, XGBoost, LightGBM for Fraud Detection](https://www.researchsquare.com/article/rs-7539803/v1)
- [Application of ML in Fraud Identification](https://www.preprints.org/manuscript/202503.1199)
- [High-accuracy ETA Prediction: Hybrid ML Approach](https://www.sciencedirect.com/science/article/pii/S2666822X2500005X)
- [Optimizing LightGBM for Regression](https://www.sciencedirect.com/science/article/pii/S2405896325011486)

### Industry Applications
- [Uber DeepETA: Deep Learning for Arrival Times](https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/)
- [DoorDash: Deep Learning for ETA Predictions](https://careersatdoordash.com/blog/deep-learning-for-smarter-eta-predictions/)
- [Riskified: Boosting Algorithm Comparison](https://www.riskified.com/resources/article/boosting-comparison/)

### Library Documentation
- [SHAP TreeExplainer Documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- [SHAP Speed Comparison](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Perfomance%20Comparison.html)
- [dtreeviz GitHub](https://github.com/parrt/dtreeviz)
- [YellowBrick Third-Party Wrapper](https://www.scikit-yb.org/en/latest/api/contrib/wrapper.html)
- [CatBoost Imbalanced Classes](https://www.geeksforgeeks.org/machine-learning/handling-imbalanced-classes-in-catboost-techniques-and-solutions/)
- [LightGBM Parameter Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
- [scikit-learn HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)

### Tutorials & Guides
- [Neptune.ai: When to Choose CatBoost](https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm)
- [Neptune.ai: LightGBM Parameters Guide](https://neptune.ai/blog/lightgbm-parameters-guide)
- [Customer Segmentation Using K-Means](https://www.dataquest.io/blog/customer-segmentation-using-k-means-clustering/)
- [Fraud Detection Handbook: Ensemble Methods](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/Ensembling.html)
