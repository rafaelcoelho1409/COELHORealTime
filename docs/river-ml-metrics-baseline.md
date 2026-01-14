# River ML Metrics Baseline

This document captures the baseline metrics for River ML models after configuring them based on River ML documentation research.

**Date:** 2026-01-14
**Configuration:** Research-based optimal metrics configuration

---

## Transaction Fraud Detection (RandomUnderSampler + ARFClassifier)

### Current Model Configuration

```python
# Base classifier: Adaptive Random Forest
classifier = forest.ARFClassifier(
    n_models=10,
    max_features="sqrt",
    lambda_value=6,
    metric=metrics.ROCAUC(),  # Recommended by River for imbalanced fraud detection
    disable_weighted_vote=False,
    drift_detector=drift.ADWIN(delta=0.002),  # Default sensitivity
    warning_detector=drift.ADWIN(delta=0.01),
    grace_period=50,
    max_depth=None,
    split_criterion="info_gain",
    delta=0.01,
    tau=0.05,
    leaf_prediction="nba",
    nb_threshold=0,
    binary_split=False,
    min_branch_fraction=0.01,
    max_share_to_split=0.99,
    max_size=100.0,
    memory_estimate_period=2000000,
    merit_preprune=True,
    seed=42,
)

# Imbalanced learning wrapper: RandomUnderSampler
# Reference: https://riverml.xyz/latest/examples/imbalanced-learning/
# River's guide achieved 96.52% ROCAUC with this approach
model = imblearn.RandomUnderSampler(
    classifier=classifier,
    desired_dist={0: 0.5, 1: 0.5},  # Balance fraud/non-fraud during training
    seed=42,
)
```

### Metrics Configuration (2026-01-14)

**Class-based metrics** (use `predict_one`):
| Metric | Arguments | Purpose |
|--------|-----------|---------|
| Recall | `cm=shared_cm, pos_val=1` | Catch rate of fraud (TP / TP+FN) |
| Precision | `cm=shared_cm, pos_val=1` | Accuracy of fraud alerts (TP / TP+FP) |
| F1 | `cm=shared_cm, pos_val=1` | Balanced harmonic mean |
| FBeta | `beta=2.0, cm=shared_cm, pos_val=1` | Weighted (2x Recall priority) |
| Accuracy | `cm=shared_cm` | Overall correct predictions |
| BalancedAccuracy | `cm=shared_cm` | Mean recall per class |
| MCC | `cm=shared_cm, pos_val=1` | Matthews Correlation Coefficient |
| GeometricMean | `cm=shared_cm` | sqrt(TPR * TNR) - imbalance robust |
| CohenKappa | `cm=shared_cm` | Agreement beyond chance |
| Jaccard | `cm=shared_cm, pos_val=1` | IoU for positive class |

**Probability-based metrics** (use `predict_proba_one`):
| Metric | Arguments | Purpose |
|--------|-----------|---------|
| ROCAUC | `n_thresholds=50, pos_val=1` | Area under ROC curve |
| RollingROCAUC | `window_size=5000, pos_val=1` | Windowed AUC for drift detection |
| LogLoss | `{}` | Probability calibration |

**Report metrics** (serialized to MLflow):
| Metric | Arguments | Purpose |
|--------|-----------|---------|
| ConfusionMatrix | `{}` | TP/TN/FP/FN distribution |
| ClassificationReport | `decimals=4, cm=shared_cm` | Per-class statistics |

### Baseline Metrics

*Pending collection after fresh training run*

| Metric | Value | Assessment |
|--------|-------|------------|
| **Recall** | - | Target: >80% |
| **Precision** | - | Target: >90% |
| **F1** | - | Target: >85% |
| **FBeta (β=2)** | - | Primary metric for fraud detection |
| **ROCAUC** | - | Target: >90% (industry standard) |
| **MCC** | - | Best for imbalanced data |
| **LogLoss** | - | Target: <0.1 (random=0.693) |

### Analysis

*To be updated after fresh training run*

### Improvement Opportunities

1. ~~**RandomUnderSampler**: River's imbalanced-learning guide achieved 96.52% ROCAUC using `imblearn.RandomUnderSampler` wrapper~~ ✅ **IMPLEMENTED** (2026-01-14)
2. **Weighted Loss**: Combine undersampling with weighted loss for best results
3. **More Training Data**: Online learning improves over time
4. **Hyperparameter Tuning**: Adjust `n_models`, `grace_period`, `max_depth` based on validation

### References

- [River ARFClassifier Documentation](https://riverml.xyz/latest/api/forest/ARFClassifier/)
- [River Imbalanced Learning Guide](https://riverml.xyz/latest/examples/imbalanced-learning/)

---

## Estimated Time of Arrival (ARFRegressor)

### Current Configuration

```python
forest.ARFRegressor(
    n_models=10,
    max_features="sqrt",
    aggregation_method="median",  # Robust to outliers
    lambda_value=6,
    metric=metrics.MAE(),
    disable_weighted_vote=True,
    drift_detector=drift.ADWIN(delta=0.002),
    warning_detector=drift.ADWIN(delta=0.01),
    grace_period=50,
    max_depth=None,
    delta=0.01,
    tau=0.05,
    leaf_prediction="adaptive",
    model_selector_decay=0.95,
    min_samples_split=5,
    binary_split=False,
    max_size=500.0,
    memory_estimate_period=2000000,
    seed=42,
)
```

### Baseline Metrics

*Not yet trained - pending baseline collection*

### References

- [River ARFRegressor Documentation](https://riverml.xyz/latest/api/forest/ARFRegressor/)

---

## E-Commerce Customer Interactions (DBSTREAM)

### Current Configuration

```python
cluster.DBSTREAM(
    clustering_threshold=1.5,  # From River documentation example
    fading_factor=0.05,
    cleanup_interval=4,
    intersection_factor=0.5,
    minimum_weight=1.0,
)
```

### Baseline Metrics

*Not yet trained - pending baseline collection*

### References

- [River DBSTREAM Documentation](https://riverml.xyz/latest/api/cluster/DBSTREAM/)

---

## Sales Forecasting (SNARIMAX)

### Current Configuration

```python
regressor = linear_model.PARegressor(
    C=1.0,
    mode=1,
    eps=0.1,
    learn_intercept=True,
)

time_series.SNARIMAX(
    p=7,   # Past 7 days
    d=1,   # First-order differencing
    q=2,   # Error terms
    m=7,   # Weekly seasonality
    sp=1,  # Seasonal AR order
    sd=1,  # Seasonal differencing
    sq=1,  # Seasonal MA order
    regressor=regressor,
)
```

### Baseline Metrics

*Not yet trained - pending baseline collection*

### References

- [River SNARIMAX Documentation](https://riverml.xyz/latest/api/time-series/SNARIMAX/)
- [River PARegressor Documentation](https://riverml.xyz/latest/api/linear-model/PARegressor/)

---

## Future Improvements Roadmap

### Completed (2026-01-14)

1. [x] **Comprehensive metrics configuration for TFD** - 13 metrics with research-based optimal args
2. [x] **Split metrics into class-based and probability-based** - proper use of predict_one vs predict_proba_one
3. [x] **Shared ConfusionMatrix implementation** - efficiency optimization
4. [x] **Report metrics serialization to MLflow** - ConfusionMatrix + ClassificationReport as pickle
5. [x] **MLflow experiments cleanup** - deleted old TFD/ETA experiments for fresh start
6. [x] **Best model selection updated to FBeta** - prioritizes catching fraud over precision
7. [x] **TFD Metrics Dashboard complete** - All 15 metrics displayed with Plotly visualizations
8. [x] **KPI indicators with delta** - Shows change from baseline (previous best model)
9. [x] **Confusion Matrix heatmap** - Plotly visualization with TP/TN/FP/FN counts
10. [x] **Classification Report heatmap** - YellowBrick-style per-class Precision/Recall/F1
11. [x] **RollingROCAUC displayed** - For drift detection monitoring
12. [x] **Real-time updates via refresh button** - All metrics update when clicked during training
13. [x] **RandomUnderSampler for imbalanced learning** - Wrapped ARFClassifier with imblearn.RandomUnderSampler (desired_dist={0: 0.5, 1: 0.5})

### High Priority

1. [x] Implement `RandomUnderSampler` for TFD to improve ROCAUC ✅ **DONE** (2026-01-14)
2. [ ] Collect baseline metrics for ETA, ECCI, and Sales Forecasting
3. [ ] Add weighted loss for imbalanced classification

### Medium Priority

4. [ ] Experiment with different `n_models` values (15, 20, 25)
5. [ ] Test different ADWIN `delta` values for drift sensitivity
6. [ ] Add preprocessing pipeline (StandardScaler) for numerical features

### Low Priority

7. [ ] Explore alternative models (HoeffdingAdaptiveTreeClassifier, DenStream)
8. [ ] Implement A/B testing framework for model comparison
9. [ ] Add automated hyperparameter tuning
