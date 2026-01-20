# Unified Metrics Tab Design for All Projects

> Research completed: January 2025
> Covers: TFD (Classification), ETA (Regression), ECCI (Clustering)

---

## Library Deployment Matrix

| Page | YellowBrick | SHAP | SHAP-IQ | scikit-plot | PDPBox | dtreeviz | Plotly | sklearn Native | Evidently |
|------|:-----------:|:----:|:-------:|:-----------:|:------:|:--------:|:------:|:--------------:|:---------:|
| **TFD** (Classification) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚪ |
| **ETA** (Regression) | ✅ | ✅ | ⚪ | ⚪ | ✅ | ⚪ | ✅ | ✅ | ⚪ |
| **ECCI** (Clustering) | ✅ | ⚪ | ⚪ | ✅ | ⚪ | ⚪ | ✅ | ⚪ | ⚪ |

**Legend**: ✅ = Will be deployed | ⚪ = Not applicable / Future consideration

### Library Usage Details

| Library | TFD Usage | ETA Usage | ECCI Usage |
|---------|-----------|-----------|------------|
| **YellowBrick** | Classification, Feature Analysis, Target, Model Selection (24 visualizers) | Regression (ResidualsPlot, PredictionError, AlphaSelection, CooksDistance) | Clustering (Silhouette, KElbow, InterclusterDistance) |
| **SHAP** | Global/Local explainability (Bar, Beeswarm, Waterfall, Force, Dependence) | Regression explainability (values in seconds) | Not applicable (clustering) |
| **SHAP-IQ** | Feature interactions (fraud patterns) | Not needed | Not applicable |
| **scikit-plot** | Lift Curve, Cumulative Gain, KS Statistic, Calibration | Not needed | Silhouette analysis |
| **PDPBox** | Partial dependence plots | Distance/traffic effects on ETA | Not applicable |
| **dtreeviz** | Tree visualization for CatBoost | Not needed (XGBoost less visual) | Not applicable |
| **Plotly** | Interactive Parallel Coordinates, scatter | Error distributions, time series | Radar charts, Cluster scatter, Segment profiles |
| **sklearn Native** | ConfusionMatrix, ROC, PR, Calibration Display | PredictionError, Residuals Display | Not applicable |
| **Evidently** | Future: Data drift monitoring | Future: Data drift monitoring | Future: Data drift monitoring |

---

## Executive Summary

Each project has different ML task types requiring different visualizations:

| Project | Task Type | Target | Key Question |
|---------|-----------|--------|--------------|
| **TFD** | Binary Classification | is_fraud (0/1) | "Is this transaction fraudulent?" |
| **ETA** | Regression | delivery_time (seconds) | "How long will delivery take?" |
| **ECCI** | Clustering | cluster_id (0-4) | "Which customer segment?" |

This document proposes a **unified tab structure** that adapts to each task type while maintaining consistent UX.

---

## Unified Tab Structure

### Common Tabs (All Projects)
```
├── Overview (Performance Dashboard)
├── Explainability (SHAP)
├── Feature Analysis
└── Model Diagnostics
```

### Task-Specific Tabs
```
TFD (Classification):
├── Performance (Classification metrics)
└── Threshold Analysis

ETA (Regression):
├── Performance (Regression metrics)
└── Error Analysis

ECCI (Clustering):
├── Cluster Analysis
└── Segment Profiles
```

---

## Tab Design by Project

---

## TFD - Transaction Fraud Detection (Classification)

### Tab 1: Overview
**Purpose**: Executive dashboard with key classification metrics

| Component | Chart Type | Metrics |
|-----------|------------|---------|
| Primary KPIs | Indicator | Recall, Precision, F1, F-beta |
| Ranking KPIs | Indicator | ROC-AUC, Avg Precision |
| Secondary | Gauge | Accuracy, Balanced Acc, MCC, Cohen Kappa, Jaccard, G-Mean |
| Calibration | Bullet | Log Loss, Brier, D² Log Loss, D² Brier |

### Tab 2: Performance (Classification)
**Purpose**: Classification-specific diagnostic visualizations

| Visualizer | Library | Description |
|------------|---------|-------------|
| ConfusionMatrix | YellowBrick | TP/TN/FP/FN distribution |
| ClassificationReport | YellowBrick | Per-class P/R/F1 heatmap |
| ROCAUC | YellowBrick | ROC curve with AUC |
| PrecisionRecallCurve | YellowBrick | PR curve (best for imbalanced) |
| ClassPredictionError | YellowBrick | Error distribution by class |
| DiscriminationThreshold | YellowBrick | Threshold optimization |
| **NEW: CalibrationCurve** | sklearn | Probability reliability |
| **NEW: LiftCurve** | scikit-plot | Business value metric |
| **NEW: CumulativeGain** | scikit-plot | Targeting efficiency |
| **NEW: KSStatistic** | scikit-plot | Distribution separation |

### Tab 3: Explainability (SHAP)
**Purpose**: Explain predictions - "Why flagged as fraud?"

| Visualizer | Scope | Description |
|------------|-------|-------------|
| Bar Plot | Global | Mean |SHAP| feature ranking |
| Beeswarm | Global | SHAP distribution by feature value |
| Waterfall | Local | Single prediction breakdown |
| Force Plot | Local | Compact additive explanation |
| Dependence | Global | Feature value vs SHAP |
| Decision Plot | Local | Cumulative SHAP path |

### Tab 4: Feature Analysis
**Purpose**: Understand feature distributions and relationships

| Visualizer | Library | Description |
|------------|---------|-------------|
| Rank1D | YellowBrick | Single feature ranking |
| Rank2D | YellowBrick | Correlation matrix |
| PCA | YellowBrick | 2D projection |
| Manifold (t-SNE) | YellowBrick | Non-linear projection |
| ParallelCoordinates | Plotly | Interactive multi-feature |
| RadViz | YellowBrick | Radial visualization |
| JointPlot | YellowBrick | 2-feature correlation |

### Tab 5: Target Analysis
**Purpose**: Understand class distribution

| Visualizer | Library | Description |
|------------|---------|-------------|
| ClassBalance | YellowBrick | Class distribution |
| FeatureCorrelation (MI) | YellowBrick | Mutual info with target |
| FeatureCorrelation (Pearson) | YellowBrick | Linear correlation |
| BalancedBinningReference | YellowBrick | Quantile binning |

### Tab 6: Model Diagnostics
**Purpose**: Model health and selection

| Visualizer | Library | Description |
|------------|---------|-------------|
| FeatureImportances | YellowBrick | Model feature ranking |
| LearningCurve | YellowBrick | Performance vs data size |
| ValidationCurve | YellowBrick | Performance vs hyperparameter |
| CVScores | YellowBrick | Cross-validation stability |
| RFECV | YellowBrick | Recursive feature elimination |
| DroppingCurve | YellowBrick | Feature robustness |

---

## ETA - Estimated Time of Arrival (Regression)

### Tab 1: Overview
**Purpose**: Executive dashboard with key regression metrics

| Component | Chart Type | Metrics |
|-----------|------------|---------|
| Primary KPIs | Indicator | MAE, RMSE, R², MAPE |
| Secondary | Gauge | MSE, RMSLE, SMAPE, MedAE |
| Rolling | Bullet | Rolling MAE, Rolling RMSE |

**NEW Metrics for ETA**:
- **MAE** (Mean Absolute Error): Average error in seconds
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²** (Coefficient of Determination): Variance explained
- **MAPE** (Mean Absolute Percentage Error): Relative error
- **MedAE** (Median Absolute Error): Robust to outliers
- **SMAPE** (Symmetric MAPE): Bounded 0-100%

### Tab 2: Performance (Regression)
**Purpose**: Regression-specific diagnostic visualizations

| Visualizer | Library | Description |
|------------|---------|-------------|
| **ResidualsPlot** | YellowBrick | Residuals vs predicted (detect heteroscedasticity) |
| **PredictionError** | YellowBrick | Actual vs predicted scatter |
| **NEW: QQ Plot** | scipy/matplotlib | Residual normality check |
| **NEW: Residual Histogram** | Plotly | Error distribution |
| **NEW: Error by Feature** | Custom | Which features cause errors? |
| **NEW: Error Over Time** | Plotly | Time-based error patterns |

**YellowBrick Regression Visualizers**:
```python
from yellowbrick.regressor import ResidualsPlot, PredictionError, AlphaSelection, CooksDistance

# Residuals Plot - detect heteroscedasticity
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)

# Prediction Error - actual vs predicted
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
```

### Tab 3: Explainability (SHAP)
**Purpose**: Explain predictions - "Why this ETA?"

| Visualizer | Scope | Description |
|------------|-------|-------------|
| Bar Plot | Global | Mean |SHAP| in seconds |
| Beeswarm | Global | SHAP distribution colored by feature |
| Waterfall | Local | "This delivery is +5min because traffic" |
| Dependence | Global | Distance vs SHAP (non-linear effects) |

**Regression SHAP Interpretation**:
- SHAP values are in **target units** (seconds for ETA)
- "Traffic factor adds +300 seconds to ETA"
- "Short distance reduces ETA by -200 seconds"

### Tab 4: Feature Analysis
**Purpose**: Understand feature distributions

| Visualizer | Library | Description |
|------------|---------|-------------|
| Rank1D | YellowBrick | Feature distribution ranking |
| Rank2D | YellowBrick | Feature correlation matrix |
| PCA | YellowBrick | 2D projection |
| ParallelCoordinates | Plotly | Interactive multi-feature |
| JointPlot | YellowBrick | 2-feature correlation |
| **NEW: 1D PDP** | PDPBox | How distance affects ETA |
| **NEW: 2D PDP** | PDPBox | Distance + traffic interaction |

### Tab 5: Target Analysis
**Purpose**: Understand ETA distribution

| Visualizer | Library | Description |
|------------|---------|-------------|
| **TargetDistribution** | Plotly | Histogram of delivery times |
| **FeatureCorrelation** | YellowBrick | Features vs delivery time |
| BalancedBinningReference | YellowBrick | ETA quantile bins |
| **NEW: BoxPlot by Category** | Plotly | ETA by weather/traffic conditions |

### Tab 6: Model Diagnostics
**Purpose**: Model health and regularization

| Visualizer | Library | Description |
|------------|---------|-------------|
| FeatureImportances | YellowBrick | Model feature ranking |
| LearningCurve | YellowBrick | Performance vs data size |
| ValidationCurve | YellowBrick | Performance vs hyperparameter |
| CVScores | YellowBrick | Cross-validation stability |
| **AlphaSelection** | YellowBrick | Regularization tuning (for Ridge/Lasso) |
| **CooksDistance** | YellowBrick | Influential outlier detection |

---

## ECCI - E-Commerce Customer Interactions (Clustering)

### Tab 1: Overview
**Purpose**: Executive dashboard with key clustering metrics

| Component | Chart Type | Metrics |
|-----------|------------|---------|
| Primary KPIs | Indicator | Silhouette Score, Inertia, n_clusters |
| Secondary | Gauge | Calinski-Harabasz, Davies-Bouldin |
| Cluster Sizes | Bar | Samples per cluster |

**NEW Metrics for ECCI**:
- **Silhouette Score** (-1 to 1): Cluster cohesion vs separation
- **Inertia**: Within-cluster sum of squares (lower = tighter)
- **Calinski-Harabasz**: Ratio of between/within cluster dispersion
- **Davies-Bouldin**: Average similarity between clusters (lower = better)

### Tab 2: Cluster Analysis
**Purpose**: Clustering-specific diagnostic visualizations

| Visualizer | Library | Description |
|------------|---------|-------------|
| **SilhouetteVisualizer** | YellowBrick | Per-cluster silhouette scores |
| **KElbowVisualizer** | YellowBrick | Optimal K selection |
| **InterclusterDistance** | YellowBrick | Cluster separation map |
| **NEW: Cluster Scatter (PCA)** | Plotly | 2D cluster visualization |
| **NEW: Cluster Scatter (t-SNE)** | Plotly | Non-linear cluster visualization |
| **NEW: Cluster Sizes** | Plotly | Bar chart of cluster populations |

**YellowBrick Clustering Visualizers**:
```python
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance

# Elbow Method - find optimal K
visualizer = KElbowVisualizer(KMeans(), k=(2, 10))
visualizer.fit(X)

# Silhouette - evaluate cluster quality
visualizer = SilhouetteVisualizer(KMeans(n_clusters=5))
visualizer.fit(X)

# Intercluster Distance - cluster relationships
visualizer = InterclusterDistance(KMeans(n_clusters=5))
visualizer.fit(X)
```

### Tab 3: Segment Profiles
**Purpose**: Understand what defines each cluster

| Visualizer | Library | Description |
|------------|---------|-------------|
| **RadarChart** | Plotly | Feature profile per cluster |
| **Heatmap** | Plotly | Cluster centroids comparison |
| **BoxPlot by Cluster** | Plotly | Feature distribution per cluster |
| **Parallel Coordinates** | Plotly | Multi-feature cluster comparison |

**Cluster Profile Example**:
```
Cluster 0 (High-Value): High total_spent, high frequency, low recency
Cluster 1 (At-Risk): Medium spent, high recency (inactive)
Cluster 2 (New): Low spent, low frequency, low recency
Cluster 3 (Loyal): High frequency, medium spent, low recency
Cluster 4 (Occasional): Low frequency, high recency
```

### Tab 4: Feature Analysis
**Purpose**: Understand feature distributions for clustering

| Visualizer | Library | Description |
|------------|---------|-------------|
| Rank1D | YellowBrick | Feature variance ranking |
| Rank2D | YellowBrick | Feature correlation matrix |
| PCA | YellowBrick | 2D projection with cluster colors |
| Manifold (t-SNE) | YellowBrick | Non-linear projection |
| ParallelCoordinates | Plotly | Feature patterns by cluster |

### Tab 5: Model Diagnostics
**Purpose**: Clustering model health

| Visualizer | Library | Description |
|------------|---------|-------------|
| **KElbow** | YellowBrick | Optimal K selection |
| **SilhouetteScore vs K** | Custom | Quality across K values |
| LearningCurve | Custom | Stability with data size |
| **NEW: Cluster Stability** | Custom | Bootstrap cluster consistency |

---

## Visualization Library Summary by Task

| Library | Classification (TFD) | Regression (ETA) | Clustering (ECCI) |
|---------|---------------------|------------------|-------------------|
| **YellowBrick** | ConfusionMatrix, ROCAUC, PR, ClassPredictionError, DiscriminationThreshold | ResidualsPlot, PredictionError, AlphaSelection, CooksDistance | SilhouetteVisualizer, KElbow, InterclusterDistance |
| **SHAP** | All plots (classification mode) | All plots (regression mode) | Limited (cluster assignment) |
| **scikit-plot** | Lift, Cumulative Gain, KS, Calibration | - | Silhouette |
| **Plotly** | Parallel Coords, Interactive scatter | Error distributions, PDP | Radar charts, Cluster scatter |
| **PDPBox** | Feature effects | Distance/traffic effects | Feature effects on segments |
| **sklearn native** | Confusion, ROC, PR, Calibration | PredictionError, Residuals | - |

---

## Proposed Tab Structure Summary

### TFD (Classification) - 6 Tabs
```
├── Overview (16 metrics)
├── Performance (10 visualizers)
│   ├── ConfusionMatrix, ClassificationReport, ROCAUC
│   ├── PrecisionRecallCurve, ClassPredictionError
│   ├── DiscriminationThreshold
│   └── NEW: Calibration, Lift, CumulativeGain, KS
├── Explainability (6 SHAP plots)
├── Feature Analysis (7 visualizers)
├── Target Analysis (4 visualizers)
└── Model Diagnostics (6 visualizers)
```

### ETA (Regression) - 6 Tabs
```
├── Overview (10 metrics)
├── Performance (6 visualizers)
│   ├── ResidualsPlot, PredictionError
│   ├── QQ Plot, Residual Histogram
│   └── Error by Feature, Error Over Time
├── Explainability (4 SHAP plots)
├── Feature Analysis (7 visualizers + PDP)
├── Target Analysis (4 visualizers)
└── Model Diagnostics (6 visualizers + AlphaSelection, CooksDistance)
```

### ECCI (Clustering) - 5 Tabs
```
├── Overview (5 metrics + cluster sizes)
├── Cluster Analysis (6 visualizers)
│   ├── SilhouetteVisualizer, KElbow, InterclusterDistance
│   └── Cluster Scatter (PCA), Cluster Scatter (t-SNE), Cluster Sizes
├── Segment Profiles (4 visualizers)
│   ├── RadarChart, Heatmap
│   └── BoxPlot by Cluster, Parallel Coordinates
├── Feature Analysis (5 visualizers)
└── Model Diagnostics (4 visualizers)
```

---

## Implementation Priority

### Phase 1: Foundation (Current + Minor Enhancements)
- TFD: Keep current tabs, add info buttons ✅
- ETA: Add YellowBrick regression visualizers (ResidualsPlot, PredictionError)
- ECCI: Add YellowBrick clustering visualizers (Silhouette, KElbow, Intercluster)

### Phase 2: SHAP Integration
- All projects: Add SHAP explainability tab
- Classification: Bar, Beeswarm, Waterfall, Force
- Regression: Bar, Beeswarm, Waterfall (in target units)
- Clustering: Limited (explain cluster assignment)

### Phase 3: Enhanced Visualizations
- TFD: Add scikit-plot (Lift, Gain, KS, Calibration)
- ETA: Add residual analysis (QQ, histogram, by-feature)
- ECCI: Add segment profiles (Radar, Heatmap)

### Phase 4: Advanced Features
- All: PDPBox partial dependence
- All: Interactive Plotly upgrades
- ECCI: Cluster stability analysis

---

## Backend Changes Required

### New YellowBrick Functions (functions.py)

```python
# ETA Regression Visualizers
def yellowbrick_regression_kwargs(metric_name, model, X_train, y_train, X_test, y_test):
    """Generate kwargs for YellowBrick regression visualizers."""

def yellowbrick_regression_visualizers(kwargs, X, y):
    """Create YellowBrick regression visualizer instances."""

# ECCI Clustering Visualizers
def yellowbrick_clustering_kwargs(metric_name, model, X):
    """Generate kwargs for YellowBrick clustering visualizers."""

def yellowbrick_clustering_visualizers(kwargs, X):
    """Create YellowBrick clustering visualizer instances."""
```

### New Endpoints (app.py)

```python
@app.post("/yellowbrick_metric")
async def yellowbrick_metric(payload: dict):
    # Already exists - extend to handle:
    # - "Regression" module for ETA
    # - "Clustering" module for ECCI
```

---

## New Dependencies

```toml
[project.dependencies]
# Existing
yellowbrick = "^1.5"
scikit-learn = "^1.5"
catboost = "^1.2"  # TFD
xgboost = "^2.0"   # ETA

# New for all projects
shap = "^0.50"
plotly = "^5.18"

# New for enhanced visualizations
scikit-plot = "^0.3"    # Classification extras
pdpbox = "^0.3"         # Partial dependence

# Optional
shapiq = "^1.3"         # Feature interactions
dtreeviz = "^2.3"       # Tree visualization
```

---

## Sources

### Regression Visualizations
- [YellowBrick Regression API](https://www.scikit-yb.org/en/latest/api/regressor/index.html)
- [sklearn PredictionErrorDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html)
- [SHAP for Regression](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)

### Clustering Visualizations
- [YellowBrick Cluster API](https://www.scikit-yb.org/en/latest/api/cluster/index.html)
- [scikit-plot Silhouette](https://scikit-plot.readthedocs.io/en/stable/metrics.html#scikitplot.metrics.plot_silhouette)

### Classification Visualizations
- [YellowBrick Classifier API](https://www.scikit-yb.org/en/latest/api/classifier/index.html)
- [scikit-plot Metrics](https://scikit-plot.readthedocs.io/en/stable/metrics.html)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
