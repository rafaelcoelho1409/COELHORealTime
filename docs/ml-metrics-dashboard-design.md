# ML Metrics Dashboard Design

> Design specification for real-time ML metrics visualization across all project pages.

---

## Table of Contents

- [Classification Metrics (TFD)](#classification-metrics-tfd)
- [Regression Metrics (ETA)](#regression-metrics-eta)
- [Clustering Metrics (ECCI)](#clustering-metrics-ecci)
- [UI Design Specification](#ui-design-specification)
- [Plotly Components](#plotly-components)

---

## Classification Metrics (TFD)

**Project:** Transaction Fraud Detection
**Model:** ARFClassifier (Adaptive Random Forest)
**Task:** Binary Classification

### Available River Metrics

| Metric | Class | Description | Range | Optimize |
|--------|-------|-------------|-------|----------|
| Accuracy | `metrics.Accuracy` | (TP+TN) / Total | 0-1 | Maximize |
| Balanced Accuracy | `metrics.BalancedAccuracy` | Avg recall per class | 0-1 | Maximize |
| Precision | `metrics.Precision` | TP / (TP+FP) | 0-1 | Maximize |
| Recall | `metrics.Recall` | TP / (TP+FN) | 0-1 | Maximize |
| F1 | `metrics.F1` | Harmonic mean P & R | 0-1 | Maximize |
| FBeta | `metrics.FBeta` | Weighted F-score | 0-1 | Maximize |
| ROC AUC | `metrics.ROCAUC` | Area under ROC | 0-1 | Maximize |
| Rolling ROC AUC | `metrics.RollingROCAUC` | Windowed ROC AUC | 0-1 | Maximize |
| MCC | `metrics.MCC` | Matthews Correlation | -1 to 1 | Maximize |
| Cohen Kappa | `metrics.CohenKappa` | Agreement beyond chance | -1 to 1 | Maximize |
| Geometric Mean | `metrics.GeometricMean` | sqrt(Sens * Spec) | 0-1 | Maximize |
| Log Loss | `metrics.LogLoss` | Cross-entropy | 0-inf | Minimize |
| Jaccard | `metrics.Jaccard` | IoU score | 0-1 | Maximize |
| Confusion Matrix | `metrics.ConfusionMatrix` | TP/TN/FP/FN | N/A | N/A |

### Priority Ranking

```
PRIMARY (Dashboard KPIs):
  1. F1 Score        - Balance precision/recall
  2. ROC AUC         - Overall discrimination
  3. Precision       - Minimize false alarms
  4. Recall          - Catch all frauds

SECONDARY (Gauges):
  5. MCC             - Best for imbalanced data
  6. Balanced Acc    - Class-weighted accuracy

DIAGNOSTIC (Charts):
  7. Confusion Matrix - Detailed breakdown
  8. Rolling ROC AUC  - Drift detection
```

---

## Regression Metrics (ETA)

**Project:** Estimated Time of Arrival
**Model:** ARFRegressor (Adaptive Random Forest)
**Task:** Regression

### Available River Metrics

| Metric | Class | Description | Range | Optimize |
|--------|-------|-------------|-------|----------|
| MAE | `metrics.MAE` | Mean Absolute Error | 0-inf | Minimize |
| MAPE | `metrics.MAPE` | Mean Absolute % Error | 0-inf | Minimize |
| MSE | `metrics.MSE` | Mean Squared Error | 0-inf | Minimize |
| RMSE | `metrics.RMSE` | Root MSE | 0-inf | Minimize |
| RMSLE | `metrics.RMSLE` | Root MSE Log Error | 0-inf | Minimize |
| SMAPE | `metrics.SMAPE` | Symmetric MAPE | 0-200% | Minimize |
| R2 | `metrics.R2` | Coefficient of Determination | -inf to 1 | Maximize |

### Priority Ranking

```
PRIMARY (Dashboard KPIs):
  1. MAE             - Interpretable error (seconds)
  2. R2              - Explained variance
  3. RMSE            - Penalizes large errors

SECONDARY (Additional context):
  4. MAPE            - Percentage error
  5. SMAPE           - Symmetric percentage

DIAGNOSTIC:
  6. MSE             - For optimization
  7. RMSLE           - Log-scale errors
```

---

## Clustering Metrics (ECCI)

**Project:** E-Commerce Customer Interactions
**Model:** DBSTREAM (Density-Based Clustering)
**Task:** Unsupervised Clustering

### Available River Metrics

| Metric | Class | Description | Applicability |
|--------|-------|-------------|---------------|
| Silhouette | `metrics.Silhouette` | Cluster cohesion/separation | Requires labels |

### Note on Clustering

DBSTREAM is an unsupervised algorithm - traditional metrics don't apply. Instead, track:

- **Cluster Count** - Number of active clusters
- **Samples per Cluster** - Distribution balance
- **Feature Distribution** - Per-cluster characteristics

---

## UI Design Specification

### Layout Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  [ Prediction ]  [ Metrics ]                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ROW 1: KPI INDICATORS                                  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ Metric1 │ │ Metric2 │ │ Metric3 │ │ Metric4 │       │   │
│  │  │  0.87   │ │  0.94   │ │  0.82   │ │  0.91   │       │   │
│  │  │  ▲+2%   │ │  ▲+1%   │ │  ▼-3%   │ │  ▲+5%   │       │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────┐ ┌─────────────────────────────────┐   │
│  │  ROW 2: GAUGES      │ │  ROW 2: DIAGNOSTIC CHART        │   │
│  │  ┌───────────────┐  │ │  (Confusion Matrix / Error Dist)│   │
│  │  │   [GAUGE]     │  │ │                                 │   │
│  │  │     85%       │  │ │    ┌───────┬───────┐            │   │
│  │  └───────────────┘  │ │    │  TN   │  FP   │            │   │
│  │  ┌───────────────┐  │ │    ├───────┼───────┤            │   │
│  │  │   [GAUGE]     │  │ │    │  FN   │  TP   │            │   │
│  │  │     82%       │  │ │    └───────┴───────┘            │   │
│  │  └───────────────┘  │ │                                 │   │
│  └─────────────────────┘ └─────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ROW 3: TIMELINE CHART                                  │   │
│  │  ════════════════════════════════════════════════════   │   │
│  │      Metric evolution over training iterations          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Specifications

| Component | Type | Purpose | Size |
|-----------|------|---------|------|
| KPI Indicator | `go.Indicator` number+delta | Primary metrics at glance | 120px height |
| Gauge | `go.Indicator` gauge+number | Secondary metrics with ranges | 200px height |
| Heatmap | `go.Heatmap` | Confusion matrix | 300px height |
| Line Chart | `go.Scatter` | Metrics over time | 250px height |

---

## Plotly Components

### 1. KPI Indicator

```python
def create_indicator(value: float, delta_ref: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=value,
        delta={"reference": delta_ref, "relative": True, "valueformat": ".1%"},
        title={"text": title},
        number={"valueformat": ".3f"}
    ))
    fig.update_layout(height=120, margin=dict(l=20, r=20, t=40, b=20))
    return fig
```

### 2. Gauge with Ranges

```python
def create_gauge(value: float, title: str, ranges: list) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, 0.5], "color": "#ef4444"},
                {"range": [0.5, 0.7], "color": "#eab308"},
                {"range": [0.7, 0.85], "color": "#22c55e"},
                {"range": [0.85, 1], "color": "#3b82f6"}
            ],
            "threshold": {"value": 0.8, "line": {"color": "black", "width": 2}}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    return fig
```

### 3. Confusion Matrix

```python
def create_confusion_matrix(tp: int, tn: int, fp: int, fn: int) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Pred: Not Fraud", "Pred: Fraud"],
        y=["Actual: Not Fraud", "Actual: Fraud"],
        colorscale="Blues",
        text=[[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]],
        texttemplate="%{text}"
    ))
    fig.update_layout(height=300, title="Confusion Matrix")
    return fig
```

### 4. Timeline Chart

```python
def create_timeline(history: dict) -> go.Figure:
    fig = go.Figure()
    for name, values in history.items():
        fig.add_trace(go.Scatter(y=values, mode="lines+markers", name=name))
    fig.update_layout(
        height=250,
        title="Metrics Over Training",
        xaxis_title="Iterations",
        yaxis_title="Value"
    )
    return fig
```

---

## Color Scheme

| Range | Color | Hex | Usage |
|-------|-------|-----|-------|
| Poor | Red | `#ef4444` | 0-50% |
| Fair | Yellow | `#eab308` | 50-70% |
| Good | Green | `#22c55e` | 70-85% |
| Excellent | Blue | `#3b82f6` | 85-100% |

---

## Implementation Status

- [x] Basic metric cards (current)
- [x] Plotly KPI indicators with delta (FBeta, ROC AUC, Precision, Recall, Rolling AUC)
- [x] Gauge charts with color ranges (MCC, Balanced Accuracy)
- [x] Confusion matrix heatmap (Plotly)
- [x] Classification report heatmap (YellowBrick-style with Plotly)
- [x] Rolling metrics for drift detection (RollingROCAUC displayed as KPI)
- [x] Real-time updates via refresh button
- [x] Delta from baseline (shows improvement/decline from previous best model)
- [ ] ~~Metrics timeline chart~~ (Available in MLflow UI - not duplicated)
