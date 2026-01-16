# Batch ML Visualization Architecture Analysis

> Generated: January 2026
> Context: Analysis of whether YellowBrick and other libraries can plot without training, plus architecture recommendations

---

## Can Charts Be Plotted Without Training Within the Library?

### YellowBrick

| Category | Needs Training Inside? | Alternative |
|----------|----------------------|-------------|
| **ConfusionMatrix** | No - has `from_predictions()` | sklearn `ConfusionMatrixDisplay.from_predictions()` |
| **ROCAUC** | No - has `from_predictions()` | sklearn `RocCurveDisplay.from_predictions()` |
| **PrecisionRecallCurve** | No - has `from_predictions()` | sklearn `PrecisionRecallDisplay.from_predictions()` |
| **ClassificationReport** | Needs fit() | Manual: `classification_report()` + seaborn heatmap |
| **LearningCurve** | **Yes - trains multiple times** | sklearn `LearningCurveDisplay.from_estimator()` (also trains) |
| **ValidationCurve** | **Yes - trains multiple times** | sklearn `ValidationCurveDisplay.from_estimator()` (also trains) |
| **CVScores** | **Yes - cross-validates** | Manual: `cross_val_score()` + bar chart |
| **FeatureImportances** | No - uses fitted model | Direct from XGBoost `.feature_importances_` |
| **ClassBalance** | No - just data viz | Plotly/seaborn bar chart |
| **ParallelCoordinates** | No - just data viz | Plotly `px.parallel_coordinates()` (better) |
| **PCA** | No - just transforms | sklearn `PCA` + Plotly scatter |

**Verdict**: Most classification/visualization charts **can work with pre-trained model + predictions**. Only LearningCurve, ValidationCurve, and CVScores require training internally.

### Other Libraries

| Library | Needs Training? | What It Needs |
|---------|-----------------|---------------|
| **SHAP** | No | Trained model + data → explains predictions |
| **SHAP-IQ** | No | Trained model + data → computes interactions |
| **dtreeviz** | No | Trained tree model → visualizes structure |
| **PDPBox** | No | Trained model + data → partial dependence |
| **ELI5** | No | Trained model → permutation importance |
| **Plotly** | No | Just data → interactive charts |

**All explainability libraries work with a pre-trained model.** They don't train - they explain.

---

## Recommended Architecture

### Best Approach: **Hybrid (Backend-Heavy)**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BATCH ML ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    FastAPI SKLEARN SERVICE                             │ │
│  │                    (Heavy computation, model lives here)               │ │
│  │                                                                        │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │ │
│  │  │  XGBoost Model  │  │  SHAP Explainer │  │  Computed Cache │        │ │
│  │  │  (loaded once)  │  │  (TreeExplainer)│  │  (SHAP values)  │        │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │ │
│  │                                                                        │ │
│  │  Endpoints:                                                            │ │
│  │  • /shap/values      → Returns SHAP values as JSON (for Plotly)       │ │
│  │  • /shap/waterfall   → Returns PNG base64 (single prediction)         │ │
│  │  • /shap/beeswarm    → Returns PNG base64 (matplotlib)                │ │
│  │  • /dtreeviz/tree    → Returns SVG (tree visualization)               │ │
│  │  • /pdp/plot         → Returns PNG base64 (partial dependence)        │ │
│  │  • /yellowbrick/*    → Returns PNG base64 (existing)                  │ │
│  │  • /metrics/raw      → Returns raw metrics JSON (for custom Plotly)   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         REFLEX FRONTEND                                │ │
│  │                    (Rendering, interactivity)                          │ │
│  │                                                                        │ │
│  │  Static Images:               Interactive Plotly:                      │ │
│  │  • rx.image(base64)           • SHAP bar/beeswarm (from JSON values)  │ │
│  │  • YellowBrick charts         • Parallel coordinates                  │ │
│  │  • dtreeviz SVG               • Feature correlation heatmap          │ │
│  │  • SHAP waterfall             • Class balance                        │ │
│  │                               • Custom metric visualizations          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Backend-Heavy?

| Factor | Backend (sklearn service) | Frontend (Reflex) |
|--------|--------------------------|-------------------|
| **Model loading** | Already loaded, cached | Would need to load 200MB+ model |
| **SHAP computation** | TreeExplainer is fast | Heavy computation blocks UI |
| **dtreeviz** | Needs graphviz, model access | Complex dependencies |
| **Caching** | Cache SHAP values server-side | No persistent cache |
| **Security** | Model stays on server | Model exposed to client |
| **Plotly interactive** | Return JSON → Reflex renders | Native Plotly support |

---

## Proposed Endpoint Structure

```python
# apps/sklearn/app.py

# === SHAP Endpoints ===
@app.post("/shap/values")
async def get_shap_values(request: ShapRequest):
    """Compute SHAP values, return as JSON for Plotly rendering in Reflex."""
    # Cache these - expensive to compute
    return {"shap_values": [...], "feature_names": [...], "base_value": ...}

@app.post("/shap/waterfall")
async def get_shap_waterfall(request: ShapWaterfallRequest):
    """Single prediction SHAP waterfall - return as base64 PNG."""
    return {"image": "data:image/png;base64,..."}

@app.post("/shap/beeswarm")
async def get_shap_beeswarm():
    """Global SHAP beeswarm - return as base64 PNG."""
    return {"image": "data:image/png;base64,..."}

# === SHAP-IQ Endpoints ===
@app.post("/shapiq/interactions")
async def get_shapiq_interactions():
    """Feature interaction values - return as JSON for Plotly."""
    return {"interactions": [...], "feature_pairs": [...]}

# === dtreeviz Endpoints ===
@app.post("/dtreeviz/tree")
async def get_tree_viz(request: TreeVizRequest):
    """Decision tree visualization - return as SVG."""
    return {"svg": "<svg>...</svg>"}

@app.post("/dtreeviz/prediction_path")
async def get_prediction_path(request: PredictionPathRequest):
    """Single sample prediction path through tree."""
    return {"svg": "<svg>...</svg>"}

# === PDPBox Endpoints ===
@app.post("/pdp/single")
async def get_pdp_single(request: PDPRequest):
    """1D Partial Dependence Plot - return as base64 PNG."""
    return {"image": "data:image/png;base64,..."}

@app.post("/pdp/interaction")
async def get_pdp_interaction(request: PDPInteractionRequest):
    """2D Partial Dependence Plot - return as base64 PNG."""
    return {"image": "data:image/png;base64,..."}
```

---

## Reflex Rendering Strategy

```python
# In Reflex - render static images
rx.image(src=TFDState.shap_waterfall_image)  # base64 PNG
rx.html(TFDState.dtreeviz_svg)                # SVG string

# In Reflex - render interactive Plotly from SHAP values
rx.plotly(data=TFDState.shap_bar_figure)      # Built from JSON values
rx.plotly(data=TFDState.parallel_coords)      # Interactive exploration
```

---

## Summary Recommendations

1. **Keep all ML computation in sklearn service** - Model, SHAP, dtreeviz, PDPBox all stay backend
2. **Return PNG/SVG for static charts** - YellowBrick, SHAP waterfall, dtreeviz, PDPBox
3. **Return JSON for interactive charts** - SHAP values → Plotly in Reflex
4. **Cache expensive computations** - SHAP values (compute once, reuse)
5. **Plotly-only charts can be Reflex-native** - Class balance, correlation heatmap, parallel coordinates (just need data, not model)

---

## Charts by Generation Location

### Generated in sklearn service (backend)

| Chart | Library | Output Format | Reason |
|-------|---------|---------------|--------|
| SHAP Waterfall | SHAP | PNG base64 | Needs model + matplotlib |
| SHAP Beeswarm | SHAP | PNG base64 | Needs model + matplotlib |
| SHAP Force Plot | SHAP | HTML | Interactive, needs model |
| SHAP-IQ Interactions | SHAP-IQ | JSON → Plotly | Needs model |
| dtreeviz Tree | dtreeviz | SVG | Needs model + graphviz |
| dtreeviz Path | dtreeviz | SVG | Needs model + graphviz |
| PDP 1D/2D | PDPBox | PNG base64 | Needs model |
| YellowBrick (all) | YellowBrick | PNG base64 | Needs model/data |
| Feature Importances | XGBoost | JSON → Plotly | Just model attribute |

### Generated in Reflex (frontend)

| Chart | Library | Data Source | Reason |
|-------|---------|-------------|--------|
| Class Balance | Plotly | Raw data | No model needed |
| Parallel Coordinates | Plotly | Raw data | No model needed, interactive |
| Correlation Heatmap | Plotly | Raw data | No model needed |
| SHAP Bar (interactive) | Plotly | SHAP values JSON | Built from backend values |
| SHAP Dependence (interactive) | Plotly | SHAP values JSON | Built from backend values |
| Custom Metrics Dashboard | Plotly | Metrics JSON | Just visualization |

---

## Next Steps

1. Add SHAP, SHAP-IQ, dtreeviz, PDPBox to sklearn service dependencies
2. Implement `/shap/*` endpoints with caching
3. Implement `/dtreeviz/*` endpoints
4. Implement `/pdp/*` endpoints
5. Update Reflex TFD Batch ML tab with new visualization tabs
6. Add interactive Plotly charts built from backend JSON data
