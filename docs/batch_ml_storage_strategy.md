# Batch ML Visualization Storage Strategy

> Generated: January 2026
> Context: Where to store visualizations - FastAPI, Reflex, or MLflow?

---

## Visualization Storage Comparison

| Factor | FastAPI (Redis/Memory) | Reflex (State) | MLflow Artifacts |
|--------|------------------------|----------------|------------------|
| **Persistence** | Session/TTL based | Session only | Permanent |
| **Tied to model version** | No | No | **Yes** |
| **Reproducibility** | Regenerate each time | Regenerate each time | **Always available** |
| **Storage cost** | RAM (expensive) | Client memory | MinIO (cheap) |
| **Load time** | Fast (if cached) | Instant (if in state) | ~200-500ms download |
| **Audit trail** | No | No | **Yes (with run_id)** |

---

## Recommendation: Hybrid Approach

### Store in MLflow Artifacts (Best for most cases)

**When model is trained**, save visualizations as artifacts:

```python
# During XGBoost training in sklearn service
with mlflow.start_run():
    model.fit(X_train, y_train)

    # Save model
    mlflow.sklearn.log_model(model, "model")

    # Generate and save visualizations as artifacts
    shap_values = explainer(X_sample)

    # Save SHAP beeswarm
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("/tmp/shap_beeswarm.png")
    mlflow.log_artifact("/tmp/shap_beeswarm.png", "visualizations")

    # Save confusion matrix
    mlflow.log_artifact("/tmp/confusion_matrix.png", "visualizations")

    # Save feature importances as JSON (for Plotly)
    mlflow.log_dict({"importances": [...], "features": [...]}, "shap_values.json")
```

**Benefits:**
- Visualizations are **versioned with the model**
- Can compare visualizations across model versions
- No recomputation needed - just download artifact
- Audit trail: "This is what SHAP looked like for model v3"

### Use FastAPI Cache for Dynamic/Interactive

For visualizations that depend on **user input** (single prediction waterfall):

```python
# FastAPI endpoint - compute on demand, cache briefly
@lru_cache(maxsize=100)  # or Redis with TTL
def get_waterfall_for_sample(sample_hash: str):
    # Compute SHAP waterfall for specific sample
    return base64_image
```

### Reflex State for Display Only

Reflex just holds the current visualization to display - doesn't store permanently:

```python
class TFDState(SharedState):
    # Loaded from MLflow artifact or computed by FastAPI
    shap_beeswarm_image: str = ""  # base64 from MLflow artifact
    shap_waterfall_image: str = ""  # base64 from FastAPI (per-sample)
```

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION STORAGE STRATEGY                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                 MLflow Artifacts (Permanent)                     │    │
│  │                                                                  │    │
│  │  Stored at training time:                                       │    │
│  │  • SHAP beeswarm (global feature importance)                    │    │
│  │  • SHAP bar plot (mean absolute values)                         │    │
│  │  • Confusion matrix                                              │    │
│  │  • ROC-AUC curve                                                 │    │
│  │  • Precision-Recall curve                                        │    │
│  │  • Feature importances (JSON for Plotly)                        │    │
│  │  • SHAP values matrix (JSON - for interactive Plotly)           │    │
│  │  • dtreeviz tree structure (SVG)                                │    │
│  │  • Learning curve, Validation curve                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              FastAPI sklearn (On-demand + Cache)                 │    │
│  │                                                                  │    │
│  │  Computed per request (with caching):                           │    │
│  │  • SHAP waterfall (single prediction) ← user input              │    │
│  │  • SHAP force plot (single prediction) ← user input             │    │
│  │  • dtreeviz prediction path ← user input                        │    │
│  │  • PDP for specific feature ← user selection                    │    │
│  │                                                                  │    │
│  │  Downloads from MLflow:                                          │    │
│  │  • Pre-computed visualizations (serve to Reflex)                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Reflex (Display only)                         │    │
│  │                                                                  │    │
│  │  • Receives images/SVG/JSON from FastAPI                        │    │
│  │  • Renders with rx.image(), rx.html(), rx.plotly()              │    │
│  │  • No permanent storage                                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Visualization Categories by Storage Location

### MLflow Artifacts (Permanent - Generated at Training Time)

| Visualization | Format | Library | Why MLflow |
|---------------|--------|---------|------------|
| SHAP Beeswarm | PNG | SHAP | Global insight, tied to model |
| SHAP Bar Plot | PNG | SHAP | Global insight, tied to model |
| SHAP Values Matrix | JSON | SHAP | Reuse for interactive Plotly |
| Confusion Matrix | PNG | YellowBrick/sklearn | Model performance snapshot |
| ROC-AUC Curve | PNG | YellowBrick/sklearn | Model performance snapshot |
| Precision-Recall Curve | PNG | YellowBrick/sklearn | Model performance snapshot |
| Classification Report | PNG | YellowBrick | Model performance snapshot |
| Feature Importances | JSON | XGBoost | Reuse for Plotly |
| Learning Curve | PNG | YellowBrick | Training diagnostic |
| Validation Curve | PNG | YellowBrick | Training diagnostic |
| dtreeviz Tree | SVG | dtreeviz | Model structure |
| SHAP-IQ Interactions | JSON | SHAP-IQ | Expensive to compute |

### FastAPI Cache (Dynamic - Computed Per Request)

| Visualization | Format | Library | Why FastAPI |
|---------------|--------|---------|-------------|
| SHAP Waterfall | PNG | SHAP | Per-sample, user input |
| SHAP Force Plot | HTML | SHAP | Per-sample, user input |
| dtreeviz Prediction Path | SVG | dtreeviz | Per-sample, user input |
| PDP Single Feature | PNG | PDPBox | User selects feature |
| PDP Interaction | PNG | PDPBox | User selects features |

### Reflex State (Display Only - No Storage)

| Data | Source | Rendered As |
|------|--------|-------------|
| Static images | Downloaded from MLflow via FastAPI | `rx.image(src=base64)` |
| SVG graphics | Downloaded from MLflow via FastAPI | `rx.html(svg_string)` |
| JSON data | Downloaded from MLflow via FastAPI | `rx.plotly(figure)` |
| Dynamic images | Computed by FastAPI | `rx.image(src=base64)` |

---

## Implementation Flow

### At Training Time (sklearn training script)

```python
import mlflow
import shap
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix, ROCAUC

def train_and_log_visualizations(X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Train model
        model = XGBClassifier(...)
        model.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # === Generate and log visualizations ===

        # 1. SHAP Beeswarm
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test[:1000])  # Sample for speed

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig("/tmp/shap_beeswarm.png", dpi=150)
        mlflow.log_artifact("/tmp/shap_beeswarm.png", "visualizations")
        plt.close()

        # 2. SHAP values as JSON (for interactive Plotly)
        shap_data = {
            "values": shap_values.values.tolist(),
            "base_values": shap_values.base_values.tolist(),
            "feature_names": list(X_test.columns),
            "data": X_test[:1000].values.tolist()
        }
        mlflow.log_dict(shap_data, "visualizations/shap_values.json")

        # 3. Confusion Matrix
        y_pred = model.predict(X_test)
        viz = ConfusionMatrix(model, classes=[0, 1])
        viz.score(X_test, y_test)
        viz.finalize()
        plt.savefig("/tmp/confusion_matrix.png", dpi=150)
        mlflow.log_artifact("/tmp/confusion_matrix.png", "visualizations")
        plt.close()

        # 4. ROC-AUC
        viz = ROCAUC(model, classes=[0, 1])
        viz.score(X_test, y_test)
        viz.finalize()
        plt.savefig("/tmp/roc_auc.png", dpi=150)
        mlflow.log_artifact("/tmp/roc_auc.png", "visualizations")
        plt.close()

        # ... more visualizations
```

### At Serving Time (FastAPI sklearn service)

```python
# Endpoint to get pre-computed visualization from MLflow
@app.get("/visualization/{viz_name}")
async def get_visualization(viz_name: str, project_name: str):
    run_id = get_best_mlflow_run(project_name, model_name)

    # Download artifact from MLflow
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=f"visualizations/{viz_name}"
    )

    # Return as base64
    with open(artifact_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    return {"image": f"data:image/png;base64,{image_data}"}

# Endpoint for dynamic visualization (per-sample)
@app.post("/shap/waterfall")
async def get_shap_waterfall(request: WaterfallRequest):
    # Load model and explainer (cached)
    model = load_model_from_mlflow(request.project_name)
    explainer = get_cached_explainer(model)

    # Compute SHAP for single sample
    sample = pd.DataFrame([request.sample])
    shap_values = explainer(sample)

    # Generate waterfall plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)

    # Return as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode()
    plt.close()

    return {"image": f"data:image/png;base64,{image_data}"}
```

### In Reflex (Display)

```python
class TFDState(SharedState):
    shap_beeswarm_image: str = ""
    shap_waterfall_image: str = ""

    @rx.event(background=True)
    async def load_shap_beeswarm(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SKLEARN_URL}/visualization/shap_beeswarm.png",
                params={"project_name": "Transaction Fraud Detection"}
            )
            data = response.json()
            async with self:
                self.shap_beeswarm_image = data["image"]

# In component
def shap_beeswarm_card():
    return rx.card(
        rx.image(src=TFDState.shap_beeswarm_image, width="100%"),
        header="SHAP Feature Importance"
    )
```

---

## Summary

| Storage Location | Use For | Persistence | Tied to Model |
|------------------|---------|-------------|---------------|
| **MLflow Artifacts** | Global model insights | Permanent | Yes |
| **FastAPI Cache** | Per-sample explanations | TTL (minutes) | No |
| **Reflex State** | Current display | Session | No |

**Bottom line: MLflow for permanent, FastAPI for dynamic, Reflex for display.**
