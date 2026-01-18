# YellowBrick + Reflex Architecture

## Problem Analysis

| Visualizer | Typical Time | Category |
|------------|-------------|----------|
| Rank1D, Rank2D | <1s | Fast |
| JointPlot, RadViz | 1-3s | Fast |
| PCA, ParallelCoordinates | 2-5s | Medium |
| Manifold (t-SNE) | 30-120s | Slow |
| RFECV | 60-300s | Very Slow |

Running all during training is problematic because:
1. Blocks the training completion response
2. User may not need all visualizations
3. Wastes compute if user never views them

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REFLEX UI                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Rank1D    │  │   Rank2D    │  │     PCA     │  │  Manifold   │  ...   │
│  │   [View]    │  │   [View]    │  │   [View]    │  │  [Generate] │        │
│  │  (cached)   │  │  (cached)   │  │  (cached)   │  │  (pending)  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FastAPI sklearn Service                                  │
│                                                                              │
│  GET  /yellowbrick/{visualizer}?run_id=...     → Return cached or generate  │
│  POST /yellowbrick/{visualizer}/generate        → Async task for slow ones   │
│  GET  /yellowbrick/status/{task_id}             → Poll task status           │
│  GET  /yellowbrick/available                    → List all + cache status    │
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   Fast Path      │    │   Async Queue    │    │   Cache Layer    │       │
│  │  (sync return)   │    │  (background)    │    │  (check first)   │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLflow + MinIO                                     │
│                                                                              │
│  artifacts/                                                                  │
│    └── {run_id}/                                                            │
│        └── yellowbrick/                                                     │
│            ├── Rank1D.png                                                   │
│            ├── Rank2D.png                                                   │
│            ├── PCA.png                                                      │
│            └── Manifold_tsne.png                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Single Endpoint with Visualizer Parameter (not one per visualizer)

```python
# Better: Single endpoint
GET /yellowbrick/{visualizer_name}?run_id=abc123&params=...

# Instead of: Multiple endpoints
GET /yellowbrick/rank1d
GET /yellowbrick/rank2d
GET /yellowbrick/pca
# ... 9 more endpoints
```

### 2. Smart Sync/Async Based on Visualizer Speed

```python
# Categorize by expected execution time
FAST_VISUALIZERS = {"Rank1D", "Rank2D", "JointPlot", "RadViz"}  # <5s
MEDIUM_VISUALIZERS = {"PCA", "ParallelCoordinates"}             # 5-15s
SLOW_VISUALIZERS = {"Manifold", "FeatureImportances", "RFECV"}  # >15s

# Fast: Generate synchronously, return image
# Slow: Return task_id, generate in background, poll for status
```

### 3. Cache in MLflow Artifacts (tied to model run)

- Each visualization is linked to a specific MLflow run
- When user trains new model → new run_id → visualizations regenerated on demand
- Old visualizations remain accessible for historical runs

---

## Implementation

### FastAPI Endpoint (`apps/sklearn/app.py`)

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional
import uuid

app = FastAPI()

# Thread pool for CPU-bound visualizations
executor = ThreadPoolExecutor(max_workers=2)

# In-memory task tracking (use Redis in production)
tasks: dict[str, dict] = {}

class VisualizerSpeed(str, Enum):
    FAST = "fast"      # <5s - sync
    MEDIUM = "medium"  # 5-15s - sync with timeout warning
    SLOW = "slow"      # >15s - async background

VISUALIZER_CONFIG = {
    "Rank1D": {"speed": VisualizerSpeed.FAST, "timeout": 5},
    "Rank2D": {"speed": VisualizerSpeed.FAST, "timeout": 5},
    "JointPlot": {"speed": VisualizerSpeed.FAST, "timeout": 5},
    "RadViz": {"speed": VisualizerSpeed.FAST, "timeout": 10},
    "PCA": {"speed": VisualizerSpeed.MEDIUM, "timeout": 15},
    "ParallelCoordinates": {"speed": VisualizerSpeed.MEDIUM, "timeout": 15},
    "Manifold": {"speed": VisualizerSpeed.SLOW, "timeout": 180},
    "FeatureImportances": {"speed": VisualizerSpeed.MEDIUM, "timeout": 30},
    "RFECV": {"speed": VisualizerSpeed.SLOW, "timeout": 300},
}


@app.get("/yellowbrick/available")
async def list_visualizers(run_id: Optional[str] = None):
    """List all visualizers with their cache status."""
    result = []
    for name, config in VISUALIZER_CONFIG.items():
        cached = check_mlflow_artifact_exists(run_id, f"yellowbrick/{name}.png") if run_id else False
        result.append({
            "name": name,
            "speed": config["speed"],
            "cached": cached,
            "estimated_time": config["timeout"],
        })
    return {"visualizers": result, "run_id": run_id}


@app.get("/yellowbrick/{visualizer_name}")
async def get_visualization(
    visualizer_name: str,
    run_id: Optional[str] = None,
    force_regenerate: bool = False,
):
    """
    Get or generate a YellowBrick visualization.

    - Fast visualizers: Generated synchronously, returned immediately
    - Slow visualizers: Returns task_id for polling
    """
    if visualizer_name not in VISUALIZER_CONFIG:
        raise HTTPException(404, f"Unknown visualizer: {visualizer_name}")

    config = VISUALIZER_CONFIG[visualizer_name]

    # Check cache first (unless force_regenerate)
    if not force_regenerate and run_id:
        cached_path = get_mlflow_artifact_path(run_id, f"yellowbrick/{visualizer_name}.png")
        if cached_path:
            return FileResponse(cached_path, media_type="image/png")

    # Fast/Medium: Generate synchronously
    if config["speed"] in [VisualizerSpeed.FAST, VisualizerSpeed.MEDIUM]:
        try:
            image_path = generate_visualization_sync(visualizer_name, run_id)
            return FileResponse(image_path, media_type="image/png")
        except TimeoutError:
            raise HTTPException(504, f"{visualizer_name} timed out")

    # Slow: Generate asynchronously
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "visualizer": visualizer_name, "progress": 0}

    # Queue background task
    asyncio.get_event_loop().run_in_executor(
        executor,
        generate_visualization_background,
        task_id, visualizer_name, run_id
    )

    return JSONResponse({
        "status": "accepted",
        "task_id": task_id,
        "message": f"{visualizer_name} is generating in background",
        "poll_url": f"/yellowbrick/status/{task_id}",
    }, status_code=202)


@app.get("/yellowbrick/status/{task_id}")
async def get_task_status(task_id: str):
    """Poll status of background visualization task."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")

    task = tasks[task_id]
    response = {
        "task_id": task_id,
        "status": task["status"],
        "visualizer": task["visualizer"],
        "progress": task.get("progress", 0),
    }

    if task["status"] == "completed":
        response["image_url"] = f"/yellowbrick/result/{task_id}"
    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")

    return response


@app.get("/yellowbrick/result/{task_id}")
async def get_task_result(task_id: str):
    """Get completed visualization image."""
    if task_id not in tasks:
        raise HTTPException(404, "Task not found")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(400, f"Task not completed: {task['status']}")

    return FileResponse(task["image_path"], media_type="image/png")


def generate_visualization_sync(visualizer_name: str, run_id: str) -> str:
    """Generate visualization synchronously (for fast visualizers)."""
    # Load data from Delta Lake or cached training data
    X, y = load_training_data(run_id)

    # Get visualizer kwargs
    kwargs = get_visualizer_kwargs(visualizer_name, X, y)

    # Generate
    image_path = generate_single_visualizer(visualizer_name, kwargs, X, y)

    # Save to MLflow artifacts
    if run_id:
        log_artifact_to_mlflow(run_id, image_path, "yellowbrick")

    return image_path


def generate_visualization_background(task_id: str, visualizer_name: str, run_id: str):
    """Generate visualization in background thread (for slow visualizers)."""
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["progress"] = 10

        X, y = load_training_data(run_id)
        tasks[task_id]["progress"] = 30

        kwargs = get_visualizer_kwargs(visualizer_name, X, y)
        tasks[task_id]["progress"] = 50

        image_path = generate_single_visualizer(visualizer_name, kwargs, X, y)
        tasks[task_id]["progress"] = 90

        if run_id:
            log_artifact_to_mlflow(run_id, image_path, "yellowbrick")

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["image_path"] = image_path

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
```

### Reflex UI Component

```python
# apps/reflex/components/yellowbrick_panel.py
import reflex as rx
from typing import Optional

class YellowBrickState(rx.State):
    visualizers: list[dict] = []
    selected_run_id: Optional[str] = None
    loading: dict[str, bool] = {}
    task_ids: dict[str, str] = {}
    images: dict[str, str] = {}

    async def load_visualizers(self):
        """Fetch available visualizers and cache status."""
        response = await fetch(f"/api/yellowbrick/available?run_id={self.selected_run_id}")
        self.visualizers = response["visualizers"]

    async def generate_visualization(self, visualizer_name: str):
        """Generate or fetch a visualization."""
        self.loading[visualizer_name] = True

        response = await fetch(f"/api/yellowbrick/{visualizer_name}?run_id={self.selected_run_id}")

        if response.status == 200:
            # Fast visualizer - got image directly
            self.images[visualizer_name] = response.image_url
            self.loading[visualizer_name] = False
        elif response.status == 202:
            # Slow visualizer - need to poll
            self.task_ids[visualizer_name] = response["task_id"]
            await self.poll_task(visualizer_name)

    async def poll_task(self, visualizer_name: str):
        """Poll for background task completion."""
        task_id = self.task_ids[visualizer_name]
        while True:
            response = await fetch(f"/api/yellowbrick/status/{task_id}")

            if response["status"] == "completed":
                self.images[visualizer_name] = response["image_url"]
                self.loading[visualizer_name] = False
                break
            elif response["status"] == "failed":
                self.loading[visualizer_name] = False
                # Show error toast
                break

            await asyncio.sleep(2)  # Poll every 2 seconds


def yellowbrick_panel() -> rx.Component:
    """YellowBrick visualization panel with on-demand generation."""
    return rx.vstack(
        rx.heading("Feature Analysis Visualizations", size="lg"),
        rx.text(f"MLflow Run: {YellowBrickState.selected_run_id}"),

        rx.grid(
            rx.foreach(
                YellowBrickState.visualizers,
                lambda viz: visualizer_card(viz),
            ),
            columns="3",
            spacing="4",
        ),

        # Modal for viewing full-size visualization
        visualization_modal(),
    )


def visualizer_card(viz: dict) -> rx.Component:
    """Card for a single visualizer."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(viz["name"], weight="bold"),
                rx.cond(
                    viz["cached"],
                    rx.badge("Cached", color="green"),
                    rx.badge(viz["speed"], color="blue"),
                ),
            ),
            rx.text(f"~{viz['estimated_time']}s", size="sm", color="gray"),

            rx.cond(
                YellowBrickState.loading[viz["name"]],
                rx.spinner(),
                rx.cond(
                    YellowBrickState.images[viz["name"]],
                    rx.button("View", on_click=lambda: show_modal(viz["name"])),
                    rx.button(
                        "Generate",
                        on_click=lambda: YellowBrickState.generate_visualization(viz["name"]),
                    ),
                ),
            ),
        ),
    )
```

---

## Best Practices Summary

| Aspect | Recommendation |
|--------|----------------|
| **API Design** | Single parameterized endpoint, not one per visualizer |
| **Sync vs Async** | Fast (<15s) = sync, Slow (>15s) = background task + polling |
| **Caching** | Store in MLflow artifacts, linked to model run_id |
| **Data Loading** | Cache X_train/y_train in memory or Redis after training |
| **Task Queue** | ThreadPoolExecutor for simple; Celery/RQ for production scale |
| **UI Pattern** | Button per visualizer, show loading state, poll for slow ones |
| **Invalidation** | New training run = new run_id = fresh visualizations |

---

## Additional Improvements

1. **WebSocket for real-time progress** (instead of polling)
2. **Redis for task state** (instead of in-memory dict)
3. **Pre-generate fast visualizers** after training completes
4. **Thumbnail previews** for quick browsing before full generation
5. **Parameter customization** (e.g., choose Manifold algorithm: t-SNE vs Isomap)

---

## YellowBrick Feature Visualizers Reference

From official docs: https://www.scikit-yb.org/en/latest/api/features/index.html

| Visualizer | Fit Method | Description |
|------------|------------|-------------|
| **Rank1D** | fit + transform | Single feature ranking (Shapiro-Wilk normality) |
| **Rank2D** | fit + transform | Pairwise correlation (pearson, spearman, kendalltau, covariance) |
| **PCA** | fit_transform | Principal Component Analysis projection (2D/3D) |
| **Manifold** | fit_transform | Non-linear reduction (t-SNE, MDS, Isomap, LLE, etc.) |
| **ParallelCoordinates** | fit_transform | Multi-dimensional line plot |
| **RadViz** | fit + transform | Radial visualization |
| **JointPlot** | fit | 2D feature correlation plot |
| **FeatureImportances** | fit | Feature importance from estimator (sklearn only) |
| **RFECV** | fit | Recursive Feature Elimination with CV (sklearn only) |

### CatBoost vs XGBoost Compatibility

| Visualizer | CatBoost | XGBoost |
|------------|----------|---------|
| Rank1D, Rank2D, PCA, Manifold, ParallelCoordinates, RadViz, JointPlot | ✓ | ✓ |
| FeatureImportances, RFECV | ✗ | ✓ |

**Note:** CatBoost is not sklearn-compatible. Use XGBoost for estimator-based visualizers.
