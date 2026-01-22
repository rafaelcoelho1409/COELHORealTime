"""
Unified FastAPI Service - COELHO RealTime

This service unifies the incremental ML (River) and batch ML (Scikit-Learn)
services into a single API with versioned endpoints using APIRouter.

Endpoint Structure:
    /api/v1/incremental/...  - Incremental ML (River) endpoints
    /api/v1/batch/...        - Batch ML (Scikit-Learn/CatBoost) endpoints
    /api/v1/sql/...          - Delta Lake SQL query endpoints
    /health                     - Health check endpoint
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
import mlflow

from config import MLFLOW_TRACKING_URI, PROJECT_NAMES
from routers.v1 import incremental, batch, sql


# =============================================================================
# Lifespan (startup/shutdown)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown tasks."""
    print("Starting Unified ML Service...", flush=True)

    # Configure MLflow
    try:
        print("Configuring MLflow...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"Error configuring MLflow: {e}")

    # Ensure MLflow experiments exist
    print("Ensuring MLflow experiments exist...")
    for project_name in PROJECT_NAMES:
        try:
            experiment = mlflow.get_experiment_by_name(project_name)
            if experiment is None or experiment.lifecycle_stage == "deleted":
                experiment_id = mlflow.create_experiment(project_name)
                print(f"Created MLflow experiment: {project_name} (ID: {experiment_id})")
            else:
                print(f"MLflow experiment exists: {project_name} (ID: {experiment.experiment_id})")
        except Exception as e:
            print(f"Error creating MLflow experiment {project_name}: {e}")

    print("Unified ML Service startup complete.", flush=True)

    yield

    print("Unified ML Service shutting down...", flush=True)


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Unified ML Service - COELHO RealTime",
    description="""
Unified API for Incremental ML (River) and Batch ML (Scikit-Learn/CatBoost).

## Endpoints

### Incremental ML (`/api/v1/incremental`)
- Real-time streaming ML with River library
- Kafka consumer integration
- Live model predictions during training

### Batch ML (`/api/v1/batch`)
- Batch training with CatBoost/Scikit-Learn
- YellowBrick visualizations
- MLflow model versioning

### Delta Lake SQL (`/api/v1/sql`)
- SQL queries against Delta Lake tables
- DuckDB and Polars query engines
- Table schema inspection
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)


# =============================================================================
# Include Routers
# =============================================================================
app.include_router(
    incremental.router,
    prefix="/api/v1/incremental",
    tags=["Incremental ML (River)"],
)

app.include_router(
    batch.router,
    prefix="/api/v1/batch",
    tags=["Batch ML (Scikit-Learn)"],
)

app.include_router(
    sql.router,
    prefix="/api/v1/sql",
    tags=["Delta Lake SQL"],
)


# =============================================================================
# Root Endpoints
# =============================================================================
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Unified ML Service - COELHO RealTime",
        "version": "1.0.0",
        "endpoints": {
            "incremental": "/api/v1/incremental",
            "batch": "/api/v1/batch",
            "sql": "/api/v1/sql",
            "docs": "/docs",
            "health": "/health",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "unified-ml-service"}
