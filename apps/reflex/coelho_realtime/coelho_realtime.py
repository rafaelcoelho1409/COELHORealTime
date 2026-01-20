import reflex as rx
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator
from .pages import home, tfd, eta, ecci

# Create a FastAPI app with Prometheus instrumentation
# This will be passed to Reflex as the api_transformer
api = FastAPI()
Instrumentator().instrument(api).expose(api)

# Pass the instrumented FastAPI as api_transformer
# Reflex will mount its internal API to this app
app = rx.App(api_transformer=api)

# =============================================================================
# Home Page
# =============================================================================
app.add_page(
    home.index,
    route="/",
    title="Home - COELHO RealTime"
)

# =============================================================================
# Transaction Fraud Detection (TFD) - Split Pages
# =============================================================================
app.add_page(
    tfd.incremental_index,
    route="/tfd/incremental",
    title="TFD Incremental ML - COELHO RealTime"
)
app.add_page(
    tfd.sql_index,
    route="/tfd/sql",
    title="TFD Delta Lake SQL - COELHO RealTime"
)

# TFD Batch ML Sub-Pages
app.add_page(
    tfd.batch_prediction_index,
    route="/tfd/batch/prediction",
    title="TFD Batch ML Prediction - COELHO RealTime"
)
app.add_page(
    tfd.batch_metrics_index,
    route="/tfd/batch/metrics",
    title="TFD Batch ML Metrics - COELHO RealTime"
)

# =============================================================================
# Estimated Time of Arrival (ETA) - Split Pages
# =============================================================================
app.add_page(
    eta.incremental_index,
    route="/eta/incremental",
    title="ETA Incremental ML - COELHO RealTime"
)
app.add_page(
    eta.sql_index,
    route="/eta/sql",
    title="ETA Delta Lake SQL - COELHO RealTime"
)

# ETA Batch ML Sub-Pages
app.add_page(
    eta.batch_prediction_index,
    route="/eta/batch/prediction",
    title="ETA Batch ML Prediction - COELHO RealTime"
)
app.add_page(
    eta.batch_metrics_index,
    route="/eta/batch/metrics",
    title="ETA Batch ML Metrics - COELHO RealTime"
)

# =============================================================================
# E-Commerce Customer Interactions (ECCI) - Split Pages
# =============================================================================
app.add_page(
    ecci.incremental_index,
    route="/ecci/incremental",
    title="ECCI Incremental ML - COELHO RealTime"
)
app.add_page(
    ecci.sql_index,
    route="/ecci/sql",
    title="ECCI Delta Lake SQL - COELHO RealTime"
)

# ECCI Batch ML Sub-Pages
app.add_page(
    ecci.batch_prediction_index,
    route="/ecci/batch/prediction",
    title="ECCI Batch ML Prediction - COELHO RealTime"
)
app.add_page(
    ecci.batch_metrics_index,
    route="/ecci/batch/metrics",
    title="ECCI Batch ML Metrics - COELHO RealTime"
)

# =============================================================================
# Redirects - Base routes redirect to /incremental
# =============================================================================
@api.get("/tfd")
async def redirect_tfd():
    return RedirectResponse(url="/tfd/incremental", status_code=302)

@api.get("/eta")
async def redirect_eta():
    return RedirectResponse(url="/eta/incremental", status_code=302)

@api.get("/ecci")
async def redirect_ecci():
    return RedirectResponse(url="/ecci/incremental", status_code=302)

# Batch ML redirects to prediction sub-page
@api.get("/tfd/batch")
async def redirect_tfd_batch():
    return RedirectResponse(url="/tfd/batch/prediction", status_code=302)

@api.get("/eta/batch")
async def redirect_eta_batch():
    return RedirectResponse(url="/eta/batch/prediction", status_code=302)

@api.get("/ecci/batch")
async def redirect_ecci_batch():
    return RedirectResponse(url="/ecci/batch/prediction", status_code=302)

# Legacy redirects (old routes to new routes)
@api.get("/transaction-fraud-detection")
async def redirect_tfd_legacy():
    return RedirectResponse(url="/tfd/incremental", status_code=302)

@api.get("/estimated-time-of-arrival")
async def redirect_eta_legacy():
    return RedirectResponse(url="/eta/incremental", status_code=302)

@api.get("/e-commerce-customer-interactions")
async def redirect_ecci_legacy():
    return RedirectResponse(url="/ecci/incremental", status_code=302)