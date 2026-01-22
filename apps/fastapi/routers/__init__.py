"""
API Routers for the Unified FastAPI Service.

Structure:
    routers/
    └── v1/            # API version 1
        ├── incremental  # River-based incremental/streaming ML
        ├── batch        # Scikit-Learn/CatBoost batch ML
        └── sql          # SQL queries against Delta Lake

Usage in app.py:
    from routers.v1 import incremental, batch, sql

Future versions:
    routers/
    ├── v1/
    └── v2/  # Add new version here without breaking v1
"""
from . import v1

__all__ = ["v1"]
