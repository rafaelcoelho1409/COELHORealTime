"""
API v1 Routers

This module contains all v1 API routers:
- incremental: River-based incremental/streaming ML
- batch: Scikit-Learn/CatBoost batch ML
- sql: SQL queries against Delta Lake
"""
from . import incremental
from . import batch
from . import sql

__all__ = ["incremental", "batch", "sql"]
