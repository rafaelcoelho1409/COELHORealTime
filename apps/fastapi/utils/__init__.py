"""
Utility Functions for ML Training

Modules:
    incremental.py   - Functions for incremental ML (River library)
    batch.py         - Functions for batch ML (Scikit-Learn/CatBoost/Scikit-plot)

These modules contain:
    - MLflow integration helpers
    - Delta Lake data loading functions
    - Feature engineering utilities
    - Model caching and prediction functions
"""
from . import incremental
from . import batch

__all__ = ["incremental", "batch"]
