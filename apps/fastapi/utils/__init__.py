"""
Utility Functions for ML Training

Modules:
    river.py   - Functions for incremental ML (River library)
    sklearn.py - Functions for batch ML (Scikit-Learn/CatBoost)

These modules contain:
    - MLflow integration helpers
    - Delta Lake data loading functions
    - Feature engineering utilities
    - Model caching and prediction functions
"""
from . import river
from . import sklearn

__all__ = ["river", "sklearn"]
