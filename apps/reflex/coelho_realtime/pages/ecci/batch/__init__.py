"""ECCI Batch ML sub-pages module.

Provides pages for Batch ML functionality:
- prediction: Form + prediction results + training controls
- metrics: sklearn overview + YellowBrick visualizations (with tabs)
"""
from .prediction import index as prediction_index
from .metrics import index as metrics_index

__all__ = [
    "prediction_index",
    "metrics_index",
]
