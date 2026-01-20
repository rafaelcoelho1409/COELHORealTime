"""ECCI (E-Commerce Customer Interactions) sub-pages module."""
from .incremental import index as incremental_index
from .sql import index as sql_index

# Batch sub-pages
from .batch import (
    prediction_index as batch_prediction_index,
    metrics_index as batch_metrics_index,
)

__all__ = [
    "incremental_index",
    "sql_index",
    # Batch sub-pages
    "batch_prediction_index",
    "batch_metrics_index",
]
