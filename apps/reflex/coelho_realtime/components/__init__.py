"""
Components module - reusable UI components for COELHO RealTime.

This module provides a clean import interface for all UI components.

Usage:
    from coelho_realtime.components import (
        coelho_realtime_navbar,
        page_tabs,
        ml_training_switch,
        delta_lake_sql_tab,
        ...
    )
"""

# Shared components (migrated)
from .shared import (
    coelho_realtime_navbar,
    page_tabs,
    page_sub_nav,
    batch_sub_nav,
    ml_training_switch,
    batch_ml_training_box,
    batch_ml_run_and_training_box,
    metric_info_dialog,
    yellowbrick_info_dialog,
    delta_lake_sql_tab,
    CONTEXT_TITLES,
)

# TFD components (migrated)
from .tfd import (
    transaction_fraud_detection_form,
    transaction_fraud_detection_batch_form,
    transaction_fraud_detection_metrics,
)

# ETA components (migrated)
from .eta import (
    estimated_time_of_arrival_form,
    estimated_time_of_arrival_batch_form,
    estimated_time_of_arrival_metrics,
    eta_map,
)

# ECCI components (migrated)
from .ecci import (
    e_commerce_customer_interactions_form,
    e_commerce_customer_interactions_batch_form,
    e_commerce_customer_interactions_metrics,
    ecci_map,
)

__all__ = [
    # Shared components (migrated)
    "coelho_realtime_navbar",
    "page_tabs",
    "page_sub_nav",
    "batch_sub_nav",
    "ml_training_switch",
    "batch_ml_training_box",
    "batch_ml_run_and_training_box",
    "metric_info_dialog",
    "yellowbrick_info_dialog",
    "delta_lake_sql_tab",
    "CONTEXT_TITLES",

    # TFD components (migrated)
    "transaction_fraud_detection_form",
    "transaction_fraud_detection_batch_form",
    "transaction_fraud_detection_metrics",

    # ETA components (migrated)
    "estimated_time_of_arrival_form",
    "estimated_time_of_arrival_batch_form",
    "estimated_time_of_arrival_metrics",
    "eta_map",

    # ECCI components (migrated)
    "e_commerce_customer_interactions_form",
    "e_commerce_customer_interactions_batch_form",
    "e_commerce_customer_interactions_metrics",
    "ecci_map",
]
