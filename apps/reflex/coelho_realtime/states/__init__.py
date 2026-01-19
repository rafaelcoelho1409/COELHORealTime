"""
States module - centralized state management for COELHO RealTime.

This module provides a clean import interface for all state classes.

Usage:
    from coelho_realtime.states import SharedState, TFDState, ETAState, ECCIState
    from coelho_realtime.states import METRIC_INFO, DROPDOWN_OPTIONS, YELLOWBRICK_INFO
"""

# Shared base state and utilities
from .shared import (
    # Base state class
    SharedState,

    # Pre-loaded constants
    METRIC_INFO,
    YELLOWBRICK_INFO,
    DROPDOWN_OPTIONS,

    # API base URLs
    RIVER_BASE_URL,
    SKLEARN_BASE_URL,

    # Helper functions
    safe_str,
    safe_int_str,
    safe_float_str,
    safe_bool,
    get_str,
    get_nested_str,
    parse_json_field,
    load_metric_info,
    load_yellowbrick_info,
    load_dropdown_options,
)

# Domain-specific state classes
from .tfd import TFDState
from .eta import ETAState
from .ecci import ECCIState

# Re-export all for convenience
__all__ = [
    # State classes
    "SharedState",  # Base state
    "TFDState",  # Transaction Fraud Detection
    "ETAState",  # Estimated Time of Arrival
    "ECCIState",  # E-Commerce Customer Interactions

    # Constants
    "METRIC_INFO",
    "YELLOWBRICK_INFO",
    "DROPDOWN_OPTIONS",
    "RIVER_BASE_URL",
    "SKLEARN_BASE_URL",

    # Helper functions
    "safe_str",
    "safe_int_str",
    "safe_float_str",
    "safe_bool",
    "get_str",
    "get_nested_str",
    "parse_json_field",
    "load_metric_info",
    "load_yellowbrick_info",
    "load_dropdown_options",
]
