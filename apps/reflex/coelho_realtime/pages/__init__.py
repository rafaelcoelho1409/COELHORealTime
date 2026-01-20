"""Pages module - all application pages.

New routing structure:
- /tfd/incremental, /tfd/batch, /tfd/sql - Transaction Fraud Detection
- /eta/incremental, /eta/batch, /eta/sql - Estimated Time of Arrival
- /ecci/incremental, /ecci/batch, /ecci/sql - E-Commerce Customer Interactions
"""
from . import home
from . import tfd
from . import eta
from . import ecci

# Legacy pages (for backwards compatibility during transition)
from . import transaction_fraud_detection
from . import estimated_time_of_arrival
from . import e_commerce_customer_interactions

__all__ = [
    "home",
    "tfd",
    "eta",
    "ecci",
    # Legacy
    "transaction_fraud_detection",
    "estimated_time_of_arrival",
    "e_commerce_customer_interactions",
]
