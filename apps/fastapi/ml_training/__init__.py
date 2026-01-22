"""
ML Training Scripts

Structure:
    ml_training/
    ├── river/      # Incremental ML (River library)
    │   ├── tfd.py  # Transaction Fraud Detection
    │   ├── eta.py  # Estimated Time of Arrival
    │   └── ecci.py # E-Commerce Customer Interactions
    └── sklearn/    # Batch ML (Scikit-Learn/CatBoost)
        ├── tfd.py  # Transaction Fraud Detection
        ├── eta.py  # Estimated Time of Arrival
        └── ecci.py # E-Commerce Customer Interactions

These scripts are executed via subprocess from the routers.
"""
