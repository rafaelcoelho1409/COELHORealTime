"""
River ML Regression Metrics Exploration Script

This script explores all available regression metrics in River ML,
analyzes their arguments, and determines the best configuration
for the Estimated Time of Arrival (ETA) project.

Usage:
    python explore_river_regression_metrics.py
"""

from river import metrics
import inspect


def get_all_metrics():
    """Get all items from river.metrics module."""
    all_items = dir(metrics)
    print("=" * 80)
    print("ALL ITEMS IN river.metrics MODULE")
    print("=" * 80)
    for item in sorted(all_items):
        if not item.startswith('_'):
            obj = getattr(metrics, item)
            obj_type = type(obj).__name__
            print(f"  {item}: {obj_type}")
    return all_items


def identify_regression_metrics():
    """Identify metrics suitable for regression tasks."""
    regression_metrics = []
    classification_metrics = []
    other_items = []

    # Known regression metrics based on River documentation
    known_regression = [
        'MAE', 'MAPE', 'MSE', 'R2', 'RMSE', 'RMSLE', 'SMAPE',
        'Rolling',  # Wrapper for rolling window
    ]

    # Known classification metrics
    known_classification = [
        'Accuracy', 'BalancedAccuracy', 'ClassificationReport',
        'CohenKappa', 'ConfusionMatrix', 'F1', 'FBeta', 'FowlkesMallows',
        'GeometricMean', 'Jaccard', 'LogLoss', 'MCC', 'Precision',
        'Recall', 'ROCAUC', 'RollingROCAUC',
    ]

    print("\n" + "=" * 80)
    print("REGRESSION METRICS (for ETA prediction)")
    print("=" * 80)

    for item in dir(metrics):
        if item.startswith('_'):
            continue
        obj = getattr(metrics, item)
        if not inspect.isclass(obj):
            continue

        # Check if it's a regression metric by inheritance or name
        try:
            # Try to instantiate and check if it accepts y_true, y_pred as floats
            if item in known_regression:
                regression_metrics.append(item)
                print(f"\n  {item}")
            elif item in known_classification:
                classification_metrics.append(item)
        except:
            other_items.append(item)

    return regression_metrics


def analyze_metric_details(metric_name):
    """Analyze a specific metric's signature and documentation."""
    print("\n" + "-" * 80)
    print(f"METRIC: {metric_name}")
    print("-" * 80)

    metric_class = getattr(metrics, metric_name)

    # Get signature
    try:
        sig = inspect.signature(metric_class)
        print(f"\nSignature: {metric_name}{sig}")

        # Get parameters
        print("\nParameters:")
        for param_name, param in sig.parameters.items():
            default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
            print(f"  - {param_name}: {annotation} = {default}")
    except Exception as e:
        print(f"  Could not get signature: {e}")

    # Get docstring
    print(f"\nDocstring:")
    doc = metric_class.__doc__
    if doc:
        # Print first 500 chars of docstring
        doc_lines = doc.strip().split('\n')
        for line in doc_lines[:20]:
            print(f"  {line}")
        if len(doc_lines) > 20:
            print("  ...")
    else:
        print("  No docstring available")

    return metric_class


def explore_rolling_wrapper():
    """Explore the Rolling wrapper for windowed metrics."""
    print("\n" + "=" * 80)
    print("ROLLING WRAPPER (for windowed metrics)")
    print("=" * 80)

    try:
        sig = inspect.signature(metrics.Rolling)
        print(f"\nSignature: Rolling{sig}")

        print("\nParameters:")
        for param_name, param in sig.parameters.items():
            default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
            print(f"  - {param_name} = {default}")

        print("\nDocstring:")
        doc = metrics.Rolling.__doc__
        if doc:
            for line in doc.strip().split('\n')[:15]:
                print(f"  {line}")
    except Exception as e:
        print(f"Error: {e}")


def get_full_help(metric_name):
    """Get full help() output for a metric."""
    print("\n" + "=" * 80)
    print(f"FULL HELP: {metric_name}")
    print("=" * 80)

    metric_class = getattr(metrics, metric_name)
    help(metric_class)


def main():
    print("=" * 80)
    print("RIVER ML REGRESSION METRICS EXPLORATION")
    print("For Estimated Time of Arrival (ETA) Project")
    print("=" * 80)

    # Step 1: List all metrics
    all_items = get_all_metrics()

    # Step 2: Identify regression metrics
    regression_metrics = identify_regression_metrics()

    # Step 3: Analyze each regression metric in detail
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF REGRESSION METRICS")
    print("=" * 80)

    regression_metric_names = ['MAE', 'MAPE', 'MSE', 'R2', 'RMSE', 'RMSLE', 'SMAPE']

    for metric_name in regression_metric_names:
        try:
            analyze_metric_details(metric_name)
        except Exception as e:
            print(f"Error analyzing {metric_name}: {e}")

    # Step 4: Explore Rolling wrapper
    explore_rolling_wrapper()

    # Step 5: Print full help for each metric
    print("\n" + "=" * 80)
    print("FULL DOCUMENTATION FOR EACH METRIC")
    print("=" * 80)

    for metric_name in regression_metric_names:
        try:
            get_full_help(metric_name)
        except Exception as e:
            print(f"Error getting help for {metric_name}: {e}")

    # Also get help for Rolling
    try:
        get_full_help('Rolling')
    except Exception as e:
        print(f"Error getting help for Rolling: {e}")


if __name__ == "__main__":
    main()
