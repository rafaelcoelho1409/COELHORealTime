#!/usr/bin/env python3
"""
Explore all items in river.metrics module.
Run: python explore_river_metrics.py
"""

from river import metrics
import inspect

print("=" * 80)
print("RIVER METRICS MODULE EXPLORATION")
print("=" * 80)

# =============================================================================
# 1. GET ALL ITEMS
# =============================================================================
all_items = [name for name in dir(metrics) if not name.startswith('_')]
print(f"\nTotal items in metrics module: {len(all_items)}")

# =============================================================================
# 2. CATEGORIZE ITEMS
# =============================================================================
classes = {}
functions = {}
submodules = {}
other = {}

for name in all_items:
    item = getattr(metrics, name)
    if inspect.isclass(item):
        classes[name] = item
    elif inspect.ismodule(item):
        submodules[name] = item
    elif callable(item):
        functions[name] = item
    else:
        other[name] = item

print(f"\nClasses: {len(classes)}")
print(f"Submodules: {len(submodules)}")
print(f"Functions/Callables: {len(functions)}")
print(f"Other: {len(other)}")

# =============================================================================
# 3. ANALYZE CLASSES
# =============================================================================
print("\n" + "=" * 80)
print("CLASSES")
print("=" * 80)

for name, cls in sorted(classes.items()):
    # Get __init__ signature
    try:
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters.keys() if p != 'self']
    except:
        params = ['<unknown>']

    # Get first line of docstring
    doc = cls.__doc__.split('\n')[0] if cls.__doc__ else "No docstring"

    print(f"\n{name}")
    print(f"  Args: {', '.join(params) if params else 'None'}")
    print(f"  Doc: {doc[:70]}...")

# =============================================================================
# 4. ANALYZE SUBMODULES
# =============================================================================
print("\n" + "=" * 80)
print("SUBMODULES")
print("=" * 80)

for name, mod in sorted(submodules.items()):
    contents = [x for x in dir(mod) if not x.startswith('_')]
    classes_in_mod = [x for x in contents if inspect.isclass(getattr(mod, x))]
    print(f"\n{name}:")
    print(f"  Classes: {classes_in_mod}")

# =============================================================================
# 5. ANALYZE FUNCTIONS - CHECK IF ALIASES
# =============================================================================
print("\n" + "=" * 80)
print("FUNCTIONS/CALLABLES - ALIAS CHECK")
print("=" * 80)

for name, func in sorted(functions.items()):
    # Try to find corresponding PascalCase class
    pascal_name = ''.join(word.capitalize() for word in name.split('_'))

    if hasattr(metrics, pascal_name):
        cls = getattr(metrics, pascal_name)
        is_alias = func is cls
        status = "ALIAS" if is_alias else "DIFFERENT"
    else:
        status = "UNIQUE"

    print(f"  {name} -> {status}")

# =============================================================================
# 6. DETAILED ANALYSIS FOR TFD-RELEVANT METRICS
# =============================================================================
print("\n" + "=" * 80)
print("DETAILED ANALYSIS: TFD-RELEVANT METRICS")
print("=" * 80)

tfd_metrics = [
    'Accuracy', 'BalancedAccuracy', 'Precision', 'Recall', 'F1', 'FBeta',
    'ROCAUC', 'RollingROCAUC', 'MCC', 'CohenKappa', 'GeometricMean',
    'LogLoss', 'CrossEntropy', 'Jaccard', 'ConfusionMatrix', 'ClassificationReport'
]

for name in tfd_metrics:
    if hasattr(metrics, name):
        cls = getattr(metrics, name)
        print(f"\n{'=' * 40}")
        print(f"{name}")
        print(f"{'=' * 40}")

        # Get full signature
        try:
            sig = inspect.signature(cls.__init__)
            print(f"Signature: {name}{sig}")

            # Get parameter details
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
                print(f"  - {param_name}: default={default}")
        except Exception as e:
            print(f"  Could not get signature: {e}")

        # Check if it needs probabilities
        if hasattr(cls, '__doc__') and cls.__doc__:
            doc_lower = cls.__doc__.lower()
            needs_proba = 'probab' in doc_lower or 'proba' in doc_lower
            if needs_proba:
                print(f"  ** LIKELY NEEDS PROBABILITIES **")

# =============================================================================
# 7. CHECK FOR WRAPPER/COMPOSITE METRICS
# =============================================================================
print("\n" + "=" * 80)
print("WRAPPER/COMPOSITE METRICS")
print("=" * 80)

wrapper_candidates = ['Metrics', 'Rolling', 'Wrapper', 'Multi']
for name, cls in classes.items():
    for keyword in wrapper_candidates:
        if keyword.lower() in name.lower():
            print(f"  {name}")
            break

# =============================================================================
# 8. MULTIOUTPUT SUBMODULE DEEP DIVE
# =============================================================================
print("\n" + "=" * 80)
print("MULTIOUTPUT SUBMODULE DEEP DIVE")
print("=" * 80)

if 'multioutput' in submodules:
    mod = submodules['multioutput']
    for name in dir(mod):
        if not name.startswith('_'):
            item = getattr(mod, name)
            if inspect.isclass(item):
                try:
                    sig = inspect.signature(item.__init__)
                    print(f"\n{name}{sig}")
                    doc = item.__doc__.split('\n')[0] if item.__doc__ else ""
                    print(f"  {doc[:70]}")
                except:
                    print(f"\n{name}: Could not get signature")

# =============================================================================
# 9. SUMMARY: RECOMMENDED ARGS FOR TFD
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: OPTIMAL ARGS FOR TFD (Binary Fraud Detection)")
print("=" * 80)

print("""
Based on analysis, recommended configuration:

SHARED CONFUSION MATRIX:
  shared_cm = metrics.ConfusionMatrix()
  # Benefits: Reduces computation, metrics share TP/TN/FP/FN counts

CLASS-BASED METRICS (use predict_one):
  Accuracy:         cm=shared_cm
  BalancedAccuracy: cm=shared_cm
  Precision:        cm=shared_cm, pos_val=1  (fraud=positive)
  Recall:           cm=shared_cm, pos_val=1  (fraud=positive)
  F1:               cm=shared_cm, pos_val=1  (fraud=positive)
  FBeta:            cm=shared_cm, pos_val=1, beta=2.0  (weight Recall 2x)
  MCC:              cm=shared_cm, pos_val=1
  GeometricMean:    cm=shared_cm
  CohenKappa:       cm=shared_cm
  Jaccard:          cm=shared_cm, pos_val=1

PROBABILITY-BASED METRICS (use predict_proba_one):
  ROCAUC:           n_thresholds=20  (more accurate than default 10)
  RollingROCAUC:    window_size=1000, pos_val=1  (drift detection)
  LogLoss:          (no args, requires probabilities)

MATRIX METRICS (no .get(), display separately):
  ConfusionMatrix:  (no args)

OPTIONAL - COMPREHENSIVE REPORT:
  ClassificationReport: (no args, provides full summary)
""")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
