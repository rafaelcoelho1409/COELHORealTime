# Batch ML Visualization Libraries Research

> Research conducted: January 2025
> Purpose: Evaluate YellowBrick alternatives and design an enhanced Batch ML visualization stack

---

## Table of Contents

1. [Current Implementation Analysis](#current-implementation-analysis)
2. [YellowBrick Status](#yellowbrick-status)
3. [Alternative Libraries Overview](#alternative-libraries-overview)
4. [SHAP vs SHAP-IQ Comparison](#shap-vs-shap-iq-comparison)
5. [Complete Library Comparison Matrix](#complete-library-comparison-matrix)
6. [YellowBrick Feature Overlap Analysis](#yellowbrick-feature-overlap-analysis)
7. [Proposed Architecture](#proposed-architecture)
8. [Proposed Tab Structure](#proposed-tab-structure)
9. [Dependencies](#dependencies)
10. [Sources](#sources)

---

## Current Implementation Analysis

### Location
- **Backend**: `/apps/sklearn/app.py` and `/apps/sklearn/functions.py`
- **Frontend**: `/apps/reflex/coelho_realtime/components/tfd.py` and `states/tfd.py`

### Current YellowBrick Visualizers

#### Classification Visualizers
| Visualizer | Description |
|------------|-------------|
| ClassificationReport | Precision, recall, F1 scores as heatmap |
| ConfusionMatrix | Binary classification confusion matrix |
| ROCAUC | Receiver Operating Characteristic curve |
| PrecisionRecallCurve | Precision-recall tradeoff visualization |
| ClassPredictionError | Actual vs. predicted classification errors |

#### Feature Analysis Visualizers
| Visualizer | Description |
|------------|-------------|
| ParallelCoordinates | Multi-dimensional feature visualization (5% sampling) |
| PCA | Principal Component Analysis with auto-scaling |

#### Target Visualizers
| Visualizer | Description |
|------------|-------------|
| BalancedBinningReference | Class balance reference visualization |
| ClassBalance | Class distribution for imbalanced datasets |

#### Model Selection Visualizers
| Visualizer | Description |
|------------|-------------|
| ValidationCurve | Parameter tuning evaluation (gamma) |
| LearningCurve | Training sample size impact |
| CVScores | Cross-validation score distributions |
| FeatureImportances | XGBoost feature importance ranking |
| DroppingCurve | Recursive feature elimination analysis |

### Current Model
- **Algorithm**: XGBClassifier (Gradient Boosting)
- **Configuration**:
  - objective: 'binary:logistic'
  - n_estimators: 200
  - learning_rate: 0.05
  - max_depth: 5
  - subsample: 0.8
  - colsample_bytree: 0.8
  - eval_metric: 'auc'
  - scale_pos_weight: Auto-calculated for imbalanced data

---

## YellowBrick Status

### Maintenance Status: ⚠️ INACTIVE

According to Snyk security analysis:
- **Last Release**: v1.5 (no new releases in 12+ months)
- **Status**: Considered inactive/discontinued project
- **Security**: 0 vulnerabilities detected (as of April 2025)
- **GitHub Stars**: 4,331

### Implications
- Library is functional but not actively developed
- No new features or bug fixes expected
- May have compatibility issues with future sklearn versions
- Consider supplementing with actively maintained libraries

---

## Alternative Libraries Overview

### Direct Alternatives for ML Model Visualization

| Library | Description | Best For |
|---------|-------------|----------|
| **SHAP** | Game theoretic approach to explain ML model output | Model explainability, feature importance |
| **SHAP-IQ (shapiq)** | Extension of SHAP for feature interactions | Understanding feature synergies |
| **scikit-plots** | Drop-in plotting for scikit-learn | Quick, simple ML plots |
| **dtreeviz** | Decision tree visualization and interpretation | Tree-based model analysis |
| **ELI5** | Debug ML classifiers and explain predictions | Pipeline debugging, text explanations |
| **PDPBox** | Partial dependence plots | Feature effect isolation |
| **sklearn-evaluation** | ML model evaluation with reports | HTML reports, experiment tracking |

### For EDA & Data Visualization

| Library | Description |
|---------|-------------|
| **Plotly** | Interactive visualizations, dashboards |
| **Seaborn** | Statistical graphics with matplotlib |
| **Sweetviz** | Automated EDA visualizations |
| **Missingno** | Missing data visualizations |

---

## SHAP vs SHAP-IQ Comparison

### Overview

| Aspect | SHAP | SHAP-IQ (shapiq) |
|--------|------|------------------|
| **Focus** | Individual feature contributions | Feature **interactions** (synergies) |
| **Order** | 1st order only | Any-order (1st, 2nd, 3rd... N-th) |
| **Question Answered** | "How much does feature X contribute?" | "How do features X and Y **together** affect the prediction?" |
| **XGBoost Support** | TreeExplainer (fast) | TreeSHAP-IQ (also fast, linear time) |
| **Overlap** | N/A | Can compute standard SHAP values with `index='SV'` |
| **Latest Version** | v0.50.0 (Nov 2025) | v1.3.2 (Jan 2025) |
| **Use Case** | General explainability | Deep interaction analysis |

### SHAP Plot Types

| Plot Type | Purpose | Scope |
|-----------|---------|-------|
| Beeswarm | Distribution of SHAP values per feature | Global |
| Bar Plot | Mean absolute SHAP values | Global |
| Waterfall | Single prediction breakdown | Local |
| Force Plot | Compact prediction visualization | Local |
| Dependence | Feature value vs SHAP value | Global |

### SHAP-IQ Plot Types

| Plot Type | Purpose | Scope |
|-----------|---------|-------|
| Interaction Bar | Top feature pairs importance | Global |
| Interaction Force | Synergy breakdown for single prediction | Local |
| N-Order Analysis | Higher-order interactions | Global |
| Interaction Heatmap | Pairwise interaction matrix | Global |

### Do They Overlap?

**Yes, partially.** SHAP-IQ can compute standard SHAP values (`index='SV'`), so it's a **superset** of SHAP for interaction analysis.

**Recommendation**:
- **Use SHAP** for: Standard explainability, waterfall plots, force plots, beeswarm
- **Use SHAP-IQ** for: When you need to understand how features work together

For **fraud detection**, SHAP-IQ is particularly valuable because fraud patterns often involve **feature combinations**, not just individual features.

---

## Complete Library Comparison Matrix

| Library | Primary Purpose | sklearn Integration | XGBoost | Maintenance | Output Format | Unique Strength |
|---------|-----------------|---------------------|---------|-------------|---------------|-----------------|
| **YellowBrick** | ML diagnostics & model selection | ✅ Excellent | Generic | ⚠️ Inactive | PNG (matplotlib) | Most complete diagnostic suite |
| **scikit-plots** | Quick ML plotting | ✅ Excellent | Generic | ✅ Active | PNG (matplotlib) | Drop-in simplicity |
| **SHAP** | Feature explainability | ✅ Excellent | ✅ TreeExplainer | ✅ Active | HTML/PNG/Interactive | Industry standard XAI |
| **SHAP-IQ** | Feature interactions | ✅ Good | ✅ TreeSHAP-IQ | ✅ Active | PNG/Interactive | Any-order interactions |
| **dtreeviz** | Tree visualization | ✅ Excellent | ✅ Native | ✅ Active | SVG only | Beautiful tree diagrams + AI chat |
| **ELI5** | Model debugging | ✅ Excellent | ✅ Native | ⚠️ Moderate | HTML/Text/DataFrame | Text explanations, pipeline debug |
| **PDPBox** | Partial dependence | ✅ Good | Generic | ⚠️ Moderate | PNG | 2D interaction plots |
| **Plotly** | Interactive viz | Via pandas | Generic | ✅ Active | HTML/Interactive | Brushing, interactivity |
| **sklearn native** | Core metrics | ✅ Native | Generic | ✅ Active | PNG (matplotlib) | Zero dependencies |

---

## YellowBrick Feature Overlap Analysis

### Classification Visualizers

| YellowBrick Feature | Alternative Library | Native sklearn? | Notes |
|---------------------|---------------------|-----------------|-------|
| **ConfusionMatrix** | `sklearn.metrics.ConfusionMatrixDisplay` | ✅ Yes (1.0+) | `from_estimator()` or `from_predictions()` |
| **ROCAUC** | `sklearn.metrics.RocCurveDisplay` | ✅ Yes (1.0+) | Supports multi-class, chance level |
| **PrecisionRecallCurve** | `sklearn.metrics.PrecisionRecallDisplay` | ✅ Yes (1.0+) | Native support |
| **ClassificationReport** | `sklearn.metrics.classification_report` + Seaborn heatmap | ⚠️ Partial | Needs manual heatmap plotting |
| **ClassPredictionError** | Custom matplotlib bar chart | ❌ No | Simple stacked bar chart from confusion matrix |
| **DiscriminationThreshold** | `sklearn.model_selection.TunedThresholdClassifierCV` | ✅ Yes (1.5+) | Or custom plot with threshold loop |

### Model Selection Visualizers

| YellowBrick Feature | Alternative Library | Native sklearn? | Notes |
|---------------------|---------------------|-----------------|-------|
| **LearningCurve** | `sklearn.model_selection.LearningCurveDisplay` | ✅ Yes (1.2+) | Native support |
| **ValidationCurve** | `sklearn.model_selection.ValidationCurveDisplay` | ✅ Yes (1.3+) | Native support |
| **CVScores** | Custom matplotlib bar chart | ❌ No | Simple: `cross_val_score()` + bar plot |
| **FeatureImportances** | **SHAP** `shap.plots.bar()` | ❌ No | SHAP is superior for this |
| **DroppingCurve (RFECV)** | `sklearn.feature_selection.RFECV` + custom plot | ⚠️ Partial | RFECV provides scores, manual plotting |

### Target Visualizers

| YellowBrick Feature | Alternative Library | Native sklearn? | Notes |
|---------------------|---------------------|-----------------|-------|
| **ClassBalance** | Seaborn `countplot()` or Plotly bar | ❌ No | One-liner with seaborn |
| **BalancedBinningReference** | Custom matplotlib histogram + `np.percentile()` | ❌ No | Simple histogram with vertical lines |

### Feature Analysis Visualizers

| YellowBrick Feature | Alternative Library | Native sklearn? | Notes |
|---------------------|---------------------|-----------------|-------|
| **ParallelCoordinates** | **Plotly** `px.parallel_coordinates()` | ❌ No | Plotly is superior (interactive, brushing) |
| **PCA Visualizer** | **Plotly** scatter + `sklearn.decomposition.PCA` | ⚠️ Partial | Easy combo |
| **RadViz** | `pandas.plotting.radviz()` | ❌ No | Pandas has this built-in |
| **Rank2D** | Seaborn `heatmap()` + correlation matrix | ❌ No | Standard approach |
| **FeatureCorrelation** | Seaborn `heatmap()` | ❌ No | Common pattern |

### Clustering Visualizers

| YellowBrick Feature | Alternative Library | Native sklearn? | Notes |
|---------------------|---------------------|-----------------|-------|
| **KElbowVisualizer** | Custom plot + `sklearn.cluster.KMeans` | ❌ No | Loop over k, plot inertia |
| **SilhouetteVisualizer** | `sklearn.metrics.silhouette_samples()` + custom plot | ⚠️ Partial | Manual but straightforward |
| **InterclusterDistance** | Custom matplotlib | ❌ No | Compute pairwise distances |

### Regression Visualizers

| YellowBrick Feature | Alternative Library | Native sklearn? | Notes |
|---------------------|---------------------|-----------------|-------|
| **ResidualsPlot** | `sklearn.metrics.PredictionErrorDisplay` | ✅ Yes (1.2+) | `kind="residual_vs_predicted"` |
| **PredictionError** | `sklearn.metrics.PredictionErrorDisplay` | ✅ Yes (1.2+) | `kind="actual_vs_predicted"` |
| **AlphaSelection** | Custom plot with `RidgeCV` or `LassoCV` | ❌ No | Plot `alphas_` vs scores |
| **CooksDistance** | `statsmodels.stats.outliers_influence.OLSInfluence` | ❌ No | statsmodels has this |

### Text Visualizers

| YellowBrick Feature | Alternative Library | Native sklearn? | Notes |
|---------------------|---------------------|-----------------|-------|
| **FreqDistVisualizer** | Custom bar chart + `CountVectorizer` | ❌ No | Simple implementation |
| **TSNEVisualizer** | **Plotly** scatter + `sklearn.manifold.TSNE` | ⚠️ Partial | Easy combo |
| **UMAPVisualizer** | **Plotly** scatter + `umap-learn` | ❌ No | UMAP library required |
| **DispersionPlot** | Custom matplotlib | ❌ No | Manual but simple |

### Verdict

**YellowBrick is NOT unique** - every feature can be replaced:
- **70% covered by sklearn native** (1.2+ has most Display classes)
- **20% covered by seaborn/matplotlib** (simple plots)
- **10% improved by SHAP/Plotly** (feature importance, parallel coords)

**However**, YellowBrick remains valuable as the **most cohesive, complete diagnostic suite** with a unified API.

---

## Proposed Architecture

### The Ultimate Batch ML Visualization Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BATCH ML VISUALIZATION STACK                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 1: CORE DIAGNOSTICS                        │   │
│  │                         (YellowBrick)                               │   │
│  │                                                                     │   │
│  │  • Classification: ConfusionMatrix, ROCAUC, PrecisionRecall        │   │
│  │  • Model Selection: LearningCurve, ValidationCurve, CVScores       │   │
│  │  • Target Analysis: ClassBalance, BalancedBinning                  │   │
│  │  • Clustering: Elbow, Silhouette, InterclusterDistance             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  LAYER 2: EXPLAINABILITY (XAI)                      │   │
│  │                       (SHAP + SHAP-IQ)                              │   │
│  │                                                                     │   │
│  │  SHAP:                          SHAP-IQ:                           │   │
│  │  • Beeswarm (global importance) • Interaction bar plots            │   │
│  │  • Waterfall (single predict)   • Pairwise synergy analysis        │   │
│  │  • Force plot (compact)         • N-order interaction values       │   │
│  │  • Dependence plots             • Interaction force plots          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                LAYER 3: TREE MODEL DEEP DIVE                        │   │
│  │                        (dtreeviz)                                   │   │
│  │                                                                     │   │
│  │  • Decision path visualization                                     │   │
│  │  • Feature split histograms                                        │   │
│  │  • Decision boundaries                                             │   │
│  │  • AI-powered tree explanations (chat)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              LAYER 4: INTERACTIVE EXPLORATION                       │   │
│  │                   (Plotly + PDPBox)                                 │   │
│  │                                                                     │   │
│  │  Plotly:                        PDPBox:                            │   │
│  │  • Parallel coordinates         • 1D partial dependence            │   │
│  │  • Interactive scatter          • 2D partial dependence            │   │
│  │  • Brushing & filtering         • Feature effect isolation         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                LAYER 5: DEBUG & TEXT ANALYSIS                       │   │
│  │                        (ELI5)                                       │   │
│  │                                                                     │   │
│  │  • Pipeline debugging                                              │   │
│  │  • Permutation importance                                          │   │
│  │  • Text feature highlighting                                       │   │
│  │  • Human-readable explanations                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Tab Structure

### Tab 1: Model Performance (YellowBrick)

| Visualization | Library | Purpose |
|---------------|---------|---------|
| Classification Report | YellowBrick | Precision/Recall/F1 heatmap |
| Confusion Matrix | YellowBrick | Class-wise accuracy |
| ROC-AUC Curve | YellowBrick | Discrimination power |
| Precision-Recall Curve | YellowBrick | Imbalanced data performance |
| Class Prediction Error | YellowBrick | Error distribution |
| Discrimination Threshold | YellowBrick | Optimal threshold selection |

### Tab 2: Model Selection (YellowBrick)

| Visualization | Library | Purpose |
|---------------|---------|---------|
| Learning Curve | YellowBrick | Bias-variance tradeoff |
| Validation Curve | YellowBrick | Hyperparameter sensitivity |
| CV Scores | YellowBrick | Cross-validation stability |
| Feature Importances | YellowBrick | Model-based ranking |
| Recursive Feature Elimination | YellowBrick | Optimal feature subset |

### Tab 3: Explainability (SHAP) ⭐ NEW

| Visualization | Library | Purpose |
|---------------|---------|---------|
| Beeswarm Plot | SHAP | Global feature importance distribution |
| Bar Plot | SHAP | Mean absolute SHAP values |
| Waterfall Plot | SHAP | Single prediction breakdown |
| Force Plot | SHAP | Compact prediction explanation |
| Dependence Plot | SHAP | Feature effect on prediction |

### Tab 4: Feature Interactions (SHAP-IQ) ⭐ NEW

| Visualization | Library | Purpose |
|---------------|---------|---------|
| Interaction Bar Plot | SHAP-IQ | Top feature pairs importance |
| Interaction Force Plot | SHAP-IQ | Synergy breakdown for single prediction |
| N-Order Analysis | SHAP-IQ | Higher-order interactions |
| Interaction Heatmap | SHAP-IQ | Pairwise interaction matrix |

### Tab 5: Tree Analysis (dtreeviz) ⭐ NEW

| Visualization | Library | Purpose |
|---------------|---------|---------|
| Decision Tree Viz | dtreeviz | Full tree structure with splits |
| Prediction Path | dtreeviz | How a sample traverses the tree |
| Decision Boundaries | dtreeviz | 2D feature space visualization |
| Leaf Statistics | dtreeviz | Class distribution in leaves |

### Tab 6: Data Exploration (Plotly + YellowBrick)

| Visualization | Library | Purpose |
|---------------|---------|---------|
| Parallel Coordinates | Plotly | Multi-feature relationships (interactive) |
| Class Balance | YellowBrick | Target distribution |
| PCA Projection | YellowBrick + Plotly | Dimensionality reduction |
| Feature Correlation | Plotly heatmap | Feature relationships |

### Tab 7: Partial Dependence (PDPBox) ⭐ NEW

| Visualization | Library | Purpose |
|---------------|---------|---------|
| 1D PDP | PDPBox | Single feature effect |
| 2D PDP | PDPBox | Two-feature interaction effect |
| ICE Plots | PDPBox | Individual conditional expectation |

---

## Why This Combination is Powerful for Fraud Detection

| Question | Library | Answer |
|----------|---------|--------|
| "Is the model accurate?" | YellowBrick | Classification metrics, confusion matrix |
| "Is the model overfitting?" | YellowBrick | Learning curve, validation curve |
| "Which features matter most?" | SHAP | Beeswarm, bar plot |
| "Why was THIS transaction flagged?" | SHAP | Waterfall, force plot |
| "Do features work together for fraud?" | SHAP-IQ | Interaction analysis |
| "How does the XGBoost tree decide?" | dtreeviz | Tree visualization |
| "How does amount affect fraud probability?" | PDPBox | Partial dependence |
| "Can I explore the data interactively?" | Plotly | Parallel coordinates |

---

## Dependencies

### Proposed pyproject.toml additions

```toml
[project.dependencies]
# Core (existing)
yellowbrick = "^1.5"
scikit-learn = "^1.5"
xgboost = "^2.0"
matplotlib = "^3.8"

# Explainability (new)
shap = "^0.50"
shapiq = "^1.3"        # SHAP-IQ

# Tree visualization (new)
dtreeviz = "^2.3"

# Partial dependence (new)
pdpbox = "^0.3"

# Interactive (new)
plotly = "^5.18"

# Optional debugging
eli5 = "^0.13"
```

---

## Library Role Summary

| Library | Role in Stack | Overlap with Others | Unique Value |
|---------|---------------|---------------------|--------------|
| **YellowBrick** | Foundation | Some with sklearn native | Most complete diagnostic suite |
| **SHAP** | Explainability | Superset of feature importance | Individual prediction explanations |
| **SHAP-IQ** | Interactions | Superset of SHAP for interactions | Feature synergy analysis |
| **dtreeviz** | Tree deep-dive | None | Beautiful tree viz + AI |
| **PDPBox** | Effect isolation | Some with SHAP dependence | Clean PDP interface |
| **Plotly** | Interactivity | None | Brushing, filtering |
| **ELI5** | Debugging | Some with SHAP | Text explanations, pipeline debug |

---

## Sources

### Official Documentation
- [YellowBrick Documentation](https://www.scikit-yb.org/)
- [YellowBrick API Reference](https://www.scikit-yb.org/en/latest/api/index.html)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [SHAP GitHub](https://github.com/shap/shap)
- [scikit-learn Visualizations](https://scikit-learn.org/stable/visualizations.html)
- [scikit-plots Documentation](https://scikit-plot.readthedocs.io/en/stable/)
- [scikit-plots GitHub](https://github.com/scikit-plots/scikit-plots)
- [dtreeviz GitHub](https://github.com/parrt/dtreeviz)
- [ELI5 Documentation](https://eli5.readthedocs.io/en/latest/overview.html)
- [sklearn-evaluation GitHub](https://github.com/ploomber/sklearn-evaluation)
- [Plotly Parallel Coordinates](https://plotly.com/python/parallel-coordinates-plot/)

### SHAP-IQ Resources
- [SHAP-IQ GitHub](https://github.com/mmschlk/shapiq)
- [SHAP-IQ Documentation](https://shapiq.readthedocs.io/)
- [SHAP-IQ Tutorial - MarkTechPost](https://www.marktechpost.com/2025/08/03/tutorial-exploring-shap-iq-visualizations/)
- [SHAP-IQ Paper (NeurIPS 2023)](https://arxiv.org/abs/2303.01179)
- [SHAP-IQ NeurIPS Poster](https://neurips.cc/virtual/2023/poster/72134)

### Analysis & Comparisons
- [YellowBrick Health Analysis - Snyk](https://snyk.io/advisor/python/yellowbrick)
- [sklearn TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [SHAP Plot Types - DeepWiki](https://deepwiki.com/shap/shap/6.1-plot-types)

### Tutorials & Articles
- [Introduction to SHAP - Towards Data Science](https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454/)
- [Machine Learning Explainability via ELI5](https://towardsdatascience.com/machine-learning-explainability-introduction-via-eli5-99c767f017e2/)
- [Beautiful Decision Tree Visualizations - KDnuggets](https://www.kdnuggets.com/2021/03/beautiful-decision-tree-visualizations-dtreeviz.html)
- [Interpreting Black Box Models with SHAP](https://medium.com/@akashanandani.56/interpreting-black-box-models-with-shap-a-beginners-guide-c50f45b3161d)

---

## Conclusion

**YellowBrick as the foundation, enhanced with SHAP for explainability, SHAP-IQ for interactions, dtreeviz for tree analysis, and Plotly for interactivity creates the most comprehensive Batch ML visualization system possible.**

This combination provides:
1. **Complete ML diagnostics** (YellowBrick)
2. **Industry-standard explainability** (SHAP)
3. **Deep interaction analysis** (SHAP-IQ)
4. **Beautiful tree visualizations** (dtreeviz)
5. **Interactive exploration** (Plotly)
6. **Feature effect isolation** (PDPBox)
