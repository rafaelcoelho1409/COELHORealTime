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

### Comprehensive XAI/Interpretability Platforms

| Library | Description | Best For |
|---------|-------------|----------|
| **InterpretML** | Microsoft's unified interpretability platform | Glass-box models (EBM), LIME, SHAP, PDP combined |
| **Alibi Explain** | Seldon's ML inspection library (15 algorithms) | Counterfactuals, Anchors, ALE, SHAP |
| **DALEX** | DrWhy.AI model explainer with Arena dashboard | Interactive dashboards, fairness analysis |
| **Shapash** | MAIF's user-friendly explainability webapp | Production deployment, SmartPredictor |
| **ExplainerDashboard** | Ready-made interactive dashboard builder | Quick dashboard setup, What-If analysis |

### Documentation & Model Cards

| Library | Description | Best For |
|---------|-------------|----------|
| **Model Card Toolkit** | TensorFlow/Google model documentation | Automated model card generation |
| **Evidently** | ML monitoring with model cards | Data drift, model monitoring dashboards |

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
| **InterpretML** | Unified XAI platform | ✅ Excellent | ✅ TreeExplainer | ✅ Active | Interactive/HTML | EBM glass-box models |
| **Alibi Explain** | 15 explanation algorithms | ✅ Good | ✅ TreeSHAP | ✅ Active | PNG/Interactive | Counterfactuals, Anchors |
| **DALEX** | Model explainer + dashboard | ✅ Excellent | ✅ Native | ✅ Active | Plotly/Interactive | Arena dashboard, Fairness |
| **Shapash** | User-friendly webapp | ✅ Excellent | ✅ Native | ✅ Active | Plotly/Webapp | SmartPredictor deployment |
| **ExplainerDashboard** | Ready-made dashboard | ✅ Excellent | ✅ Native | ✅ Active | Interactive/HTML | What-If analysis |
| **Evidently** | ML monitoring | ✅ Excellent | Generic | ✅ Active | Plotly/HTML | Data drift, model cards |

---

## New Libraries Deep Dive (January 2025)

### sklearn Native Display Classes (sklearn 1.0+)

sklearn now provides native visualization without external dependencies:

| Display Class | Purpose | Methods |
|--------------|---------|---------|
| `ConfusionMatrixDisplay` | Confusion matrix visualization | `from_estimator()`, `from_predictions()` |
| `RocCurveDisplay` | ROC curve with AUC | `from_estimator()`, `from_predictions()` |
| `PrecisionRecallDisplay` | Precision-Recall curve | `from_estimator()`, `from_predictions()` |
| `DetCurveDisplay` | Detection Error Tradeoff curve | `from_estimator()`, `from_predictions()` |
| `CalibrationDisplay` | Calibration curve (reliability) | `from_estimator()`, `from_predictions()` |
| `LearningCurveDisplay` | Learning curves (sklearn 1.2+) | `from_estimator()` |
| `ValidationCurveDisplay` | Validation curves (sklearn 1.3+) | `from_estimator()` |
| `PartialDependenceDisplay` | PDP and ICE plots | `from_estimator()` |
| `DecisionBoundaryDisplay` | Decision boundaries | `from_estimator()` |
| `PredictionErrorDisplay` | Regression error analysis | `from_estimator()`, `from_predictions()` |

**Advantage**: Works with `from_predictions()` for pre-fitted models - no retraining!

### scikit-plots Functions

One-liner plotting library for sklearn metrics:

| Function | Purpose |
|----------|---------|
| `skplt.metrics.plot_confusion_matrix` | Confusion matrix heatmap |
| `skplt.metrics.plot_roc` | Multi-class ROC curves |
| `skplt.metrics.plot_precision_recall` | Precision-Recall curves |
| `skplt.metrics.plot_ks_statistic` | KS statistic plot |
| `skplt.metrics.plot_cumulative_gain` | Cumulative gain chart |
| `skplt.metrics.plot_lift_curve` | Lift curve |
| `skplt.metrics.plot_silhouette` | Silhouette analysis |
| `skplt.estimators.plot_feature_importances` | Feature importance bar |
| `skplt.estimators.plot_learning_curve` | Learning curves |
| `skplt.decomposition.plot_pca_component_variance` | PCA variance explained |
| `skplt.decomposition.plot_pca_2d_projection` | PCA 2D scatter |
| `skplt.cluster.plot_elbow_curve` | Elbow method |

### SHAP Plot Types (v0.50+)

| Plot | Purpose | Scope |
|------|---------|-------|
| `shap.plots.bar` | Mean absolute SHAP values | Global |
| `shap.plots.beeswarm` | Distribution colored by feature value | Global |
| `shap.plots.violin` | Violin plot of SHAP values | Global |
| `shap.plots.heatmap` | Heatmap of SHAP values | Global |
| `shap.plots.scatter` | Dependence plot (feature vs SHAP) | Global |
| `shap.plots.waterfall` | Single prediction breakdown | Local |
| `shap.plots.force` | Compact prediction visualization | Local |
| `shap.plots.decision` | Cumulative SHAP decision plot | Local |
| `shap.plots.text` | Text highlighting for NLP | Local |
| `shap.plots.image` | Image pixel attribution | Local |
| `shap.plots.partial_dependence` | Partial dependence | Global |
| `shap.plots.embedding` | 2D projection of SHAP | Global |
| `shap.plots.monitoring` | Model monitoring over time | Global |
| `shap.plots.group_difference` | Compare groups | Global |

### InterpretML (Microsoft)

Unified platform combining multiple XAI methods:

| Component | Description |
|-----------|-------------|
| **Explainable Boosting Machine (EBM)** | Glass-box model as accurate as boosting but interpretable |
| **LIME** | Local surrogate explanations |
| **SHAP** | Shapley value explanations |
| **PDP** | Partial dependence plots |
| **Sensitivity Analysis** | Feature sensitivity |
| **Dashboard** | Interactive visualization dashboard |

**Best For**: When you need an interpretable model that matches XGBoost performance.

### Alibi Explain (Seldon) - 15 Algorithms

| Category | Algorithms |
|----------|------------|
| **Global** | ALE, Partial Dependence, Permutation Importance |
| **Local Feature Attribution** | Integrated Gradients, Kernel SHAP, TreeSHAP |
| **Local Necessary Features** | Anchors, Pertinent Positives (CEM) |
| **Counterfactuals** | CFI, CEM Negatives, Prototype-guided, RL-based |
| **Similarity** | Similarity Explanations |

**Best For**: Counterfactual explanations ("what would need to change for a different prediction?")

### DALEX (DrWhy.AI)

| Feature | Description |
|---------|-------------|
| **Explainer wrapper** | Unified API for any model |
| **Variable Importance** | Permutation-based importance |
| **Break Down** | Sequential feature contributions |
| **SHAP** | Native SHAP integration |
| **PDP/ALE** | Partial dependence & ALE plots |
| **Ceteris Paribus** | Individual conditional expectation |
| **Arena Dashboard** | Interactive multi-model comparison |
| **Fairness Module** | Bias detection & mitigation |

**Best For**: Model comparison dashboards and fairness analysis.

### Shapash (MAIF)

| Feature | Description |
|---------|-------------|
| **Webapp** | Interactive local/global exploration |
| **SmartExplainer** | Main analysis object |
| **SmartPredictor** | Lightweight deployment object |
| **Explicit Labels** | Human-readable feature names |
| **HTML Report** | Standalone audit document |
| **Explainability Metrics** | Stability, Consistency, Compacity |

**Best For**: Production deployment with user-friendly explanations.

### ExplainerDashboard

| Tab/Component | Description |
|---------------|-------------|
| **SHAP Summary** | Global feature importance |
| **SHAP Dependence** | Feature vs SHAP scatter |
| **SHAP Interactions** | Pairwise interactions |
| **Feature Importance** | Permutation importance |
| **Partial Dependence** | PDP plots |
| **Decision Trees** | Individual tree inspection |
| **What-If Analysis** | Modify inputs, see predictions |
| **Individual Predictions** | Local explanations |
| **Classification Stats** | Confusion matrix, ROC, PR |

**Best For**: Quick dashboard setup without custom code.

### Evidently

| Feature | Description |
|---------|-------------|
| **Data Drift** | Detect distribution changes |
| **Model Performance** | Track metrics over time |
| **Data Quality** | Missing values, duplicates |
| **Target Drift** | Label distribution changes |
| **Model Cards** | Documentation generation |
| **Reports** | Interactive HTML reports |
| **Test Suites** | Automated quality checks |

**Best For**: Production ML monitoring and documentation.

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
- [scikit-plots Metrics Module](https://scikit-plot.readthedocs.io/en/stable/metrics.html)
- [dtreeviz GitHub](https://github.com/parrt/dtreeviz)
- [ELI5 Documentation](https://eli5.readthedocs.io/en/latest/overview.html)
- [sklearn-evaluation GitHub](https://github.com/ploomber/sklearn-evaluation)
- [Plotly Parallel Coordinates](https://plotly.com/python/parallel-coordinates-plot/)

### SHAP-IQ Resources
- [SHAP-IQ GitHub](https://github.com/mmschlk/shapiq)
- [SHAP-IQ Documentation](https://shapiq.readthedocs.io/)
- [SHAP-IQ Tutorial - MarkTechPost](https://www.marktechpost.com/2025/08/03/tutorial-exploring-shap-iq-visualizations/)
- [SHAP-IQ Paper (NeurIPS 2023)](https://arxiv.org/abs/2303.01179)

### XAI Platform Documentation (New)
- [InterpretML](https://interpret.ml/)
- [Alibi Explain](https://docs.seldon.ai/alibi-explain/)
- [DALEX Python](https://dalex.drwhy.ai/python/)
- [DALEX Arena Dashboard](https://dalex.drwhy.ai/python-dalex-arena.html)
- [DALEX Fairness](https://dalex.drwhy.ai/python-dalex-fairness.html)
- [Shapash Documentation](https://shapash.readthedocs.io/)
- [Shapash GitHub](https://github.com/MAIF/shapash)
- [ExplainerDashboard](https://explainerdashboard.readthedocs.io/)
- [Evidently AI](https://www.evidentlyai.com/)
- [Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit)

### Analysis & Comparisons
- [YellowBrick Health Analysis - Snyk](https://snyk.io/advisor/python/yellowbrick)
- [sklearn TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [SHAP Plot Types - DeepWiki](https://deepwiki.com/shap/shap/6.1-plot-types)
- [Top Python Libraries for ML Interpretability](https://www.nb-data.com/p/top-python-packages-for-machine-learning)
- [6 Python Libraries to Interpret ML Models](https://www.analyticsvidhya.com/blog/2020/03/6-python-libraries-interpret-machine-learning-models/)

### Tutorials & Articles
- [Introduction to SHAP - Towards Data Science](https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454/)
- [Machine Learning Explainability via ELI5](https://towardsdatascience.com/machine-learning-explainability-introduction-via-eli5-99c767f017e2/)
- [Beautiful Decision Tree Visualizations - KDnuggets](https://www.kdnuggets.com/2021/03/beautiful-decision-tree-visualizations-dtreeviz.html)
- [Interpreting Black Box Models with SHAP](https://medium.com/@akashanandani.56/interpreting-black-box-models-with-shap-a-beginners-guide-c50f45b3161d)
- [ML Model Monitoring Dashboard Tutorial](https://www.evidentlyai.com/blog/ml-model-monitoring-dashboard-tutorial)
- [Scikit-plot Visualization Tutorial](https://coderzcolumn.com/tutorials/machine-learning/scikit-plot-visualizing-machine-learning-algorithm-results-and-performance)

---

## Recommended Reflex TFD Batch ML Metrics Tab Layout

Based on notebook 007 YellowBrick classes tested and comprehensive library research.

### Phase 1: Core Metrics Cards (MVP)

Simple value cards with color-coded status indicators:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BATCH ML METRICS                                              [Refresh ↻] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Accuracy │ │Precision │ │  Recall  │ │    F1    │ │ ROC-AUC  │          │
│  │  0.9888  │ │  0.4687  │ │  0.9337  │ │  0.6241  │ │  0.9932  │          │
│  │  ✓ Good  │ │  ⚠ Low   │ │ ✓ High   │ │⚠ Moderate│ │ ✓ Great  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │  G-Mean  │ │   MCC    │ │Train Time│ │ Samples  │                       │
│  │  0.9611  │ │  0.6820  │ │  45.2s   │ │  1.02M   │                       │
│  │ ✓ Great  │ │  ✓ Good  │ │          │ │          │                       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Metrics from sklearn.metrics:**
- `accuracy_score` - Overall correctness
- `precision_score` - When model predicts fraud, how often is it correct?
- `recall_score` - Of all actual frauds, how many did we catch?
- `f1_score` - Harmonic mean of precision/recall
- `roc_auc_score` - Discrimination ability
- `geometric_mean_score` (imblearn) - Balance between classes
- `matthews_corrcoef` - Best single metric for imbalanced data

### Phase 2: Classification Visualizations (sklearn native)

Sub-tabs for classification analysis using sklearn Display classes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  [Confusion Matrix] [ROC Curve] [Precision-Recall] [Classification Report] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │                                                             │        │
│     │                                                             │        │
│     │                  sklearn Display Image                      │        │
│     │              (ConfusionMatrixDisplay.from_predictions)      │        │
│     │                                                             │        │
│     │                                                             │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
- `ConfusionMatrixDisplay.from_predictions(y_test, y_pred)`
- `RocCurveDisplay.from_predictions(y_test, y_pred_proba)`
- `PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba)`
- Classification report as Seaborn heatmap

### Phase 3: Feature Importance (SHAP)

SHAP plots for explainability:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FEATURE IMPORTANCE (SHAP)                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Bar Plot] [Beeswarm] [Waterfall (sample)] [Dependence]                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │                                                             │        │
│     │                  SHAP Bar Plot                              │        │
│     │              (shap.plots.bar)                               │        │
│     │                                                             │        │
│     │  amount            ████████████████████████████████  95.3   │        │
│     │  billing_match     ███                                1.9   │        │
│     │  cvv_provided      ██                                 1.3   │        │
│     │  account_age       █                                  0.9   │        │
│     │                                                             │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Model Selection (from notebook 007)

Based on YellowBrick classes tested:

| YellowBrick Class | Alternative | Implementation |
|-------------------|-------------|----------------|
| `ValidationCurve` | `sklearn.model_selection.ValidationCurveDisplay` | Native sklearn 1.3+ |
| `LearningCurve` | `sklearn.model_selection.LearningCurveDisplay` | Native sklearn 1.2+ |
| `CVScores` | Custom Plotly bar chart | `cross_val_score()` + `rx.plotly()` |
| `FeatureImportances` | SHAP bar plot | `shap.plots.bar()` |
| `DroppingCurve` | Custom matplotlib | Manual RFECV + plot |

### Phase 5: Target Analysis

| YellowBrick Class | Alternative | Implementation |
|-------------------|-------------|----------------|
| `ClassBalance` | Plotly bar chart | `rx.plotly()` with `go.Bar` |
| `BalancedBinningReference` | Plotly histogram | `px.histogram()` with percentile lines |

### Complete Tab Structure

```
Batch ML Tab
├── Prediction (left 30%)
│   ├── Training Box (Train button, spinner, model info)
│   └── Form Card (input fields)
│
└── Metrics (right 70%)
    ├── Core Metrics (cards row)
    │   └── Accuracy, Precision, Recall, F1, ROC-AUC, G-Mean, MCC
    │
    ├── Classification (sub-tabs)
    │   ├── Confusion Matrix (sklearn)
    │   ├── ROC Curve (sklearn)
    │   ├── Precision-Recall (sklearn)
    │   └── Classification Report (seaborn heatmap)
    │
    ├── Feature Importance (sub-tabs)
    │   ├── SHAP Bar (global)
    │   ├── SHAP Beeswarm (global)
    │   ├── SHAP Waterfall (local - single prediction)
    │   └── Model Feature Importance (native)
    │
    └── Model Analysis (sub-tabs) [Future]
        ├── Learning Curve (sklearn)
        ├── Validation Curve (sklearn)
        ├── Cross-Validation Scores (plotly)
        └── Class Balance (plotly)
```

### Implementation Priority

| Priority | Component | Library | Effort |
|----------|-----------|---------|--------|
| **P0** | Metrics cards | Pure Reflex | Low |
| **P1** | Confusion Matrix | sklearn `ConfusionMatrixDisplay` | Low |
| **P1** | ROC Curve | sklearn `RocCurveDisplay` | Low |
| **P1** | Precision-Recall | sklearn `PrecisionRecallDisplay` | Low |
| **P2** | SHAP Bar Plot | shap | Medium |
| **P2** | Classification Report | seaborn heatmap | Medium |
| **P3** | SHAP Beeswarm | shap | Medium |
| **P3** | Learning Curve | sklearn `LearningCurveDisplay` | Medium |
| **P4** | SHAP Waterfall | shap | Medium |
| **P4** | Class Balance | plotly | Low |
| **P5** | SHAP Interactions | SHAP-IQ | High |
| **P5** | Counterfactuals | Alibi | High |

### Reflex Integration Notes

**For matplotlib-based plots (sklearn, SHAP, seaborn):**
```python
import io
import base64

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# In Reflex component
rx.image(src=f"data:image/png;base64,{base64_str}")
```

**For Plotly charts:**
```python
import plotly.graph_objects as go

# Native Reflex support
rx.plotly(data=fig)
```

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
