# Metrics Tab Redesign - Comprehensive Visualization Stack

> Research completed: January 2025
> Based on: YellowBrick, SHAP, SHAP-IQ, dtreeviz, scikit-plot, PDPBox, sklearn native, InterpretML, Alibi, DALEX, Shapash, ExplainerDashboard, Evidently

---

## Executive Summary

After comprehensive research on 12+ visualization libraries, the optimal Metrics tab structure should be reorganized from **task-based** (current) to **insight-based** categories that answer specific questions data scientists and stakeholders ask.

### Current Structure (Task-Based)
```
Metrics Tab
├── Overview (sklearn metrics cards)
├── Classification (YellowBrick)
├── Feature Analysis (YellowBrick)
├── Target (YellowBrick)
└── Model Selection (YellowBrick)
```

### Proposed Structure (Insight-Based)
```
Metrics Tab
├── Overview (Performance Dashboard)
├── Performance (Classification Diagnostics)
├── Explainability (Why Predictions?)
├── Feature Interactions (How Features Combine?)
├── Data Quality (Is Data OK?)
├── Model Diagnostics (Is Model OK?)
└── What-If Analysis (Scenario Testing)
```

---

## Detailed Tab Design

### Tab 1: Overview (Performance Dashboard)

**Purpose**: Executive summary - quick health check of model performance

**Current**: 16 sklearn metric cards (KPIs, gauges, bullets)

**Proposed Enhancement**: Keep current + add trend sparklines

| Component | Library | Visualization |
|-----------|---------|---------------|
| Primary KPIs | Plotly | Recall, Precision, F1, F-beta indicators with delta |
| Ranking KPIs | Plotly | ROC-AUC, Avg Precision indicators |
| Secondary Gauges | Plotly | Accuracy, Balanced Acc, MCC, Cohen Kappa, Jaccard, G-Mean |
| Calibration Bullets | Plotly | Log Loss, Brier, D2 Log Loss, D2 Brier |
| **NEW: Confusion Matrix Mini** | sklearn | Small confusion matrix heatmap |
| **NEW: Class Balance Bar** | Plotly | Quick class distribution view |

**Design Notes**:
- Keep current layout - it's effective
- Add mini confusion matrix for at-a-glance error understanding
- Info buttons already implemented

---

### Tab 2: Performance (Classification Diagnostics)

**Purpose**: Deep dive into classification performance - "How accurate is the model?"

**Combines**: Current "Classification" tab + sklearn native + scikit-plot enhancements

| Subtab | Library | Visualizations |
|--------|---------|----------------|
| **Confusion Matrix** | YellowBrick | ConfusionMatrix (current) |
| **ROC Curve** | YellowBrick | ROCAUC with multi-class support |
| **Precision-Recall** | YellowBrick | PrecisionRecallCurve (critical for imbalanced data) |
| **Classification Report** | YellowBrick | Heatmap of P/R/F1 per class |
| **Prediction Error** | YellowBrick | ClassPredictionError bar chart |
| **Threshold Tuning** | YellowBrick | DiscriminationThreshold |
| **NEW: Calibration Curve** | sklearn | CalibrationDisplay - probability reliability |
| **NEW: Lift Curve** | scikit-plot | plot_lift_curve - business value |
| **NEW: Cumulative Gain** | scikit-plot | plot_cumulative_gain - targeting efficiency |
| **NEW: KS Statistic** | scikit-plot | plot_ks_statistic - distribution separation |

**New Visualizations Explained**:

1. **Calibration Curve** (sklearn `CalibrationDisplay`)
   - Shows if predicted probabilities match actual outcomes
   - Critical for fraud scoring thresholds
   - "When model says 70% fraud probability, is it actually 70%?"

2. **Lift Curve** (scikit-plot `plot_lift_curve`)
   - Business value metric: "How many times better than random?"
   - Essential for stakeholder communication
   - Shows ROI of using the model

3. **Cumulative Gain** (scikit-plot `plot_cumulative_gain`)
   - "If I review top 20% of predictions, what % of fraud do I catch?"
   - Directly maps to operational capacity

4. **KS Statistic** (scikit-plot `plot_ks_statistic`)
   - Maximum separation between fraud/non-fraud distributions
   - Single number summary of discrimination power

---

### Tab 3: Explainability (SHAP) - NEW

**Purpose**: Answer "Why did the model make this prediction?"

**Library**: SHAP (primary), with SHAP-IQ extension

| Subtab | Function | Scope | Description |
|--------|----------|-------|-------------|
| **Bar Plot** | `shap.plots.bar` | Global | Mean absolute SHAP values - feature ranking |
| **Beeswarm** | `shap.plots.beeswarm` | Global | Distribution of SHAP values colored by feature value |
| **Waterfall** | `shap.plots.waterfall` | Local | Single prediction breakdown - "why this transaction?" |
| **Force Plot** | `shap.plots.force` | Local | Compact additive explanation |
| **Dependence** | `shap.plots.scatter` | Global | Feature value vs SHAP value - non-linear effects |
| **Heatmap** | `shap.plots.heatmap` | Global | Matrix view across samples |
| **Decision Plot** | `shap.plots.decision` | Local | Cumulative SHAP path to prediction |

**Fraud Detection Value**:
- **Bar**: "Amount and account_age are top fraud indicators"
- **Beeswarm**: "High amounts have high SHAP values (push toward fraud)"
- **Waterfall**: "This specific transaction was flagged because amount=9999 (+0.3) and cvv_missing (+0.2)"
- **Dependence**: "Non-linear: fraud probability jumps sharply above $5000"

**Implementation Notes**:
```python
import shap
explainer = shap.TreeExplainer(model)  # Fast for CatBoost/XGBoost
shap_values = explainer(X_test)

# Global
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)

# Local (single prediction)
shap.plots.waterfall(shap_values[0])
shap.plots.force(shap_values[0])
```

---

### Tab 4: Feature Interactions (SHAP-IQ) - NEW

**Purpose**: Answer "How do features work together?"

**Library**: SHAP-IQ (shapiq)

| Subtab | Function | Description |
|--------|----------|-------------|
| **Interaction Bar** | `shapiq.plot.bar` | Top feature pairs by interaction strength |
| **SI Graph** | `shapiq.plot.si_graph` | Network visualization of feature synergies |
| **Interaction Heatmap** | Custom | Pairwise interaction matrix |
| **UpSet Plot** | `shapiq.plot.upset` | Set intersections of interacting features |
| **N-Order Analysis** | `shapiq.plot.stacked_bar` | 1st, 2nd, 3rd order contributions |

**Fraud Detection Value**:
- "Amount + Account_Age interaction: New accounts with high amounts are 5x more suspicious"
- "CVV_Missing + Foreign_Currency synergy detected"
- Reveals fraud patterns that single features miss

**Implementation Notes**:
```python
import shapiq
explainer = shapiq.TreeExplainer(model, index="k-SII", max_order=2)
interaction_values = explainer(X_test)

# Visualize
shapiq.plot.bar(interaction_values)
shapiq.plot.si_graph(interaction_values)
```

---

### Tab 5: Feature Analysis (Enhanced)

**Purpose**: Understand feature distributions and relationships

**Combines**: Current YellowBrick + PDPBox + Plotly enhancements

| Subtab | Library | Description |
|--------|---------|-------------|
| **Rank1D** | YellowBrick | Single feature ranking (Shapiro) |
| **Rank2D** | YellowBrick | Correlation matrix heatmap |
| **PCA** | YellowBrick | 2D principal component projection |
| **Manifold (t-SNE)** | YellowBrick | Non-linear 2D projection |
| **Parallel Coordinates** | Plotly | Interactive multi-feature view (UPGRADE) |
| **RadViz** | YellowBrick | Radial feature visualization |
| **JointPlot** | YellowBrick | 2-feature correlation |
| **NEW: 1D PDP** | PDPBox | Single feature partial dependence |
| **NEW: 2D PDP** | PDPBox | Two-feature interaction PDP |
| **NEW: ICE Plots** | sklearn | Individual Conditional Expectation |

**Upgrade: Parallel Coordinates with Plotly**
- Current: YellowBrick (static image)
- Proposed: Plotly `px.parallel_coordinates` (interactive, brushing, filtering)
- Enables: "Select fraud cases and see their feature patterns"

**New: Partial Dependence Plots (PDPBox)**
- Shows marginal effect of features on predictions
- 2D PDP reveals interaction effects
- Complements SHAP dependence plots

---

### Tab 6: Target Analysis (Enhanced)

**Purpose**: Understand target distribution and feature-target relationships

**Combines**: Current YellowBrick + Evidently data quality

| Subtab | Library | Description |
|--------|---------|-------------|
| **Class Balance** | YellowBrick | Target class distribution |
| **Feature Correlation (MI)** | YellowBrick | Mutual information with target |
| **Feature Correlation (Pearson)** | YellowBrick | Linear correlation with target |
| **Balanced Binning** | YellowBrick | Quantile-based binning reference |
| **NEW: Target Drift** | Evidently | Compare train vs test target distribution |
| **NEW: Feature-Target Plots** | PDPBox | TargetPlot per feature |

---

### Tab 7: Model Diagnostics (Enhanced)

**Purpose**: Answer "Is the model healthy? Overfitting? Needs more data?"

**Combines**: Current YellowBrick Model Selection + sklearn + dtreeviz

| Subtab | Library | Description |
|--------|---------|-------------|
| **Feature Importances** | YellowBrick | Model-based feature ranking |
| **Learning Curve** | YellowBrick | Performance vs training size |
| **Validation Curve** | YellowBrick | Performance vs hyperparameter |
| **CV Scores** | YellowBrick | Cross-validation stability |
| **RFECV** | YellowBrick | Recursive feature elimination |
| **Dropping Curve** | YellowBrick | Feature robustness |
| **NEW: Tree Visualization** | dtreeviz | Single tree structure (for ensemble sample) |
| **NEW: Prediction Path** | dtreeviz | How a sample traverses the tree |
| **NEW: Decision Boundaries** | sklearn | 2D feature space visualization |

**New: dtreeviz Integration**
```python
import dtreeviz
# For CatBoost, extract single tree
viz = dtreeviz.model(
    model.get_booster(),
    X_train, y_train,
    feature_names=feature_names,
    class_names=["Non-Fraud", "Fraud"]
)
viz.view()  # Full tree
viz.explain_prediction_path(X_test[0])  # Single prediction
```

---

### Tab 8: What-If Analysis - NEW (Future)

**Purpose**: Interactive scenario testing - "What if I change this feature?"

**Library**: ExplainerDashboard or Custom Reflex

| Component | Description |
|-----------|-------------|
| **Feature Editor** | Modify input values interactively |
| **Live Prediction** | See prediction update in real-time |
| **Counterfactual** | "What minimal change would flip the prediction?" |
| **Sensitivity Analysis** | How sensitive is prediction to each feature? |

**Implementation Options**:
1. **ExplainerDashboard** - Ready-made, but separate dashboard
2. **Custom Reflex** - Integrated, uses existing form + SHAP

**Fraud Detection Value**:
- "If account_age was 365 days instead of 1 day, would this still be flagged?"
- "What's the minimum amount that would trigger fraud alert?"

---

## Library Comparison Matrix

| Library | Strengths | Weaknesses | Best For |
|---------|-----------|------------|----------|
| **YellowBrick** | Unified API, complete diagnostics | Inactive maintenance | Foundation, model selection |
| **SHAP** | Industry standard, TreeExplainer fast | Memory intensive | Global/local explainability |
| **SHAP-IQ** | Unique interaction analysis | Newer, less docs | Feature synergies |
| **dtreeviz** | Beautiful tree viz, AI chat | Trees only | Ensemble inspection |
| **scikit-plot** | Drop-in simple | Limited scope | Quick metrics |
| **PDPBox** | Clean PDP interface | Slower for large data | Effect isolation |
| **sklearn native** | Zero dependencies, from_predictions | Basic styling | Core metrics |
| **Plotly** | Interactive, brushing | Setup complexity | Parallel coords |
| **Evidently** | Production monitoring | Overkill for batch | Data drift |

---

## Proposed Dependencies

```toml
[project.dependencies]
# Core (existing)
yellowbrick = "^1.5"
scikit-learn = "^1.5"
catboost = "^1.2"
matplotlib = "^3.8"
plotly = "^5.18"

# Explainability (new)
shap = "^0.50"
shapiq = "^1.3"

# Tree visualization (new)
dtreeviz = "^2.3"

# Partial dependence (new)
pdpbox = "^0.3"

# Quick metrics (new)
scikit-plot = "^0.3"

# Optional: Production monitoring
# evidently = "^0.4"
```

---

## Implementation Priority

| Phase | Tab | Effort | Value |
|-------|-----|--------|-------|
| **Phase 1** | Overview (current) | Done | High |
| **Phase 1** | Performance (enhanced) | Medium | High |
| **Phase 2** | Explainability (SHAP) | Medium | Very High |
| **Phase 2** | Feature Analysis (enhanced) | Low | Medium |
| **Phase 3** | Feature Interactions (SHAP-IQ) | Medium | High |
| **Phase 3** | Model Diagnostics (enhanced) | Medium | Medium |
| **Phase 4** | What-If Analysis | High | Medium |

---

## Visual Tab Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BATCH ML METRICS                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Overview] [Performance] [Explainability] [Interactions] [Features]        │
│  [Target] [Diagnostics] [What-If]                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─── Overview ─────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                        │  │
│  │  │ Recall │ │Precision│ │   F1   │ │ F-beta │  <- Primary KPIs      │  │
│  │  │  0.93  │ │  0.47  │ │  0.62  │ │  0.78  │                        │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘                        │  │
│  │                                                                       │  │
│  │  ┌────────┐ ┌────────┐                                               │  │
│  │  │ROC-AUC │ │Avg Prec│  <- Ranking KPIs                             │  │
│  │  │  0.99  │ │  0.85  │                                               │  │
│  │  └────────┘ └────────┘                                               │  │
│  │                                                                       │  │
│  │  [Gauge] [Gauge] [Gauge] [Gauge] [Gauge] [Gauge]  <- Secondary      │  │
│  │                                                                       │  │
│  │  [Bullet] [Bullet] [Bullet] [Bullet]  <- Calibration                │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─── Explainability (SHAP) ────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  [Bar Plot ▼] [ℹ️]                                                    │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                                                                 │ │  │
│  │  │  amount          ████████████████████████████████████  0.45    │ │  │
│  │  │  account_age     ████████████                          0.18    │ │  │
│  │  │  cvv_provided    ██████████                            0.15    │ │  │
│  │  │  billing_match   ████████                              0.12    │ │  │
│  │  │  hour            ████                                  0.05    │ │  │
│  │  │                                                                 │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Questions for User Before Implementation

1. **Priority**: Which tabs are most important for your use case?
   - Explainability (SHAP) for model interpretation?
   - Feature Interactions (SHAP-IQ) for fraud pattern discovery?
   - What-If Analysis for scenario testing?

2. **Interactive vs Static**:
   - Should Parallel Coordinates be upgraded to interactive Plotly?
   - Is What-If Analysis needed in Phase 1?

3. **Dependencies**:
   - OK to add ~5 new Python packages (shap, shapiq, dtreeviz, pdpbox, scikit-plot)?

4. **Performance**:
   - SHAP computation can be slow for large datasets. Cache strategy?
   - t-SNE/Manifold already slow - acceptable?

---

## Sources

### Core Libraries
- [YellowBrick Documentation](https://www.scikit-yb.org/)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [SHAP-IQ Documentation](https://shapiq.readthedocs.io/)
- [dtreeviz GitHub](https://github.com/parrt/dtreeviz)
- [scikit-plot Documentation](https://scikit-plot.readthedocs.io/)
- [PDPBox Documentation](https://pdpbox.readthedocs.io/)
- [sklearn Visualizations](https://scikit-learn.org/stable/visualizations.html)

### XAI Platforms (Reference)
- [InterpretML](https://interpret.ml/)
- [Alibi Explain](https://docs.seldon.ai/alibi-explain/)
- [DALEX Python](https://dalex.drwhy.ai/python/)
- [Shapash Documentation](https://shapash.readthedocs.io/)
- [ExplainerDashboard](https://explainerdashboard.readthedocs.io/)

### Monitoring
- [Evidently AI](https://www.evidentlyai.com/)
- [Evidently GitHub](https://github.com/evidentlyai/evidently)
