"""ECCI Batch ML Metrics page - sklearn overview + YellowBrick visualizations."""
import reflex as rx
from ....components import (
    coelho_realtime_navbar,
    page_sub_nav,
    batch_sub_nav,
    batch_ml_run_and_training_box,
)
from ....components.ecci import yellowbrick_ecci_dynamic_info_button
from ....components.shared import metric_info_dialog
from ....states import ECCIState, SharedState


PROJECT_NAME = "E-Commerce Customer Interactions"
BASE_ROUTE = "/ecci"
BATCH_ROUTE = "/ecci/batch"


# =============================================================================
# METRICS DISPLAY COMPONENTS
# =============================================================================
def _metric_card(title: str, value_key: str, description: str, color: str = "purple") -> rx.Component:
    """Build a metric card with title, value, description, and info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(title, size="2", weight="bold"),
                rx.spacer(),
                metric_info_dialog(value_key, "ecci"),
                width="100%",
                align="center"
            ),
            rx.text(
                ECCIState.ecci_batch_metrics.get(value_key, "--"),
                size="6",
                weight="bold",
                color=color
            ),
            rx.text(description, size="1", color="gray"),
            spacing="1",
            align_items="center",
            width="100%"
        ),
        width="100%"
    )


def _kpi_indicator(title: str, value_key: str, color: str = "purple") -> rx.Component:
    """Build a KPI indicator showing a single metric with info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(title, size="1", color="gray"),
                rx.spacer(),
                metric_info_dialog(value_key, "ecci"),
                width="100%",
                align="center"
            ),
            rx.text(
                ECCIState.ecci_batch_metrics.get(value_key, "--"),
                size="5",
                weight="bold",
                color=color
            ),
            spacing="1",
            align_items="center",
            width="100%"
        ),
        width="100%"
    )


# =============================================================================
# YELLOWBRICK CONTENT
# =============================================================================
def _yellowbrick_content(category: str, description: str) -> rx.Component:
    """Build YellowBrick visualization content for a specific category."""
    return rx.cond(
        SharedState.batch_model_available["E-Commerce Customer Interactions"],
        rx.vstack(
            rx.text(description, size="2", color="gray"),
            rx.hstack(
                rx.box(
                    rx.select(
                        ECCIState.yellowbrick_metrics_options[category],
                        value=ECCIState.yellowbrick_metric_name,
                        on_change=lambda v: ECCIState.set_yellowbrick_visualization(category, v),
                        placeholder="Select visualization...",
                        width="100%"
                    ),
                    flex="1"
                ),
                rx.cond(
                    ECCIState.yellowbrick_metric_name != "",
                    yellowbrick_ecci_dynamic_info_button(),
                    rx.fragment()
                ),
                spacing="2",
                align_items="center",
                width="100%"
            ),
            rx.cond(
                ECCIState.yellowbrick_loading,
                rx.vstack(
                    rx.hstack(
                        rx.spinner(size="3"),
                        rx.text("Loading visualization...", size="2", color="gray"),
                        spacing="2",
                        align_items="center",
                    ),
                    rx.button(
                        rx.hstack(
                            rx.icon("square", size=12),
                            rx.text("Stop", size="1"),
                            spacing="1",
                            align_items="center"
                        ),
                        on_click=ECCIState.cancel_yellowbrick_loading,
                        size="1",
                        color_scheme="red",
                        variant="soft"
                    ),
                    spacing="3",
                    align_items="center",
                    justify="center",
                    width="100%",
                    padding="4em"
                ),
                rx.fragment()
            ),
            rx.cond(
                ECCIState.yellowbrick_error != "",
                rx.callout(
                    ECCIState.yellowbrick_error,
                    icon="triangle-alert",
                    color="red",
                    width="100%"
                ),
                rx.fragment()
            ),
            rx.cond(
                ECCIState.yellowbrick_image_base64 != "",
                rx.image(
                    src=f"data:image/png;base64,{ECCIState.yellowbrick_image_base64}",
                    width="100%",
                    alt="YellowBrick visualization"
                ),
                rx.fragment()
            ),
            spacing="4",
            width="100%",
        ),
        rx.callout(
            "Train a model first to view visualizations. Go to the Prediction page and click Train.",
            icon="info",
            color="blue",
            width="100%"
        )
    )


# =============================================================================
# TAB CONTENT FUNCTIONS
# =============================================================================
def _sklearn_overview_tab() -> rx.Component:
    """Build the sklearn metrics overview tab content with clustering metrics."""
    return rx.vstack(
        rx.hstack(
            rx.hstack(
                rx.icon("layout-dashboard", size=20, color=rx.color("accent", 10)),
                rx.heading("Clustering Metrics Overview", size="5", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            rx.button(
                rx.icon("refresh-cw", size=16),
                on_click=SharedState.refresh_batch_mlflow_metrics(PROJECT_NAME),
                size="1",
                variant="ghost",
                cursor="pointer",
                title="Refresh metrics"
            ),
            align_items="center",
            spacing="2",
            width="100%"
        ),
        # Training Data Info
        rx.cond(
            SharedState.batch_training_total_rows["E-Commerce Customer Interactions"] > 0,
            rx.callout(
                rx.hstack(
                    rx.text("Training Data:", size="2", weight="medium"),
                    rx.badge(
                        SharedState.batch_training_total_rows["E-Commerce Customer Interactions"].to(str) + " events",
                        color_scheme="purple",
                        variant="solid",
                        size="2"
                    ),
                    spacing="2",
                    align_items="center"
                ),
                icon="database",
                color="purple",
                variant="soft",
                width="100%"
            ),
            rx.fragment()
        ),
        # Primary Clustering Metrics - KPIs
        rx.hstack(
            rx.icon("target", size=16, color="purple"),
            rx.text("Primary Metrics", size="2", weight="bold"),
            rx.text("(Cluster quality indicators)", size="1", color="gray"),
            spacing="2",
            align_items="center"
        ),
        rx.grid(
            _kpi_indicator("Silhouette Score", "silhouette_score", "purple"),
            _kpi_indicator("Calinski-Harabasz", "calinski_harabasz_score", "blue"),
            _kpi_indicator("Davies-Bouldin", "davies_bouldin_score", "orange"),
            _kpi_indicator("Number of Clusters", "n_clusters", "green"),
            columns="4",
            spacing="2",
            width="100%"
        ),
        rx.divider(size="4", width="100%"),
        # Secondary Metrics
        rx.hstack(
            rx.icon("activity", size=16, color="blue"),
            rx.text("Model Metrics", size="2", weight="bold"),
            rx.text("(KMeans internal metrics)", size="1", color="gray"),
            spacing="2",
            align_items="center"
        ),
        rx.grid(
            _metric_card(
                "Inertia",
                "inertia",
                "Sum of squared distances to cluster centers (lower is better)",
                "blue"
            ),
            _metric_card(
                "Preprocessing Time",
                "preprocessing_time_seconds",
                "Time taken to load and preprocess data",
                "gray"
            ),
            columns="2",
            spacing="2",
            width="100%"
        ),
        rx.divider(size="4", width="100%"),
        # Metric Interpretation Guide
        rx.callout(
            rx.vstack(
                rx.text("**Metric Interpretation Guide:**", weight="bold"),
                rx.text("• **Silhouette Score** [-1, 1]: Higher is better. Measures cluster cohesion and separation."),
                rx.text("• **Calinski-Harabasz** [0, ∞): Higher is better. Ratio of between-cluster to within-cluster variance."),
                rx.text("• **Davies-Bouldin** [0, ∞): Lower is better. Average similarity of each cluster with its most similar cluster."),
                rx.text("• **Inertia** [0, ∞): Lower is better. Sum of squared distances to nearest cluster center."),
                spacing="1",
                align_items="start"
            ),
            icon="info",
            color="purple",
            variant="soft",
            width="100%"
        ),
        spacing="4",
        width="100%",
    )


def _clustering_tab() -> rx.Component:
    """Build the YellowBrick Clustering tab content."""
    return rx.vstack(
        rx.hstack(
            rx.icon("circle-dot", size=20, color=rx.color("accent", 10)),
            rx.heading("Clustering", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Clustering", "Select a clustering visualization to analyze cluster quality and optimal K."),
        spacing="4",
        width="100%",
    )


def _features_tab() -> rx.Component:
    """Build the YellowBrick Feature Analysis tab content."""
    return rx.vstack(
        rx.hstack(
            rx.icon("scatter-chart", size=20, color=rx.color("accent", 10)),
            rx.heading("Feature Analysis", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Feature Analysis", "Select a feature analysis visualization to understand feature relationships."),
        spacing="4",
        width="100%",
    )


def _target_tab() -> rx.Component:
    """Build the YellowBrick Target/Cluster Analysis tab content."""
    return rx.vstack(
        rx.hstack(
            rx.icon("crosshair", size=20, color=rx.color("accent", 10)),
            rx.heading("Target Analysis", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Target", "Select a target analysis visualization to understand cluster distribution."),
        spacing="4",
        width="100%",
    )


def _diagnostics_tab() -> rx.Component:
    """Build the YellowBrick Model Diagnostics tab content."""
    return rx.vstack(
        rx.hstack(
            rx.icon("settings-2", size=20, color=rx.color("accent", 10)),
            rx.heading("Model Diagnostics", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Model Selection", "Select a model selection visualization to analyze model performance."),
        spacing="4",
        width="100%",
    )


def _text_analysis_tab() -> rx.Component:
    """Build the YellowBrick Text Analysis tab content."""
    return rx.vstack(
        rx.hstack(
            rx.icon("text", size=20, color=rx.color("accent", 10)),
            rx.heading("Text Modeling", size="5", weight="bold"),
            spacing="2",
            align_items="center"
        ),
        _yellowbrick_content("Text Analysis", "Select a text analysis visualization to explore search query patterns."),
        rx.callout(
            rx.vstack(
                rx.text("**Data Sampling:** Some visualizations use sampled data for performance:", size="2"),
                rx.text("• FreqDistVisualizer, DispersionPlot, WordCorrelationPlot: Full dataset", size="1", color="gray"),
                rx.text("• TSNEVisualizer, UMAPVisualizer: 2,000 samples (slow)", size="1", color="gray"),
                rx.text("• PosTagVisualizer: 1,000 samples (slow)", size="1", color="gray"),
                spacing="1",
                align_items="start"
            ),
            icon="info",
            color="blue",
            variant="soft",
            size="1",
            width="100%"
        ),
        spacing="4",
        width="100%",
    )


# =============================================================================
# METRICS TABS
# =============================================================================
def _metrics_tabs() -> rx.Component:
    """Build the metrics tabs (sklearn overview + YellowBrick categories)."""
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger("Overview", value="overview", flex="1"),
            rx.tabs.trigger("Clustering", value="clustering", flex="1"),
            rx.tabs.trigger("Feature Analysis", value="features", flex="1"),
            rx.tabs.trigger("Target Analysis", value="target", flex="1"),
            rx.tabs.trigger("Model Diagnostics", value="diagnostics", flex="1"),
            rx.tabs.trigger("Text Modeling", value="text", flex="1"),
            size="2",
            width="100%",
        ),
        rx.tabs.content(
            rx.box(_sklearn_overview_tab(), padding_top="1em"),
            value="overview",
        ),
        rx.tabs.content(
            rx.box(_clustering_tab(), padding_top="1em"),
            value="clustering",
        ),
        rx.tabs.content(
            rx.box(_features_tab(), padding_top="1em"),
            value="features",
        ),
        rx.tabs.content(
            rx.box(_target_tab(), padding_top="1em"),
            value="target",
        ),
        rx.tabs.content(
            rx.box(_diagnostics_tab(), padding_top="1em"),
            value="diagnostics",
        ),
        rx.tabs.content(
            rx.box(_text_analysis_tab(), padding_top="1em"),
            value="text",
        ),
        default_value="overview",
        width="100%",
    )


# =============================================================================
# PAGE INDEX
# =============================================================================
def index() -> rx.Component:
    """ECCI Batch ML Metrics page with sklearn overview and YellowBrick visualizations."""
    return rx.vstack(
        coelho_realtime_navbar(),
        rx.box(
            page_sub_nav(BASE_ROUTE, "batch"),
            padding_x="2em",
            padding_top="1em",
            width="100%"
        ),
        rx.box(
            rx.hstack(
                # Left column - Training controls
                rx.vstack(
                    batch_ml_run_and_training_box("KMeans", PROJECT_NAME),
                    spacing="4",
                    width="30%"
                ),
                # Right column - Metrics tabs
                rx.vstack(
                    batch_sub_nav(BATCH_ROUTE, "metrics"),
                    _metrics_tabs(),
                    align_items="start",
                    spacing="4",
                    width="70%"
                ),
                spacing="6",
                align_items="start",
                width="100%"
            ),
            padding="2em",
            width="100%"
        ),
        on_mount=[
            ECCIState.init_ecci_form_if_empty,
            SharedState.init_batch_page(PROJECT_NAME),
            ECCIState.clear_yellowbrick_visualization,
        ],
        on_unmount=[
            ECCIState.clear_yellowbrick_visualization,
            ECCIState.cleanup_on_page_leave(PROJECT_NAME),
        ],
        spacing="0",
        width="100%"
    )
