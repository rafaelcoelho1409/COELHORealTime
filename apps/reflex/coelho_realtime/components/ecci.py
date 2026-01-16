"""
E-Commerce Customer Interactions (ECCI) components module.

This module contains all ECCI-specific UI components:
- e_commerce_customer_interactions_form() - Customer interaction form
- e_commerce_customer_interactions_metrics() - Clustering metrics dashboard
- ecci_map() - Customer location map
- ECCI-specific helper components

All components use ECCIState for ECCI-specific state.
"""
import reflex as rx
from ..states import ECCIState, SharedState
from .shared import metric_info_dialog, ml_training_switch, batch_ml_training_box


# =============================================================================
# ECCI Card Helper Functions
# =============================================================================
def ecci_metric_card(label: str, value_var, metric_key: str) -> rx.Component:
    """Create a metric card with info button for ECCI."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(label, size="2", weight="bold", color="gray"),
                rx.spacer(),
                metric_info_dialog(metric_key, "ecci"),
                width="100%",
                align="center"
            ),
            rx.text(value_var, size="5", weight="bold"),
            spacing="1",
            align="center",
            width="100%"
        ),
        size="1"
    )


def ecci_kpi_card_with_info(plotly_key: str, metric_key: str) -> rx.Component:
    """Create a KPI card with Plotly chart and info button for ECCI."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.spacer(),
                metric_info_dialog(metric_key, "ecci"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=ECCIState.ecci_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1"
    )


def ecci_gauge_card_with_info(plotly_key: str, metric_key: str) -> rx.Component:
    """Create a gauge card with Plotly chart and info button for ECCI."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.spacer(),
                metric_info_dialog(metric_key, "ecci"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=ECCIState.ecci_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1",
        width="50%"
    )


def mlflow_run_info_badge(project_name: str) -> rx.Component:
    """Display MLflow experiment run info (run_id, status, start_time) for a project."""
    run_info = ECCIState.mlflow_run_info[project_name]
    return rx.hstack(
        # MLflow source badge
        rx.badge(
            rx.hstack(
                rx.icon("flask-conical", size=12),
                rx.text("MLflow", size="1", weight="bold"),
                spacing="1",
                align="center"
            ),
            color_scheme="purple",
            variant="soft"
        ),
        # Status badge with conditional styling
        rx.cond(
            run_info["is_live"],
            rx.badge(
                rx.hstack(
                    rx.box(
                        width="8px",
                        height="8px",
                        border_radius="50%",
                        background_color="green",
                        class_name="animate-pulse"
                    ),
                    rx.text("LIVE", size="1"),
                    spacing="1",
                    align="center"
                ),
                color_scheme="green",
                variant="surface"
            ),
            rx.badge(
                run_info["status"],
                color_scheme=rx.cond(
                    run_info["status"] == "FINISHED",
                    "blue",
                    "gray"
                ),
                variant="surface"
            )
        ),
        # Run ID
        rx.cond(
            run_info["run_id"] != "",
            rx.hstack(
                rx.text("Run:", size="1", color="gray"),
                rx.code(run_info["run_id"], size="1"),
                spacing="1",
                align="center"
            ),
            rx.fragment()
        ),
        # Start time
        rx.cond(
            run_info["start_time"] != "",
            rx.hstack(
                rx.text("Started:", size="1", color="gray"),
                rx.text(run_info["start_time"], size="1"),
                spacing="1",
                align="center"
            ),
            rx.fragment()
        ),
        spacing="3",
        align="center",
        padding="2"
    )


# =============================================================================
# ECCI Map Component
# =============================================================================
def ecci_map() -> rx.Component:
    """
    Map component for ECCI page showing customer location.
    Uses Folium with OpenStreetMap tiles embedded via iframe.
    """
    return rx.box(
        rx.html(ECCIState.ecci_folium_map_html),
        width="100%",
        height="250px",
        overflow="hidden",
        border_radius="8px",
    )


# =============================================================================
# ECCI Metrics Component
# =============================================================================
def e_commerce_customer_interactions_metrics() -> rx.Component:
    """Display MLflow clustering metrics for ECCI with Plotly dashboard layout."""
    return rx.vstack(
        # Run info badge
        mlflow_run_info_badge("E-Commerce Customer Interactions"),
        # ROW 1: KPI Indicators (primary metrics with delta from baseline)
        rx.grid(
            ecci_kpi_card_with_info("kpi_silhouette", "silhouette"),
            ecci_kpi_card_with_info("kpi_rolling_silhouette", "rolling_silhouette"),
            ecci_kpi_card_with_info("kpi_n_clusters", "n_clusters"),
            ecci_kpi_card_with_info("kpi_n_micro_clusters", "n_micro_clusters"),
            columns="4",
            spacing="2",
            width="100%"
        ),
        # ROW 2: Rolling metrics (text cards with info buttons)
        rx.grid(
            ecci_metric_card("Silhouette", ECCIState.ecci_metrics["silhouette"], "silhouette"),
            ecci_metric_card("Rolling Silhouette", ECCIState.ecci_metrics["rolling_silhouette"], "rolling_silhouette"),
            ecci_metric_card("Time Rolling Silhouette", ECCIState.ecci_metrics["time_rolling_silhouette"], "time_rolling_silhouette"),
            columns="3",
            spacing="2",
            width="100%"
        ),
        # ROW 3: Gauge + Cluster Stats (both as Plotly indicators, same height)
        rx.hstack(
            ecci_gauge_card_with_info("gauge_silhouette", "silhouette"),
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.text("Cluster Statistics", size="2", weight="bold", color="gray"),
                        metric_info_dialog("n_clusters", "ecci"),
                        width="100%",
                        justify="end"
                    ),
                    rx.plotly(data=ECCIState.ecci_dashboard_figures["cluster_stats"], width="100%"),
                    spacing="0",
                    width="100%"
                ),
                size="1",
                width="50%"
            ),
            spacing="2",
            width="100%"
        ),
        spacing="3",
        width="100%"
    )


# =============================================================================
# ECCI Form Component
# =============================================================================
def e_commerce_customer_interactions_form(model_key: str = None, project_name: str = None) -> rx.Component:
    """Customer interaction form for E-Commerce clustering prediction."""
    # Build form card with 3-column layout for compact display
    form_card = rx.card(
        rx.vstack(
            # Form Legend
            rx.hstack(
                rx.icon("shopping-cart", size=20, color=rx.color("accent", 10)),
                rx.heading("Customer Interaction", size="4", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            # Predict button at TOP for visibility
            rx.button(
                "Predict",
                on_click=ECCIState.predict_ecci,
                size="2",
                width="100%",
                disabled=~ECCIState.incremental_model_available["E-Commerce Customer Interactions"]
            ),
            # Randomize button below Predict
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size=14),
                    rx.text("Randomize All Fields", size="2"),
                    spacing="1",
                    align_items="center"
                ),
                on_click=ECCIState.randomize_ecci_form,
                variant="soft",
                color_scheme="blue",
                size="2",
                width="100%"
            ),
            rx.divider(),
            # Form fields in 3-column grid
            rx.grid(
                # Browser
                rx.vstack(
                    rx.text("Browser", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["browser"],
                        value=ECCIState.ecci_form_data.get("browser", ""),
                        on_change=lambda v: ECCIState.update_ecci("browser", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Device Type
                rx.vstack(
                    rx.text("Device", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["device_type"],
                        value=ECCIState.ecci_form_data.get("device_type", ""),
                        on_change=lambda v: ECCIState.update_ecci("device_type", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # OS
                rx.vstack(
                    rx.text("OS", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["os"],
                        value=ECCIState.ecci_form_data.get("os", ""),
                        on_change=lambda v: ECCIState.update_ecci("os", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Event Type
                rx.vstack(
                    rx.text("Event Type", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["event_type"],
                        value=ECCIState.ecci_form_data.get("event_type", ""),
                        on_change=lambda v: ECCIState.update_ecci("event_type", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Product Category
                rx.vstack(
                    rx.text("Category", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["product_category"],
                        value=ECCIState.ecci_form_data.get("product_category", ""),
                        on_change=lambda v: ECCIState.update_ecci("product_category", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Price
                rx.vstack(
                    rx.text("Price", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("price", ""),
                        on_change=lambda v: ECCIState.update_ecci("price", v),
                        min=0.0,
                        step=0.01,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Date
                rx.vstack(
                    rx.text("Date", size="1", color="gray"),
                    rx.input(
                        type="date",
                        value=ECCIState.ecci_form_data.get("timestamp_date", ""),
                        on_change=lambda v: ECCIState.update_ecci("timestamp_date", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Time
                rx.vstack(
                    rx.text("Time", size="1", color="gray"),
                    rx.input(
                        type="time",
                        value=ECCIState.ecci_form_data.get("timestamp_time", ""),
                        on_change=lambda v: ECCIState.update_ecci("timestamp_time", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Product ID
                rx.vstack(
                    rx.text("Product ID", size="1", color="gray"),
                    rx.input(
                        value=ECCIState.ecci_form_data.get("product_id", ""),
                        on_change=lambda v: ECCIState.update_ecci("product_id", v),
                        placeholder="prod_1050",
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Latitude
                rx.vstack(
                    rx.text("Latitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("lat", ""),
                        on_change=lambda v: ECCIState.update_ecci("lat", v),
                        min=29.5,
                        max=30.1,
                        step=0.001,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Longitude
                rx.vstack(
                    rx.text("Longitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("lon", ""),
                        on_change=lambda v: ECCIState.update_ecci("lon", v),
                        min=-95.8,
                        max=-95.0,
                        step=0.001,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Random Coordinates button (with label for alignment)
                rx.vstack(
                    rx.text("Coords", size="1", color="gray"),
                    rx.button(
                        rx.hstack(
                            rx.icon("shuffle", size=12),
                            rx.text("Random", size="1"),
                            spacing="1",
                            align_items="center"
                        ),
                        on_click=ECCIState.generate_random_ecci_coordinates,
                        variant="outline",
                        size="1",
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Quantity
                rx.vstack(
                    rx.text("Quantity", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("quantity", ""),
                        on_change=lambda v: ECCIState.update_ecci("quantity", v),
                        min=1,
                        step=1,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Time on Page
                rx.vstack(
                    rx.text("Time (s)", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("time_on_page_seconds", ""),
                        on_change=lambda v: ECCIState.update_ecci("time_on_page_seconds", v),
                        min=0,
                        step=1,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Session Event Sequence
                rx.vstack(
                    rx.text("Sequence", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("session_event_sequence", ""),
                        on_change=lambda v: ECCIState.update_ecci("session_event_sequence", v),
                        min=1,
                        step=1,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Referrer URL
                rx.vstack(
                    rx.text("Referrer", size="1", color="gray"),
                    rx.input(
                        value=ECCIState.ecci_form_data.get("referrer_url", ""),
                        on_change=lambda v: ECCIState.update_ecci("referrer_url", v),
                        placeholder="google.com",
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            # Display fields (read-only info stacked vertically)
            rx.vstack(
                rx.text(
                    f"Customer ID: {ECCIState.ecci_form_data.get('customer_id', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"Event ID: {ECCIState.ecci_form_data.get('event_id', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"Page URL: {ECCIState.ecci_form_data.get('page_url', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"Search Query: {ECCIState.ecci_form_data.get('search_query', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"Session ID: {ECCIState.ecci_form_data.get('session_id', '')}",
                    size="1",
                    color="gray"
                ),
                spacing="1",
                align_items="start",
                width="100%"
            ),
            spacing="2",
            align_items="start",
            width="100%"
        ),
        width="100%"
    )

    # Build left column with optional ML training switch
    if model_key and project_name:
        left_column = rx.vstack(
            ml_training_switch(model_key, project_name),
            form_card,
            spacing="4",
            width="30%"
        )
    else:
        left_column = rx.vstack(
            form_card,
            spacing="4",
            width="30%"
        )

    # Right column - Tabs for Prediction, Metrics and Analytics
    right_column = rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("target", size=14),
                        rx.text("Prediction"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="prediction"
                ),
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("chart-bar", size=14),
                        rx.text("Metrics"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="metrics"
                ),
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("layers", size=14),
                        rx.text("Analytics"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="analytics"
                ),
            ),
            # Tab 1: Cluster Prediction
            rx.tabs.content(
                rx.vstack(
                    # Customer Location map - always visible
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("map-pin", size=16, color=rx.color("accent", 10)),
                                rx.text("Customer Location", size="3", weight="bold"),
                                spacing="2",
                                align_items="center"
                            ),
                            ecci_map(),
                            spacing="2",
                            width="100%"
                        ),
                        variant="surface",
                        width="100%"
                    ),
                    # Prediction boxes - always visible
                    rx.hstack(
                        # Left: Cluster prediction
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("target", size=16, color=rx.color("accent", 10)),
                                    rx.text("Predicted Cluster", size="3", weight="bold"),
                                    spacing="2",
                                    align_items="center"
                                ),
                                rx.cond(
                                    ECCIState.ecci_prediction_show,
                                    rx.box(
                                        rx.plotly(data=ECCIState.ecci_prediction_figure, width="100%"),
                                        width="100%",
                                        flex="1",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center"
                                    ),
                                    rx.box(
                                        rx.cond(
                                            ECCIState.incremental_model_available["E-Commerce Customer Interactions"],
                                            rx.callout(
                                                "Click **Predict** to identify the customer segment.",
                                                icon="info",
                                                color="blue",
                                                width="100%"
                                            ),
                                            rx.callout(
                                                "No trained model available. Toggle **Real-time ML Training** to train first.",
                                                icon="triangle-alert",
                                                color="orange",
                                                width="100%"
                                            )
                                        ),
                                        width="100%",
                                        flex="1",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center"
                                    )
                                ),
                                spacing="2",
                                width="100%",
                                height="100%"
                            ),
                            variant="surface",
                            width="50%",
                            height="320px"
                        ),
                        # Right: Feature distribution for predicted cluster
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("bar-chart-2", size=16, color=rx.color("accent", 10)),
                                    rx.text("Cluster Behavior", size="3", weight="bold"),
                                    spacing="2",
                                    align_items="center"
                                ),
                                rx.cond(
                                    ECCIState.ecci_prediction_show,
                                    rx.vstack(
                                        rx.select(
                                            ECCIState.ecci_feature_options,
                                            value=ECCIState.ecci_selected_feature,
                                            on_change=ECCIState.set_ecci_selected_feature,
                                            size="1",
                                            width="100%"
                                        ),
                                        rx.plotly(data=ECCIState.ecci_selected_cluster_feature_figure, width="100%"),
                                        spacing="2",
                                        width="100%",
                                        flex="1"
                                    ),
                                    rx.box(
                                        rx.callout(
                                            rx.text("Cluster behavior shown after prediction."),
                                            icon="info",
                                            color="blue",
                                            width="100%"
                                        ),
                                        width="100%",
                                        flex="1",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center"
                                    )
                                ),
                                spacing="2",
                                width="100%",
                                height="100%"
                            ),
                            variant="surface",
                            width="50%",
                            height="320px"
                        ),
                        spacing="3",
                        width="100%",
                        align_items="stretch"
                    ),
                    # Cluster interpretation - only after prediction
                    rx.cond(
                        ECCIState.ecci_prediction_show,
                        rx.card(
                            rx.hstack(
                                rx.icon("info", size=16, color=rx.color("blue", 9)),
                                rx.text(
                                    f"This customer interaction was assigned to Cluster {ECCIState.ecci_predicted_cluster}. Clusters represent groups of similar customer behaviors based on their browsing patterns, device usage, and purchase activities.",
                                    size="2"
                                ),
                                spacing="2",
                                align_items="start"
                            ),
                            variant="surface",
                            width="100%"
                        ),
                        rx.box()
                    ),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="prediction"
            ),
            # Tab 2: Clustering Metrics
            rx.tabs.content(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Clustering Metrics", size="5"),
                        rx.button(
                            rx.icon("refresh-cw", size=16),
                            on_click=ECCIState.refresh_mlflow_metrics("E-Commerce Customer Interactions"),
                            size="1",
                            variant="ghost",
                            cursor="pointer",
                            title="Refresh metrics"
                        ),
                        align_items="center",
                        spacing="2"
                    ),
                    e_commerce_customer_interactions_metrics(),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="metrics"
            ),
            # Tab 3: Cluster Analytics
            rx.tabs.content(
                rx.vstack(
                    # Run info badge
                    mlflow_run_info_badge("E-Commerce Customer Interactions"),
                    # Samples per cluster chart
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("pie-chart", size=16, color=rx.color("accent", 10)),
                                rx.text("Samples per Cluster", size="3", weight="bold"),
                                rx.spacer(),
                                rx.button(
                                    rx.hstack(
                                        rx.icon("refresh-cw", size=14),
                                        rx.text("Refresh", size="1"),
                                        spacing="1"
                                    ),
                                    on_click=ECCIState.fetch_ecci_cluster_counts,
                                    variant="outline",
                                    size="1"
                                ),
                                spacing="2",
                                align_items="center",
                                width="100%"
                            ),
                            rx.plotly(data=ECCIState.ecci_cluster_counts_figure, width="100%"),
                            spacing="2",
                            width="100%"
                        ),
                        variant="surface",
                        width="100%"
                    ),
                    # Feature distribution across all clusters
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("bar-chart-2", size=16, color=rx.color("accent", 10)),
                                rx.text("Feature Distribution", size="3", weight="bold"),
                                rx.spacer(),
                                rx.select(
                                    ECCIState.ecci_feature_options,
                                    value=ECCIState.ecci_selected_feature,
                                    on_change=ECCIState.set_ecci_selected_feature,
                                    size="1",
                                    width="200px"
                                ),
                                spacing="2",
                                align_items="center",
                                width="100%"
                            ),
                            rx.plotly(data=ECCIState.ecci_all_clusters_feature_figure, width="100%"),
                            spacing="2",
                            width="100%"
                        ),
                        variant="surface",
                        width="100%"
                    ),
                    # Info callout
                    rx.callout(
                        rx.text(
                            "This tab shows aggregated statistics across all clusters. Use the feature selector to explore how different attributes are distributed across customer segments."
                        ),
                        icon="info",
                        color="blue",
                        width="100%"
                    ),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="analytics"
            ),
            default_value="prediction",
            width="100%"
        ),
        align_items="start",
        spacing="4",
        width="70%"
    )

    return rx.hstack(
        left_column,
        right_column,
        spacing="6",
        align_items="start",
        width="100%"
    )


# =============================================================================
# ECCI Batch ML Form
# =============================================================================
def e_commerce_customer_interactions_batch_form(model_key: str = None, project_name: str = None) -> rx.Component:
    """Batch ML form for E-Commerce Customer Interactions using KMeans clustering.

    Mirrors the Incremental ML form layout with batch_ml_training_box instead of ml_training_switch.
    """
    # Build form card with 3-column layout (same as incremental ML)
    form_card = rx.card(
        rx.vstack(
            # Form Legend
            rx.hstack(
                rx.icon("shopping-cart", size=20, color=rx.color("accent", 10)),
                rx.heading("Customer Interaction", size="4", weight="bold"),
                spacing="2",
                align_items="center"
            ),
            # Predict button at TOP for visibility
            rx.button(
                "Predict",
                on_click=ECCIState.predict_batch_ecci,
                size="2",
                width="100%",
                disabled=~SharedState.batch_model_available["E-Commerce Customer Interactions"]
            ),
            # Randomize button below Predict
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size=14),
                    rx.text("Randomize All Fields", size="2"),
                    spacing="1",
                    align_items="center"
                ),
                on_click=ECCIState.randomize_ecci_form,
                variant="soft",
                color_scheme="blue",
                size="2",
                width="100%"
            ),
            rx.divider(),
            # Form fields in 3-column grid (same as incremental ML)
            rx.grid(
                # Browser
                rx.vstack(
                    rx.text("Browser", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["browser"],
                        value=ECCIState.ecci_form_data.get("browser", ""),
                        on_change=lambda v: ECCIState.update_ecci("browser", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Device Type
                rx.vstack(
                    rx.text("Device", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["device_type"],
                        value=ECCIState.ecci_form_data.get("device_type", ""),
                        on_change=lambda v: ECCIState.update_ecci("device_type", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # OS
                rx.vstack(
                    rx.text("OS", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["os"],
                        value=ECCIState.ecci_form_data.get("os", ""),
                        on_change=lambda v: ECCIState.update_ecci("os", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Event Type
                rx.vstack(
                    rx.text("Event Type", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["event_type"],
                        value=ECCIState.ecci_form_data.get("event_type", ""),
                        on_change=lambda v: ECCIState.update_ecci("event_type", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Product Category
                rx.vstack(
                    rx.text("Category", size="1", color="gray"),
                    rx.select(
                        ECCIState.ecci_options["product_category"],
                        value=ECCIState.ecci_form_data.get("product_category", ""),
                        on_change=lambda v: ECCIState.update_ecci("product_category", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Price
                rx.vstack(
                    rx.text("Price", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("price", ""),
                        on_change=lambda v: ECCIState.update_ecci("price", v),
                        min=0.0,
                        step=0.01,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Date
                rx.vstack(
                    rx.text("Date", size="1", color="gray"),
                    rx.input(
                        type="date",
                        value=ECCIState.ecci_form_data.get("timestamp_date", ""),
                        on_change=lambda v: ECCIState.update_ecci("timestamp_date", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Time
                rx.vstack(
                    rx.text("Time", size="1", color="gray"),
                    rx.input(
                        type="time",
                        value=ECCIState.ecci_form_data.get("timestamp_time", ""),
                        on_change=lambda v: ECCIState.update_ecci("timestamp_time", v),
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Product ID
                rx.vstack(
                    rx.text("Product ID", size="1", color="gray"),
                    rx.input(
                        value=ECCIState.ecci_form_data.get("product_id", ""),
                        on_change=lambda v: ECCIState.update_ecci("product_id", v),
                        placeholder="prod_1050",
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Latitude
                rx.vstack(
                    rx.text("Latitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("lat", ""),
                        on_change=lambda v: ECCIState.update_ecci("lat", v),
                        min=29.5,
                        max=30.1,
                        step=0.001,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Longitude
                rx.vstack(
                    rx.text("Longitude", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("lon", ""),
                        on_change=lambda v: ECCIState.update_ecci("lon", v),
                        min=-95.8,
                        max=-95.0,
                        step=0.001,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Random Coordinates button
                rx.vstack(
                    rx.text("Coords", size="1", color="gray"),
                    rx.button(
                        rx.hstack(
                            rx.icon("shuffle", size=12),
                            rx.text("Random", size="1"),
                            spacing="1",
                            align_items="center"
                        ),
                        on_click=ECCIState.generate_random_ecci_coordinates,
                        variant="outline",
                        size="1",
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Quantity
                rx.vstack(
                    rx.text("Quantity", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("quantity", ""),
                        on_change=lambda v: ECCIState.update_ecci("quantity", v),
                        min=1,
                        step=1,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Time on Page
                rx.vstack(
                    rx.text("Time (s)", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("time_on_page_seconds", ""),
                        on_change=lambda v: ECCIState.update_ecci("time_on_page_seconds", v),
                        min=0,
                        step=1,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Session Event Sequence
                rx.vstack(
                    rx.text("Sequence", size="1", color="gray"),
                    rx.input(
                        type="number",
                        value=ECCIState.ecci_form_data.get("session_event_sequence", ""),
                        on_change=lambda v: ECCIState.update_ecci("session_event_sequence", v),
                        min=1,
                        step=1,
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                # Referrer URL
                rx.vstack(
                    rx.text("Referrer", size="1", color="gray"),
                    rx.input(
                        value=ECCIState.ecci_form_data.get("referrer_url", ""),
                        on_change=lambda v: ECCIState.update_ecci("referrer_url", v),
                        placeholder="google.com",
                        width="100%"
                    ),
                    spacing="1",
                    align_items="start",
                    width="100%"
                ),
                columns="3",
                spacing="2",
                width="100%"
            ),
            # Display fields
            rx.vstack(
                rx.text(
                    f"Customer ID: {ECCIState.ecci_form_data.get('customer_id', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"Event ID: {ECCIState.ecci_form_data.get('event_id', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"Page URL: {ECCIState.ecci_form_data.get('page_url', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"Search Query: {ECCIState.ecci_form_data.get('search_query', '')}",
                    size="1",
                    color="gray"
                ),
                rx.text(
                    f"Session ID: {ECCIState.ecci_form_data.get('session_id', '')}",
                    size="1",
                    color="gray"
                ),
                spacing="1",
                align_items="start",
                width="100%"
            ),
            spacing="2",
            align_items="start",
            width="100%"
        ),
        width="100%"
    )

    # Build left column with batch ML training box
    if model_key and project_name:
        left_column = rx.vstack(
            batch_ml_training_box(model_key, project_name),
            form_card,
            spacing="4",
            width="30%"
        )
    else:
        left_column = rx.vstack(
            batch_ml_training_box("KMeans", "E-Commerce Customer Interactions"),
            form_card,
            spacing="4",
            width="30%"
        )

    # Right column - Tabs for Prediction and Metrics (same structure as incremental ML)
    right_column = rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("target", size=14),
                        rx.text("Prediction"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="prediction"
                ),
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("chart-bar", size=14),
                        rx.text("Metrics"),
                        spacing="2",
                        align_items="center"
                    ),
                    value="metrics"
                ),
            ),
            # Tab 1: Cluster Prediction
            rx.tabs.content(
                rx.vstack(
                    # Customer Location map - always visible
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("map-pin", size=16, color=rx.color("accent", 10)),
                                rx.text("Customer Location", size="3", weight="bold"),
                                spacing="2",
                                align_items="center"
                            ),
                            ecci_map(),
                            spacing="2",
                            width="100%"
                        ),
                        variant="surface",
                        width="100%"
                    ),
                    # Prediction boxes - always visible
                    rx.hstack(
                        # Left: Cluster prediction
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("target", size=16, color=rx.color("accent", 10)),
                                    rx.text("Predicted Cluster", size="3", weight="bold"),
                                    spacing="2",
                                    align_items="center"
                                ),
                                # Model info badge
                                rx.hstack(
                                    rx.badge(
                                        rx.hstack(
                                            rx.icon("brain", size=12),
                                            rx.text("KMeans Clustering (Scikit-Learn)", size="1"),
                                            spacing="1",
                                            align_items="center"
                                        ),
                                        color_scheme="purple",
                                        variant="soft",
                                        size="1"
                                    ),
                                    rx.badge("Batch ML", color_scheme="blue", variant="soft", size="1"),
                                    spacing="2"
                                ),
                                rx.cond(
                                    ECCIState.ecci_batch_prediction_show,
                                    rx.box(
                                        rx.plotly(data=ECCIState.ecci_batch_prediction_figure, width="100%"),
                                        width="100%",
                                        flex="1",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center"
                                    ),
                                    rx.box(
                                        rx.cond(
                                            SharedState.batch_model_available["E-Commerce Customer Interactions"],
                                            rx.callout(
                                                "Click **Predict** to identify the customer segment.",
                                                icon="info",
                                                color="blue",
                                                width="100%"
                                            ),
                                            rx.callout(
                                                "No trained model available. Click **Train** to train the batch model first.",
                                                icon="triangle-alert",
                                                color="orange",
                                                width="100%"
                                            )
                                        ),
                                        width="100%",
                                        flex="1",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center"
                                    )
                                ),
                                spacing="2",
                                width="100%",
                                height="100%"
                            ),
                            variant="surface",
                            width="50%",
                            height="320px"
                        ),
                        # Right: Feature distribution for predicted cluster
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("bar-chart-2", size=16, color=rx.color("accent", 10)),
                                    rx.text("Cluster Behavior", size="3", weight="bold"),
                                    spacing="2",
                                    align_items="center"
                                ),
                                rx.cond(
                                    ECCIState.ecci_batch_prediction_show,
                                    rx.vstack(
                                        rx.select(
                                            ECCIState.ecci_feature_options,
                                            value=ECCIState.ecci_selected_feature,
                                            on_change=ECCIState.set_ecci_selected_feature,
                                            size="1",
                                            width="100%"
                                        ),
                                        rx.plotly(data=ECCIState.ecci_selected_cluster_feature_figure, width="100%"),
                                        spacing="2",
                                        width="100%",
                                        flex="1"
                                    ),
                                    rx.box(
                                        rx.callout(
                                            rx.text("Cluster behavior shown after prediction."),
                                            icon="info",
                                            color="blue",
                                            width="100%"
                                        ),
                                        width="100%",
                                        flex="1",
                                        display="flex",
                                        align_items="center",
                                        justify_content="center"
                                    )
                                ),
                                spacing="2",
                                width="100%",
                                height="100%"
                            ),
                            variant="surface",
                            width="50%",
                            height="320px"
                        ),
                        spacing="3",
                        width="100%",
                        align_items="stretch"
                    ),
                    # Cluster interpretation - only after prediction
                    rx.cond(
                        ECCIState.ecci_batch_prediction_show,
                        rx.card(
                            rx.hstack(
                                rx.icon("info", size=16, color=rx.color("blue", 9)),
                                rx.text(
                                    f"This customer interaction was assigned to Cluster {ECCIState.ecci_batch_predicted_cluster}. Clusters represent groups of similar customer behaviors based on their browsing patterns, device usage, and purchase activities.",
                                    size="2"
                                ),
                                spacing="2",
                                align_items="start"
                            ),
                            variant="surface",
                            width="100%"
                        ),
                        rx.box()
                    ),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="prediction"
            ),
            # Tab 2: Clustering Metrics
            rx.tabs.content(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Clustering Metrics", size="5"),
                        rx.button(
                            rx.icon("refresh-cw", size=16),
                            on_click=SharedState.get_batch_mlflow_metrics("E-Commerce Customer Interactions"),
                            size="1",
                            variant="ghost",
                            cursor="pointer",
                            title="Refresh metrics"
                        ),
                        align_items="center",
                        spacing="2"
                    ),
                    # Model info badge
                    rx.hstack(
                        rx.badge(
                            rx.hstack(
                                rx.icon("brain", size=12),
                                rx.text("KMeans Clustering (Scikit-Learn)", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            color_scheme="purple",
                            variant="soft",
                            size="1"
                        ),
                        rx.badge("Batch ML", color_scheme="blue", variant="soft", size="1"),
                        spacing="2"
                    ),
                    # Batch metrics display (reuse the same metrics component for now)
                    e_commerce_customer_interactions_metrics(),
                    spacing="4",
                    width="100%",
                    padding_top="1em"
                ),
                value="metrics"
            ),
            default_value="prediction",
            width="100%"
        ),
        align_items="start",
        spacing="4",
        width="70%"
    )

    return rx.hstack(
        left_column,
        right_column,
        spacing="6",
        align_items="start",
        width="100%"
    )
