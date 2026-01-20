"""
Shared components module - common UI elements used across all pages.

This module contains:
- metric_info_dialog: Info button with metric formula and explanation
- ml_training_switch: Toggle for real-time ML training
- coelho_realtime_navbar: Main navigation bar
- page_tabs: Tab navigation (Incremental ML, Batch ML, Delta Lake SQL)
"""

import reflex as rx
from ..states.shared import SharedState, METRIC_INFO, YELLOWBRICK_INFO


# Context titles for each project type
CONTEXT_TITLES = {
    "tfd": "In Fraud Detection",
    "eta": "In ETA Prediction",
    "ecci": "In Customer Clustering",
}


def metric_info_dialog(metric_key: str, project_key: str = "tfd") -> rx.Component:
    """Create an info dialog button showing metric formula and explanation."""
    info = METRIC_INFO.get(project_key, {}).get("metrics", {}).get(metric_key, {})
    if not info:
        return rx.fragment()

    context_title = CONTEXT_TITLES.get(project_key, "Context")

    return rx.dialog.root(
        rx.dialog.trigger(
            rx.icon_button(
                rx.icon("info", size=12),
                size="1",
                variant="ghost",
                color_scheme="gray",
                cursor="pointer",
                title=f"Learn about {info.get('name', metric_key)}"
            )
        ),
        rx.dialog.content(
            rx.hstack(
                rx.dialog.title(
                    rx.hstack(
                        rx.icon("calculator", size=20),
                        rx.text(info.get("name", metric_key)),
                        spacing="2",
                        align="center"
                    ),
                    margin="0"
                ),
                rx.spacer(),
                rx.dialog.close(
                    rx.icon_button(
                        rx.icon("x", size=16),
                        size="1",
                        variant="ghost",
                        color_scheme="gray",
                        cursor="pointer"
                    )
                ),
                width="100%",
                align="center"
            ),
            rx.separator(size="4"),
            rx.vstack(
                # Formula section
                rx.box(
                    rx.vstack(
                        rx.text("Formula", weight="bold", size="2", color="gray"),
                        rx.markdown(info.get("formula", "")),
                        spacing="1",
                        align="start",
                        width="100%"
                    ),
                    padding="3",
                    background=rx.color("gray", 2),
                    border_radius="8px",
                    width="100%"
                ),
                # Explanation section
                rx.vstack(
                    rx.text("What it means", weight="bold", size="2", color="gray"),
                    rx.text(info.get("explanation", ""), size="2"),
                    spacing="1",
                    align="start",
                    width="100%"
                ),
                # Context section
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("target", size=14),
                            rx.text(context_title, weight="bold", size="2"),
                            spacing="1",
                            align="center"
                        ),
                        rx.markdown(info.get("context", ""), component_map={"p": lambda text: rx.text(text, size="2")}),
                        spacing="1",
                        align="start",
                        width="100%"
                    ),
                    padding="3",
                    background=rx.color("accent", 2),
                    border_radius="8px",
                    width="100%"
                ),
                # Range info
                rx.hstack(
                    rx.badge(f"Range: {info.get('range', 'N/A')}", color_scheme="blue", variant="soft"),
                    rx.badge(info.get("optimal", ""), color_scheme="green", variant="soft"),
                    spacing="2"
                ),
                spacing="3",
                width="100%",
                align="start"
            ),
            max_width="500px"
        )
    )


def yellowbrick_info_dialog(visualizer_key: str, project_key: str = "tfd") -> rx.Component:
    """Create an info dialog button for YellowBrick visualizers.

    Shows description, interpretation, fraud context, and when to use
    for each YellowBrick visualization.
    """
    info = YELLOWBRICK_INFO.get(project_key, {}).get("visualizers", {}).get(visualizer_key, {})
    if not info:
        return rx.fragment()

    return rx.dialog.root(
        rx.dialog.trigger(
            rx.icon_button(
                rx.icon("info", size=12),
                size="1",
                variant="ghost",
                color_scheme="gray",
                cursor="pointer",
                title=f"Learn about {info.get('name', visualizer_key)}"
            )
        ),
        rx.dialog.content(
            rx.hstack(
                rx.dialog.title(
                    rx.hstack(
                        rx.icon("bar-chart-2", size=20),
                        rx.text(info.get("name", visualizer_key)),
                        spacing="2",
                        align="center"
                    ),
                    margin="0"
                ),
                rx.spacer(),
                rx.dialog.close(
                    rx.icon_button(
                        rx.icon("x", size=16),
                        size="1",
                        variant="ghost",
                        color_scheme="gray",
                        cursor="pointer"
                    )
                ),
                width="100%",
                align="center"
            ),
            rx.separator(size="4"),
            rx.vstack(
                # Category badge
                rx.badge(
                    info.get("category", "Visualizer"),
                    color_scheme="purple",
                    variant="soft",
                    size="1"
                ),
                # Description section
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("eye", size=14),
                            rx.text("What it shows", weight="bold", size="2"),
                            spacing="1",
                            align="center"
                        ),
                        rx.text(info.get("description", ""), size="2"),
                        spacing="1",
                        align="start",
                        width="100%"
                    ),
                    padding="3",
                    background=rx.color("gray", 2),
                    border_radius="8px",
                    width="100%"
                ),
                # Interpretation section
                rx.vstack(
                    rx.hstack(
                        rx.icon("scan-eye", size=14),
                        rx.text("How to read it", weight="bold", size="2"),
                        spacing="1",
                        align="center"
                    ),
                    rx.markdown(info.get("interpretation", ""), component_map={"p": lambda text: rx.text(text, size="2")}),
                    spacing="1",
                    align="start",
                    width="100%"
                ),
                # Fraud context section
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("shield-alert", size=14),
                            rx.text("In Fraud Detection", weight="bold", size="2"),
                            spacing="1",
                            align="center"
                        ),
                        rx.markdown(info.get("fraud_context", ""), component_map={"p": lambda text: rx.text(text, size="2")}),
                        spacing="1",
                        align="start",
                        width="100%"
                    ),
                    padding="3",
                    background=rx.color("accent", 2),
                    border_radius="8px",
                    width="100%"
                ),
                # When to use section
                rx.vstack(
                    rx.hstack(
                        rx.icon("lightbulb", size=14),
                        rx.text("When to use", weight="bold", size="2"),
                        spacing="1",
                        align="center"
                    ),
                    rx.text(info.get("when_to_use", ""), size="2"),
                    spacing="1",
                    align="start",
                    width="100%"
                ),
                # Parameters info
                rx.cond(
                    info.get("parameters", "") != "",
                    rx.hstack(
                        rx.badge("Key params:", color_scheme="gray", variant="soft", size="1"),
                        rx.code(info.get("parameters", ""), size="1"),
                        spacing="2",
                        wrap="wrap"
                    ),
                    rx.fragment()
                ),
                spacing="3",
                width="100%",
                align="start"
            ),
            max_width="550px"
        )
    )


def ml_training_switch(model_key: str, project_name: str) -> rx.Component:
    """
    A switch component to control real-time ML training.
    When enabled, starts Kafka consumer to process live data.
    When disabled or on page leave, stops the consumer.
    Training continues from best model in MLflow.
    """
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.hstack(
                        rx.icon("activity", size=18, color=rx.cond(
                            SharedState.ml_training_enabled,
                            rx.color("green", 9),
                            rx.color("gray", 9)
                        )),
                        rx.text(
                            "Real-time ML Training",
                            size="3",
                            weight="medium"
                        ),
                        spacing="2",
                        align_items="center"
                    ),
                    rx.text(
                        rx.cond(
                            SharedState.ml_training_enabled,
                            "Processing live Kafka stream data",
                            "Toggle to start processing live data"
                        ),
                        size="1",
                        color="gray"
                    ),
                    spacing="1",
                    align_items="start"
                ),
                rx.switch(
                    checked=SharedState.ml_training_enabled,
                    on_change=lambda checked: SharedState.toggle_ml_training(
                        checked,
                        model_key,
                        project_name
                    ),
                    size="3"
                ),
                justify="between",
                align_items="center",
                width="100%"
            ),
            rx.divider(size="4", width="100%"),
            # Model name badge and MLflow button
            rx.hstack(
                rx.badge(
                    rx.hstack(
                        rx.icon("brain", size=12),
                        rx.text(SharedState.incremental_ml_model_name[project_name], size="1"),
                        spacing="1",
                        align_items="center"
                    ),
                    color_scheme="blue",
                    variant="soft",
                    size="1"
                ),
                # MLflow button - links to experiment (fills remaining space)
                rx.cond(
                    SharedState.mlflow_experiment_url[project_name] != "",
                    rx.link(
                        rx.button(
                            rx.hstack(
                                rx.image(
                                    src="https://cdn.simpleicons.org/mlflow/0194E2",
                                    width="14px",
                                    height="14px"
                                ),
                                rx.text("MLflow", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            size="1",
                            variant="soft",
                            color_scheme="cyan",
                            width="100%"
                        ),
                        href=SharedState.mlflow_experiment_url[project_name],
                        is_external=True,
                        flex="1"
                    ),
                    rx.button(
                        rx.hstack(
                            rx.image(
                                src="https://cdn.simpleicons.org/mlflow/0194E2",
                                width="14px",
                                height="14px",
                                opacity="0.5"
                            ),
                            rx.text("MLflow", size="1"),
                            spacing="1",
                            align_items="center"
                        ),
                        size="1",
                        variant="soft",
                        color_scheme="gray",
                        disabled=True,
                        flex="1"
                    )
                ),
                spacing="2",
                align_items="center",
                width="100%"
            ),
            spacing="2",
            align_items="start",
            width="100%"
        ),
        variant="surface",
        size="2",
        width="100%"
    )


def _run_select_item(run) -> rx.Component:
    """Helper to render a single run option in the selector (Pydantic MLflowRunInfo)."""
    # Format: "★ run_id" for best, "run_id" for others
    return rx.select.item(
        rx.cond(
            run.is_best,
            "★ " + run.run_id,
            run.run_id
        ),
        value=run.run_id
    )


def batch_ml_run_and_training_box(model_key: str, project_name: str) -> rx.Component:
    """
    Unified box for Batch ML with MLflow Run selector and Training controls.
    Combines MLflow run selection with training functionality, separated by a divider.
    """
    return rx.card(
        rx.vstack(
            # === MLflow Run Section ===
            rx.hstack(
                rx.icon("git-branch", size=16, color=rx.color("blue", 9)),
                rx.text("MLflow Run", size="2", weight="medium"),
                rx.cond(
                    SharedState.batch_runs_loading[project_name],
                    rx.spinner(size="1"),
                    rx.fragment()
                ),
                spacing="2",
                align_items="center"
            ),
            rx.cond(
                SharedState.batch_mlflow_runs[project_name].length() > 0,
                rx.select.root(
                    rx.select.trigger(placeholder="Select MLflow run...", width="100%"),
                    rx.select.content(
                        rx.foreach(
                            SharedState.batch_mlflow_runs[project_name],
                            _run_select_item
                        )
                    ),
                    value=SharedState.selected_batch_run[project_name],
                    on_change=lambda v: SharedState.select_batch_run(project_name, v),
                    size="2",
                    width="100%"
                ),
                rx.text("No runs available. Train a model first.", size="1", color="gray")
            ),
            # === Divider between MLflow Run and Training ===
            rx.divider(size="4", width="100%"),
            # === Batch ML Training Section ===
            rx.hstack(
                rx.vstack(
                    rx.hstack(
                        rx.icon("layers", size=18, color=rx.cond(
                            SharedState.batch_model_available[project_name],
                            rx.color("blue", 9),
                            rx.color("gray", 9)
                        )),
                        rx.text(
                            "Batch ML Training",
                            size="3",
                            weight="medium"
                        ),
                        spacing="2",
                        align_items="center"
                    ),
                    rx.text(
                        rx.cond(
                            SharedState.batch_training_loading[project_name],
                            "Training in progress...",
                            rx.cond(
                                SharedState.batch_model_available[project_name],
                                "Model trained and ready",
                                "Click Train to create model"
                            )
                        ),
                        size="1",
                        color="gray"
                    ),
                    spacing="1",
                    align_items="start"
                ),
                rx.cond(
                    SharedState.batch_training_loading[project_name],
                    # Training in progress: show spinner + stop button
                    rx.hstack(
                        rx.spinner(size="3"),
                        rx.button(
                            rx.hstack(
                                rx.icon("square", size=12),
                                rx.text("Stop", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            on_click=SharedState.stop_batch_training(project_name),
                            size="1",
                            color_scheme="red",
                            variant="soft"
                        ),
                        spacing="2",
                        align_items="center"
                    ),
                    # Not training: show Train button
                    rx.button(
                        rx.hstack(
                            rx.icon("play", size=14),
                            rx.text("Train", size="2"),
                            spacing="1",
                            align_items="center"
                        ),
                        on_click=SharedState.train_batch_model(model_key, project_name),
                        size="2",
                        color_scheme="blue",
                        variant="solid"
                    )
                ),
                justify="between",
                align_items="center",
                width="100%"
            ),
            # Live training status (shown only during training)
            rx.cond(
                SharedState.batch_training_loading[project_name],
                rx.vstack(
                    # Progress bar
                    rx.progress(
                        value=SharedState.batch_training_progress[project_name],
                        max=100,
                        width="100%",
                        size="1",
                        color_scheme="blue",
                    ),
                    # Status message with stage icon
                    rx.hstack(
                        rx.cond(
                            SharedState.batch_training_stage[project_name] == "loading_data",
                            rx.icon("database", size=12, color=rx.color("blue", 9)),
                            rx.cond(
                                SharedState.batch_training_stage[project_name] == "training",
                                rx.icon("brain", size=12, color=rx.color("purple", 9)),
                                rx.cond(
                                    SharedState.batch_training_stage[project_name] == "evaluating",
                                    rx.icon("bar-chart-2", size=12, color=rx.color("green", 9)),
                                    rx.cond(
                                        SharedState.batch_training_stage[project_name] == "logging_mlflow",
                                        rx.icon("cloud-upload", size=12, color=rx.color("cyan", 9)),
                                        rx.icon("loader", size=12, color=rx.color("gray", 9)),
                                    )
                                )
                            )
                        ),
                        rx.text(
                            SharedState.batch_training_status[project_name],
                            size="1",
                            color="gray",
                            style={"font_style": "italic"}
                        ),
                        rx.text(
                            f"{SharedState.batch_training_progress[project_name]}%",
                            size="1",
                            color="blue",
                            weight="medium"
                        ),
                        spacing="2",
                        align_items="center",
                        width="100%"
                    ),
                    # CatBoost training log (shown only during training stage)
                    rx.cond(
                        SharedState.batch_training_catboost_log[project_name].length() > 0,
                        rx.box(
                            rx.vstack(
                                # Iteration
                                rx.hstack(
                                    rx.text("Iteration", size="1", color="gray", width="70px"),
                                    rx.text(
                                        SharedState.batch_training_catboost_log[project_name]["iteration"],
                                        size="1",
                                        weight="bold",
                                        color="blue"
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                # Test score
                                rx.hstack(
                                    rx.text("Test", size="1", color="gray", width="70px"),
                                    rx.text(
                                        SharedState.batch_training_catboost_log[project_name]["test"],
                                        size="1",
                                        weight="bold",
                                        color="green"
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                # Best score
                                rx.hstack(
                                    rx.text("Best", size="1", color="gray", width="70px"),
                                    rx.text(
                                        SharedState.batch_training_catboost_log[project_name]["best"],
                                        size="1",
                                        weight="bold",
                                        color="purple"
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                # Total time
                                rx.hstack(
                                    rx.text("Total", size="1", color="gray", width="70px"),
                                    rx.text(
                                        SharedState.batch_training_catboost_log[project_name]["total"],
                                        size="1",
                                        weight="medium",
                                        color="cyan"
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                # Time remaining
                                rx.hstack(
                                    rx.text("Remaining", size="1", color="gray", width="70px"),
                                    rx.hstack(
                                        rx.icon("clock", size=10, color="orange"),
                                        rx.text(
                                            SharedState.batch_training_catboost_log[project_name]["remaining"],
                                            size="1",
                                            weight="medium",
                                            color="orange"
                                        ),
                                        spacing="1",
                                        align_items="center",
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                spacing="1",
                                align_items="start",
                                width="100%",
                            ),
                            padding="2",
                            background=rx.color("gray", 2),
                            border_radius="4px",
                            width="100%",
                        ),
                        rx.fragment()
                    ),
                    spacing="1",
                    width="100%"
                ),
                rx.fragment()
            ),
            # Training data percentage input (hidden during training)
            rx.cond(
                SharedState.batch_training_loading[project_name],
                rx.fragment(),
                rx.hstack(
                    rx.icon("database", size=14, color="gray"),
                    rx.text("Data %", size="1", color="gray"),
                    rx.input(
                        value=SharedState.batch_training_data_percentage[project_name],
                        on_change=lambda v: SharedState.set_batch_training_percentage(project_name, v),
                        type="number",
                        min=1,
                        max=100,
                        size="1",
                        width="60px",
                    ),
                    rx.text(
                        rx.cond(
                            SharedState.batch_training_data_percentage[project_name] < 100,
                            "faster",
                            "full"
                        ),
                        size="1",
                        color="gray"
                    ),
                    align_items="center",
                    spacing="2",
                    width="100%"
                ),
            ),
            rx.divider(size="4", width="100%"),
            # Model name, Batch ML badges, and MLflow button
            rx.hstack(
                rx.badge(
                    rx.hstack(
                        rx.icon("brain", size=12),
                        rx.text(SharedState.batch_ml_model_name[project_name], size="1"),
                        spacing="1",
                        align_items="center"
                    ),
                    color_scheme="purple",
                    variant="soft",
                    size="1"
                ),
                rx.badge("Batch ML", color_scheme="blue", variant="soft", size="1"),
                rx.spacer(),
                # MLflow button - links to experiment
                rx.cond(
                    SharedState.batch_mlflow_experiment_url[project_name] != "",
                    rx.link(
                        rx.button(
                            rx.hstack(
                                rx.image(
                                    src="https://cdn.simpleicons.org/mlflow/0194E2",
                                    width="14px",
                                    height="14px"
                                ),
                                rx.text("MLflow", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            size="1",
                            variant="soft",
                            color_scheme="cyan",
                        ),
                        href=SharedState.batch_mlflow_experiment_url[project_name],
                        is_external=True,
                    ),
                    rx.button(
                        rx.hstack(
                            rx.image(
                                src="https://cdn.simpleicons.org/mlflow/0194E2",
                                width="14px",
                                height="14px",
                                opacity="0.5"
                            ),
                            rx.text("MLflow", size="1"),
                            spacing="1",
                            align_items="center"
                        ),
                        size="1",
                        variant="soft",
                        color_scheme="gray",
                        disabled=True,
                    )
                ),
                spacing="2",
                align_items="center",
                width="100%"
            ),
            spacing="2",
            align_items="start",
            width="100%"
        ),
        variant="surface",
        size="2",
        width="100%"
    )


def batch_ml_training_box(model_key: str, project_name: str) -> rx.Component:
    """
    A training box component for Batch ML (Scikit-Learn).
    Unlike Incremental ML's switch, this uses a button to trigger one-time training.
    Shows spinner during training, live status updates, and model info when available.

    Note: For TFD page, use batch_ml_run_and_training_box() which includes MLflow run selector.
    This standalone function is kept for ETA and ECCI pages.
    """
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.hstack(
                        rx.icon("layers", size=18, color=rx.cond(
                            SharedState.batch_model_available[project_name],
                            rx.color("blue", 9),
                            rx.color("gray", 9)
                        )),
                        rx.text(
                            "Batch ML Training",
                            size="3",
                            weight="medium"
                        ),
                        spacing="2",
                        align_items="center"
                    ),
                    rx.text(
                        rx.cond(
                            SharedState.batch_training_loading[project_name],
                            "Training in progress...",
                            rx.cond(
                                SharedState.batch_model_available[project_name],
                                "Model trained and ready",
                                "Click Train to create model"
                            )
                        ),
                        size="1",
                        color="gray"
                    ),
                    spacing="1",
                    align_items="start"
                ),
                rx.cond(
                    SharedState.batch_training_loading[project_name],
                    # Training in progress: show spinner + stop button
                    rx.hstack(
                        rx.spinner(size="3"),
                        rx.button(
                            rx.hstack(
                                rx.icon("square", size=12),
                                rx.text("Stop", size="1"),
                                spacing="1",
                                align_items="center"
                            ),
                            on_click=SharedState.stop_batch_training(project_name),
                            size="1",
                            color_scheme="red",
                            variant="soft"
                        ),
                        spacing="2",
                        align_items="center"
                    ),
                    # Not training: show Train button
                    rx.button(
                        rx.hstack(
                            rx.icon("play", size=14),
                            rx.text("Train", size="2"),
                            spacing="1",
                            align_items="center"
                        ),
                        on_click=SharedState.train_batch_model(model_key, project_name),
                        size="2",
                        color_scheme="blue",
                        variant="solid"
                    )
                ),
                justify="between",
                align_items="center",
                width="100%"
            ),
            # Live training status (shown only during training)
            rx.cond(
                SharedState.batch_training_loading[project_name],
                rx.vstack(
                    # Progress bar
                    rx.progress(
                        value=SharedState.batch_training_progress[project_name],
                        max=100,
                        width="100%",
                        size="1",
                        color_scheme="blue",
                    ),
                    # Status message with stage icon
                    rx.hstack(
                        rx.cond(
                            SharedState.batch_training_stage[project_name] == "loading_data",
                            rx.icon("database", size=12, color=rx.color("blue", 9)),
                            rx.cond(
                                SharedState.batch_training_stage[project_name] == "training",
                                rx.icon("brain", size=12, color=rx.color("purple", 9)),
                                rx.cond(
                                    SharedState.batch_training_stage[project_name] == "evaluating",
                                    rx.icon("bar-chart-2", size=12, color=rx.color("green", 9)),
                                    rx.cond(
                                        SharedState.batch_training_stage[project_name] == "logging_mlflow",
                                        rx.icon("cloud-upload", size=12, color=rx.color("cyan", 9)),
                                        rx.icon("loader", size=12, color=rx.color("gray", 9)),
                                    )
                                )
                            )
                        ),
                        rx.text(
                            SharedState.batch_training_status[project_name],
                            size="1",
                            color="gray",
                            style={"font_style": "italic"}
                        ),
                        rx.text(
                            f"{SharedState.batch_training_progress[project_name]}%",
                            size="1",
                            color="blue",
                            weight="medium"
                        ),
                        spacing="2",
                        align_items="center",
                        width="100%"
                    ),
                    # CatBoost training log (shown only during training stage)
                    rx.cond(
                        SharedState.batch_training_catboost_log[project_name].length() > 0,
                        rx.box(
                            rx.vstack(
                                # Iteration
                                rx.hstack(
                                    rx.text("Iteration", size="1", color="gray", width="70px"),
                                    rx.text(
                                        SharedState.batch_training_catboost_log[project_name]["iteration"],
                                        size="1",
                                        weight="bold",
                                        color="blue"
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                # Test score
                                rx.hstack(
                                    rx.text("Test", size="1", color="gray", width="70px"),
                                    rx.text(
                                        SharedState.batch_training_catboost_log[project_name]["test"],
                                        size="1",
                                        weight="bold",
                                        color="green"
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                # Best score
                                rx.hstack(
                                    rx.text("Best", size="1", color="gray", width="70px"),
                                    rx.text(
                                        SharedState.batch_training_catboost_log[project_name]["best"],
                                        size="1",
                                        weight="bold",
                                        color="purple"
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                # Total time
                                rx.hstack(
                                    rx.text("Total", size="1", color="gray", width="70px"),
                                    rx.text(
                                        SharedState.batch_training_catboost_log[project_name]["total"],
                                        size="1",
                                        weight="medium",
                                        color="cyan"
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                # Time remaining
                                rx.hstack(
                                    rx.text("Remaining", size="1", color="gray", width="70px"),
                                    rx.hstack(
                                        rx.icon("clock", size=10, color="orange"),
                                        rx.text(
                                            SharedState.batch_training_catboost_log[project_name]["remaining"],
                                            size="1",
                                            weight="medium",
                                            color="orange"
                                        ),
                                        spacing="1",
                                        align_items="center",
                                    ),
                                    spacing="2",
                                    align_items="center",
                                    width="100%",
                                ),
                                spacing="1",
                                align_items="start",
                                width="100%",
                            ),
                            padding="2",
                            background=rx.color("gray", 2),
                            border_radius="4px",
                            width="100%",
                        ),
                        rx.fragment()
                    ),
                    spacing="1",
                    width="100%"
                ),
                rx.fragment()
            ),
            # Training data percentage input (hidden during training)
            rx.cond(
                SharedState.batch_training_loading[project_name],
                rx.fragment(),
                rx.hstack(
                    rx.icon("database", size=14, color="gray"),
                    rx.text("Data %", size="1", color="gray"),
                    rx.input(
                        value=SharedState.batch_training_data_percentage[project_name],
                        on_change=lambda v: SharedState.set_batch_training_percentage(project_name, v),
                        type="number",
                        min=1,
                        max=100,
                        size="1",
                        width="60px",
                    ),
                    rx.text(
                        rx.cond(
                            SharedState.batch_training_data_percentage[project_name] < 100,
                            "faster",
                            "full"
                        ),
                        size="1",
                        color="gray"
                    ),
                    align_items="center",
                    spacing="2",
                    width="100%"
                ),
            ),
            rx.divider(size="4", width="100%"),
            # Model name and Batch ML badges
            rx.hstack(
                rx.badge(
                    rx.hstack(
                        rx.icon("brain", size=12),
                        rx.text(SharedState.batch_ml_model_name[project_name], size="1"),
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
            spacing="2",
            align_items="start",
            width="100%"
        ),
        variant="surface",
        size="2",
        width="100%"
    )


def coelho_realtime_navbar() -> rx.Component:
    return rx.box(
        rx.box(
            rx.hstack(
                # Left section - Logo and Title
                rx.hstack(
                    rx.image(
                        src="/coelho_realtime_logo.png",
                        width="5em",
                        height="auto",
                        border_radius="12px"
                    ),
                    rx.vstack(
                        rx.heading(
                            "COELHO RealTime",
                            size="6",
                            weight="bold",
                            color=rx.color("accent", 11)
                        ),
                        rx.text(
                            SharedState.page_name,
                            size="2",
                            weight="medium",
                            color=rx.color("gray", 11)
                        ),
                        spacing="1",
                        align_items="start"
                    ),
                    spacing="4",
                    align_items="center"
                ),
                # Right section - Navigation
                rx.hstack(
                    rx.link(
                        rx.hstack(
                            rx.icon("home", size=16),
                            rx.text("Home", size="3", weight="medium"),
                            spacing="2",
                            align_items="center"
                        ),
                        href="/",
                        color=rx.color("gray", 11),
                        _hover={"color": rx.color("accent", 11)}
                    ),
                    rx.menu.root(
                        rx.menu.trigger(
                            rx.button(
                                rx.hstack(
                                    rx.text("Applications", size="3", weight="medium"),
                                    rx.icon("chevron-down", size=16),
                                    spacing="2",
                                    align_items="center"
                                ),
                                variant="soft",
                                size="2",
                                color_scheme="gray"
                            )
                        ),
                        rx.menu.content(
                            # TFD Sub-menu
                            rx.menu.sub(
                                rx.menu.sub_trigger(
                                    rx.hstack(
                                        rx.icon("credit-card", size=16, color=rx.color("accent", 10)),
                                        rx.text("Transaction Fraud Detection", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    )
                                ),
                                rx.menu.sub_content(
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("activity", size=16, color=rx.color("accent", 10)),
                                                rx.text("Incremental ML", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/tfd/incremental"
                                        )
                                    ),
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("layers", size=16, color=rx.color("accent", 10)),
                                                rx.text("Batch ML", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/tfd/batch"
                                        )
                                    ),
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("database", size=16, color=rx.color("accent", 10)),
                                                rx.text("Delta Lake SQL", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/tfd/sql"
                                        )
                                    )
                                )
                            ),
                            # ETA Sub-menu
                            rx.menu.sub(
                                rx.menu.sub_trigger(
                                    rx.hstack(
                                        rx.icon("clock", size=16, color=rx.color("accent", 10)),
                                        rx.text("Estimated Time of Arrival", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    )
                                ),
                                rx.menu.sub_content(
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("activity", size=16, color=rx.color("accent", 10)),
                                                rx.text("Incremental ML", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/eta/incremental"
                                        )
                                    ),
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("layers", size=16, color=rx.color("accent", 10)),
                                                rx.text("Batch ML", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/eta/batch"
                                        )
                                    ),
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("database", size=16, color=rx.color("accent", 10)),
                                                rx.text("Delta Lake SQL", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/eta/sql"
                                        )
                                    )
                                )
                            ),
                            # ECCI Sub-menu
                            rx.menu.sub(
                                rx.menu.sub_trigger(
                                    rx.hstack(
                                        rx.icon("shopping-cart", size=16, color=rx.color("accent", 10)),
                                        rx.text("E-Commerce Customer Interactions", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    )
                                ),
                                rx.menu.sub_content(
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("activity", size=16, color=rx.color("accent", 10)),
                                                rx.text("Incremental ML", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/ecci/incremental"
                                        )
                                    ),
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("layers", size=16, color=rx.color("accent", 10)),
                                                rx.text("Batch ML", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/ecci/batch"
                                        )
                                    ),
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.icon("database", size=16, color=rx.color("accent", 10)),
                                                rx.text("Delta Lake SQL", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="/ecci/sql"
                                        )
                                    )
                                )
                            ),
                            size="2"
                        )
                    ),
                    rx.menu.root(
                        rx.menu.trigger(
                            rx.button(
                                rx.hstack(
                                    rx.text("Services", size="3", weight="medium"),
                                    rx.icon("chevron-down", size=16),
                                    spacing="2",
                                    align_items="center"
                                ),
                                variant="soft",
                                size="2",
                                color_scheme="gray"
                            )
                        ),
                        rx.menu.content(
                            rx.menu.sub(
                                rx.menu.sub_trigger(
                                    rx.hstack(
                                        rx.image(
                                            src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg",
                                            width="16px",
                                            height="16px"
                                        ),
                                        rx.text("FastAPI", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    )
                                ),
                                rx.menu.sub_content(
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.image(
                                                    src="https://riverml.xyz/latest/img/icon.png",
                                                    width="16px",
                                                    height="16px"
                                                ),
                                                rx.text("River (Incremental ML)", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="http://localhost:8002/docs",
                                            is_external=True
                                        )
                                    ),
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.image(
                                                    src="https://scikit-learn.org/stable/_static/scikit-learn-logo-without-subtitle.svg",
                                                    width="16px",
                                                    height="16px"
                                                ),
                                                rx.text("Scikit-Learn (Batch ML)", size="3", weight="medium"),
                                                spacing="2",
                                                align_items="center"
                                            ),
                                            href="http://localhost:8003/docs",
                                            is_external=True
                                        )
                                    )
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/apachespark/apachespark-original.svg",
                                            width="16px",
                                            height="16px"
                                        ),
                                        rx.text("Spark", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    ),
                                    href="http://localhost:4040",
                                    is_external=True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src="https://cdn.simpleicons.org/mlflow/0194E2",
                                            width="16px",
                                            height="16px"
                                        ),
                                        rx.text("MLflow", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    ),
                                    href="http://localhost:5001",
                                    is_external=True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src="https://cdn.simpleicons.org/minio/C72E49",
                                            width="16px",
                                            height="16px"
                                        ),
                                        rx.text("MinIO", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    ),
                                    href="http://localhost:9001",
                                    is_external=True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg",
                                            width="16px",
                                            height="16px"
                                        ),
                                        rx.text("Prometheus", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    ),
                                    href="http://localhost:9090",
                                    is_external=True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/grafana/grafana-original.svg",
                                            width="16px",
                                            height="16px"
                                        ),
                                        rx.text("Grafana", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    ),
                                    href="http://localhost:3001",
                                    is_external=True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg",
                                            width="16px",
                                            height="16px"
                                        ),
                                        rx.text("Alertmanager", size="3", weight="medium"),
                                        spacing="2",
                                        align_items="center"
                                    ),
                                    href="http://localhost:9094",
                                    is_external=True
                                )
                            ),
                            size="2"
                        )
                    ),
                    spacing="4",
                    align_items="center"
                ),
                justify="between",
                align_items="center",
                width="100%"
            ),
            max_width="1400px",
            width="100%",
            padding_x="2em",
            padding_y="1.2em"
        ),
        bg=rx.color("accent", 2),
        border_bottom=f"1px solid {rx.color('gray', 6)}",
        width="100%",
        position="sticky",
        top="0",
        z_index="1000",
        backdrop_filter="blur(10px)"
    )


def page_tabs() -> rx.Component:
    """Legacy tab navigation - kept for backwards compatibility."""
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger(
                rx.hstack(
                    rx.icon("activity", size=14),
                    rx.text("Incremental ML"),
                    spacing="1",
                    align="center"
                ),
                value="incremental_ml"
            ),
            rx.tabs.trigger(
                rx.hstack(
                    rx.icon("layers", size=14),
                    rx.text("Batch ML"),
                    spacing="1",
                    align="center"
                ),
                value="batch_ml"
            ),
            rx.tabs.trigger(
                rx.hstack(
                    rx.icon("database", size=14),
                    rx.text("Delta Lake SQL"),
                    spacing="1",
                    align="center"
                ),
                value="delta_lake_sql"
            ),
        ),
        value=SharedState.tab_name,
        on_change=SharedState.set_tab,
        default_value="incremental_ml",
        width="100%",
    )


def page_sub_nav(base_route: str, active: str) -> rx.Component:
    """Sub-navigation for project pages (Incremental ML, Batch ML, Delta Lake SQL).

    Uses links for proper routing instead of tabs for better performance.
    Each sub-page loads only its own components.

    Args:
        base_route: Base route for the project (e.g., "/tfd", "/eta", "/ecci")
        active: Currently active sub-page ("incremental", "batch", "sql")
    """
    def nav_link(label: str, icon_name: str, route: str, is_active: bool) -> rx.Component:
        return rx.link(
            rx.hstack(
                rx.icon(icon_name, size=14),
                rx.text(label, size="2", weight="medium"),
                spacing="2",
                align="center",
                justify="center",
                padding_x="16px",
                padding_y="8px",
                border_radius="6px",
                width="100%",
                background=rx.cond(
                    is_active,
                    rx.color("accent", 3),
                    "transparent"
                ),
                color=rx.cond(
                    is_active,
                    rx.color("accent", 11),
                    rx.color("gray", 11)
                ),
                _hover={
                    "background": rx.color("accent", 4) if not is_active else rx.color("accent", 3),
                },
            ),
            href=route,
            underline="none",
            width="100%",
        )

    return rx.hstack(
        nav_link("Incremental ML", "activity", f"{base_route}/incremental", active == "incremental"),
        nav_link("Batch ML", "layers", f"{base_route}/batch/prediction", active == "batch"),
        nav_link("Delta Lake SQL", "database", f"{base_route}/sql", active == "sql"),
        spacing="1",
        padding="4px",
        background=rx.color("gray", 2),
        border_radius="8px",
        width="100%",
    )


def batch_sub_nav(base_route: str, active: str) -> rx.Component:
    """Sub-navigation for Batch ML sub-pages.

    Provides navigation between:
    - Prediction: Form + prediction results + training controls
    - Metrics: sklearn overview + YellowBrick visualizations (with inner tabs)

    Args:
        base_route: Base route for batch pages (e.g., "/tfd/batch", "/eta/batch")
        active: Currently active sub-page ("prediction" or "metrics")
    """
    def nav_link(label: str, icon_name: str, route: str, is_active: bool) -> rx.Component:
        return rx.link(
            rx.hstack(
                rx.icon(icon_name, size=14),
                rx.text(label, size="2", weight="medium"),
                spacing="2",
                align="center",
                justify="center",
                padding_x="16px",
                padding_y="8px",
                border_radius="6px",
                width="100%",
                background=rx.cond(
                    is_active,
                    rx.color("accent", 3),
                    "transparent"
                ),
                color=rx.cond(
                    is_active,
                    rx.color("accent", 11),
                    rx.color("gray", 11)
                ),
                _hover={
                    "background": rx.color("accent", 4) if not is_active else rx.color("accent", 3),
                },
            ),
            href=route,
            underline="none",
            width="100%",
        )

    return rx.hstack(
        nav_link("Prediction", "target", f"{base_route}/prediction", active == "prediction"),
        nav_link("Metrics", "bar-chart-2", f"{base_route}/metrics", active == "metrics"),
        spacing="1",
        padding="4px",
        background=rx.color("gray", 2),
        border_radius="8px",
        width="100%",
    )


# =============================================================================
# Delta Lake SQL Tab Component
# =============================================================================
def delta_lake_sql_tab() -> rx.Component:
    """
    Delta Lake SQL query interface.
    Allows users to execute SQL queries against Delta Lake tables via DuckDB/Polars.
    """
    return rx.hstack(
        # Left column - Query editor and controls (35%)
        rx.vstack(
            # SQL Query Editor Card
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("terminal", size=16, color=rx.color("accent", 10)),
                        rx.heading("SQL Editor", size="3", weight="bold"),
                        rx.spacer(),
                        # Engine selector - compact
                        rx.select(
                            ["polars", "duckdb"],
                            value=SharedState.current_sql_engine,
                            on_change=SharedState.set_sql_engine,
                            size="1",
                            variant="soft",
                        ),
                        spacing="2",
                        align="center",
                        width="100%",
                    ),
                    rx.divider(),
                    # Query textarea - monospace, dark mode friendly
                    rx.text_area(
                        value=SharedState.current_sql_query,
                        on_change=SharedState.update_sql_query,
                        placeholder="SELECT * FROM data LIMIT 100",
                        min_height="180px",
                        width="100%",
                        font_family="'SF Mono', 'Fira Code', 'Consolas', monospace",
                        font_size="12px",
                        style={
                            "background": "var(--gray-a2)",
                            "border": "1px solid var(--gray-a5)",
                            "border_radius": "6px",
                        },
                    ),
                    # Action buttons - compact
                    rx.hstack(
                        rx.button(
                            rx.hstack(
                                rx.cond(
                                    SharedState.current_sql_loading,
                                    rx.spinner(size="1"),
                                    rx.icon("play", size=14)
                                ),
                                rx.text("Run", size="2"),
                                spacing="1",
                                align="center"
                            ),
                            on_click=SharedState.execute_sql_query,
                            color_scheme="green",
                            size="2",
                            disabled=SharedState.current_sql_loading,
                        ),
                        rx.button(
                            rx.icon("eraser", size=14),
                            on_click=SharedState.clear_sql_query,
                            variant="soft",
                            size="2",
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    spacing="2",
                    width="100%",
                ),
                width="100%",
                style={"background": "var(--color-panel-solid)"},
            ),
            # Table Info Card - compact
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("table-2", size=14, color=rx.color("gray", 10)),
                        rx.text("Table: ", size="1", color="gray", weight="medium"),
                        rx.code("data", size="1"),
                        rx.spacer(),
                        rx.cond(
                            SharedState.current_table_row_count > 0,
                            rx.text(f"~{SharedState.current_table_row_count:,} rows", size="1", color="gray"),
                            rx.button(
                                rx.icon("refresh-cw", size=12),
                                on_click=SharedState.fetch_table_schema,
                                variant="ghost",
                                size="1",
                            ),
                        ),
                        spacing="1",
                        align="center",
                        width="100%",
                    ),
                    # Column list (collapsible)
                    rx.cond(
                        SharedState.current_table_columns.length() > 0,
                        rx.accordion.root(
                            rx.accordion.item(
                                header=rx.text(f"Columns ({SharedState.current_table_columns.length()})", size="1", color="gray"),
                                content=rx.vstack(
                                    rx.foreach(
                                        SharedState.current_table_columns,
                                        lambda col: rx.hstack(
                                            rx.code(col["name"], size="1"),
                                            rx.text(col["type"], size="1", color="gray"),
                                            spacing="2",
                                        )
                                    ),
                                    spacing="0",
                                    max_height="150px",
                                    overflow_y="auto",
                                    padding_y="1",
                                ),
                                value="columns"
                            ),
                            type="single",
                            collapsible=True,
                            width="100%",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    spacing="1",
                    width="100%",
                ),
                width="100%",
                padding="2",
                style={"background": "var(--color-panel-solid)"},
            ),
            # Query Templates Card - compact
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("bookmark", size=14, color=rx.color("gray", 10)),
                        rx.text("Templates", size="1", color="gray", weight="medium"),
                        spacing="1",
                        align="center",
                    ),
                    rx.vstack(
                        rx.foreach(
                            SharedState.current_sql_templates,
                            lambda template: rx.button(
                                rx.text(template["name"], size="1"),
                                on_click=SharedState.set_sql_query_from_template(template["query"]),
                                variant="ghost",
                                size="1",
                                width="100%",
                                justify="start",
                                style={"padding": "4px 8px"},
                            )
                        ),
                        spacing="0",
                        width="100%",
                    ),
                    spacing="1",
                    width="100%",
                ),
                width="100%",
                padding="2",
                style={"background": "var(--color-panel-solid)"},
            ),
            spacing="3",
            width="35%",
            align="start",
        ),
        # Right column - Results (65%)
        rx.vstack(
            # Results Card
            rx.card(
                rx.vstack(
                    # Header row with title, search, and badges
                    rx.hstack(
                        rx.hstack(
                            rx.icon("table", size=18, color=rx.color("accent", 10)),
                            rx.heading("Query Results", size="3", weight="bold"),
                            spacing="2",
                            align="center",
                        ),
                        rx.spacer(),
                        # Search input in header
                        rx.cond(
                            SharedState.sql_has_results,
                            rx.hstack(
                                rx.input(
                                    value=SharedState.current_sql_search_filter,
                                    on_change=SharedState.set_sql_search_filter,
                                    placeholder="Filter results...",
                                    size="1",
                                    width="160px",
                                    variant="soft",
                                ),
                                rx.badge(
                                    f"{SharedState.sql_filtered_row_count}/{SharedState.sql_results_row_count}",
                                    color_scheme="gray",
                                    variant="soft",
                                    size="1",
                                ),
                                rx.badge(
                                    f"{SharedState.current_sql_execution_time:.0f}ms",
                                    color_scheme="green",
                                    variant="soft",
                                    size="1",
                                ),
                                spacing="2",
                                align="center",
                            ),
                            rx.fragment(),
                        ),
                        spacing="2",
                        align="center",
                        width="100%",
                    ),
                    rx.divider(),
                    # Error display
                    rx.cond(
                        SharedState.sql_has_error,
                        rx.callout(
                            SharedState.current_sql_error,
                            icon="alert-triangle",
                            color="red",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    # Loading state
                    rx.cond(
                        SharedState.current_sql_loading,
                        rx.center(
                            rx.vstack(
                                rx.spinner(size="3"),
                                rx.text("Executing query...", size="2", color="gray"),
                                spacing="2",
                                align="center",
                            ),
                            height="300px",
                            width="100%",
                        ),
                        # Results table or empty state
                        rx.cond(
                            SharedState.sql_has_results,
                            # Scrollable table container (both X and Y)
                            rx.scroll_area(
                                rx.table.root(
                                    rx.table.header(
                                        rx.table.row(
                                            rx.foreach(
                                                SharedState.sql_results_columns,
                                                lambda col: rx.table.column_header_cell(
                                                    rx.hstack(
                                                        rx.text(col, size="1", weight="bold"),
                                                        rx.cond(
                                                            SharedState.current_sql_sort_column == col,
                                                            rx.cond(
                                                                SharedState.current_sql_sort_direction == "asc",
                                                                rx.icon("chevron-up", size=12, color=rx.color("accent", 10)),
                                                                rx.icon("chevron-down", size=12, color=rx.color("accent", 10)),
                                                            ),
                                                            rx.icon("chevrons-up-down", size=12, color=rx.color("gray", 8)),
                                                        ),
                                                        spacing="1",
                                                        align="center",
                                                    ),
                                                    on_click=SharedState.toggle_sql_sort(col),
                                                    style={
                                                        "white_space": "nowrap",
                                                        "background": "var(--color-background)",
                                                        "position": "sticky",
                                                        "top": "0",
                                                        "z_index": "1",
                                                        "padding": "10px 14px",
                                                        "font_size": "11px",
                                                        "text_transform": "uppercase",
                                                        "letter_spacing": "0.5px",
                                                        "border_bottom": "2px solid var(--gray-a6)",
                                                        "border_right": "1px solid var(--gray-a4)",
                                                        "cursor": "pointer",
                                                        "_hover": {"background": "var(--gray-a3)"},
                                                    },
                                                )
                                            )
                                        )
                                    ),
                                    rx.table.body(
                                        rx.foreach(
                                            SharedState.sql_results_filtered,
                                            lambda row: rx.table.row(
                                                rx.foreach(
                                                    SharedState.sql_results_columns,
                                                    lambda col: rx.table.cell(
                                                        rx.text(row[col], size="1"),
                                                        style={
                                                            "white_space": "nowrap",
                                                            "padding": "8px 14px",
                                                            "font_size": "12px",
                                                            "font_family": "'SF Mono', 'Fira Code', monospace",
                                                            "border_bottom": "1px solid var(--gray-a4)",
                                                            "border_right": "1px solid var(--gray-a4)",
                                                        },
                                                    )
                                                ),
                                                style={
                                                    "_hover": {"background": "var(--accent-a3)"},
                                                },
                                            )
                                        )
                                    ),
                                    width="max-content",
                                    min_width="100%",
                                    size="1",
                                    style={
                                        "border_collapse": "separate",
                                        "border_spacing": "0",
                                    },
                                ),
                                type="auto",
                                scrollbars="both",
                                style={
                                    "height": "400px",
                                    "width": "100%",
                                    "border": "1px solid var(--gray-a5)",
                                    "border_radius": "8px",
                                    "background": "var(--color-background)",
                                },
                            ),
                            rx.center(
                                rx.vstack(
                                    rx.icon("database", size=40, color=rx.color("gray", 6)),
                                    rx.text("No results yet", size="2", color="gray"),
                                    rx.text("Run a query to see results", size="1", color="gray"),
                                    spacing="2",
                                    align="center",
                                ),
                                height="250px",
                                width="100%",
                            ),
                        ),
                    ),
                    spacing="2",
                    width="100%",
                ),
                width="100%",
                height="100%",
                style={"background": "var(--color-panel-solid)"},
            ),
            # Info callout - more compact
            rx.callout(
                rx.text(
                    "Query 'data' table. SELECT only. Max 10K rows.",
                    size="1"
                ),
                icon="info",
                size="1",
                width="100%",
                variant="soft",
            ),
            spacing="3",
            width="65%",
            height="100%",
        ),
        spacing="4",
        width="100%",
        align="start",
        padding="4",
    )
