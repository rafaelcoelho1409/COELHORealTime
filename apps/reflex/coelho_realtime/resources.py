import reflex as rx
from .state import State, METRIC_INFO


## METRIC INFO DIALOG COMPONENT
def metric_info_dialog(metric_key: str, project_key: str = "tfd") -> rx.Component:
    """Create an info dialog button showing metric formula and explanation."""
    info = METRIC_INFO.get(project_key, {}).get("metrics", {}).get(metric_key, {})
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
                            rx.text("In Fraud Detection", weight="bold", size="2"),
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


## MAP COMPONENTS (using Folium embedded via rx.html)
def eta_map() -> rx.Component:
    """
    Map component for ETA page showing origin and destination.
    Uses Folium with OpenStreetMap tiles embedded via iframe.
    """
    return rx.box(
        rx.html(State.eta_folium_map_html),
        width="100%",
        height="300px",
        overflow="hidden",
        border_radius="8px",
    )


def ecci_map() -> rx.Component:
    """
    Map component for ECCI page showing customer location.
    Uses Folium with OpenStreetMap tiles embedded via iframe.
    """
    return rx.box(
        rx.html(State.ecci_folium_map_html),
        width="100%",
        height="250px",
        overflow="hidden",
        border_radius="8px",
    )


## COMPONENTS
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
                        rx.icon("activity", size = 18, color = rx.cond(
                            State.ml_training_enabled,
                            rx.color("green", 9),
                            rx.color("gray", 9)
                        )),
                        rx.text(
                            "Real-time ML Training",
                            size = "3",
                            weight = "medium"
                        ),
                        spacing = "2",
                        align_items = "center"
                    ),
                    rx.text(
                        rx.cond(
                            State.ml_training_enabled,
                            "Processing live Kafka stream data",
                            "Toggle to start processing live data"
                        ),
                        size = "1",
                        color = "gray"
                    ),
                    spacing = "1",
                    align_items = "start"
                ),
                rx.switch(
                    checked = State.ml_training_enabled,
                    on_change = lambda checked: State.toggle_ml_training(
                        checked,
                        model_key,
                        project_name
                    ),
                    size = "3"
                ),
                justify = "between",
                align_items = "center",
                width = "100%"
            ),
            rx.divider(size = "4", width = "100%"),
            # Model name badge and MLflow button
            rx.hstack(
                rx.badge(
                    rx.hstack(
                        rx.icon("brain", size = 12),
                        rx.text(State.incremental_ml_model_name[project_name], size = "1"),
                        spacing = "1",
                        align_items = "center"
                    ),
                    color_scheme = "blue",
                    variant = "soft",
                    size = "1"
                ),
                # MLflow button - links to experiment (fills remaining space)
                rx.cond(
                    State.mlflow_experiment_url[project_name] != "",
                    rx.link(
                        rx.button(
                            rx.hstack(
                                rx.image(
                                    src = "https://cdn.simpleicons.org/mlflow/0194E2",
                                    width = "14px",
                                    height = "14px"
                                ),
                                rx.text("MLflow", size = "1"),
                                spacing = "1",
                                align_items = "center"
                            ),
                            size = "1",
                            variant = "soft",
                            color_scheme = "cyan",
                            width = "100%"
                        ),
                        href = State.mlflow_experiment_url[project_name],
                        is_external = True,
                        flex = "1"
                    ),
                    rx.button(
                        rx.hstack(
                            rx.image(
                                src = "https://cdn.simpleicons.org/mlflow/0194E2",
                                width = "14px",
                                height = "14px",
                                opacity = "0.5"
                            ),
                            rx.text("MLflow", size = "1"),
                            spacing = "1",
                            align_items = "center"
                        ),
                        size = "1",
                        variant = "soft",
                        color_scheme = "gray",
                        disabled = True,
                        flex = "1"
                    )
                ),
                spacing = "2",
                align_items = "center",
                width = "100%"
            ),
            spacing = "2",
            align_items = "start",
            width = "100%"
        ),
        variant = "surface",
        size = "2",
        width = "100%"
    )


def coelho_realtime_navbar() -> rx.Component:
    return rx.box(
        rx.box(
            rx.hstack(
                # Left section - Logo and Title
                rx.hstack(
                    rx.image(
                        src = "/coelho_realtime_logo.png",
                        width = "5em",
                        height = "auto",
                        border_radius = "12px"
                    ),
                    rx.vstack(
                        rx.heading(
                            "COELHO RealTime",
                            size = "6",
                            weight = "bold",
                            color = rx.color("accent", 11)
                        ),
                        rx.text(
                            State.page_name,
                            size = "2",
                            weight = "medium",
                            color = rx.color("gray", 11)
                        ),
                        spacing = "1",
                        align_items = "start"
                    ),
                    spacing = "4",
                    align_items = "center"
                ),
                # Right section - Navigation
                rx.hstack(
                    rx.link(
                        rx.hstack(
                            rx.icon("home", size = 16),
                            rx.text("Home", size = "3", weight = "medium"),
                            spacing = "2",
                            align_items = "center"
                        ),
                        href = "/",
                        color = rx.color("gray", 11),
                        _hover = {"color": rx.color("accent", 11)}
                    ),
                    rx.menu.root(
                        rx.menu.trigger(
                            rx.button(
                                rx.hstack(
                                    rx.text("Applications", size = "3", weight = "medium"),
                                    rx.icon("chevron-down", size = 16),
                                    spacing = "2",
                                    align_items = "center"
                                ),
                                variant = "soft",
                                size = "2",
                                color_scheme = "gray"
                            )
                        ),
                        rx.menu.content(
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.icon("credit-card", size = 16, color = rx.color("accent", 10)),
                                        rx.text("Transaction Fraud Detection", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "/transaction-fraud-detection"
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.icon("clock", size = 16, color = rx.color("accent", 10)),
                                        rx.text("Estimated Time of Arrival", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "/estimated-time-of-arrival"
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.icon("shopping-cart", size = 16, color = rx.color("accent", 10)),
                                        rx.text("E-Commerce Customer Interactions", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "/e-commerce-customer-interactions"
                                )
                            ),
                            size = "2"
                        )
                    ),
                    rx.menu.root(
                        rx.menu.trigger(
                            rx.button(
                                rx.hstack(
                                    rx.text("Services", size = "3", weight = "medium"),
                                    rx.icon("chevron-down", size = 16),
                                    spacing = "2",
                                    align_items = "center"
                                ),
                                variant = "soft",
                                size = "2",
                                color_scheme = "gray"
                            )
                        ),
                        rx.menu.content(
                            rx.menu.sub(
                                rx.menu.sub_trigger(
                                    rx.hstack(
                                        rx.image(
                                            src = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg",
                                            width = "16px",
                                            height = "16px"
                                        ),
                                        rx.text("FastAPI", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    )
                                ),
                                rx.menu.sub_content(
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.image(
                                                    src = "https://riverml.xyz/latest/img/icon.png",
                                                    width = "16px",
                                                    height = "16px"
                                                ),
                                                rx.text("River (Incremental ML)", size = "3", weight = "medium"),
                                                spacing = "2",
                                                align_items = "center"
                                            ),
                                            href = "http://localhost:8002/docs",
                                            is_external = True
                                        )
                                    ),
                                    rx.menu.item(
                                        rx.link(
                                            rx.hstack(
                                                rx.image(
                                                    src = "https://scikit-learn.org/stable/_static/scikit-learn-logo-without-subtitle.svg",
                                                    width = "16px",
                                                    height = "16px"
                                                ),
                                                rx.text("Scikit-Learn (Batch ML)", size = "3", weight = "medium"),
                                                spacing = "2",
                                                align_items = "center"
                                            ),
                                            href = "http://localhost:8003/docs",
                                            is_external = True
                                        )
                                    )
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/apachespark/apachespark-original.svg",
                                            width = "16px",
                                            height = "16px"
                                        ),
                                        rx.text("Spark", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "http://localhost:4040",
                                    is_external = True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src = "https://cdn.simpleicons.org/mlflow/0194E2",
                                            width = "16px",
                                            height = "16px"
                                        ),
                                        rx.text("MLflow", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "http://localhost:5001",
                                    is_external = True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src = "https://cdn.simpleicons.org/minio/C72E49",
                                            width = "16px",
                                            height = "16px"
                                        ),
                                        rx.text("MinIO", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "http://localhost:9001",
                                    is_external = True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg",
                                            width = "16px",
                                            height = "16px"
                                        ),
                                        rx.text("Prometheus", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "http://localhost:9090",
                                    is_external = True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/grafana/grafana-original.svg",
                                            width = "16px",
                                            height = "16px"
                                        ),
                                        rx.text("Grafana", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "http://localhost:3001",
                                    is_external = True
                                )
                            ),
                            rx.menu.item(
                                rx.link(
                                    rx.hstack(
                                        rx.image(
                                            src = "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg",
                                            width = "16px",
                                            height = "16px"
                                        ),
                                        rx.text("Alertmanager", size = "3", weight = "medium"),
                                        spacing = "2",
                                        align_items = "center"
                                    ),
                                    href = "http://localhost:9094",
                                    is_external = True
                                )
                            ),
                            size = "2"
                        )
                    ),
                    spacing = "4",
                    align_items = "center"
                ),
                justify = "between",
                align_items = "center",
                width = "100%"
            ),
            max_width = "1400px",
            width = "100%",
            padding_x = "2em",
            padding_y = "1.2em"
        ),
        bg = rx.color("accent", 2),
        border_bottom = f"1px solid {rx.color('gray', 6)}",
        width = "100%",
        position = "sticky",
        top = "0",
        z_index = "1000",
        backdrop_filter = "blur(10px)"
    )

def page_tabs() -> rx.Component:
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
        value=State.tab_name,
        on_change=State.set_tab,
        default_value="incremental_ml",
        width="100%",
    )


## DELTA LAKE SQL TAB
def delta_lake_sql_tab() -> rx.Component:
    """
    Delta Lake SQL query interface.
    Allows users to execute SQL queries against Delta Lake tables via DuckDB.
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
                            value=State.current_sql_engine,
                            on_change=State.set_sql_engine,
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
                        value=State.current_sql_query,
                        on_change=State.update_sql_query,
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
                                    State.current_sql_loading,
                                    rx.spinner(size="1"),
                                    rx.icon("play", size=14)
                                ),
                                rx.text("Run", size="2"),
                                spacing="1",
                                align="center"
                            ),
                            on_click=State.execute_sql_query,
                            color_scheme="green",
                            size="2",
                            disabled=State.current_sql_loading,
                        ),
                        rx.button(
                            rx.icon("eraser", size=14),
                            on_click=State.clear_sql_query,
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
                            State.current_table_row_count > 0,
                            rx.text(f"~{State.current_table_row_count:,} rows", size="1", color="gray"),
                            rx.button(
                                rx.icon("refresh-cw", size=12),
                                on_click=State.fetch_table_schema,
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
                        State.current_table_columns.length() > 0,
                        rx.accordion.root(
                            rx.accordion.item(
                                header=rx.text(f"Columns ({State.current_table_columns.length()})", size="1", color="gray"),
                                content=rx.vstack(
                                    rx.foreach(
                                        State.current_table_columns,
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
                            State.current_sql_templates,
                            lambda template: rx.button(
                                rx.text(template["name"], size="1"),
                                on_click=State.set_sql_query_from_template(template["query"]),
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
                            State.sql_has_results,
                            rx.hstack(
                                rx.input(
                                    value=State.current_sql_search_filter,
                                    on_change=State.set_sql_search_filter,
                                    placeholder="Filter results...",
                                    size="1",
                                    width="160px",
                                    variant="soft",
                                ),
                                rx.badge(
                                    f"{State.sql_filtered_row_count}/{State.sql_results_row_count}",
                                    color_scheme="gray",
                                    variant="soft",
                                    size="1",
                                ),
                                rx.badge(
                                    f"{State.current_sql_execution_time:.0f}ms",
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
                        State.sql_has_error,
                        rx.callout(
                            State.current_sql_error,
                            icon="alert-triangle",
                            color="red",
                            size="1",
                        ),
                        rx.fragment(),
                    ),
                    # Loading state
                    rx.cond(
                        State.current_sql_loading,
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
                            State.sql_has_results,
                            # Scrollable table container (both X and Y)
                            rx.scroll_area(
                                rx.table.root(
                                    rx.table.header(
                                        rx.table.row(
                                            rx.foreach(
                                                State.sql_results_columns,
                                                lambda col: rx.table.column_header_cell(
                                                    rx.hstack(
                                                        rx.text(col, size="1", weight="bold"),
                                                        rx.cond(
                                                            State.current_sql_sort_column == col,
                                                            rx.cond(
                                                                State.current_sql_sort_direction == "asc",
                                                                rx.icon("chevron-up", size=12, color=rx.color("accent", 10)),
                                                                rx.icon("chevron-down", size=12, color=rx.color("accent", 10)),
                                                            ),
                                                            rx.icon("chevrons-up-down", size=12, color=rx.color("gray", 8)),
                                                        ),
                                                        spacing="1",
                                                        align="center",
                                                    ),
                                                    on_click=State.toggle_sql_sort(col),
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
                                            State.sql_results_filtered,
                                            lambda row: rx.table.row(
                                                rx.foreach(
                                                    State.sql_results_columns,
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


## TRANSACTION FRAUD DETECTION
def transaction_fraud_detection_form(model_key: str = None, project_name: str = None) -> rx.Component:
    # Build form card with 3-column layout for compact display
    form_card = rx.card(
        rx.vstack(
            # Form Legend
            rx.hstack(
                rx.icon("credit-card", size = 20, color = rx.color("accent", 10)),
                rx.heading("Transaction Details", size = "4", weight = "bold"),
                spacing = "2",
                align_items = "center"
            ),
            # Predict button at TOP for visibility
            rx.button(
                "Predict",
                on_click = State.predict_transaction_fraud_detection,
                size = "2",
                width = "100%",
                disabled = ~State.incremental_model_available["Transaction Fraud Detection"]
            ),
            # Randomize button below Predict
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size = 14),
                    rx.text("Randomize All Fields", size = "2"),
                    spacing = "1",
                    align_items = "center"
                ),
                on_click = State.randomize_tfd_form,
                variant = "soft",
                color_scheme = "blue",
                size = "2",
                width = "100%"
            ),
            rx.divider(),
            # Form fields in 3-column grid
            rx.grid(
                # Amount
                rx.vstack(
                    rx.text("Amount", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.tfd_form_data.get("amount", ""),
                        on_change = lambda v: State.update_tfd("amount", v),
                        step = 0.01,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Account Age
                rx.vstack(
                    rx.text("Account Age", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.tfd_form_data.get("account_age_days", ""),
                        on_change = lambda v: State.update_tfd("account_age_days", v),
                        min = 0,
                        step = 1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Currency
                rx.vstack(
                    rx.text("Currency", size = "1", color = "gray"),
                    rx.select(
                        State.tfd_options["currency"],
                        value = State.tfd_form_data.get("currency", ""),
                        on_change = lambda v: State.update_tfd("currency", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Date
                rx.vstack(
                    rx.text("Date", size = "1", color = "gray"),
                    rx.input(
                        type = "date",
                        value = State.tfd_form_data.get("timestamp_date", ""),
                        on_change = lambda v: State.update_tfd("timestamp_date", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Time
                rx.vstack(
                    rx.text("Time", size = "1", color = "gray"),
                    rx.input(
                        type = "time",
                        value = State.tfd_form_data.get("timestamp_time", ""),
                        on_change = lambda v: State.update_tfd("timestamp_time", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Merchant ID
                rx.vstack(
                    rx.text("Merchant ID", size = "1", color = "gray"),
                    rx.select(
                        State.tfd_options["merchant_id"],
                        value = State.tfd_form_data.get("merchant_id", ""),
                        on_change = lambda v: State.update_tfd("merchant_id", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Product Category
                rx.vstack(
                    rx.text("Category", size = "1", color = "gray"),
                    rx.select(
                        State.tfd_options["product_category"],
                        value = State.tfd_form_data.get("product_category", ""),
                        on_change = lambda v: State.update_tfd("product_category", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Transaction Type
                rx.vstack(
                    rx.text("Type", size = "1", color = "gray"),
                    rx.select(
                        State.tfd_options["transaction_type"],
                        value = State.tfd_form_data.get("transaction_type", ""),
                        on_change = lambda v: State.update_tfd("transaction_type", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Payment Method
                rx.vstack(
                    rx.text("Payment", size = "1", color = "gray"),
                    rx.select(
                        State.tfd_options["payment_method"],
                        value = State.tfd_form_data.get("payment_method", ""),
                        on_change = lambda v: State.update_tfd("payment_method", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Latitude
                rx.vstack(
                    rx.text("Latitude", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.tfd_form_data.get("lat", ""),
                        on_change = lambda v: State.update_tfd("lat", v),
                        min = -90.0,
                        max = 90.0,
                        step = 0.0001,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Longitude
                rx.vstack(
                    rx.text("Longitude", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.tfd_form_data.get("lon", ""),
                        on_change = lambda v: State.update_tfd("lon", v),
                        min = -180.0,
                        max = 180.0,
                        step = 0.0001,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Browser
                rx.vstack(
                    rx.text("Browser", size = "1", color = "gray"),
                    rx.select(
                        State.tfd_options["browser"],
                        value = State.tfd_form_data.get("browser", ""),
                        on_change = lambda v: State.update_tfd("browser", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # OS
                rx.vstack(
                    rx.text("OS", size = "1", color = "gray"),
                    rx.select(
                        State.tfd_options["os"],
                        value = State.tfd_form_data.get("os", ""),
                        on_change = lambda v: State.update_tfd("os", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # CVV Provided (with label for alignment)
                rx.vstack(
                    rx.text("CVV", size = "1", color = "gray"),
                    rx.checkbox(
                        "Provided",
                        checked = State.tfd_form_data.get("cvv_provided", False),
                        on_change = lambda v: State.update_tfd("cvv_provided", v),
                        size = "1"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Billing Address Match (with label for alignment)
                rx.vstack(
                    rx.text("Billing", size = "1", color = "gray"),
                    rx.checkbox(
                        "Address Match",
                        checked = State.tfd_form_data.get("billing_address_match", False),
                        on_change = lambda v: State.update_tfd("billing_address_match", v),
                        size = "1"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                columns = "3",
                spacing = "2",
                width = "100%"
            ),
            # Display fields (read-only info stacked vertically)
            rx.vstack(
                rx.text(
                    f"Transaction ID: {State.tfd_form_data.get('transaction_id', '')}",
                    size = "1",
                    color = "gray"
                ),
                rx.text(
                    f"User ID: {State.tfd_form_data.get('user_id', '')}",
                    size = "1",
                    color = "gray"
                ),
                rx.text(
                    f"IP Address: {State.tfd_form_data.get('ip_address', '')}",
                    size = "1",
                    color = "gray"
                ),
                spacing = "1",
                align_items = "start",
                width = "100%"
            ),
            spacing = "2",
            align_items = "start",
            width = "100%"
        ),
        width = "100%"
    )

    # Build left column with optional ML training switch
    if model_key and project_name:
        left_column = rx.vstack(
            ml_training_switch(model_key, project_name),
            form_card,
            spacing = "4",
            width = "30%"
        )
    else:
        left_column = rx.vstack(
            form_card,
            spacing = "4",
            width = "30%"
        )

    # Right column - Tabs for Prediction and Metrics
    right_column = rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("target", size = 14),
                        rx.text("Prediction"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    value = "prediction"
                ),
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("chart-bar", size = 14),
                        rx.text("Metrics"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    value = "metrics"
                ),
            ),
            # Tab 1: Prediction
            rx.tabs.content(
                rx.vstack(
                    # Prediction section - header always visible
                    rx.hstack(
                        rx.icon("shield-alert", size = 20, color = rx.color("accent", 10)),
                        rx.heading("Prediction Result", size = "5", weight = "bold"),
                        spacing = "2",
                        align_items = "center",
                        width = "100%"
                    ),
                    # Model info row
                    rx.hstack(
                        rx.badge(
                            rx.hstack(
                                rx.icon("brain", size = 12),
                                rx.text("RandomUnderSampler + ARFClassifier", size = "1"),
                                spacing = "1",
                                align_items = "center"
                            ),
                            color_scheme = "blue",
                            variant = "soft",
                            size = "1"
                        ),
                        rx.badge("Imbalanced Learning", color_scheme = "purple", variant = "soft", size = "1"),
                        spacing = "2"
                    ),
                    # MLflow run info (LIVE/FINISHED status)
                    mlflow_run_info_badge("Transaction Fraud Detection"),
                    rx.cond(
                        State.tfd_prediction_show,
                        # Show prediction results when available
                        rx.card(
                            rx.vstack(
                                # Plotly Gauge Chart
                                rx.plotly(data = State.tfd_fraud_gauge, width = "100%"),
                                # Prediction summary cards (compact)
                                rx.hstack(
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("triangle-alert", size = 14, color = State.tfd_prediction_color),
                                                rx.text("Classification", size = "1", color = "gray"),
                                                spacing = "1",
                                                align_items = "center"
                                            ),
                                            rx.text(
                                                State.tfd_prediction_text,
                                                size = "5",
                                                weight = "bold",
                                                color = State.tfd_prediction_color,
                                                align = "center"
                                            ),
                                            spacing = "1",
                                            align_items = "center",
                                            width = "100%"
                                        ),
                                        variant = "surface",
                                        size = "1",
                                        width = "100%"
                                    ),
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("percent", size = 14, color = "red"),
                                                rx.text("Fraud", size = "1", color = "gray"),
                                                spacing = "1",
                                                align_items = "center"
                                            ),
                                            rx.text(
                                                f"{State.tfd_fraud_probability * 100:.2f}%",
                                                size = "5",
                                                weight = "bold",
                                                align = "center",
                                                color = "red"
                                            ),
                                            spacing = "1",
                                            align_items = "center",
                                            width = "100%"
                                        ),
                                        variant = "surface",
                                        size = "1",
                                        width = "100%"
                                    ),
                                    rx.card(
                                        rx.vstack(
                                            rx.hstack(
                                                rx.icon("circle-check", size = 14, color = "green"),
                                                rx.text("Not Fraud", size = "1", color = "gray"),
                                                spacing = "1",
                                                align_items = "center"
                                            ),
                                            rx.text(
                                                f"{(1 - State.tfd_fraud_probability) * 100:.2f}%",
                                                size = "5",
                                                weight = "bold",
                                                align = "center",
                                                color = "green"
                                            ),
                                            spacing = "1",
                                            align_items = "center",
                                            width = "100%"
                                        ),
                                        variant = "surface",
                                        size = "1",
                                        width = "100%"
                                    ),
                                    spacing = "2",
                                    width = "100%"
                                ),
                                spacing = "4",
                                width = "100%"
                            ),
                            variant = "classic",
                            width = "100%"
                        ),
                        # Show info or warning message when no prediction yet
                        rx.cond(
                            State.incremental_model_available["Transaction Fraud Detection"],
                            rx.callout(
                                "Fill in the transaction details and click **Predict** to get the fraud probability.",
                                icon = "info",
                                color = "blue",
                                width = "100%"
                            ),
                            rx.callout(
                                "No trained model available. Toggle **Real-time ML Training** to train first.",
                                icon = "triangle-alert",
                                color = "orange",
                                width = "100%"
                            )
                        )
                    ),
                    spacing = "4",
                    width = "100%",
                    padding_top = "1em"
                ),
                value = "prediction"
            ),
            # Tab 2: Metrics
            rx.tabs.content(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Classification Metrics", size = "5"),
                        rx.button(
                            rx.icon("refresh-cw", size = 16),
                            on_click = State.refresh_mlflow_metrics("Transaction Fraud Detection"),
                            size = "1",
                            variant = "ghost",
                            cursor = "pointer",
                            title = "Refresh metrics"
                        ),
                        align_items = "center",
                        spacing = "2"
                    ),
                    # Model info row
                    rx.hstack(
                        rx.badge(
                            rx.hstack(
                                rx.icon("brain", size = 12),
                                rx.text("RandomUnderSampler + ARFClassifier", size = "1"),
                                spacing = "1",
                                align_items = "center"
                            ),
                            color_scheme = "blue",
                            variant = "soft",
                            size = "1"
                        ),
                        rx.badge("Imbalanced Learning", color_scheme = "purple", variant = "soft", size = "1"),
                        spacing = "2"
                    ),
                    transaction_fraud_detection_metrics(),
                    spacing = "4",
                    width = "100%",
                    padding_top = "1em"
                ),
                value = "metrics"
            ),
            default_value = "prediction",
            width = "100%"
        ),
        on_mount = State.get_mlflow_metrics("Transaction Fraud Detection"),
        align_items = "start",
        spacing = "4",
        width = "70%"
    )

    return rx.hstack(
        left_column,
        right_column,
        spacing = "6",
        align_items = "start",
        width = "100%"
    )

def metric_card(label: str, value_var, metric_key: str = None, project_key: str = "tfd") -> rx.Component:
    """Create a compact styled metric card with optional info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(
                    label,
                    size = "1",
                    weight = "medium",
                    color = "gray"
                ),
                metric_info_dialog(metric_key, project_key) if metric_key else rx.fragment(),
                spacing = "1",
                align = "center",
                justify = "center"
            ),
            rx.text(
                value_var,
                size = "4",
                weight = "bold",
                align = "center"
            ),
            spacing = "1",
            align_items = "center",
            justify = "center",
            height = "100%"
        ),
        variant = "surface",
        size = "1"
    )


def mlflow_run_info_badge(project_name: str) -> rx.Component:
    """Display MLflow experiment run info (run_id, status, start_time) for a project."""
    run_info = State.mlflow_run_info[project_name]
    return rx.hstack(
        # Status badge with conditional styling
        rx.cond(
            run_info["is_live"],
            rx.badge(
                rx.hstack(
                    rx.box(
                        width = "8px",
                        height = "8px",
                        border_radius = "50%",
                        background_color = "green",
                        class_name = "animate-pulse"
                    ),
                    rx.text("LIVE", size = "1"),
                    spacing = "1",
                    align = "center"
                ),
                color_scheme = "green",
                variant = "surface"
            ),
            rx.badge(
                run_info["status"],
                color_scheme = rx.cond(
                    run_info["status"] == "FINISHED",
                    "blue",
                    "gray"
                ),
                variant = "surface"
            )
        ),
        # Run ID
        rx.cond(
            run_info["run_id"] != "",
            rx.hstack(
                rx.text("Run:", size = "1", color = "gray"),
                rx.code(run_info["run_id"], size = "1"),
                spacing = "1",
                align = "center"
            ),
            rx.fragment()
        ),
        # Start time
        rx.cond(
            run_info["start_time"] != "",
            rx.hstack(
                rx.text("Started:", size = "1", color = "gray"),
                rx.text(run_info["start_time"], size = "1"),
                spacing = "1",
                align = "center"
            ),
            rx.fragment()
        ),
        spacing = "3",
        align = "center",
        padding = "2"
    )


def kpi_card_with_info(plotly_key: str, metric_key: str) -> rx.Component:
    """Create a KPI card with Plotly chart and info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.spacer(),
                metric_info_dialog(metric_key, "tfd"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=State.tfd_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1"
    )


def gauge_card_with_info(plotly_key: str, metric_key: str) -> rx.Component:
    """Create a gauge card with Plotly chart and info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.spacer(),
                metric_info_dialog(metric_key, "tfd"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=State.tfd_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1",
        width="50%"
    )


def heatmap_card_with_info(plotly_key: str, metric_key: str) -> rx.Component:
    """Create a heatmap card with Plotly chart and info button."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.spacer(),
                metric_info_dialog(metric_key, "tfd"),
                width="100%",
                justify="end"
            ),
            rx.plotly(data=State.tfd_dashboard_figures[plotly_key], width="100%"),
            spacing="0",
            width="100%"
        ),
        size="1",
        width="50%"
    )


def transaction_fraud_detection_metrics() -> rx.Component:
    """Display MLflow classification metrics for TFD with Plotly dashboard layout."""
    return rx.vstack(
        # Run info badge
        mlflow_run_info_badge("Transaction Fraud Detection"),
        # ROW 1: KPI Indicators (primary metrics with delta from baseline)
        rx.grid(
            kpi_card_with_info("kpi_fbeta", "fbeta"),
            kpi_card_with_info("kpi_rocauc", "rocauc"),
            kpi_card_with_info("kpi_precision", "precision"),
            kpi_card_with_info("kpi_recall", "recall"),
            kpi_card_with_info("kpi_rolling_rocauc", "rolling_rocauc"),
            columns="5",
            spacing="2",
            width="100%"
        ),
        # ROW 2: Gauges (secondary metrics)
        rx.hstack(
            gauge_card_with_info("gauge_mcc", "mcc"),
            gauge_card_with_info("gauge_balanced_accuracy", "balanced_accuracy"),
            spacing="2",
            width="100%"
        ),
        # ROW 3: Confusion Matrix + Classification Report (side by side)
        rx.hstack(
            heatmap_card_with_info("confusion_matrix", "confusion_matrix"),
            heatmap_card_with_info("classification_report", "classification_report"),
            spacing="2",
            width="100%"
        ),
        # ROW 4: Additional metrics (text cards with info buttons)
        rx.grid(
            metric_card("F1", State.tfd_metrics["f1"], "f1"),
            metric_card("Accuracy", State.tfd_metrics["accuracy"], "accuracy"),
            metric_card("Geo Mean", State.tfd_metrics["geometric_mean"], "geometric_mean"),
            metric_card("Cohen κ", State.tfd_metrics["cohen_kappa"], "cohen_kappa"),
            metric_card("Jaccard", State.tfd_metrics["jaccard"], "jaccard"),
            metric_card("LogLoss", State.tfd_metrics["logloss"], "logloss"),
            columns="6",
            spacing="2",
            width="100%"
        ),
        spacing="3",
        width="100%"
    )


## TRANSACTION FRAUD DETECTION - BATCH ML
def transaction_fraud_detection_batch_form() -> rx.Component:
    """Batch ML form for Transaction Fraud Detection using XGBClassifier."""
    return rx.hstack(
        # Left column - Form (reuses same form as incremental ML)
        rx.card(
            rx.vstack(
                # Form Legend
                rx.hstack(
                    rx.icon("credit-card", size = 20, color = rx.color("accent", 10)),
                    rx.heading("Transaction Details", size = "4", weight = "bold"),
                    spacing = "2",
                    align_items = "center"
                ),
                rx.text(
                    "Enter transaction data to predict fraud probability using the batch ML model.",
                    size = "2",
                    color = "gray"
                ),
                # Randomize button
                rx.button(
                    rx.hstack(
                        rx.icon("shuffle", size = 14),
                        rx.text("Randomize All Fields", size = "2"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    on_click = State.randomize_tfd_form,
                    variant = "soft",
                    color_scheme = "blue",
                    size = "2",
                    width = "100%"
                ),
                rx.divider(),
                # Row 1: Amount and Account Age
                rx.hstack(
                    rx.vstack(
                        rx.text("Amount", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.tfd_form_data.get("amount", ""),
                            on_change = lambda v: State.update_tfd("amount", v),
                            step = 0.01,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Account Age (days)", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.tfd_form_data.get("account_age_days", ""),
                            on_change = lambda v: State.update_tfd("account_age_days", v),
                            min = 0,
                            step = 1,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 2: Date and Time
                rx.hstack(
                    rx.vstack(
                        rx.text("Date", size = "1", color = "gray"),
                        rx.input(
                            type = "date",
                            value = State.tfd_form_data.get("timestamp_date", ""),
                            on_change = lambda v: State.update_tfd("timestamp_date", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Time", size = "1", color = "gray"),
                        rx.input(
                            type = "time",
                            value = State.tfd_form_data.get("timestamp_time", ""),
                            on_change = lambda v: State.update_tfd("timestamp_time", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 3: Currency
                rx.vstack(
                    rx.text("Currency", size = "1", color = "gray"),
                    rx.select(
                        State.tfd_options["currency"],
                        value = State.tfd_form_data.get("currency", ""),
                        on_change = lambda v: State.update_tfd("currency", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Row 4: Merchant ID and Product Category
                rx.hstack(
                    rx.vstack(
                        rx.text("Merchant ID", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["merchant_id"],
                            value = State.tfd_form_data.get("merchant_id", ""),
                            on_change = lambda v: State.update_tfd("merchant_id", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Product Category", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["product_category"],
                            value = State.tfd_form_data.get("product_category", ""),
                            on_change = lambda v: State.update_tfd("product_category", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 5: Transaction Type and Payment Method
                rx.hstack(
                    rx.vstack(
                        rx.text("Transaction Type", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["transaction_type"],
                            value = State.tfd_form_data.get("transaction_type", ""),
                            on_change = lambda v: State.update_tfd("transaction_type", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Payment Method", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["payment_method"],
                            value = State.tfd_form_data.get("payment_method", ""),
                            on_change = lambda v: State.update_tfd("payment_method", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 6: Latitude and Longitude
                rx.hstack(
                    rx.vstack(
                        rx.text("Latitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.tfd_form_data.get("lat", ""),
                            on_change = lambda v: State.update_tfd("lat", v),
                            min = -90.0,
                            max = 90.0,
                            step = 0.0001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("Longitude", size = "1", color = "gray"),
                        rx.input(
                            type = "number",
                            value = State.tfd_form_data.get("lon", ""),
                            on_change = lambda v: State.update_tfd("lon", v),
                            min = -180.0,
                            max = 180.0,
                            step = 0.0001,
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 7: Browser and OS
                rx.hstack(
                    rx.vstack(
                        rx.text("Browser", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["browser"],
                            value = State.tfd_form_data.get("browser", ""),
                            on_change = lambda v: State.update_tfd("browser", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    rx.vstack(
                        rx.text("OS", size = "1", color = "gray"),
                        rx.select(
                            State.tfd_options["os"],
                            value = State.tfd_form_data.get("os", ""),
                            on_change = lambda v: State.update_tfd("os", v),
                            width = "100%"
                        ),
                        spacing = "1",
                        align_items = "start",
                        width = "100%"
                    ),
                    spacing = "3",
                    width = "100%"
                ),
                # Row 8: CVV Provided and Billing Address Match
                rx.hstack(
                    rx.checkbox(
                        "CVV Provided",
                        checked = State.tfd_form_data.get("cvv_provided", False),
                        on_change = lambda v: State.update_tfd("cvv_provided", v),
                        size = "1"
                    ),
                    rx.checkbox(
                        "Billing Address Match",
                        checked = State.tfd_form_data.get("billing_address_match", False),
                        on_change = lambda v: State.update_tfd("billing_address_match", v),
                        size = "1"
                    ),
                    spacing = "4",
                    width = "100%"
                ),
                # Display fields
                rx.vstack(
                    rx.text(
                        f"Transaction ID: {State.tfd_form_data.get('transaction_id', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    rx.text(
                        f"User ID: {State.tfd_form_data.get('user_id', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    rx.text(
                        f"IP Address: {State.tfd_form_data.get('ip_address', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    rx.text(
                        f"User Agent: {State.tfd_form_data.get('user_agent', '')}",
                        size = "1",
                        color = "gray"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Predict button (disabled if model not available)
                rx.button(
                    rx.cond(
                        State.tfd_batch_model_available,
                        rx.hstack(
                            rx.icon("brain", size=16),
                            rx.text("Predict"),
                            spacing="2",
                            align_items="center"
                        ),
                        rx.hstack(
                            rx.icon("lock", size=16),
                            rx.text("Train Model First"),
                            spacing="2",
                            align_items="center"
                        )
                    ),
                    on_click = State.predict_batch_tfd,
                    disabled = ~State.tfd_batch_model_available,
                    size = "3",
                    width = "100%"
                ),
                spacing = "3",
                align_items = "start",
                width = "100%"
            ),
            width = "30%"
        ),
        # Right column - Metrics, Results, and YellowBrick
        rx.vstack(
            rx.markdown(
                f"**Batch ML model:** {State.batch_ml_model_name['Transaction Fraud Detection']}",
                size = "2",
                color = "gray"
            ),
            # Training tile with toggle (like incremental ML)
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("graduation-cap", size=20, color=rx.color("accent", 10)),
                        rx.heading("Model Training", size="4", weight="bold"),
                        rx.spacer(),
                        # Toggle for batch ML training
                        rx.cond(
                            State.batch_training_loading,
                            rx.hstack(
                                rx.spinner(size="2"),
                                rx.text("Training...", size="2", color="gray"),
                                spacing="2"
                            ),
                            rx.switch(
                                checked=State.tfd_batch_ml_enabled,
                                on_change=lambda _: State.toggle_batch_ml_training(
                                    "Transaction Fraud Detection"
                                ),
                                size="2"
                            )
                        ),
                        spacing="2",
                        align_items="center",
                        width="100%"
                    ),
                    # Training status display
                    rx.cond(
                        State.batch_training_loading,
                        # Training in progress
                        rx.vstack(
                            rx.text(
                                "Training in progress. This may take a few minutes.",
                                size="2",
                                color="gray"
                            ),
                            rx.text(
                                "Toggle off to stop training.",
                                size="1",
                                color="gray"
                            ),
                            spacing="1",
                            width="100%"
                        ),
                        # Not training - show status
                        rx.vstack(
                            # Show trained status if model available
                            rx.cond(
                                State.tfd_batch_model_available,
                                rx.hstack(
                                    rx.icon("circle-check", size=16, color="green"),
                                    rx.text("Model trained and ready", size="2", color="green"),
                                    spacing="2",
                                    align_items="center"
                                ),
                                rx.hstack(
                                    rx.icon("circle-alert", size=16, color="orange"),
                                    rx.text("No trained model available", size="2", color="orange"),
                                    spacing="2",
                                    align_items="center"
                                )
                            ),
                            # Last trained timestamp
                            rx.cond(
                                State.tfd_batch_last_trained != "",
                                rx.text(
                                    f"Last trained: {State.tfd_batch_last_trained}",
                                    size="1",
                                    color="gray"
                                ),
                                rx.fragment()
                            ),
                            rx.text(
                                "Toggle on to start training.",
                                size="1",
                                color="gray"
                            ),
                            # Training error display
                            rx.cond(
                                State.batch_training_error != "",
                                rx.callout(
                                    State.batch_training_error,
                                    icon="triangle-alert",
                                    color="red",
                                    size="1"
                                ),
                                rx.fragment()
                            ),
                            spacing="2",
                            align_items="start",
                            width="100%"
                        )
                    ),
                    spacing="3",
                    align_items="start",
                    width="100%"
                ),
                width="100%"
            ),
            # Tabs for Predictions and Detailed Metrics
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("Predictions", value = "predictions"),
                    rx.tabs.trigger("Detailed Metrics", value = "detailed_metrics"),
                ),
                # Tab 1: Predictions
                rx.tabs.content(
                    rx.vstack(
                        rx.hstack(
                            rx.heading("Classification Metrics", size = "6"),
                            rx.button(
                                rx.icon("refresh-cw", size = 16),
                                on_click = State.refresh_batch_mlflow_metrics("Transaction Fraud Detection"),
                                size = "1",
                                variant = "ghost",
                                cursor = "pointer",
                                title = "Refresh metrics"
                            ),
                            align_items = "center",
                            spacing = "2"
                        ),
                        transaction_fraud_detection_batch_metrics(),
                        rx.divider(),
                        # Prediction section
                        rx.hstack(
                            rx.icon("shield-alert", size = 20, color = rx.color("accent", 10)),
                            rx.heading("Prediction Result", size = "5", weight = "bold"),
                            spacing = "2",
                            align_items = "center",
                            width = "100%"
                        ),
                        rx.cond(
                            State.tfd_batch_prediction_show,
                            # Show prediction results when available
                            rx.card(
                                rx.vstack(
                                    # Plotly Gauge Chart
                                    rx.plotly(data = State.tfd_batch_fraud_gauge, width = "100%"),
                                    # Prediction summary cards
                                    rx.hstack(
                                        rx.card(
                                            rx.vstack(
                                                rx.hstack(
                                                    rx.icon("triangle-alert", size = 16, color = State.tfd_batch_prediction_color),
                                                    rx.text("Classification", size = "2", color = "gray"),
                                                    spacing = "1",
                                                    align_items = "center"
                                                ),
                                                rx.text(
                                                    State.tfd_batch_prediction_text,
                                                    size = "7",
                                                    weight = "bold",
                                                    color = State.tfd_batch_prediction_color,
                                                    align = "center"
                                                ),
                                                spacing = "2",
                                                align_items = "center",
                                                width = "100%"
                                            ),
                                            variant = "surface",
                                            size = "3",
                                            width = "100%"
                                        ),
                                        rx.card(
                                            rx.vstack(
                                                rx.hstack(
                                                    rx.icon("percent", size = 16, color = "red"),
                                                    rx.text("Fraud", size = "2", color = "gray"),
                                                    spacing = "1",
                                                    align_items = "center"
                                                ),
                                                rx.text(
                                                    f"{State.tfd_batch_fraud_probability * 100:.2f}%",
                                                    size = "7",
                                                    weight = "bold",
                                                    align = "center",
                                                    color = "red"
                                                ),
                                                spacing = "2",
                                                align_items = "center",
                                                width = "100%"
                                            ),
                                            variant = "surface",
                                            size = "3",
                                            width = "100%"
                                        ),
                                        rx.card(
                                            rx.vstack(
                                                rx.hstack(
                                                    rx.icon("circle-check", size = 16, color = "green"),
                                                    rx.text("Not Fraud", size = "2", color = "gray"),
                                                    spacing = "1",
                                                    align_items = "center"
                                                ),
                                                rx.text(
                                                    f"{(1 - State.tfd_batch_fraud_probability) * 100:.2f}%",
                                                    size = "7",
                                                    weight = "bold",
                                                    align = "center",
                                                    color = "green"
                                                ),
                                                spacing = "2",
                                                align_items = "center",
                                                width = "100%"
                                            ),
                                            variant = "surface",
                                            size = "3",
                                            width = "100%"
                                        ),
                                        spacing = "3",
                                        width = "100%"
                                    ),
                                    spacing = "4",
                                    width = "100%"
                                ),
                                variant = "classic",
                                width = "100%"
                            ),
                            # Show info message when no prediction yet
                            rx.callout(
                                "Fill in the transaction details and click **Predict** to get the fraud probability.",
                                icon = "info",
                                color = "blue",
                                width = "100%"
                            )
                        ),
                        spacing = "4",
                        width = "100%",
                        padding_top = "1em"
                    ),
                    value = "predictions"
                ),
                # Tab 2: Detailed Metrics (YellowBrick)
                rx.tabs.content(
                    rx.vstack(
                        rx.hstack(
                            rx.heading("Classification Metrics", size = "6"),
                            rx.button(
                                rx.icon("refresh-cw", size = 16),
                                on_click = State.refresh_batch_mlflow_metrics("Transaction Fraud Detection"),
                                size = "1",
                                variant = "ghost",
                                cursor = "pointer",
                                title = "Refresh metrics"
                            ),
                            align_items = "center",
                            spacing = "2"
                        ),
                        transaction_fraud_detection_batch_metrics(),
                        rx.divider(),
                        rx.heading("Detailed Metrics", size = "5"),
                        # YellowBrick metric selectors
                        rx.hstack(
                            rx.vstack(
                                rx.text("Metric Type", size = "1", color = "gray"),
                                rx.select(
                                    State.yellowbrick_metric_types,
                                    value = State.yellowbrick_metric_type,
                                    on_change = State.set_yellowbrick_metric_type,
                                    width = "100%"
                                ),
                                spacing = "1",
                                align_items = "start",
                                width = "50%"
                            ),
                            rx.vstack(
                                rx.text("Metric Name", size = "1", color = "gray"),
                                rx.select(
                                    State.yellowbrick_metric_options,
                                    value = State.yellowbrick_metric_name,
                                    on_change = State.set_yellowbrick_metric_name,
                                    width = "100%"
                                ),
                                spacing = "1",
                                align_items = "start",
                                width = "50%"
                            ),
                            spacing = "3",
                            width = "100%"
                        ),
                        # YellowBrick visualization display
                        rx.cond(
                            State.yellowbrick_loading,
                            rx.center(
                                rx.spinner(size = "3"),
                                width = "100%",
                                padding = "4em"
                            ),
                            rx.cond(
                                State.yellowbrick_error != "",
                                rx.callout(
                                    State.yellowbrick_error,
                                    icon = "circle-alert",
                                    color = "red",
                                    width = "100%"
                                ),
                                rx.cond(
                                    State.yellowbrick_image_base64 != "",
                                    rx.card(
                                        rx.image(
                                            src = f"data:image/png;base64,{State.yellowbrick_image_base64}",
                                            width = "100%",
                                            height = "auto"
                                        ),
                                        variant = "surface",
                                        width = "100%"
                                    ),
                                    rx.callout(
                                        "Select a metric type and metric name to display the YellowBrick visualization.",
                                        icon = "info",
                                        color = "blue",
                                        width = "100%"
                                    )
                                )
                            )
                        ),
                        spacing = "4",
                        width = "100%",
                        padding_top = "1em"
                    ),
                    value = "detailed_metrics"
                ),
                default_value = "predictions",
                width = "100%"
            ),
            on_mount = State.get_batch_mlflow_metrics("Transaction Fraud Detection"),
            align_items = "start",
            spacing = "4",
            width = "70%"
        ),
        spacing = "6",
        align_items = "start",
        width = "100%"
    )


def transaction_fraud_detection_batch_metrics() -> rx.Component:
    """Display MLflow classification metrics for TFD batch ML as individual cards."""
    return rx.grid(
        rx.foreach(
            State.tfd_batch_metric_names,
            lambda name: metric_card(
                name.replace("_", " ").title(),
                State.tfd_batch_metrics[name]
            )
        ),
        columns = "5",
        spacing = "3",
        width = "100%"
    )


## ESTIMATED TIME OF ARRIVAL
def estimated_time_of_arrival_form(model_key: str = None, project_name: str = None) -> rx.Component:
    # Build form card with 3-column layout for compact display
    form_card = rx.card(
        rx.vstack(
            # Form Legend
            rx.hstack(
                rx.icon("clock", size = 20, color = rx.color("accent", 10)),
                rx.heading("Trip Details", size = "4", weight = "bold"),
                spacing = "2",
                align_items = "center"
            ),
            # Predict button at TOP for visibility
            rx.button(
                "Predict",
                on_click = State.predict_eta,
                size = "2",
                width = "100%",
                disabled = ~State.incremental_model_available["Estimated Time of Arrival"]
            ),
            # Randomize button below Predict
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size = 14),
                    rx.text("Randomize All Fields", size = "2"),
                    spacing = "1",
                    align_items = "center"
                ),
                on_click = State.randomize_eta_form,
                variant = "soft",
                color_scheme = "blue",
                size = "2",
                width = "100%"
            ),
            rx.divider(),
            # Form fields in 3-column grid
            rx.grid(
                # Driver ID
                rx.vstack(
                    rx.text("Driver ID", size = "1", color = "gray"),
                    rx.select(
                        State.eta_options["driver_id"],
                        value = State.eta_form_data.get("driver_id", ""),
                        on_change = lambda v: State.update_eta("driver_id", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Vehicle ID
                rx.vstack(
                    rx.text("Vehicle ID", size = "1", color = "gray"),
                    rx.select(
                        State.eta_options["vehicle_id"],
                        value = State.eta_form_data.get("vehicle_id", ""),
                        on_change = lambda v: State.update_eta("vehicle_id", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Weather
                rx.vstack(
                    rx.text("Weather", size = "1", color = "gray"),
                    rx.select(
                        State.eta_options["weather"],
                        value = State.eta_form_data.get("weather", ""),
                        on_change = lambda v: State.update_eta("weather", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Date
                rx.vstack(
                    rx.text("Date", size = "1", color = "gray"),
                    rx.input(
                        type = "date",
                        value = State.eta_form_data.get("timestamp_date", ""),
                        on_change = lambda v: State.update_eta("timestamp_date", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Time
                rx.vstack(
                    rx.text("Time", size = "1", color = "gray"),
                    rx.input(
                        type = "time",
                        value = State.eta_form_data.get("timestamp_time", ""),
                        on_change = lambda v: State.update_eta("timestamp_time", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Vehicle Type
                rx.vstack(
                    rx.text("Vehicle Type", size = "1", color = "gray"),
                    rx.select(
                        State.eta_options["vehicle_type"],
                        value = State.eta_form_data.get("vehicle_type", ""),
                        on_change = lambda v: State.update_eta("vehicle_type", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Origin Lat
                rx.vstack(
                    rx.text("Origin Lat", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("origin_lat", ""),
                        on_change = lambda v: State.update_eta("origin_lat", v),
                        min = 29.5,
                        max = 30.1,
                        step = 0.0001,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Origin Lon
                rx.vstack(
                    rx.text("Origin Lon", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("origin_lon", ""),
                        on_change = lambda v: State.update_eta("origin_lon", v),
                        min = -95.8,
                        max = -95.0,
                        step = 0.0001,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Random Coordinates button (with label for alignment)
                rx.vstack(
                    rx.text("Coords", size = "1", color = "gray"),
                    rx.button(
                        rx.hstack(
                            rx.icon("shuffle", size = 12),
                            rx.text("Random", size = "1"),
                            spacing = "1",
                            align_items = "center"
                        ),
                        on_click = State.generate_random_eta_coordinates,
                        variant = "outline",
                        size = "1",
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Dest Lat
                rx.vstack(
                    rx.text("Dest Lat", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("destination_lat", ""),
                        on_change = lambda v: State.update_eta("destination_lat", v),
                        min = 29.5,
                        max = 30.1,
                        step = 0.0001,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Dest Lon
                rx.vstack(
                    rx.text("Dest Lon", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("destination_lon", ""),
                        on_change = lambda v: State.update_eta("destination_lon", v),
                        min = -95.8,
                        max = -95.0,
                        step = 0.0001,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Hour of Day
                rx.vstack(
                    rx.text("Hour", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("hour_of_day", ""),
                        on_change = lambda v: State.update_eta("hour_of_day", v),
                        min = 0,
                        max = 23,
                        step = 1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Driver Rating
                rx.vstack(
                    rx.text("Rating", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("driver_rating", ""),
                        on_change = lambda v: State.update_eta("driver_rating", v),
                        min = 3.5,
                        max = 5.0,
                        step = 0.1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Temperature
                rx.vstack(
                    rx.text("Temp °C", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("temperature_celsius", ""),
                        on_change = lambda v: State.update_eta("temperature_celsius", v),
                        min = -50.0,
                        max = 50.0,
                        step = 0.1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Debug Traffic Factor
                rx.vstack(
                    rx.text("Traffic Factor", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("debug_traffic_factor", ""),
                        on_change = lambda v: State.update_eta("debug_traffic_factor", v),
                        min = 0.3,
                        max = 1.9,
                        step = 0.1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Debug Weather Factor
                rx.vstack(
                    rx.text("Weather Factor", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("debug_weather_factor", ""),
                        on_change = lambda v: State.update_eta("debug_weather_factor", v),
                        min = 1.0,
                        max = 2.0,
                        step = 0.1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Debug Driver Factor
                rx.vstack(
                    rx.text("Driver Factor", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("debug_driver_factor", ""),
                        on_change = lambda v: State.update_eta("debug_driver_factor", v),
                        min = 0.85,
                        max = 1.15,
                        step = 0.01,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Debug Incident Delay
                rx.vstack(
                    rx.text("Incident (s)", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.eta_form_data.get("debug_incident_delay_seconds", ""),
                        on_change = lambda v: State.update_eta("debug_incident_delay_seconds", v),
                        min = 0,
                        max = 1800,
                        step = 1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                columns = "3",
                spacing = "2",
                width = "100%"
            ),
            # Display fields (read-only info stacked vertically)
            rx.vstack(
                rx.text(
                    f"Trip ID: {State.eta_form_data.get('trip_id', '')}",
                    size = "1",
                    color = "gray"
                ),
                rx.text(
                    f"Estimated Distance: {State.eta_estimated_distance_km} km",
                    size = "1",
                    color = "gray"
                ),
                rx.text(
                    f"Initial Estimated Travel Time: {State.eta_initial_estimated_travel_time_seconds} s",
                    size = "1",
                    color = "gray"
                ),
                spacing = "1",
                align_items = "start",
                width = "100%"
            ),
            spacing = "2",
            align_items = "start",
            width = "100%"
        ),
        width = "100%"
    )

    # Build left column with optional ML training switch
    if model_key and project_name:
        left_column = rx.vstack(
            ml_training_switch(model_key, project_name),
            form_card,
            spacing = "4",
            width = "30%"
        )
    else:
        left_column = rx.vstack(
            form_card,
            spacing = "4",
            width = "30%"
        )

    # Right column - Tabs for Prediction and Metrics
    right_column = rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("target", size = 14),
                        rx.text("Prediction"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    value = "prediction"
                ),
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("chart-bar", size = 14),
                        rx.text("Metrics"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    value = "metrics"
                ),
            ),
            # Tab 1: Prediction
            rx.tabs.content(
                rx.vstack(
                    # Prediction section - always show both cards
                    rx.hstack(
                        # Left: Map - always visible with current coordinates
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("map-pin", size = 16, color = rx.color("accent", 10)),
                                    rx.text("Origin and Destination", size = "3", weight = "bold"),
                                    spacing = "2",
                                    align_items = "center"
                                ),
                                eta_map(),
                                rx.text(
                                    f"Estimated Distance: {State.eta_estimated_distance_km} km",
                                    size = "2",
                                    color = "gray"
                                ),
                                rx.text(
                                    f"Initial Estimated Travel Time: {State.eta_initial_estimated_travel_time_seconds} s",
                                    size = "2",
                                    color = "gray"
                                ),
                                spacing = "2",
                                width = "100%",
                                height = "100%"
                            ),
                            variant = "surface",
                            width = "50%",
                            height = "400px"
                        ),
                        # Right: ETA Prediction - shows info or results
                        rx.card(
                            rx.vstack(
                                rx.hstack(
                                    rx.icon("clock", size = 16, color = rx.color("accent", 10)),
                                    rx.text("ETA - Prediction", size = "3", weight = "bold"),
                                    spacing = "2",
                                    align_items = "center"
                                ),
                                rx.cond(
                                    State.eta_prediction_show,
                                    # Show prediction results when available
                                    rx.box(
                                        rx.plotly(data = State.eta_prediction_figure, width = "100%"),
                                        width = "100%",
                                        flex = "1",
                                        display = "flex",
                                        align_items = "center",
                                        justify_content = "center"
                                    ),
                                    # Show info or warning message when no prediction yet
                                    rx.box(
                                        rx.cond(
                                            State.incremental_model_available["Estimated Time of Arrival"],
                                            rx.callout(
                                                "Click **Predict** to get the estimated time of arrival.",
                                                icon = "info",
                                                color = "blue",
                                                width = "100%"
                                            ),
                                            rx.callout(
                                                "No trained model available. Toggle **Real-time ML Training** to train first.",
                                                icon = "triangle-alert",
                                                color = "orange",
                                                width = "100%"
                                            )
                                        ),
                                        width = "100%",
                                        flex = "1",
                                        display = "flex",
                                        align_items = "center",
                                        justify_content = "center"
                                    )
                                ),
                                spacing = "2",
                                width = "100%",
                                height = "100%"
                            ),
                            variant = "surface",
                            width = "50%",
                            height = "380px"
                        ),
                        spacing = "3",
                        width = "100%",
                        align_items = "stretch"
                    ),
                    spacing = "4",
                    width = "100%",
                    padding_top = "1em"
                ),
                value = "prediction"
            ),
            # Tab 2: Metrics
            rx.tabs.content(
                rx.vstack(
                    rx.hstack(
                        rx.heading("Regression Metrics", size = "5"),
                        rx.button(
                            rx.icon("refresh-cw", size = 16),
                            on_click = State.refresh_mlflow_metrics("Estimated Time of Arrival"),
                            size = "1",
                            variant = "ghost",
                            cursor = "pointer",
                            title = "Refresh metrics"
                        ),
                        align_items = "center",
                        spacing = "2"
                    ),
                    estimated_time_of_arrival_metrics(),
                    spacing = "4",
                    width = "100%",
                    padding_top = "1em"
                ),
                value = "metrics"
            ),
            default_value = "prediction",
            width = "100%"
        ),
        on_mount = State.get_mlflow_metrics("Estimated Time of Arrival"),
        align_items = "start",
        spacing = "4",
        width = "70%"
    )

    return rx.hstack(
        left_column,
        right_column,
        spacing = "6",
        align_items = "start",
        width = "100%"
    )


def estimated_time_of_arrival_metrics() -> rx.Component:
    """Display MLflow regression metrics for Estimated Time of Arrival as individual cards."""
    return rx.vstack(
        # Run info badge
        mlflow_run_info_badge("Estimated Time of Arrival"),
        # Metrics grid
        rx.grid(
            metric_card("MAE", State.eta_metrics["mae"]),
            metric_card("MAPE", State.eta_metrics["mape"]),
            metric_card("MSE", State.eta_metrics["mse"]),
            metric_card("R²", State.eta_metrics["r2"]),
            metric_card("RMSE", State.eta_metrics["rmse"]),
            metric_card("RMSLE", State.eta_metrics["rmsle"]),
            metric_card("SMAPE", State.eta_metrics["smape"]),
            columns = "7",
            spacing = "2",
            width = "100%"
        ),
        spacing = "3",
        width = "100%"
    )


## E-COMMERCE CUSTOMER INTERACTIONS
def e_commerce_customer_interactions_form(model_key: str = None, project_name: str = None) -> rx.Component:
    # Build form card with 3-column layout for compact display
    form_card = rx.card(
        rx.vstack(
            # Form Legend
            rx.hstack(
                rx.icon("shopping-cart", size = 20, color = rx.color("accent", 10)),
                rx.heading("Customer Interaction", size = "4", weight = "bold"),
                spacing = "2",
                align_items = "center"
            ),
            # Predict button at TOP for visibility
            rx.button(
                "Predict",
                on_click = State.predict_ecci,
                size = "2",
                width = "100%",
                disabled = ~State.incremental_model_available["E-Commerce Customer Interactions"]
            ),
            # Randomize button below Predict
            rx.button(
                rx.hstack(
                    rx.icon("shuffle", size = 14),
                    rx.text("Randomize All Fields", size = "2"),
                    spacing = "1",
                    align_items = "center"
                ),
                on_click = State.randomize_ecci_form,
                variant = "soft",
                color_scheme = "blue",
                size = "2",
                width = "100%"
            ),
            rx.divider(),
            # Form fields in 3-column grid
            rx.grid(
                # Browser
                rx.vstack(
                    rx.text("Browser", size = "1", color = "gray"),
                    rx.select(
                        State.ecci_options["browser"],
                        value = State.ecci_form_data.get("browser", ""),
                        on_change = lambda v: State.update_ecci("browser", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Device Type
                rx.vstack(
                    rx.text("Device", size = "1", color = "gray"),
                    rx.select(
                        State.ecci_options["device_type"],
                        value = State.ecci_form_data.get("device_type", ""),
                        on_change = lambda v: State.update_ecci("device_type", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # OS
                rx.vstack(
                    rx.text("OS", size = "1", color = "gray"),
                    rx.select(
                        State.ecci_options["os"],
                        value = State.ecci_form_data.get("os", ""),
                        on_change = lambda v: State.update_ecci("os", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Event Type
                rx.vstack(
                    rx.text("Event Type", size = "1", color = "gray"),
                    rx.select(
                        State.ecci_options["event_type"],
                        value = State.ecci_form_data.get("event_type", ""),
                        on_change = lambda v: State.update_ecci("event_type", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Product Category
                rx.vstack(
                    rx.text("Category", size = "1", color = "gray"),
                    rx.select(
                        State.ecci_options["product_category"],
                        value = State.ecci_form_data.get("product_category", ""),
                        on_change = lambda v: State.update_ecci("product_category", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Price
                rx.vstack(
                    rx.text("Price", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.ecci_form_data.get("price", ""),
                        on_change = lambda v: State.update_ecci("price", v),
                        min = 0.0,
                        step = 0.01,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Date
                rx.vstack(
                    rx.text("Date", size = "1", color = "gray"),
                    rx.input(
                        type = "date",
                        value = State.ecci_form_data.get("timestamp_date", ""),
                        on_change = lambda v: State.update_ecci("timestamp_date", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Time
                rx.vstack(
                    rx.text("Time", size = "1", color = "gray"),
                    rx.input(
                        type = "time",
                        value = State.ecci_form_data.get("timestamp_time", ""),
                        on_change = lambda v: State.update_ecci("timestamp_time", v),
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Product ID
                rx.vstack(
                    rx.text("Product ID", size = "1", color = "gray"),
                    rx.input(
                        value = State.ecci_form_data.get("product_id", ""),
                        on_change = lambda v: State.update_ecci("product_id", v),
                        placeholder = "prod_1050",
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Latitude
                rx.vstack(
                    rx.text("Latitude", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.ecci_form_data.get("lat", ""),
                        on_change = lambda v: State.update_ecci("lat", v),
                        min = 29.5,
                        max = 30.1,
                        step = 0.001,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Longitude
                rx.vstack(
                    rx.text("Longitude", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.ecci_form_data.get("lon", ""),
                        on_change = lambda v: State.update_ecci("lon", v),
                        min = -95.8,
                        max = -95.0,
                        step = 0.001,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Random Coordinates button (with label for alignment)
                rx.vstack(
                    rx.text("Coords", size = "1", color = "gray"),
                    rx.button(
                        rx.hstack(
                            rx.icon("shuffle", size = 12),
                            rx.text("Random", size = "1"),
                            spacing = "1",
                            align_items = "center"
                        ),
                        on_click = State.generate_random_ecci_coordinates,
                        variant = "outline",
                        size = "1",
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Quantity
                rx.vstack(
                    rx.text("Quantity", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.ecci_form_data.get("quantity", ""),
                        on_change = lambda v: State.update_ecci("quantity", v),
                        min = 1,
                        step = 1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Time on Page
                rx.vstack(
                    rx.text("Time (s)", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.ecci_form_data.get("time_on_page_seconds", ""),
                        on_change = lambda v: State.update_ecci("time_on_page_seconds", v),
                        min = 0,
                        step = 1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Session Event Sequence
                rx.vstack(
                    rx.text("Sequence", size = "1", color = "gray"),
                    rx.input(
                        type = "number",
                        value = State.ecci_form_data.get("session_event_sequence", ""),
                        on_change = lambda v: State.update_ecci("session_event_sequence", v),
                        min = 1,
                        step = 1,
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                # Referrer URL
                rx.vstack(
                    rx.text("Referrer", size = "1", color = "gray"),
                    rx.input(
                        value = State.ecci_form_data.get("referrer_url", ""),
                        on_change = lambda v: State.update_ecci("referrer_url", v),
                        placeholder = "google.com",
                        width = "100%"
                    ),
                    spacing = "1",
                    align_items = "start",
                    width = "100%"
                ),
                columns = "3",
                spacing = "2",
                width = "100%"
            ),
            # Display fields (read-only info stacked vertically)
            rx.vstack(
                rx.text(
                    f"Customer ID: {State.ecci_form_data.get('customer_id', '')}",
                    size = "1",
                    color = "gray"
                ),
                rx.text(
                    f"Event ID: {State.ecci_form_data.get('event_id', '')}",
                    size = "1",
                    color = "gray"
                ),
                rx.text(
                    f"Page URL: {State.ecci_form_data.get('page_url', '')}",
                    size = "1",
                    color = "gray"
                ),
                rx.text(
                    f"Search Query: {State.ecci_form_data.get('search_query', '')}",
                    size = "1",
                    color = "gray"
                ),
                rx.text(
                    f"Session ID: {State.ecci_form_data.get('session_id', '')}",
                    size = "1",
                    color = "gray"
                ),
                spacing = "1",
                align_items = "start",
                width = "100%"
            ),
            spacing = "2",
            align_items = "start",
            width = "100%"
        ),
        width = "100%"
    )

    # Build left column with optional ML training switch
    if model_key and project_name:
        left_column = rx.vstack(
            ml_training_switch(model_key, project_name),
            form_card,
            spacing = "4",
            width = "30%"
        )
    else:
        left_column = rx.vstack(
            form_card,
            spacing = "4",
            width = "30%"
        )

    # Right column - Tabs for Prediction and Analytics
    right_column = rx.vstack(
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("target", size = 14),
                        rx.text("Prediction"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    value = "prediction"
                ),
                rx.tabs.trigger(
                    rx.hstack(
                        rx.icon("chart-bar", size = 14),
                        rx.text("Analytics"),
                        spacing = "2",
                        align_items = "center"
                    ),
                    value = "analytics"
                ),
            ),
        # Tab 1: Cluster Prediction
        rx.tabs.content(
            rx.vstack(
                # Algorithm info
                rx.callout(
                    rx.vstack(
                        rx.text(
                            "DBSTREAM is a density-based clustering algorithm for data streams.",
                            size = "2"
                        ),
                        rx.text(
                            "It identifies clusters dynamically as data points arrive, without predefined cluster counts.",
                            size = "2",
                            color = "gray"
                        ),
                        spacing = "1",
                        align_items = "start"
                    ),
                    icon = "info",
                    color = "blue",
                    width = "100%"
                ),
                # Customer Location map - always visible
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("map-pin", size = 16, color = rx.color("accent", 10)),
                            rx.text("Customer Location", size = "3", weight = "bold"),
                            spacing = "2",
                            align_items = "center"
                        ),
                        ecci_map(),
                        spacing = "2",
                        width = "100%"
                    ),
                    variant = "surface",
                    width = "100%"
                ),
                # Prediction boxes - always visible
                rx.hstack(
                    # Left: Cluster prediction
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("target", size = 16, color = rx.color("accent", 10)),
                                rx.text("Predicted Cluster", size = "3", weight = "bold"),
                                spacing = "2",
                                align_items = "center"
                            ),
                            rx.cond(
                                State.ecci_prediction_show,
                                rx.box(
                                    rx.plotly(data = State.ecci_prediction_figure, width = "100%"),
                                    width = "100%",
                                    flex = "1",
                                    display = "flex",
                                    align_items = "center",
                                    justify_content = "center"
                                ),
                                rx.box(
                                    rx.cond(
                                        State.incremental_model_available["E-Commerce Customer Interactions"],
                                        rx.callout(
                                            "Click **Predict** to identify the customer segment.",
                                            icon = "info",
                                            color = "blue",
                                            width = "100%"
                                        ),
                                        rx.callout(
                                            "No trained model available. Toggle **Real-time ML Training** to train first.",
                                            icon = "triangle-alert",
                                            color = "orange",
                                            width = "100%"
                                        )
                                    ),
                                    width = "100%",
                                    flex = "1",
                                    display = "flex",
                                    align_items = "center",
                                    justify_content = "center"
                                )
                            ),
                            spacing = "2",
                            width = "100%",
                            height = "100%"
                        ),
                        variant = "surface",
                        width = "50%",
                        height = "320px"
                    ),
                    # Right: Feature distribution for predicted cluster
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("bar-chart-2", size = 16, color = rx.color("accent", 10)),
                                rx.text("Cluster Behavior", size = "3", weight = "bold"),
                                spacing = "2",
                                align_items = "center"
                            ),
                            rx.cond(
                                State.ecci_prediction_show,
                                rx.vstack(
                                    rx.select(
                                        State.ecci_feature_options,
                                        value = State.ecci_selected_feature,
                                        on_change = State.set_ecci_selected_feature,
                                        size = "1",
                                        width = "100%"
                                    ),
                                    rx.plotly(data = State.ecci_selected_cluster_feature_figure, width = "100%"),
                                    spacing = "2",
                                    width = "100%",
                                    flex = "1"
                                ),
                                rx.box(
                                    rx.callout(
                                        rx.text("Cluster behavior shown after prediction."),
                                        icon = "info",
                                        color = "blue",
                                        width = "100%"
                                    ),
                                    width = "100%",
                                    flex = "1",
                                    display = "flex",
                                    align_items = "center",
                                    justify_content = "center"
                                )
                            ),
                            spacing = "2",
                            width = "100%",
                            height = "100%"
                        ),
                        variant = "surface",
                        width = "50%",
                        height = "320px"
                    ),
                    spacing = "3",
                    width = "100%",
                    align_items = "stretch"
                ),
                # Cluster interpretation - only after prediction
                rx.cond(
                    State.ecci_prediction_show,
                    rx.card(
                        rx.hstack(
                            rx.icon("info", size = 16, color = rx.color("blue", 9)),
                            rx.text(
                                f"This customer interaction was assigned to Cluster {State.ecci_predicted_cluster}. Clusters represent groups of similar customer behaviors based on their browsing patterns, device usage, and purchase activities.",
                                size = "2"
                            ),
                            spacing = "2",
                            align_items = "start"
                        ),
                        variant = "surface",
                        width = "100%"
                    ),
                    rx.box()
                ),
                spacing = "4",
                width = "100%",
                padding_top = "1em"
            ),
            value = "prediction"
        ),
        # Tab 2: Cluster Analytics
        rx.tabs.content(
            rx.vstack(
                # Run info badge
                mlflow_run_info_badge("E-Commerce Customer Interactions"),
                # Samples per cluster chart
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("pie-chart", size = 16, color = rx.color("accent", 10)),
                            rx.text("Samples per Cluster", size = "3", weight = "bold"),
                            rx.spacer(),
                            rx.button(
                                rx.hstack(
                                    rx.icon("refresh-cw", size = 14),
                                    rx.text("Refresh", size = "1"),
                                    spacing = "1"
                                ),
                                on_click = State.fetch_ecci_cluster_counts,
                                variant = "outline",
                                size = "1"
                            ),
                            spacing = "2",
                            align_items = "center",
                            width = "100%"
                        ),
                        rx.plotly(data = State.ecci_cluster_counts_figure, width = "100%"),
                        spacing = "2",
                        width = "100%"
                    ),
                    variant = "surface",
                    width = "100%"
                ),
                # Feature distribution across all clusters
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("bar-chart-2", size = 16, color = rx.color("accent", 10)),
                            rx.text("Feature Distribution", size = "3", weight = "bold"),
                            rx.spacer(),
                            rx.select(
                                State.ecci_feature_options,
                                value = State.ecci_selected_feature,
                                on_change = State.set_ecci_selected_feature,
                                size = "1",
                                width = "200px"
                            ),
                            spacing = "2",
                            align_items = "center",
                            width = "100%"
                        ),
                        rx.plotly(data = State.ecci_all_clusters_feature_figure, width = "100%"),
                        spacing = "2",
                        width = "100%"
                    ),
                    variant = "surface",
                    width = "100%"
                ),
                # Info callout
                rx.callout(
                    rx.text(
                        "This tab shows aggregated statistics across all clusters. Use the feature selector to explore how different attributes are distributed across customer segments."
                    ),
                    icon = "info",
                    color = "blue",
                    width = "100%"
                ),
                spacing = "4",
                width = "100%",
                padding_top = "1em"
            ),
            value = "analytics"
        ),
        default_value = "prediction",
        width = "100%"
            ),
        align_items = "start",
        spacing = "4",
        width = "70%"
    )

    return rx.hstack(
        left_column,
        right_column,
        spacing = "6",
        align_items = "start",
        width = "100%"
    )