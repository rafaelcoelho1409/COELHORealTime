"""
E-Commerce Customer Interactions (ECCI) state module.

This module contains:
- ECCIState class with ECCI-specific state variables and methods
- ECCI form handlers
- ECCI prediction methods (clustering)
- ECCI computed variables
- Cluster visualization state
"""
import reflex as rx
import datetime as dt
import plotly.graph_objects as go
import folium
from .shared import SharedState, RIVER_BASE_URL, safe_float_str
from ..utils import httpx_client_post, httpx_client_get


class ECCIState(SharedState):
    """E-Commerce Customer Interactions state.

    Inherits from SharedState to access common state variables while adding
    ECCI-specific computed vars and event handlers for clustering.
    """

    # ==========================================================================
    # ECCI STATE VARIABLES
    # ==========================================================================
    ecci_cluster_counts: dict = {}
    ecci_selected_feature: str = "event_type"
    ecci_cluster_feature_counts: dict = {}

    # ECCI coordinate bounds (Houston metro area)
    _ecci_lat_bounds = (29.5, 30.1)
    _ecci_lon_bounds = (-95.8, -95.0)

    # ECCI Form field type mappings
    _ecci_float_fields = {"lat", "lon", "price"}
    _ecci_int_fields = {"session_event_sequence", "quantity", "time_on_page_seconds"}

    # ==========================================================================
    # ECCI COMPUTED VARIABLES
    # ==========================================================================
    @rx.var
    def ecci_form_data(self) -> dict:
        """Get E-Commerce Customer Interactions form data."""
        return self.form_data.get("E-Commerce Customer Interactions", {})

    @rx.var
    def ecci_options(self) -> dict[str, list[str]]:
        """Get all ECCI dropdown options as a dict."""
        opts = self.dropdown_options.get("E-Commerce Customer Interactions", {})
        return opts if isinstance(opts, dict) else {}

    @rx.var
    def ecci_prediction_show(self) -> bool:
        """Check if ECCI prediction results should be shown."""
        results = self.prediction_results.get("E-Commerce Customer Interactions", {})
        if isinstance(results, dict):
            return results.get("show", False)
        return False

    @rx.var
    def ecci_predicted_cluster(self) -> int:
        """Get predicted cluster number."""
        results = self.prediction_results.get("E-Commerce Customer Interactions", {})
        if isinstance(results, dict):
            return results.get("cluster", 0)
        return 0

    @rx.var
    def ecci_prediction_figure(self) -> go.Figure:
        """Generate Plotly figure for ECCI cluster prediction display."""
        cluster = self.ecci_predicted_cluster

        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=cluster,
                title={'text': "<b>Cluster</b>", 'font': {'size': 24}},
                number={'font': {'size': 72, 'color': '#8b5cf6'}},
                domain={'row': 0, 'column': 0}
            )
        )
        fig.update_layout(
            grid={'rows': 1, 'columns': 1},
            height=250,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    @rx.var
    def ecci_metrics(self) -> dict[str, str]:
        """Get all ECCI clustering metrics as formatted strings."""
        metrics = self.mlflow_metrics.get("E-Commerce Customer Interactions", {})
        if not isinstance(metrics, dict):
            return {
                "silhouette": "0.00",
                "rolling_silhouette": "0.00",
                "time_rolling_silhouette": "0.00",
                "n_clusters": "0",
                "n_micro_clusters": "0",
            }
        return {
            "silhouette": f"{metrics.get('metrics.Silhouette', 0):.4f}",
            "rolling_silhouette": f"{metrics.get('metrics.RollingSilhouette', 0):.4f}",
            "time_rolling_silhouette": f"{metrics.get('metrics.TimeRollingSilhouette', 0):.4f}",
            "n_clusters": f"{int(metrics.get('metrics.n_clusters', 0))}",
            "n_micro_clusters": f"{int(metrics.get('metrics.n_micro_clusters', 0))}",
        }

    @rx.var
    def ecci_metrics_raw(self) -> dict[str, float]:
        """Get raw ECCI metrics as floats for dashboard calculations."""
        metrics = self.mlflow_metrics.get("E-Commerce Customer Interactions", {})
        if not isinstance(metrics, dict):
            return {
                "silhouette": 0, "rolling_silhouette": 0, "time_rolling_silhouette": 0,
                "n_clusters": 0, "n_micro_clusters": 0,
            }
        return {
            "silhouette": metrics.get("metrics.Silhouette", 0),
            "rolling_silhouette": metrics.get("metrics.RollingSilhouette", 0),
            "time_rolling_silhouette": metrics.get("metrics.TimeRollingSilhouette", 0),
            "n_clusters": metrics.get("metrics.n_clusters", 0),
            "n_micro_clusters": metrics.get("metrics.n_micro_clusters", 0),
        }

    @rx.var
    def ecci_dashboard_figures(self) -> dict:
        """Generate all ECCI dashboard Plotly figures."""
        raw = self.ecci_metrics_raw
        mlflow_data = self.mlflow_metrics.get("E-Commerce Customer Interactions", {})

        baseline = {
            "silhouette": float(mlflow_data.get("baseline_Silhouette", 0) or 0),
            "rolling_silhouette": float(mlflow_data.get("baseline_RollingSilhouette", 0) or 0),
        }

        def create_kpi_silhouette(value: float, title: str, baseline_val: float = 0) -> go.Figure:
            """Create KPI indicator for Silhouette metrics (higher is better)."""
            if value >= 0.7:
                color = "#3b82f6"
            elif value >= 0.5:
                color = "#22c55e"
            elif value >= 0.25:
                color = "#eab308"
            elif value >= 0:
                color = "#f97316"
            else:
                color = "#ef4444"

            delta_config = None
            if baseline_val != 0:
                delta_config = {
                    "reference": baseline_val,
                    "relative": True,
                    "valueformat": ".1%",
                    "increasing": {"color": "#22c55e"},
                    "decreasing": {"color": "#ef4444"}
                }

            fig = go.Figure(go.Indicator(
                mode="number+delta" if delta_config else "number",
                value=value,
                delta=delta_config,
                title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                number={"font": {"size": 28, "color": color}, "valueformat": ".4f"}
            ))
            fig.update_layout(
                height=110, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_kpi_count(value: float, title: str) -> go.Figure:
            """Create KPI indicator for cluster counts."""
            fig = go.Figure(go.Indicator(
                mode="number",
                value=value,
                title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                number={"font": {"size": 28, "color": "#3b82f6"}, "valueformat": ".0f"}
            ))
            fig.update_layout(
                height=110, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_gauge_silhouette(value: float) -> go.Figure:
            """Create Silhouette gauge (-1 to 1 scale, higher is better)."""
            steps = [
                {"range": [-1, 0], "color": "#ef4444"},
                {"range": [0, 0.25], "color": "#f97316"},
                {"range": [0.25, 0.5], "color": "#eab308"},
                {"range": [0.5, 0.7], "color": "#22c55e"},
                {"range": [0.7, 1], "color": "#3b82f6"}
            ]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": "<b>Silhouette Score</b>", "font": {"size": 14}},
                number={"valueformat": ".4f", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 1},
                    "bar": {"color": "#1e40af"},
                    "steps": steps,
                    "threshold": {"value": 0.5, "line": {"color": "black", "width": 2}, "thickness": 0.75}
                }
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_cluster_stats_indicator(n_clusters: float, n_micro: float) -> go.Figure:
            """Create dual indicator for cluster statistics."""
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number",
                value=n_clusters,
                title={"text": "<b>Macro Clusters</b>", "font": {"size": 12}},
                number={"font": {"size": 36, "color": "#3b82f6"}, "valueformat": ".0f"},
                domain={"x": [0, 0.45], "y": [0.1, 0.9]}
            ))
            fig.add_trace(go.Indicator(
                mode="number",
                value=n_micro,
                title={"text": "<b>Micro Clusters</b>", "font": {"size": 12}},
                number={"font": {"size": 36, "color": "#8b5cf6"}, "valueformat": ".0f"},
                domain={"x": [0.55, 1], "y": [0.1, 0.9]}
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        return {
            "kpi_silhouette": create_kpi_silhouette(raw["silhouette"], "Silhouette", baseline["silhouette"]),
            "kpi_rolling_silhouette": create_kpi_silhouette(raw["rolling_silhouette"], "Rolling Silhouette", baseline["rolling_silhouette"]),
            "kpi_n_clusters": create_kpi_count(raw["n_clusters"], "Clusters"),
            "kpi_n_micro_clusters": create_kpi_count(raw["n_micro_clusters"], "Micro Clusters"),
            "gauge_silhouette": create_gauge_silhouette(raw["silhouette"]),
            "cluster_stats": create_cluster_stats_indicator(raw["n_clusters"], raw["n_micro_clusters"]),
        }

    # ==========================================================================
    # MAP COORDINATE COMPUTED VARS
    # ==========================================================================
    def _safe_float(self, value, default: float) -> float:
        """Safely convert value to float."""
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @rx.var
    def ecci_lat(self) -> float:
        """Get ECCI location latitude."""
        form_data = self.form_data.get("E-Commerce Customer Interactions") or {}
        return self._safe_float(form_data.get("lat"), 29.8)

    @rx.var
    def ecci_lon(self) -> float:
        """Get ECCI location longitude."""
        form_data = self.form_data.get("E-Commerce Customer Interactions") or {}
        return self._safe_float(form_data.get("lon"), -95.4)

    @rx.var
    def ecci_folium_map_html(self) -> str:
        """Generate Folium map HTML for ECCI customer location display."""
        lat = self.ecci_lat
        lon = self.ecci_lon

        m = folium.Map(
            location=[lat, lon],
            zoom_start=10,
            tiles='cartodbpositron'
        )

        folium.Marker(
            location=[lat, lon],
            popup='Customer Location',
            icon=folium.Icon(color='purple', icon='user')
        ).add_to(m)

        html = m._repr_html_()
        html = html.replace(
            'style="position:relative;width:100%;height:0;padding-bottom:60%;"',
            'style="width:100%;height:250px;"'
        )
        html = html.replace(
            'style="position:absolute;width:100%;height:100%;left:0;top:0;',
            'style="width:100%;height:100%;'
        )
        html = html.replace(
            'Make this Notebook Trusted to load map: File -> Trust Notebook',
            ''
        )
        return html

    # ==========================================================================
    # CLUSTER ANALYTICS COMPUTED VARS
    # ==========================================================================
    @rx.var
    def ecci_feature_options(self) -> list[str]:
        """Get available features for cluster analysis."""
        return [
            "event_type",
            "product_category",
            "referrer_url",
            "quantity",
            "time_on_page_seconds",
            "session_event_sequence",
            "device_type",
            "browser",
            "os",
        ]

    @rx.var
    def ecci_cluster_counts_figure(self) -> go.Figure:
        """Generate bar chart for samples per cluster."""
        cluster_counts = self.ecci_cluster_counts
        fig = go.Figure()

        if cluster_counts:
            clusters = list(cluster_counts.keys())
            counts = list(cluster_counts.values())

            fig.add_trace(
                go.Bar(
                    x=[f"Cluster {c}" for c in clusters],
                    y=counts,
                    marker_color='#8b5cf6',
                    text=counts,
                    textposition='auto',
                )
            )

        fig.update_layout(
            title='Samples per Cluster',
            xaxis_title='Cluster',
            yaxis_title='Count',
            height=350,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    @rx.var
    def ecci_selected_cluster_feature_figure(self) -> go.Figure:
        """Generate bar chart for predicted cluster's feature distribution."""
        fig = go.Figure()
        predicted_cluster = self.ecci_predicted_cluster
        feature_counts = self.ecci_cluster_feature_counts
        selected_feature = self.ecci_selected_feature

        cluster_key = str(predicted_cluster)
        if cluster_key in feature_counts:
            cluster_data = feature_counts[cluster_key]
            if isinstance(cluster_data, dict):
                sorted_items = sorted(cluster_data.items(), key=lambda x: x[1], reverse=True)[:10]
                labels = [str(item[0]) for item in sorted_items]
                values = [item[1] for item in sorted_items]

                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color='#3b82f6',
                        text=values,
                        textposition='auto',
                    )
                )

        fig.update_layout(
            title=f'Cluster {predicted_cluster} - {selected_feature}',
            xaxis_title=selected_feature,
            yaxis_title='Count',
            height=300,
            margin=dict(l=40, r=20, t=50, b=80),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45,
        )
        return fig

    @rx.var
    def ecci_all_clusters_feature_figure(self) -> go.Figure:
        """Generate grouped bar chart for all clusters' feature distribution."""
        fig = go.Figure()
        feature_counts = self.ecci_cluster_feature_counts
        selected_feature = self.ecci_selected_feature

        if feature_counts:
            all_values = set()
            for cluster_data in feature_counts.values():
                if isinstance(cluster_data, dict):
                    all_values.update(cluster_data.keys())

            value_totals = {}
            for val in all_values:
                total = sum(
                    cluster_data.get(val, 0)
                    for cluster_data in feature_counts.values()
                    if isinstance(cluster_data, dict)
                )
                value_totals[val] = total

            top_values = sorted(value_totals.keys(), key=lambda x: value_totals[x], reverse=True)[:10]

            colors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899']
            for i, (cluster_id, cluster_data) in enumerate(sorted(feature_counts.items())):
                if isinstance(cluster_data, dict):
                    values = [cluster_data.get(val, 0) for val in top_values]
                    fig.add_trace(
                        go.Bar(
                            name=f'Cluster {cluster_id}',
                            x=[str(v) for v in top_values],
                            y=values,
                            marker_color=colors[i % len(colors)],
                        )
                    )

        fig.update_layout(
            title=f'Feature Distribution: {selected_feature}',
            xaxis_title=selected_feature,
            yaxis_title='Count',
            barmode='group',
            height=500,
            margin=dict(l=40, r=20, t=50, b=150),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45,
            legend=dict(orientation='h', yanchor='top', y=-0.35, xanchor='center', x=0.5),
        )
        return fig

    # ==========================================================================
    # ECCI EVENT HANDLERS
    # ==========================================================================
    @rx.event
    def update_ecci(self, field: str, value):
        """Update an ECCI form field with automatic type conversion."""
        try:
            if field in self._ecci_float_fields:
                value = float(value) if value else 0.0
            elif field in self._ecci_int_fields:
                value = int(value) if value else 0
        except (ValueError, TypeError):
            return
        if "E-Commerce Customer Interactions" not in self.form_data:
            self.form_data["E-Commerce Customer Interactions"] = {}
        self.form_data["E-Commerce Customer Interactions"][field] = value

    @rx.event
    def generate_random_ecci_coordinates(self):
        """Generate random coordinates within Houston metro bounds for ECCI."""
        import random
        if "E-Commerce Customer Interactions" not in self.form_data:
            self.form_data["E-Commerce Customer Interactions"] = {}
        form = self.form_data["E-Commerce Customer Interactions"]
        form["lat"] = round(random.uniform(self._ecci_lat_bounds[0], self._ecci_lat_bounds[1]), 3)
        form["lon"] = round(random.uniform(self._ecci_lon_bounds[0], self._ecci_lon_bounds[1]), 3)
        self.prediction_results["E-Commerce Customer Interactions"] = {"show": False, "cluster": 0}

    @rx.event(background=True)
    async def predict_ecci(self):
        """Make prediction for E-Commerce Customer Interactions using current form state."""
        project_name = "E-Commerce Customer Interactions"
        current_form = self.form_data.get(project_name, {})
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"
        payload = {
            "project_name": project_name,
            "model_name": "DBSTREAM",
            "customer_id": current_form.get("customer_id", ""),
            "device_info": {
                "device_type": current_form.get("device_type", ""),
                "browser": current_form.get("browser", ""),
                "os": current_form.get("os", "")
            },
            "event_id": current_form.get("event_id", ""),
            "event_type": current_form.get("event_type", ""),
            "location": {
                "lat": float(current_form.get("lat", 0)),
                "lon": float(current_form.get("lon", 0))
            },
            "page_url": current_form.get("page_url", ""),
            "price": float(current_form.get("price", 0)),
            "product_category": current_form.get("product_category", ""),
            "product_id": current_form.get("product_id", ""),
            "quantity": int(current_form.get("quantity", 1)) if current_form.get("quantity") else None,
            "referrer_url": current_form.get("referrer_url", ""),
            "search_query": current_form.get("search_query", ""),
            "session_event_sequence": int(current_form.get("session_event_sequence", 1)),
            "session_id": current_form.get("session_id", ""),
            "time_on_page_seconds": int(current_form.get("time_on_page_seconds", 0)),
            "timestamp": timestamp
        }
        try:
            print(f"Making ECCI prediction with payload: {payload}")
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/predict",
                json=payload,
                timeout=30.0
            )
            result = response.json()
            print(f"ECCI Prediction result: {result}")
            cluster = result.get("cluster", 0)
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "cluster": cluster,
                        "model_source": result.get("model_source", "mlflow"),
                        "show": True
                    }
                }
            await self._fetch_mlflow_metrics_internal(project_name)
        except Exception as e:
            print(f"Error making ECCI prediction: {e}")
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "cluster": 0,
                        "show": False
                    }
                }

    @rx.event
    def randomize_ecci_form(self):
        """Generate random values locally using loaded dropdown options (instant)."""
        import random
        import uuid
        project_name = "E-Commerce Customer Interactions"
        opts = self.dropdown_options.get(project_name, {})
        now = dt.datetime.now()
        form_data = {
            "browser": random.choice(opts.get("browser") or ["Chrome"]),
            "device_type": random.choice(opts.get("device_type") or ["Desktop"]),
            "os": random.choice(opts.get("os") or ["Windows"]),
            "event_type": random.choice(opts.get("event_type") or ["page_view"]),
            "product_category": random.choice(opts.get("product_category") or ["Electronics"]),
            "product_id": random.choice(opts.get("product_id") or ["prod_1000"]),
            "referrer_url": random.choice(opts.get("referrer_url") or ["direct"]),
            "lat": str(round(random.uniform(29.5, 30.1), 3)),
            "lon": str(round(random.uniform(-95.8, -95.0), 3)),
            "price": str(round(random.uniform(9.99, 999.99), 2)),
            "quantity": str(random.randint(1, 5)),
            "session_event_sequence": str(random.randint(1, 20)),
            "time_on_page_seconds": str(random.randint(5, 300)),
            "timestamp_date": now.strftime("%Y-%m-%d"),
            "timestamp_time": now.strftime("%H:%M"),
            "customer_id": f"cust_{uuid.uuid4().hex[:8]}",
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "session_id": f"sess_{uuid.uuid4().hex[:10]}",
            "page_url": f"https://shop.example.com/products/{random.randint(1000, 9999)}",
            "search_query": random.choice(["", "", "laptop", "phone", "headphones", "shoes"]),
        }
        self.form_data[project_name] = form_data
        self.prediction_results[project_name] = {"cluster": 0, "show": False}
        yield rx.toast.success(
            "Form randomized",
            description="All fields filled with random values.",
            duration=2000,
        )

    # ==========================================================================
    # ECCI CLUSTER ANALYTICS EVENT HANDLERS
    # ==========================================================================
    @rx.event(background=True)
    async def fetch_ecci_cluster_counts(self):
        """Fetch cluster counts from FastAPI."""
        try:
            response = await httpx_client_get(
                url=f"{RIVER_BASE_URL}/cluster_counts",
                timeout=30.0
            )
            cluster_counts = response.json()
            async with self:
                self.ecci_cluster_counts = cluster_counts
        except Exception as e:
            print(f"Error fetching ECCI cluster counts: {e}")
            async with self:
                self.ecci_cluster_counts = {}

    @rx.event(background=True)
    async def fetch_ecci_cluster_feature_counts(self, feature: str = None):
        """Fetch feature counts per cluster from FastAPI."""
        if feature is None:
            feature = self.ecci_selected_feature
        device_info_features = ["device_type", "browser", "os"]
        column_name = "device_info" if feature in device_info_features else feature
        try:
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/cluster_feature_counts",
                json={"column_name": column_name},
                timeout=30.0
            )
            raw_data = response.json()
            if feature in device_info_features:
                processed_data = {}
                for cluster_id, device_counts in raw_data.items():
                    processed_data[cluster_id] = {}
                    for device_str, count in device_counts.items():
                        try:
                            import ast
                            device_dict = ast.literal_eval(device_str)
                            feature_value = device_dict.get(feature, "unknown")
                            if feature_value in processed_data[cluster_id]:
                                processed_data[cluster_id][feature_value] += count
                            else:
                                processed_data[cluster_id][feature_value] = count
                        except:
                            pass
                async with self:
                    self.ecci_cluster_feature_counts = processed_data
            else:
                async with self:
                    self.ecci_cluster_feature_counts = raw_data
        except Exception as e:
            print(f"Error fetching ECCI cluster feature counts: {e}")
            async with self:
                self.ecci_cluster_feature_counts = {}

    @rx.event
    def set_ecci_selected_feature(self, feature: str):
        """Set the selected feature and trigger data fetch."""
        self.ecci_selected_feature = feature
        return ECCIState.fetch_ecci_cluster_feature_counts(feature)

    # ==========================================================================
    # BATCH ML PREDICTION (Scikit-Learn)
    # ==========================================================================
    @rx.var
    def ecci_batch_prediction_show(self) -> bool:
        """Check if batch ECCI prediction results should be shown."""
        results = self.batch_prediction_results.get("E-Commerce Customer Interactions", {})
        if isinstance(results, dict):
            return results.get("show", False)
        return False

    @rx.var
    def ecci_batch_predicted_cluster(self) -> int:
        """Get batch predicted cluster ID."""
        results = self.batch_prediction_results.get("E-Commerce Customer Interactions", {})
        if isinstance(results, dict):
            return results.get("cluster_id", -1)
        return -1

    @rx.var
    def ecci_batch_prediction_figure(self) -> go.Figure:
        """Generate Plotly figure for batch ECCI cluster prediction."""
        cluster_id = self.ecci_batch_predicted_cluster

        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=cluster_id,
                title={'text': "<b>Cluster</b>", 'font': {'size': 18}},
                number={'font': {'size': 72, 'color': '#8b5cf6'}},  # Purple for batch ML
            )
        )
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    @rx.event(background=True)
    async def predict_batch_ecci(self):
        """Make batch prediction for ECCI using Scikit-Learn model."""
        from .shared import SKLEARN_BASE_URL
        project_name = "E-Commerce Customer Interactions"
        current_form = self.form_data.get(project_name, {})
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"

        payload = {
            "project_name": project_name,
            "model_name": "KMeans",
            "customer_id": current_form.get("customer_id", ""),
            "event_id": current_form.get("event_id", ""),
            "session_id": current_form.get("session_id", ""),
            "timestamp": timestamp,
            "event_type": current_form.get("event_type", ""),
            "product_id": current_form.get("product_id", ""),
            "product_category": current_form.get("product_category", ""),
            "price": float(current_form.get("price", 0)),
            "quantity": int(current_form.get("quantity", 1)),
            "lat": float(current_form.get("lat", 0)),
            "lon": float(current_form.get("lon", 0)),
            "page_url": current_form.get("page_url", ""),
            "referrer_url": current_form.get("referrer_url", ""),
            "search_query": current_form.get("search_query", ""),
            "time_on_page_seconds": int(current_form.get("time_on_page_seconds", 0)),
            "device_info": {
                "device_type": current_form.get("device_type", ""),
                "browser": current_form.get("browser", ""),
                "os": current_form.get("os", "")
            },
            "session_event_sequence": int(current_form.get("session_event_sequence", 1))
        }

        # Show loading toast
        yield rx.toast.info(
            "Making prediction...",
            description="Assigning customer to cluster with ML model.",
            duration=3000,
        )

        try:
            print(f"Making batch ECCI prediction with payload: {payload}")
            response = await httpx_client_post(
                url=f"{SKLEARN_BASE_URL}/predict",
                json=payload,
                timeout=30.0
            )
            result = response.json()
            print(f"Batch ECCI Prediction result: {result}")
            cluster_id = result.get("cluster_id", -1)

            async with self:
                self.batch_prediction_results = {
                    **self.batch_prediction_results,
                    project_name: {
                        "cluster_id": cluster_id,
                        "model_source": "sklearn",
                        "show": True
                    }
                }
            yield rx.toast.success(
                "Cluster Assigned",
                description=f"Customer assigned to Cluster {cluster_id}.",
                duration=3000,
            )
        except Exception as e:
            print(f"Error making batch ECCI prediction: {e}")
            async with self:
                self.batch_prediction_results = {
                    **self.batch_prediction_results,
                    project_name: {
                        "cluster_id": -1,
                        "model_source": "sklearn",
                        "show": False
                    }
                }
            yield rx.toast.error(
                "Prediction failed",
                description=str(e)[:100],
                duration=4000,
            )
