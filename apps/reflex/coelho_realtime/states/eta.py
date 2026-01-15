"""
Estimated Time of Arrival (ETA) state module.

This module contains:
- ETAState class with ETA-specific state variables and methods
- ETA form handlers
- ETA prediction methods
- ETA computed variables
- Map-related state (Folium)
"""
import reflex as rx
import datetime as dt
import math
import plotly.graph_objects as go
import folium
from .shared import (
    SharedState,
    RIVER_BASE_URL,
    safe_float_str,
)
from ..utils import httpx_client_post


class ETAState(SharedState):
    """Estimated Time of Arrival state.

    Inherits from SharedState to access common state variables while adding
    ETA-specific computed vars and event handlers.
    """

    # ==========================================================================
    # ETA FORM FIELD TYPE MAPPINGS (for automatic conversion)
    # ==========================================================================
    _eta_float_fields = {
        "origin_lat", "origin_lon", "destination_lat", "destination_lon",
        "temperature_celsius", "driver_rating",
        "debug_traffic_factor", "debug_weather_factor", "debug_driver_factor"
    }
    _eta_int_fields = {
        "hour_of_day", "debug_incident_delay_seconds",
        "initial_estimated_travel_time_seconds", "day_of_week"
    }
    # Coordinate fields that should reset prediction when changed
    _eta_coordinate_fields = {"origin_lat", "origin_lon", "destination_lat", "destination_lon"}
    # Houston metro bounds for random coordinate generation
    _eta_lat_bounds = (29.5, 30.1)
    _eta_lon_bounds = (-95.8, -95.0)
    # Average speed for initial ETA estimate (same as Kafka producer)
    _eta_avg_speed_kmh = 40

    # ==========================================================================
    # ETA COMPUTED VARIABLES
    # ==========================================================================
    @rx.var
    def eta_form_data(self) -> dict:
        """Get Estimated Time of Arrival form data."""
        return self.form_data.get("Estimated Time of Arrival", {})

    @rx.var
    def eta_options(self) -> dict[str, list[str]]:
        """Get all ETA dropdown options as a dict."""
        opts = self.dropdown_options.get("Estimated Time of Arrival", {})
        return opts if isinstance(opts, dict) else {}

    @rx.var
    def eta_prediction_show(self) -> bool:
        """Check if ETA prediction results should be shown."""
        results = self.prediction_results.get("Estimated Time of Arrival", {})
        if isinstance(results, dict):
            return results.get("show", False)
        return False

    @rx.var
    def eta_prediction_seconds(self) -> float:
        """Get ETA prediction in seconds."""
        results = self.prediction_results.get("Estimated Time of Arrival", {})
        if isinstance(results, dict):
            return results.get("eta_seconds", 0.0)
        return 0.0

    @rx.var
    def eta_prediction_minutes(self) -> float:
        """Get ETA prediction in minutes."""
        return round(self.eta_prediction_seconds / 60, 2) if self.eta_prediction_seconds > 0 else 0.0

    @rx.var
    def eta_metrics(self) -> dict[str, str]:
        """Get all ETA regression metrics as formatted strings."""
        metrics = self.mlflow_metrics.get("Estimated Time of Arrival", {})
        if not isinstance(metrics, dict):
            return {
                "mae": "0.00", "mape": "0.00", "mse": "0.00", "r2": "0.00",
                "rmse": "0.00", "rmsle": "0.00", "smape": "0.00",
                "rolling_mae": "0.00", "rolling_rmse": "0.00", "time_rolling_mae": "0.00"
            }
        return {
            "mae": f"{metrics.get('metrics.MAE', 0):.2f}",
            "mape": f"{metrics.get('metrics.MAPE', 0):.2f}",
            "mse": f"{metrics.get('metrics.MSE', 0):.2f}",
            "r2": f"{metrics.get('metrics.R2', 0):.2f}",
            "rmse": f"{metrics.get('metrics.RMSE', 0):.2f}",
            "rmsle": f"{metrics.get('metrics.RMSLE', 0):.2f}",
            "smape": f"{metrics.get('metrics.SMAPE', 0):.2f}",
            "rolling_mae": f"{metrics.get('metrics.RollingMAE', 0):.2f}",
            "rolling_rmse": f"{metrics.get('metrics.RollingRMSE', 0):.2f}",
            "time_rolling_mae": f"{metrics.get('metrics.TimeRollingMAE', 0):.2f}",
        }

    @rx.var
    def eta_metrics_raw(self) -> dict[str, float]:
        """Get raw ETA metrics as floats for dashboard calculations."""
        metrics = self.mlflow_metrics.get("Estimated Time of Arrival", {})
        if not isinstance(metrics, dict):
            return {
                "mae": 0, "mape": 0, "mse": 0, "r2": 0, "rmse": 0, "rmsle": 0, "smape": 0,
                "rolling_mae": 0, "rolling_rmse": 0, "time_rolling_mae": 0
            }
        return {
            "mae": metrics.get("metrics.MAE", 0),
            "mape": metrics.get("metrics.MAPE", 0),
            "mse": metrics.get("metrics.MSE", 0),
            "r2": metrics.get("metrics.R2", 0),
            "rmse": metrics.get("metrics.RMSE", 0),
            "rmsle": metrics.get("metrics.RMSLE", 0),
            "smape": metrics.get("metrics.SMAPE", 0),
            "rolling_mae": metrics.get("metrics.RollingMAE", 0),
            "rolling_rmse": metrics.get("metrics.RollingRMSE", 0),
            "time_rolling_mae": metrics.get("metrics.TimeRollingMAE", 0),
        }

    @rx.var
    def eta_dashboard_figures(self) -> dict:
        """Generate all ETA dashboard Plotly figures (KPI indicators, gauges)."""
        raw = self.eta_metrics_raw
        mlflow_data = self.mlflow_metrics.get("Estimated Time of Arrival", {})

        # Extract baseline metrics for delta calculation
        baseline = {
            "mae": float(mlflow_data.get("baseline_MAE", 0) or 0),
            "rmse": float(mlflow_data.get("baseline_RMSE", 0) or 0),
            "r2": float(mlflow_data.get("baseline_R2", 0) or 0),
            "rolling_mae": float(mlflow_data.get("baseline_RollingMAE", 0) or 0),
        }

        def create_kpi_regression(value: float, title: str, unit: str = "s",
                                   baseline_val: float = 0, lower_is_better: bool = True) -> go.Figure:
            """Create KPI indicator for regression metrics with delta from baseline."""
            if lower_is_better:
                if value <= 30:
                    color = "#3b82f6"
                elif value <= 60:
                    color = "#22c55e"
                elif value <= 120:
                    color = "#eab308"
                else:
                    color = "#ef4444"
            else:
                if value >= 0.9:
                    color = "#3b82f6"
                elif value >= 0.7:
                    color = "#22c55e"
                elif value >= 0.5:
                    color = "#eab308"
                else:
                    color = "#ef4444"

            delta_config = None
            if baseline_val > 0:
                delta_config = {
                    "reference": baseline_val,
                    "relative": True,
                    "valueformat": ".1%",
                    "increasing": {"color": "#ef4444" if lower_is_better else "#22c55e"},
                    "decreasing": {"color": "#22c55e" if lower_is_better else "#ef4444"}
                }

            fig = go.Figure(go.Indicator(
                mode="number+delta" if delta_config else "number",
                value=value,
                delta=delta_config,
                title={"text": f"<b>{title}</b>", "font": {"size": 14}},
                number={"suffix": unit, "font": {"size": 28, "color": color}, "valueformat": ".1f"}
            ))
            fig.update_layout(
                height=110, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_gauge_r2(value: float) -> go.Figure:
            """Create R2 gauge (-1 to 1 scale, higher is better)."""
            steps = [
                {"range": [-1, 0], "color": "#ef4444"},
                {"range": [0, 0.5], "color": "#eab308"},
                {"range": [0.5, 0.7], "color": "#22c55e"},
                {"range": [0.7, 1], "color": "#3b82f6"}
            ]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": "<b>R2 (Goodness of Fit)</b>", "font": {"size": 14}},
                number={"valueformat": ".3f", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 1},
                    "bar": {"color": "#1e40af"},
                    "steps": steps,
                    "threshold": {"value": 0.7, "line": {"color": "black", "width": 2}, "thickness": 0.75}
                }
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        def create_gauge_mape(value: float) -> go.Figure:
            """Create MAPE gauge (0 to 100% scale, lower is better)."""
            display_value = min(value, 100)
            steps = [
                {"range": [0, 10], "color": "#3b82f6"},
                {"range": [10, 25], "color": "#22c55e"},
                {"range": [25, 50], "color": "#eab308"},
                {"range": [50, 100], "color": "#ef4444"}
            ]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=display_value,
                title={"text": "<b>MAPE (% Error)</b>", "font": {"size": 14}},
                number={"suffix": "%", "valueformat": ".1f", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#1e40af"},
                    "steps": steps,
                    "threshold": {"value": 25, "line": {"color": "black", "width": 2}, "thickness": 0.75}
                }
            ))
            fig.update_layout(
                height=180, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        return {
            "kpi_mae": create_kpi_regression(raw["mae"], "MAE", "s", baseline["mae"], lower_is_better=True),
            "kpi_rmse": create_kpi_regression(raw["rmse"], "RMSE", "s", baseline["rmse"], lower_is_better=True),
            "kpi_r2": create_kpi_regression(raw["r2"], "R2", "", baseline["r2"], lower_is_better=False),
            "kpi_rolling_mae": create_kpi_regression(raw["rolling_mae"], "Rolling MAE", "s", baseline["rolling_mae"], lower_is_better=True),
            "gauge_r2": create_gauge_r2(raw["r2"]),
            "gauge_mape": create_gauge_mape(raw["mape"]),
        }

    @rx.var
    def eta_prediction_figure(self) -> go.Figure:
        """Generate Plotly figure for ETA prediction display."""
        seconds = self.eta_prediction_seconds
        minutes = self.eta_prediction_minutes

        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=seconds,
                title={'text': "<b>Seconds</b>", 'font': {'size': 18}},
                number={'font': {'size': 48, 'color': '#3b82f6'}},
                domain={'row': 0, 'column': 0}
            )
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=minutes,
                title={'text': "<b>Minutes</b>", 'font': {'size': 18}},
                number={'font': {'size': 48, 'color': '#22c55e'}},
                domain={'row': 1, 'column': 0}
            )
        )
        fig.update_layout(
            grid={'rows': 2, 'columns': 1, 'pattern': "independent"},
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    # ==========================================================================
    # MAP COORDINATE COMPUTED VARS
    # ==========================================================================
    def _safe_float(self, value, default: float) -> float:
        """Safely convert value to float, returning default on any error."""
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @rx.var
    def eta_origin_lat(self) -> float:
        """Get ETA origin latitude."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        return self._safe_float(form_data.get("origin_lat"), 29.8)

    @rx.var
    def eta_origin_lon(self) -> float:
        """Get ETA origin longitude."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        return self._safe_float(form_data.get("origin_lon"), -95.4)

    @rx.var
    def eta_destination_lat(self) -> float:
        """Get ETA destination latitude."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        return self._safe_float(form_data.get("destination_lat"), 29.8)

    @rx.var
    def eta_destination_lon(self) -> float:
        """Get ETA destination longitude."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        return self._safe_float(form_data.get("destination_lon"), -95.4)

    @rx.var
    def eta_folium_map_html(self) -> str:
        """Generate Folium map HTML for ETA origin/destination display."""
        origin_lat = self.eta_origin_lat
        origin_lon = self.eta_origin_lon
        dest_lat = self.eta_destination_lat
        dest_lon = self.eta_destination_lon

        center_lat = (origin_lat + dest_lat) / 2
        center_lon = (origin_lon + dest_lon) / 2

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='cartodbpositron'
        )

        folium.Marker(
            location=[origin_lat, origin_lon],
            popup='Origin',
            icon=folium.Icon(color='blue', icon='play')
        ).add_to(m)

        folium.Marker(
            location=[dest_lat, dest_lon],
            popup='Destination',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

        folium.PolyLine(
            locations=[[origin_lat, origin_lon], [dest_lat, dest_lon]],
            color='#333333',
            weight=3,
            opacity=0.8
        ).add_to(m)

        html = m._repr_html_()
        html = html.replace(
            'style="position:relative;width:100%;height:0;padding-bottom:60%;"',
            'style="width:100%;height:300px;"'
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

    @rx.var
    def eta_estimated_distance_km(self) -> float:
        """Calculate estimated distance using Haversine formula."""
        form_data = self.form_data.get("Estimated Time of Arrival") or {}
        if not isinstance(form_data, dict):
            return 0.0

        origin_lat = self._safe_float(form_data.get("origin_lat"), 0)
        origin_lon = self._safe_float(form_data.get("origin_lon"), 0)
        dest_lat = self._safe_float(form_data.get("destination_lat"), 0)
        dest_lon = self._safe_float(form_data.get("destination_lon"), 0)

        lon1, lat1, lon2, lat2 = map(math.radians, [origin_lon, origin_lat, dest_lon, dest_lat])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return round(c * r, 2)

    @rx.var
    def eta_initial_estimated_travel_time_seconds(self) -> int:
        """Calculate initial estimated travel time based on distance and average speed."""
        distance = self.eta_estimated_distance_km
        if distance <= 0:
            return 60
        travel_time = int((distance / self._eta_avg_speed_kmh) * 3600)
        return max(60, travel_time)

    # ==========================================================================
    # ETA EVENT HANDLERS
    # ==========================================================================
    @rx.event
    def update_eta(self, field: str, value):
        """Update an ETA form field with automatic type conversion."""
        try:
            if field in self._eta_float_fields:
                value = float(value) if value else 0.0
            elif field in self._eta_int_fields:
                value = int(value) if value else 0
        except (ValueError, TypeError):
            return
        if "Estimated Time of Arrival" not in self.form_data:
            self.form_data["Estimated Time of Arrival"] = {}
        self.form_data["Estimated Time of Arrival"][field] = value
        if field in self._eta_coordinate_fields:
            self.prediction_results["Estimated Time of Arrival"] = {"show": False, "eta_seconds": 0.0}

    @rx.event
    def generate_random_eta_coordinates(self):
        """Generate random coordinates within Houston metro bounds."""
        import random
        if "Estimated Time of Arrival" not in self.form_data:
            self.form_data["Estimated Time of Arrival"] = {}
        form = self.form_data["Estimated Time of Arrival"]
        form["origin_lat"] = round(random.uniform(self._eta_lat_bounds[0], self._eta_lat_bounds[1]), 6)
        form["origin_lon"] = round(random.uniform(self._eta_lon_bounds[0], self._eta_lon_bounds[1]), 6)
        while True:
            dest_lat = round(random.uniform(self._eta_lat_bounds[0], self._eta_lat_bounds[1]), 6)
            dest_lon = round(random.uniform(self._eta_lon_bounds[0], self._eta_lon_bounds[1]), 6)
            if abs(form["origin_lat"] - dest_lat) >= 0.01 or abs(form["origin_lon"] - dest_lon) >= 0.01:
                break
        form["destination_lat"] = dest_lat
        form["destination_lon"] = dest_lon
        self.prediction_results["Estimated Time of Arrival"] = {"show": False, "eta_seconds": 0.0}

    @rx.event(background=True)
    async def predict_eta(self):
        """Make prediction for Estimated Time of Arrival using current form state."""
        project_name = "Estimated Time of Arrival"
        current_form = self.form_data.get(project_name, {})
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"
        payload = {
            "project_name": project_name,
            "model_name": "ARFRegressor",
            "trip_id": current_form.get("trip_id", ""),
            "driver_id": current_form.get("driver_id", ""),
            "vehicle_id": current_form.get("vehicle_id", ""),
            "timestamp": timestamp,
            "origin": {
                "lat": float(current_form.get("origin_lat", 0)),
                "lon": float(current_form.get("origin_lon", 0))
            },
            "destination": {
                "lat": float(current_form.get("destination_lat", 0)),
                "lon": float(current_form.get("destination_lon", 0))
            },
            "estimated_distance_km": self.eta_estimated_distance_km,
            "weather": current_form.get("weather", ""),
            "temperature_celsius": float(current_form.get("temperature_celsius", 0)),
            "day_of_week": int(current_form.get("day_of_week", 0)),
            "hour_of_day": int(current_form.get("hour_of_day", 0)),
            "driver_rating": float(current_form.get("driver_rating", 0)),
            "vehicle_type": current_form.get("vehicle_type", ""),
            "initial_estimated_travel_time_seconds": self.eta_initial_estimated_travel_time_seconds,
            "debug_traffic_factor": float(current_form.get("debug_traffic_factor", 0)),
            "debug_weather_factor": float(current_form.get("debug_weather_factor", 0)),
            "debug_incident_delay_seconds": int(current_form.get("debug_incident_delay_seconds", 0)),
            "debug_driver_factor": float(current_form.get("debug_driver_factor", 0))
        }
        try:
            print(f"Making ETA prediction with payload: {payload}")
            response = await httpx_client_post(
                url=f"{RIVER_BASE_URL}/predict",
                json=payload,
                timeout=30.0
            )
            result = response.json()
            print(f"ETA Prediction result: {result}")
            eta_seconds = result.get("Estimated Time of Arrival", 0.0)
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "eta_seconds": eta_seconds,
                        "model_source": result.get("model_source", "mlflow"),
                        "show": True
                    }
                }
            await self._fetch_mlflow_metrics_internal(project_name)
        except Exception as e:
            print(f"Error making ETA prediction: {e}")
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "eta_seconds": 0.0,
                        "model_source": "mlflow",
                        "show": False
                    }
                }

    @rx.event
    def randomize_eta_form(self):
        """Generate random values locally using loaded dropdown options (instant)."""
        import random
        import uuid
        project_name = "Estimated Time of Arrival"
        opts = self.dropdown_options.get(project_name, {})
        now = dt.datetime.now()
        origin_lat = round(random.uniform(29.5, 30.1), 6)
        origin_lon = round(random.uniform(-95.8, -95.0), 6)
        dest_lat = round(random.uniform(29.5, 30.1), 6)
        dest_lon = round(random.uniform(-95.8, -95.0), 6)
        distance_km = round(abs(origin_lat - dest_lat) * 111 + abs(origin_lon - dest_lon) * 85, 2)
        form_data = {
            "driver_id": random.choice(opts.get("driver_id") or ["driver_1000"]),
            "vehicle_id": random.choice(opts.get("vehicle_id") or ["vehicle_100"]),
            "weather": random.choice(opts.get("weather") or ["Clear"]),
            "vehicle_type": random.choice(opts.get("vehicle_type") or ["Sedan"]),
            "timestamp_date": now.strftime("%Y-%m-%d"),
            "timestamp_time": now.strftime("%H:%M"),
            "origin_lat": str(origin_lat),
            "origin_lon": str(origin_lon),
            "destination_lat": str(dest_lat),
            "destination_lon": str(dest_lon),
            "hour_of_day": str(random.randint(0, 23)),
            "day_of_week": str(random.randint(0, 6)),
            "driver_rating": str(round(random.uniform(3.5, 5.0), 1)),
            "temperature_celsius": str(round(random.uniform(15.0, 35.0), 1)),
            "estimated_distance_km": str(distance_km),
            "initial_estimated_travel_time_seconds": str(int(distance_km * 60)),
            "debug_traffic_factor": str(round(random.uniform(0.8, 1.5), 2)),
            "debug_weather_factor": str(round(random.uniform(0.9, 1.3), 2)),
            "debug_incident_delay_seconds": str(random.choice([0, 0, 0, 60, 120, 300])),
            "debug_driver_factor": str(round(random.uniform(0.9, 1.1), 2)),
            "trip_id": f"trip_{uuid.uuid4().hex[:12]}",
        }
        self.form_data[project_name] = form_data
        self.prediction_results[project_name] = {"eta_seconds": 0.0, "show": False}
