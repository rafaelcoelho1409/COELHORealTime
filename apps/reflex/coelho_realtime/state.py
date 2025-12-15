import reflex as rx
import os
import asyncio
import datetime as dt
import plotly.graph_objects as go
from .utils import httpx_client_post

FASTAPI_HOST = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_BASE_URL = f"http://{FASTAPI_HOST}:8001"


class State(rx.State):
    tab_name: str = "Incremental ML"
    page_name_mapping: dict = {
        "/": "Home",
        "/transaction-fraud-detection": "Transaction Fraud Detection",
        "/estimated-time-of-arrival": "Estimated Time of Arrival",
        "/e-commerce-customer-interactions": "E-Commerce Customer Interactions"
    }
    project_name: str = "Home"
    # Track the currently active Kafka producer/model
    # Only one model runs at a time to conserve resources
    activated_model: str = ""
    model_switch_message: str = ""
    model_switch_error: str = ""
    # ML training switch state - user-controlled toggle
    ml_training_enabled: bool = False
    _current_page_model_key: str = ""  # Track which model key belongs to current page
    incremental_ml_state: dict = {
        "Transaction Fraud Detection": False,
        "Estimated Time of Arrival": False,
        "E-Commerce Customer Interactions": False
    }
    incremental_ml_model_name: dict = {
        "Transaction Fraud Detection": "Adaptive Random Forest Classifier (River)",
    }
    incremental_ml_sample: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    form_data: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    dropdown_options: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    prediction_results: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }
    mlflow_metrics: dict = {
        "Transaction Fraud Detection": {},
        "Estimated Time of Arrival": {},
        "E-Commerce Customer Interactions": {}
    }

    ##==========================================================================
    ## VARS
    ##==========================================================================
    @rx.var
    def page_name(self) -> str:
        """Get the current page name based on the route."""
        current_path = self.router.url.path
        return self.page_name_mapping.get(current_path, "Home")

    @rx.var
    def tfd_form_data(self) -> dict:
        """Get Transaction Fraud Detection form data."""
        return self.form_data.get("Transaction Fraud Detection", {})

    # Transaction Fraud Detection dropdown options (consolidated)
    @rx.var
    def tfd_options(self) -> dict[str, list[str]]:
        """Get all TFD dropdown options as a dict."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        return opts if isinstance(opts, dict) else {}

    @rx.var
    def tfd_prediction_show(self) -> bool:
        """Check if prediction results should be shown."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return results.get("show", False)
        return False

    @rx.var
    def tfd_fraud_probability(self) -> float:
        """Get fraud probability."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return results.get("fraud_probability", 0.0)
        return 0.0

    @rx.var
    def tfd_prediction_text(self) -> str:
        """Get prediction result text."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return "Fraud" if results.get("prediction", 0) == 1 else "Not Fraud"
        return ""

    @rx.var
    def tfd_prediction_color(self) -> str:
        """Get prediction result color."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return "red" if results.get("prediction", 0) == 1 else "green"
        return "gray"

    @rx.var
    def tfd_fraud_gauge(self) -> go.Figure:
        """Generate Plotly gauge chart for fraud probability."""
        prob = self.tfd_fraud_probability * 100

        # Determine risk level and colors
        if prob < 30:
            risk_text = "LOW RISK"
            bar_color = "#22c55e"  # green
        elif prob < 70:
            risk_text = "MEDIUM RISK"
            bar_color = "#eab308"  # yellow
        else:
            risk_text = "HIGH RISK"
            bar_color = "#ef4444"  # red

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={
                'suffix': "%",
                'font': {'size': 40, 'color': bar_color}
            },
            title={
                'text': f"<b>{risk_text}</b><br><span style='font-size:14px;color:gray'>Fraud Probability</span>",
                'font': {'size': 18}
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "#666",
                    'tickvals': [0, 25, 50, 75, 100],
                    'ticktext': ['0%', '25%', '50%', '75%', '100%']
                },
                'bar': {'color': bar_color, 'thickness': 0.75},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.3)'},    # green zone
                    {'range': [30, 70], 'color': 'rgba(234, 179, 8, 0.3)'},   # yellow zone
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}   # red zone
                ],
                'threshold': {
                    'line': {'color': "#333", 'width': 4},
                    'thickness': 0.8,
                    'value': prob
                }
            }
        ))

        fig.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=80, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#888'}
        )

        return fig

    # Transaction Fraud Detection metrics (consolidated)
    @rx.var
    def tfd_metrics(self) -> dict[str, str]:
        """Get all TFD metrics as formatted percentage strings."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if not isinstance(metrics, dict):
            return {
                "f1": "0.00%", "accuracy": "0.00%", "recall": "0.00%",
                "precision": "0.00%", "rocauc": "0.00%", "geometric_mean": "0.00%"
            }
        return {
            "f1": f"{metrics.get('metrics.F1', 0) * 100:.2f}%",
            "accuracy": f"{metrics.get('metrics.Accuracy', 0) * 100:.2f}%",
            "recall": f"{metrics.get('metrics.Recall', 0) * 100:.2f}%",
            "precision": f"{metrics.get('metrics.Precision', 0) * 100:.2f}%",
            "rocauc": f"{metrics.get('metrics.ROCAUC', 0) * 100:.2f}%",
            "geometric_mean": f"{metrics.get('metrics.GeometricMean', 0) * 100:.2f}%",
        }

    @rx.var
    def ml_training_switch_checked(self) -> bool:
        """Check if ML training is enabled for the current page's model."""
        return self.ml_training_enabled and self.activated_model == self._current_page_model_key


    ##==========================================================================
    ## EVENTS
    ##==========================================================================
    @rx.event
    def set_current_page_model(self, model_key: str):
        """Set the model key for the current page (called on page mount)."""
        self._current_page_model_key = model_key
        # Reset switch state when entering a new page
        # Switch will be ON only if this page's model is already running
        self.ml_training_enabled = (self.activated_model == model_key)

    @rx.event(background = True)
    async def toggle_ml_training(self, enabled: bool, model_key: str, project_name: str):
        """
        Toggle ML training on/off via the switch component.
        This is the main user-facing control for starting/stopping Kafka streams.
        """
        async with self:
            self.ml_training_enabled = enabled
        if enabled:
            # Start the model
            if self.activated_model == model_key:
                yield rx.toast.info(
                    f"Already running",
                    description = f"ML training for {project_name} is already active",
                    duration = 3000,
                )
                return
            try:
                response = await httpx_client_post(
                    url = f"{FASTAPI_BASE_URL}/switch_model",
                    json = {
                        "model_key": model_key,
                        "project_name": project_name
                    },
                    timeout = 30.0
                )
                result = response.json()
                message = result.get("message", "Model switched successfully")
                async with self:
                    self.activated_model = model_key
                    self.model_switch_message = message
                    self.model_switch_error = ""
                    self.ml_training_enabled = True
                yield rx.toast.success(
                    f"Real-time ML training started",
                    description = f"Processing live data for {project_name}",
                    duration = 5000,
                    close_button = True,
                )
                print(f"Model start successful: {message}")
            except Exception as e:
                error_msg = f"Error starting model: {e}"
                print(error_msg)
                async with self:
                    self.model_switch_error = error_msg
                    self.model_switch_message = ""
                    self.ml_training_enabled = False
                    if self.activated_model == model_key:
                        self.activated_model = ""
                yield rx.toast.error(
                    f"Failed to start ML training",
                    description = str(e),
                    duration = 8000,
                    close_button = True,
                )
        else:
            # Stop the model
            if not self.activated_model:
                return
            try:
                await httpx_client_post(
                    url = f"{FASTAPI_BASE_URL}/switch_model",
                    json = {
                        "model_key": "none",
                        "project_name": ""
                    },
                    timeout = 30.0
                )
                async with self:
                    self.activated_model = ""
                    self.model_switch_message = "Model stopped"
                    self.ml_training_enabled = False
                yield rx.toast.info(
                    "Real-time ML training stopped",
                    description = f"Stopped processing for {project_name}",
                    duration = 3000,
                )
            except Exception as e:
                print(f"Error stopping model: {e}")
                yield rx.toast.warning(
                    "Could not stop ML training",
                    description = str(e),
                    duration = 5000,
                )

    @rx.event(background = True)
    async def cleanup_on_page_leave(self, project_name: str):
        """
        Called when user navigates away from a page.
        Stops the model if it was running for this page.
        """
        if self.activated_model and self._current_page_model_key:
            # Only stop if leaving the page that owns the running model
            if self.activated_model == self._current_page_model_key:
                try:
                    await httpx_client_post(
                        url = f"{FASTAPI_BASE_URL}/switch_model",
                        json = {
                            "model_key": "none",
                            "project_name": ""
                        },
                        timeout = 30.0
                    )
                    async with self:
                        self.activated_model = ""
                        self.model_switch_message = "Model stopped"
                        self.ml_training_enabled = False
                    yield rx.toast.info(
                        "Real-time ML training stopped",
                        description = f"Stopped processing for {project_name}",
                        duration = 3000,
                    )
                except Exception as e:
                    print(f"Error stopping model on page leave: {e}")

    @rx.event(background = True)
    async def update_sample(self, project_name: str):
        """Fetch initial sample from FastAPI (runs in background to avoid lock expiration)."""
        if project_name == "Home":
            async with self:
                self.incremental_ml_sample[project_name] = {}
            return
        try:
            sample = await httpx_client_post(
                url = f"{FASTAPI_BASE_URL}/initial_sample",
                json = {"project_name": project_name},
                timeout = 30.0
            )
            sample_data = sample.json()
            async with self:
                self.incremental_ml_sample[project_name] = sample_data
            # Initialize form data with sample
            if project_name == "Transaction Fraud Detection":
                await self._init_tfd_form_internal(sample_data)
        except Exception as e:
            print(f"Error fetching initial sample for {project_name}: {e}")
            async with self:
                self.incremental_ml_sample[project_name] = {}

    async def _init_tfd_form_internal(self, sample: dict):
        """Internal helper to initialize TFD form (called from background events)."""
        # Parse timestamp
        timestamp_str = sample.get("timestamp", "")
        timestamp_date = ""
        timestamp_time = ""
        if timestamp_str:
            try:
                timestamp = dt.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                timestamp_date = timestamp.strftime("%Y-%m-%d")
                timestamp_time = timestamp.strftime("%H:%M")
            except:
                timestamp_date = dt.datetime.now().strftime("%Y-%m-%d")
                timestamp_time = dt.datetime.now().strftime("%H:%M")
        form_data = {
            "amount": sample.get("amount", 0.0),
            "account_age_days": sample.get("account_age_days", 0),
            "timestamp_date": timestamp_date,
            "timestamp_time": timestamp_time,
            "currency": sample.get("currency", ""),
            "merchant_id": sample.get("merchant_id", ""),
            "product_category": sample.get("product_category", ""),
            "transaction_type": sample.get("transaction_type", ""),
            "payment_method": sample.get("payment_method", ""),
            "lat": sample.get("location", {}).get("lat", 0.0),
            "lon": sample.get("location", {}).get("lon", 0.0),
            "browser": sample.get("device_info", {}).get("browser", ""),
            "os": sample.get("device_info", {}).get("os", ""),
            "cvv_provided": sample.get("cvv_provided", False),
            "billing_address_match": sample.get("billing_address_match", False),
            "transaction_id": sample.get("transaction_id", ""),
            "user_id": sample.get("user_id", ""),
            "ip_address": sample.get("ip_address", ""),
            "user_agent": sample.get("user_agent", "")
        }
        async with self:
            self.form_data["Transaction Fraud Detection"] = form_data
        # Fetch dropdown options in parallel
        await self._fetch_tfd_options_internal()

    async def _fetch_tfd_options_internal(self):
        """Internal helper to fetch dropdown options in parallel."""
        project_name = "Transaction Fraud Detection"
        try:
            # Fetch all unique values in parallel for better performance
            responses = await asyncio.gather(
                httpx_client_post(
                    url = f"{FASTAPI_BASE_URL}/unique_values",
                    json = {"column_name": "currency", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{FASTAPI_BASE_URL}/unique_values",
                    json = {"column_name": "merchant_id", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{FASTAPI_BASE_URL}/unique_values",
                    json = {"column_name": "product_category", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{FASTAPI_BASE_URL}/unique_values",
                    json = {"column_name": "transaction_type", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{FASTAPI_BASE_URL}/unique_values",
                    json = {"column_name": "payment_method", "project_name": project_name},
                    timeout = 30.0
                ),
                httpx_client_post(
                    url = f"{FASTAPI_BASE_URL}/unique_values",
                    json = {"column_name": "device_info", "project_name": project_name},
                    timeout = 30.0
                ),
                return_exceptions = True
            )
            # Unpack responses
            currency_response, merchant_response, product_response, \
                transaction_response, payment_response, device_response = responses
            # Parse device info
            browsers = set()
            oses = set()
            if not isinstance(device_response, Exception):
                device_info_options = device_response.json().get("unique_values", [])
                for device_str in device_info_options:
                    try:
                        device_dict = eval(device_str)
                        browsers.add(device_dict.get("browser", ""))
                        oses.add(device_dict.get("os", ""))
                    except:
                        pass
            dropdown_options = {
                "currency": currency_response.json().get("unique_values", []) if not isinstance(currency_response, Exception) else [],
                "merchant_id": merchant_response.json().get("unique_values", []) if not isinstance(merchant_response, Exception) else [],
                "product_category": product_response.json().get("unique_values", []) if not isinstance(product_response, Exception) else [],
                "transaction_type": transaction_response.json().get("unique_values", []) if not isinstance(transaction_response, Exception) else [],
                "payment_method": payment_response.json().get("unique_values", []) if not isinstance(payment_response, Exception) else [],
                "browser": sorted(list(browsers)),
                "os": sorted(list(oses))
            }
            async with self:
                self.dropdown_options["Transaction Fraud Detection"] = dropdown_options
        except Exception as e:
            print(f"Error fetching dropdown options: {e}")
            async with self:
                self.dropdown_options["Transaction Fraud Detection"] = {}

    # TFD Form field update handler (consolidated)
    # Field type mappings for automatic conversion
    _tfd_float_fields = {"amount", "lat", "lon"}
    _tfd_int_fields = {"account_age_days"}
    _tfd_bool_fields = {"cvv_provided", "billing_address_match"}

    @rx.event
    def update_tfd(self, field: str, value):
        """Update a TFD form field with automatic type conversion."""
        try:
            if field in self._tfd_float_fields:
                value = float(value) if value else 0.0
            elif field in self._tfd_int_fields:
                value = int(value) if value else 0
            elif field in self._tfd_bool_fields:
                value = bool(value)
            # str fields need no conversion
        except (ValueError, TypeError):
            return  # Ignore invalid conversions
        current = self.form_data.get("Transaction Fraud Detection", {})
        current[field] = value
        self.form_data = {**self.form_data, "Transaction Fraud Detection": current}

    @rx.event
    async def init_transaction_fraud_detection_form(self, sample: dict):
        """Initialize Transaction Fraud Detection form with sample data."""
        await self._init_tfd_form_internal(sample)

    @rx.event
    async def fetch_transaction_fraud_detection_options(self):
        """Fetch unique values for all dropdown fields."""
        await self._fetch_tfd_options_internal()

    @rx.event(background = True)
    async def predict_transaction_fraud_detection(self):
        """Make prediction for transaction fraud detection using current form state."""
        project_name = "Transaction Fraud Detection"
        current_form = self.form_data.get(project_name, {})
        # Combine date and time
        timestamp = f"{current_form.get('timestamp_date', '')}T{current_form.get('timestamp_time', '')}:00.000000+00:00"
        # Prepare request payload from current form state
        payload = {
            "project_name": project_name,
            "model_name": "ARFClassifier",
            "transaction_id": current_form.get("transaction_id", ""),
            "user_id": current_form.get("user_id", ""),
            "timestamp": timestamp,
            "amount": float(current_form.get("amount", 0)),
            "currency": current_form.get("currency", ""),
            "merchant_id": current_form.get("merchant_id", ""),
            "product_category": current_form.get("product_category", ""),
            "transaction_type": current_form.get("transaction_type", ""),
            "payment_method": current_form.get("payment_method", ""),
            "location": {
                "lat": float(current_form.get("lat", 0)),
                "lon": float(current_form.get("lon", 0))
            },
            "ip_address": current_form.get("ip_address", ""),
            "device_info": {
                "os": current_form.get("os", ""),
                "browser": current_form.get("browser", "")
            },
            "user_agent": current_form.get("user_agent", ""),
            "account_age_days": int(current_form.get("account_age_days", 0)),
            "cvv_provided": bool(current_form.get("cvv_provided", False)),
            "billing_address_match": bool(current_form.get("billing_address_match", False))
        }
        # Make prediction
        try:
            print(f"Making prediction with payload: {payload}")
            response = await httpx_client_post(
                url = f"{FASTAPI_BASE_URL}/predict",
                json = payload,
                timeout = 30.0
            )
            result = response.json()
            print(f"Prediction result: {result}")
            # Create new dict to trigger reactivity
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "prediction": result.get("prediction"),
                        "fraud_probability": result.get("fraud_probability"),
                        "show": True
                    }
                }
        except Exception as e:
            print(f"Error making prediction: {e}")
            async with self:
                self.prediction_results = {
                    **self.prediction_results,
                    project_name: {
                        "prediction": None,
                        "fraud_probability": 0.0,
                        "show": False
                    }
                }

    @rx.event(background = True)
    async def get_mlflow_metrics(self, project_name: str):
        """Fetch MLflow metrics for a project (runs in background to avoid lock expiration)."""
        try:
            response = await httpx_client_post(
                url = f"{FASTAPI_BASE_URL}/mlflow_metrics",
                json = {
                    "project_name": project_name,
                    "model_name": "ARFClassifier"
                },
                timeout = 60.0
            )
            async with self:
                self.mlflow_metrics = {
                    **self.mlflow_metrics,
                    project_name: response.json()
                }
        except Exception as e:
            print(f"Error fetching MLflow metrics: {e}")
            async with self:
                self.mlflow_metrics = {
                    **self.mlflow_metrics,
                    project_name: {}
                }

    @rx.event(background = True)
    async def refresh_mlflow_metrics(self, project_name: str):
        """Force refresh MLflow metrics bypassing cache."""
        try:
            response = await httpx_client_post(
                url = f"{FASTAPI_BASE_URL}/mlflow_metrics",
                json = {
                    "project_name": project_name,
                    "model_name": "ARFClassifier",
                    "force_refresh": True
                },
                timeout = 60.0
            )
            async with self:
                self.mlflow_metrics = {
                    **self.mlflow_metrics,
                    project_name: response.json()
                }
            yield rx.toast.success(
                "Metrics refreshed",
                description = f"Latest metrics loaded for {project_name}",
                duration = 2000
            )
        except Exception as e:
            print(f"Error refreshing MLflow metrics: {e}")
            yield rx.toast.error(
                "Refresh failed",
                description = str(e),
                duration = 3000
            )