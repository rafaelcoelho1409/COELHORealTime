import reflex as rx
import os
import datetime as dt
import asyncio
from .utils import (
    httpx_client_get,
    httpx_client_post
)

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

    @rx.var
    def tfd_dropdown_options(self) -> dict:
        """Get Transaction Fraud Detection dropdown options."""
        return self.dropdown_options.get("Transaction Fraud Detection", {})

    @rx.var
    def tfd_prediction_results(self) -> dict:
        """Get Transaction Fraud Detection prediction results."""
        return self.prediction_results.get("Transaction Fraud Detection", {})

    # Transaction Fraud Detection dropdown options with proper types
    @rx.var
    def tfd_currency_options(self) -> list[str]:
        """Get currency dropdown options."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        if isinstance(opts, dict):
            return opts.get("currency", [])
        return []

    @rx.var
    def tfd_merchant_id_options(self) -> list[str]:
        """Get merchant ID dropdown options."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        if isinstance(opts, dict):
            return opts.get("merchant_id", [])
        return []

    @rx.var
    def tfd_product_category_options(self) -> list[str]:
        """Get product category dropdown options."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        if isinstance(opts, dict):
            return opts.get("product_category", [])
        return []

    @rx.var
    def tfd_transaction_type_options(self) -> list[str]:
        """Get transaction type dropdown options."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        if isinstance(opts, dict):
            return opts.get("transaction_type", [])
        return []

    @rx.var
    def tfd_payment_method_options(self) -> list[str]:
        """Get payment method dropdown options."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        if isinstance(opts, dict):
            return opts.get("payment_method", [])
        return []

    @rx.var
    def tfd_browser_options(self) -> list[str]:
        """Get browser dropdown options."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        if isinstance(opts, dict):
            return opts.get("browser", [])
        return []

    @rx.var
    def tfd_os_options(self) -> list[str]:
        """Get OS dropdown options."""
        opts = self.dropdown_options.get("Transaction Fraud Detection", {})
        if isinstance(opts, dict):
            return opts.get("os", [])
        return []

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
    def tfd_is_fraud(self) -> bool:
        """Check if prediction is fraud."""
        results = self.prediction_results.get("Transaction Fraud Detection", {})
        if isinstance(results, dict):
            return results.get("prediction", 0) == 1
        return False

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
    def tfd_mlflow_metrics(self) -> dict:
        """Get Transaction Fraud Detection MLflow metrics."""
        return self.mlflow_metrics.get("Transaction Fraud Detection", {})

    # Individual metric computed vars
    @rx.var
    def tfd_metric_f1(self) -> str:
        """Get F1 metric as percentage string."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if isinstance(metrics, dict):
            value = metrics.get("metrics.F1", 0)
            return f"{value * 100:.2f}%"
        return "0.00%"

    @rx.var
    def tfd_metric_accuracy(self) -> str:
        """Get Accuracy metric as percentage string."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if isinstance(metrics, dict):
            value = metrics.get("metrics.Accuracy", 0)
            return f"{value * 100:.2f}%"
        return "0.00%"

    @rx.var
    def tfd_metric_recall(self) -> str:
        """Get Recall metric as percentage string."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if isinstance(metrics, dict):
            value = metrics.get("metrics.Recall", 0)
            return f"{value * 100:.2f}%"
        return "0.00%"

    @rx.var
    def tfd_metric_precision(self) -> str:
        """Get Precision metric as percentage string."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if isinstance(metrics, dict):
            value = metrics.get("metrics.Precision", 0)
            return f"{value * 100:.2f}%"
        return "0.00%"

    @rx.var
    def tfd_metric_rocauc(self) -> str:
        """Get ROCAUC metric as percentage string."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if isinstance(metrics, dict):
            value = metrics.get("metrics.ROCAUC", 0)
            return f"{value * 100:.2f}%"
        return "0.00%"

    @rx.var
    def tfd_metric_geometric_mean(self) -> str:
        """Get GeometricMean metric as percentage string."""
        metrics = self.mlflow_metrics.get("Transaction Fraud Detection", {})
        if isinstance(metrics, dict):
            value = metrics.get("metrics.GeometricMean", 0)
            return f"{value * 100:.2f}%"
        return "0.00%"

    @rx.var
    def ml_training_switch_checked(self) -> bool:
        """Check if ML training is enabled for the current page's model."""
        return self.ml_training_enabled and self.activated_model == self._current_page_model_key


    ##==========================================================================
    ## EVENTS
    ##==========================================================================
    @rx.event
    def change_incremental_ml_state(self, page_name: str, state: bool):
        self.incremental_ml_state[page_name] = state

    @rx.event
    def set_current_page_model(self, model_key: str):
        """Set the model key for the current page (called on page mount)."""
        self._current_page_model_key = model_key
        # Reset switch state when entering a new page
        # Switch will be ON only if this page's model is already running
        self.ml_training_enabled = (self.activated_model == model_key)

    @rx.event(background=True)
    async def toggle_ml_training(self, enabled: bool, model_key: str, project_name: str):
        """
        Toggle ML training on/off via the switch component.
        This is the main user-facing control for starting/stopping Kafka streams.
        """
        async with self:
            self.ml_training_enabled = enabled

        if enabled:
            # Start the model - inline the logic here since we can't call other background events
            if self.activated_model == model_key:
                yield rx.toast.info(
                    f"Already running",
                    description=f"ML training for {project_name} is already active",
                    duration=3000,
                )
                return

            try:
                response = await httpx_client_post(
                    url=f"{FASTAPI_BASE_URL}/switch_model",
                    json={
                        "model_key": model_key,
                        "project_name": project_name
                    },
                    timeout=30.0
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
                    description=f"Processing live data for {project_name}",
                    duration=5000,
                    close_button=True,
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
                    description=str(e),
                    duration=8000,
                    close_button=True,
                )
        else:
            # Stop the model
            if not self.activated_model:
                return

            try:
                await httpx_client_post(
                    url=f"{FASTAPI_BASE_URL}/switch_model",
                    json={
                        "model_key": "none",
                        "project_name": ""
                    },
                    timeout=30.0
                )
                async with self:
                    self.activated_model = ""
                    self.model_switch_message = "Model stopped"
                    self.ml_training_enabled = False

                yield rx.toast.info(
                    "Real-time ML training stopped",
                    description=f"Stopped processing for {project_name}",
                    duration=3000,
                )
            except Exception as e:
                print(f"Error stopping model: {e}")
                yield rx.toast.warning(
                    "Could not stop ML training",
                    description=str(e),
                    duration=5000,
                )

    @rx.event(background=True)
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
                        url=f"{FASTAPI_BASE_URL}/switch_model",
                        json={
                            "model_key": "none",
                            "project_name": ""
                        },
                        timeout=30.0
                    )
                    async with self:
                        self.activated_model = ""
                        self.model_switch_message = "Model stopped"
                        self.ml_training_enabled = False

                    yield rx.toast.info(
                        "Real-time ML training stopped",
                        description=f"Stopped processing for {project_name}",
                        duration=3000,
                    )
                except Exception as e:
                    print(f"Error stopping model on page leave: {e}")

    @rx.event(background=True)
    async def switch_active_model(self, model_key: str, project_name: str):
        """Start a model (called from toggle or directly)."""
        if self.activated_model == model_key:
            return

        try:
            response = await httpx_client_post(
                url=f"{FASTAPI_BASE_URL}/switch_model",
                json={
                    "model_key": model_key,
                    "project_name": project_name
                },
                timeout=30.0
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
                description=f"Processing live data for {project_name}",
                duration=5000,
                close_button=True,
            )

        except Exception as e:
            error_msg = f"Error starting model: {e}"
            print(error_msg)
            async with self:
                self.model_switch_error = error_msg
                self.model_switch_message = ""
                self.ml_training_enabled = False

            yield rx.toast.error(
                f"Failed to start ML training",
                description=str(e),
                duration=8000,
                close_button=True,
            )

    @rx.event(background=True)
    async def stop_active_model(self):
        """Stop the currently active model."""
        if not self.activated_model:
            return

        try:
            await httpx_client_post(
                url=f"{FASTAPI_BASE_URL}/switch_model",
                json={
                    "model_key": "none",
                    "project_name": ""
                },
                timeout=30.0
            )
            async with self:
                self.activated_model = ""
                self.model_switch_message = "Model stopped"
                self.ml_training_enabled = False

            yield rx.toast.info(
                "Real-time ML training stopped",
                description="All background processes have been stopped",
                duration=3000,
            )
        except Exception as e:
            print(f"Error stopping model: {e}")
            yield rx.toast.warning(
                "Could not stop ML training",
                description=str(e),
                duration=5000,
            )

    def get_model_key(self, project_name: str, model_type: str = "river") -> str:
        """Generate model key from project name and type."""
        return f"{project_name.replace(' ', '_').replace('-', '_').lower()}_{model_type}.py"

    @rx.event(background=True)
    async def update_sample(self, project_name: str):
        """Fetch initial sample from FastAPI (runs in background to avoid lock expiration)."""
        if project_name == "Home":
            async with self:
                self.incremental_ml_sample[project_name] = {}
            return

        try:
            sample = await httpx_client_post(
                url=f"{FASTAPI_BASE_URL}/initial_sample",
                json={"project_name": project_name},
                timeout=30.0
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
                    url=f"{FASTAPI_BASE_URL}/unique_values",
                    json={"column_name": "currency", "project_name": project_name},
                    timeout=30.0
                ),
                httpx_client_post(
                    url=f"{FASTAPI_BASE_URL}/unique_values",
                    json={"column_name": "merchant_id", "project_name": project_name},
                    timeout=30.0
                ),
                httpx_client_post(
                    url=f"{FASTAPI_BASE_URL}/unique_values",
                    json={"column_name": "product_category", "project_name": project_name},
                    timeout=30.0
                ),
                httpx_client_post(
                    url=f"{FASTAPI_BASE_URL}/unique_values",
                    json={"column_name": "transaction_type", "project_name": project_name},
                    timeout=30.0
                ),
                httpx_client_post(
                    url=f"{FASTAPI_BASE_URL}/unique_values",
                    json={"column_name": "payment_method", "project_name": project_name},
                    timeout=30.0
                ),
                httpx_client_post(
                    url=f"{FASTAPI_BASE_URL}/unique_values",
                    json={"column_name": "device_info", "project_name": project_name},
                    timeout=30.0
                ),
                return_exceptions=True
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

    @rx.event
    async def init_transaction_fraud_detection_form(self, sample: dict):
        """Initialize Transaction Fraud Detection form with sample data."""
        await self._init_tfd_form_internal(sample)

    @rx.event
    async def fetch_transaction_fraud_detection_options(self):
        """Fetch unique values for all dropdown fields."""
        await self._fetch_tfd_options_internal()

    @rx.event
    async def predict_transaction_fraud_detection(self, form_data: dict):
        """Make prediction for transaction fraud detection."""
        project_name = "Transaction Fraud Detection"
        current_form = self.form_data[project_name]
        # Combine date and time
        timestamp = f"{form_data.get('timestamp_date', current_form.get('timestamp_date'))}T{form_data.get('timestamp_time', current_form.get('timestamp_time'))}:00.000000+00:00"
        # Prepare request payload
        payload = {
            "project_name": project_name,
            "model_name": "ARFClassifier",
            "transaction_id": current_form.get("transaction_id"),
            "user_id": current_form.get("user_id"),
            "timestamp": timestamp,
            "amount": float(form_data.get("amount", current_form.get("amount"))),
            "currency": form_data.get("currency", current_form.get("currency")),
            "merchant_id": form_data.get("merchant_id", current_form.get("merchant_id")),
            "product_category": form_data.get("product_category", current_form.get("product_category")),
            "transaction_type": form_data.get("transaction_type", current_form.get("transaction_type")),
            "payment_method": form_data.get("payment_method", current_form.get("payment_method")),
            "location": {
                "lat": float(form_data.get("lat", current_form.get("lat"))),
                "lon": float(form_data.get("lon", current_form.get("lon")))
            },
            "ip_address": current_form.get("ip_address"),
            "device_info": {
                "os": form_data.get("os", current_form.get("os")),
                "browser": form_data.get("browser", current_form.get("browser"))
            },
            "user_agent": current_form.get("user_agent"),
            "account_age_days": int(form_data.get("account_age_days", current_form.get("account_age_days"))),
            "cvv_provided": form_data.get("cvv_provided", str(current_form.get("cvv_provided"))).lower() == "true",
            "billing_address_match": form_data.get("billing_address_match", str(current_form.get("billing_address_match"))).lower() == "true"
        }
        # Make prediction
        try:
            response = await httpx_client_post(
                url=f"{FASTAPI_BASE_URL}/predict",
                json=payload,
                timeout=30.0
            )
            result = response.json()
            self.prediction_results[project_name] = {
                "prediction": result.get("prediction"),
                "fraud_probability": result.get("fraud_probability"),
                "show": True
            }
        except Exception as e:
            print(f"Error making prediction: {e}")
            self.prediction_results[project_name] = {
                "prediction": None,
                "fraud_probability": 0.0,
                "show": False
            }

    @rx.event(background=True)
    async def get_mlflow_metrics(self, project_name: str):
        """Fetch MLflow metrics for a project (runs in background to avoid lock expiration)."""
        try:
            response = await httpx_client_post(
                url=f"{FASTAPI_BASE_URL}/mlflow_metrics",
                json={
                    "project_name": project_name,
                    "model_name": "ARFClassifier"
                },
                timeout=60.0
            )
            async with self:
                self.mlflow_metrics[project_name] = response.json()
        except Exception as e:
            print(f"Error fetching MLflow metrics: {e}")
            async with self:
                self.mlflow_metrics[project_name] = {}