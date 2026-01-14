# COELHO RealTime - Project Roadmap

## Project Overview
Real-time ML platform with incremental learning, featuring:
- Transaction Fraud Detection (River ML)
- Estimated Time of Arrival (River ML)
- E-Commerce Customer Interactions (DBSTREAM Clustering)

**Tech Stack:** Reflex, River ML, Sklearn, Kafka, MLflow, k3d/Kubernetes

---

## Completed Milestones

### Phase 1: Core Infrastructure
- [x] k3d Kubernetes cluster setup
- [x] Helm chart for deployment
- [x] Kafka for real-time data streaming
- [x] FastAPI backend with River ML models
- [x] MLflow for experiment tracking

### Phase 2: Streamlit MVP (DEPRECATED - Migrated to Reflex)
- [x] Transaction Fraud Detection page → Migrated to Reflex
- [x] Estimated Time of Arrival page → Migrated to Reflex
- [x] E-Commerce Customer Interactions page → Migrated to Reflex
- [x] Real-time training toggle → Migrated to Reflex
- [x] Prediction forms and visualizations → Migrated to Reflex
- [x] **Streamlit app deleted** - fully replaced by Reflex (Phase 3)

### Phase 3: Reflex Migration (COMPLETED)
- [x] Migrate TFD page to Reflex
- [x] Migrate ETA page to Reflex
- [x] Migrate ECCI page to Reflex
- [x] Add cluster analytics tabs
- [x] Add "Randomize All Fields" buttons
- [x] Fix form field display issues (0 values, None handling)
- [x] Update navbar with official service logos
- [x] Improve page layouts (always-visible maps)

### Phase 6: ML Training Service Separation (COMPLETED)
- [x] Design new service architecture (River ML Training Service)
- [x] Create River FastAPI service with endpoints:
  - [x] `/switch_model` - Start/stop ML training
  - [x] `/predict` - Model predictions (River + sklearn)
  - [x] `/status`, `/health` - Service monitoring
- [x] Add Helm templates for River service (deployment, service, configmap)
- [x] Update Reflex to call River directly for training and predictions
- [x] Clean up FastAPI (removed training code, now Analytics-only)
- [x] Update navbar with nested FastAPI submenu (Analytics, River, Scikit-Learn)

**Current Architecture:**
```
Reflex → River (training + predictions)
      → FastAPI Analytics (data queries, yellowbrick)

Kafka → River ML Training Scripts → MLflow
```

---

## Upcoming Phases

### Phase 4: Storage & Persistence (COMPLETED)
**Goal:** Proper artifact storage with MinIO

- [x] Add MinIO Helm dependency (Bitnami chart)
- [x] Configure MinIO buckets:
  - `mlflow-artifacts` - MLflow models and artifacts
  - `delta-lake` - Delta Lake tables
  - `raw-data` - Raw Kafka data snapshots
- [x] Update MLflow to use MinIO as artifact store
- [x] Configure MLflow environment variables:
  ```
  MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  AWS_ACCESS_KEY_ID=<minio-access-key>
  AWS_SECRET_ACCESS_KEY=<minio-secret-key>
  ```
- [x] **Transfer existing saved models and encoders to MLflow/MinIO**
  - All models now saved exclusively to MLflow (backed by MinIO S3)
  - Local `apps/river/encoders/` and `apps/river/models/` directories deprecated and can be deleted
- [x] Test model logging and retrieval from MinIO
- [x] Add MinIO to Services dropdown in navbar

### Phase 5: MLflow Model Integration (Priority: HIGH) - PARTIALLY COMPLETED
**Goal:** Connect Reflex pages to latest/best models registered on MLflow instead of loading in-memory models

- [x] Update River to load models from MLflow registry:
  - [x] Query MLflow for **best model** per project (by metrics)
  - [x] Load model artifacts from MinIO via MLflow
  - [x] Implement `get_best_mlflow_run()` with metric-based selection:
    - Transaction Fraud Detection: Maximize F1
    - Estimated Time of Arrival: Minimize MAE
    - Sales Forecasting: Minimize MAE
    - E-Commerce Customer Interactions: Latest run (no metrics yet)
  - [x] Load encoders from the same best run as the model
  - [x] Load cluster data (ECCI) from MLflow artifacts
- [x] Best model continuation training:
  - [x] New training runs load the best existing model and continue training
  - [x] Encoders are loaded from the same run to maintain consistency
  - [x] Old runs are preserved (not deleted)
- [ ] Implement model hot-reloading:
  - [ ] Periodic check for new model versions
  - [ ] Graceful model swap without downtime
- [x] Update prediction endpoints:
  - [x] Use MLflow-loaded models for inference
  - [ ] Return model version in prediction response
- [ ] Add model info to Reflex UI:
  - [ ] Display current model version on each page
  - [ ] Show last model update timestamp
- [ ] **ECCI Clustering Metrics** (Priority: MEDIUM)
  - [ ] Add clustering metrics to ECCI training script:
    - Silhouette Score (via cluster membership)
    - Number of clusters formed
    - Samples per cluster distribution
  - [ ] Log metrics to MLflow for best model selection
  - [ ] Update `BEST_METRIC_CRITERIA["E-Commerce Customer Interactions"]` to use cluster metrics

### Phase 6: Scikit-Learn Service (COMPLETED)
**Goal:** Create dedicated Scikit-Learn FastAPI service for batch ML

- [x] Create Scikit-Learn FastAPI service with endpoints:
  - [x] `/predict` - Batch ML predictions (XGBClassifier via MLflow)
  - [x] `/mlflow_metrics` - Get MLflow metrics for batch models
  - [x] `/yellowbrick_metric` - Generate YellowBrick visualizations
  - [x] `/sample`, `/health` - Service monitoring
- [x] Add Helm templates for Scikit-Learn service (deployment, service, configmap)
- [x] Add Batch ML tab to Reflex TFD page
- [x] Update Reflex state with batch ML handlers (predictions, YellowBrick)
- [x] Enable Scikit-Learn option in navbar (removed "Soon" badge)
- [x] Clean up sklearn service (removed River code from copied FastAPI)

**Sklearn Service Architecture:**
```
apps/sklearn/
├── app.py              # FastAPI (port 8003)
├── functions.py        # Sklearn/YellowBrick helpers
├── Dockerfile.sklearn  # Docker image with UV
└── entrypoint.sh       # Startup script
```

### Phase 6b: Consolidate FastAPI Services (COMPLETED)
**Goal:** Deprecate FastAPI Analytics and distribute its endpoints to River and Sklearn

**Final State (2 services):**
- River (8002) - Incremental ML + data sampling + form helpers + cluster analytics
- Sklearn (8003) - Batch ML + YellowBrick + batch metrics

**Migration Completed:**

| Endpoint | From | To | Status |
|----------|------|-----|--------|
| `/sample` | FastAPI | River | ✓ Done |
| `/unique_values` | FastAPI | River | ✓ Done |
| `/initial_sample` | FastAPI | River | ✓ Done |
| `/get_ordinal_encoder` | FastAPI | River | ✓ Done |
| `/cluster_counts` | FastAPI | River | ✓ Done |
| `/cluster_feature_counts` | FastAPI | River | ✓ Done |
| `/mlflow_metrics` | FastAPI | River | ✓ Done |
| `/healthcheck` | FastAPI | River + Sklearn | ✓ Done |
| `/yellowbrick_metric` | FastAPI | Sklearn | ✓ Already there |

**Completed Tasks:**
- [x] Move data endpoints to River service (`/sample`, `/unique_values`, `/initial_sample`)
- [x] Move River-specific endpoints to River service (`/get_ordinal_encoder`, `/cluster_counts`, `/cluster_feature_counts`)
- [x] Add `/mlflow_metrics` to River service (for incremental models)
- [x] Update Reflex state.py to use River endpoints instead of FastAPI
- [x] Update Reflex resources.py - removed FASTAPI_BASE_URL references
- [x] Remove FastAPI Analytics service:
  - [x] Deleted `apps/fastapi/` directory
  - [x] Removed Helm templates (`k3d/helm/templates/fastapi/`)
  - [x] Removed `fastapi` section from `values.yaml`
- [x] Update navbar Services dropdown (removed FastAPI Analytics submenu)

**Benefits Achieved:**
- 2 services instead of 3 (simpler architecture)
- Clear separation: Incremental ML (River) vs Batch ML (Sklearn)
- No code duplication
- Easier maintenance and deployment

### Phase 7: Observability Stack (PARTIALLY COMPLETED)
**Goal:** Production-grade monitoring with Prometheus & Grafana

- [x] Add kube-prometheus-stack Helm dependency (v80.6.0)
- [x] Configure ServiceMonitors for:
  - [x] FastAPI metrics (prometheus-fastapi-instrumentator)
  - [x] River ML Training Service metrics (prometheus-fastapi-instrumentator)
  - [x] PostgreSQL metrics (Bitnami exporter)
  - [x] Redis metrics (Bitnami exporter)
  - [x] MinIO metrics (ServiceMonitor with cluster/node metrics)
  - [ ] Kafka metrics (disabled - JMX exporter incompatible with Kafka 4.0)
  - [x] MLflow metrics (enabled via Bitnami chart `tracking.metrics.enabled`)
  - [x] Reflex backend metrics (prometheus-fastapi-instrumentator added)
- [x] Create custom Grafana dashboards:
  - [x] COELHORealTime Overview (service health, CPU, memory, network)
  - [x] ML Pipeline Dashboard (FastAPI/River metrics, training, predictions)
  - [x] Kafka Dashboard (consumer lag, throughput, partitions, pipeline topics)
  - [x] PostgreSQL Dashboard (connections, queries, replication)
  - [x] Redis Dashboard (memory, connections, ops/sec)
  - [x] MinIO Dashboard (S3 operations, storage, buckets)
- [x] Configure alerting rules (30+ rules in PrometheusRule CRD)
- [x] Grafana connected to PostgreSQL for persistence
- [x] Fix Grafana datasource provisioning (timing issue with sidecar)
- [x] Port forwarding configured: Prometheus (9090), Grafana (3001), Alertmanager (9094)
- **Note:** Kafka JMX exporter disabled due to incompatibility with Kafka 4.0. Wait for Bitnami update.

#### Pending Observability Enhancements

- [x] **Set up Alertmanager notifications** (COMPLETED)
  - [x] Configured Alertmanager receivers in `kube-prometheus-stack.alertmanager.config`
  - [x] Slack, Discord, Email, PagerDuty receivers defined (commented out - not needed for now)
  - [x] Routing rules for severity levels configured:
    - `critical` → 10s group_wait, 1h repeat
    - `warning` → 1m group_wait, 6h repeat
    - `info` → 5m group_wait, 24h repeat
  - [x] Inhibition rules to prevent alert flooding:
    - Critical suppresses warning for same alertname
    - KafkaDown suppresses Kafka-related alerts
    - PostgreSQLDown suppresses MLflow alerts
    - MinIODown suppresses MLflow and River alerts
  - [x] All microservices connected to Alertmanager:
    - [x] River ML - ServiceMonitor + alerting rules
    - [x] Sklearn - ServiceMonitor + alerting rules (new)
    - [x] Kafka Producers - Pod-level alerting rules (new)
    - [x] Reflex - Prometheus instrumentation added + ServiceMonitor enabled
    - [x] MLflow - Bitnami chart metrics enabled + ServiceMonitor
    - [x] Spark - ServiceMonitor + alerting rules (new)
  - **Note:** To enable notifications, uncomment receivers in `values.yaml` and add webhook URLs
  - **Testing Alerting Rules:**
    - Test PromQL queries: `./scripts/test-alerting-rules.sh`
    - Test Alertmanager routing: `./scripts/test-alertmanager-routing.sh`
    - Validate YAML syntax: `promtool check rules k3d/helm/files/prometheus/rules/application-rules.yaml`
    - Prometheus UI (rules & alerts): `kubectl port-forward -n coelho-realtime svc/coelho-realtime-kube-prome-prometheus 9090:9090`
    - Alertmanager UI (routing & silences): `kubectl port-forward -n coelho-realtime svc/coelho-realtime-kube-prome-alertmanager 9094:9094`
    - Check resources: `kubectl get prometheusrules,servicemonitors -n coelho-realtime`

- [ ] **Create custom application dashboards** (Priority: MEDIUM)
  - [ ] FastAPI Analytics Dashboard - request latency histograms, error rates by endpoint, throughput
  - [ ] River ML Training Dashboard - samples processed, prediction latency, model accuracy metrics
  - [ ] Reflex Frontend Dashboard - WebSocket connections, page load times, backend API calls
  - [ ] Kafka Producers Dashboard - messages produced per topic, producer lag, batch sizes
  - Consider using Grafana dashboard provisioning via ConfigMaps

- [ ] **Add Resource Usage Overview to Grafana Dashboard** (Priority: MEDIUM)
  - [ ] Add total CPU usage panel (sum of all coelho-realtime-* pods)
  - [ ] Add total RAM usage panel (sum of all coelho-realtime-* pods)
  - [ ] Add per-service breakdown table:
    - Service name (coelho-realtime-river, coelho-realtime-sklearn, etc.)
    - CPU usage (cores)
    - RAM usage (MB/GB)
    - Percentage of total
  - [ ] Add time-series graph showing resource usage over time
  - [ ] Add alerts for high resource consumption (>80% of limits)
  - [ ] Place prominently in COELHORealTime Overview dashboard

### Phase 9: Data Lake & SQL Analytics (COMPLETED)
**Goal:** Enable SQL queries on streaming data with Delta Lake

- [x] Add PySpark to project (Bitnami Spark Helm chart with bitnamilegacy/spark:4.0.0)
- [x] Configure Spark to use MinIO (S3A connector with hadoop-aws, aws-java-sdk-bundle)
- [x] Implement Delta Lake tables:
  - [x] `transaction_fraud_detection` - TFD streaming data
  - [x] `estimated_time_of_arrival` - ETA streaming data
  - [x] `e_commerce_customer_interactions` - ECCI streaming data
- [x] Create Kafka → Delta Lake pipeline (Spark Structured Streaming)
- [x] Add Spark to Services dropdown in navbar
- [x] Add Spark Grafana dashboards (Performance Metrics + Structured Streaming)
- [x] Update Sklearn to read from Delta Lake instead of parquet files
- [x] Remove parquet saving from River service (freed from data persistence)
- [x] **Add SQL query interface to Reflex** (COMPLETED)
  - [x] Install DuckDB with Delta Lake extension
  - [x] Install Polars with Delta Lake support (faster alternative engine)
  - [x] Configure both engines to connect to MinIO (S3 endpoint, credentials)
  - [x] Create new "Delta Lake SQL" tab in Reflex UI with:
    - [x] SQL editor textarea with syntax highlighting
    - [x] Run/Clear buttons with keyboard shortcuts
    - [x] Results table with horizontal/vertical scrolling
    - [x] Engine selector dropdown (Polars/DuckDB for benchmarking)
    - [x] Query templates dropdown (per-project templates)
    - [x] Column sorting (ascending/descending toggle)
    - [x] Search filter for results
    - [x] Execution time and row count display
  - [x] Implement safety: read-only queries only (SELECT), query limits, blocked DDL/DML
  - [x] Added SQL execution endpoints to River service (`/sql_query`, `/table_schema`)

**Spark + Delta Lake Architecture:**
```
Kafka Topics → Spark Structured Streaming → Delta Lake (MinIO s3a://lakehouse/delta/*)
                                                    ↓
                                              Polars LazyFrame (pl.scan_delta)
                                                    ↓
                                              River / Sklearn Services
```

**Components:**
- Spark Master (bitnamilegacy/spark:4.0.0) - port 4040
- Spark Worker (1 replica)
- Spark Streaming Deployment (3 jobs: TFD, ETA, ECCI)
- Delta Lake 4.0.0 JARs (downloaded via initContainers)
- Kafka clients 3.9.0, Hadoop AWS 3.4.2, AWS SDK 1.12.790

### Phase 10: Batch ML Studies (Priority: LOW - Study phase)
**Goal:** Complement incremental ML with batch ML techniques

- [ ] Scikit-Learn experiments:
  - [ ] Batch fraud detection model
  - [ ] Compare with River incremental model
- [ ] YellowBrick visualizations:
  - [ ] Feature importance plots
  - [ ] Model selection visualizations
  - [ ] Cluster analysis visualizations
- [ ] PySpark MLlib (optional):
  - [ ] Distributed training on Delta Lake data
  - [ ] Large-scale feature engineering

### Phase 11: Analytics Tab (Priority: LOW - Future Enhancement)
**Goal:** Add a fourth top-level tab for data analytics and EDA

- [ ] Add "Analytics" as fourth top-level tab to all pages
- [ ] Exploratory Data Analysis (EDA) features:
  - [ ] Auto-generated statistics (mean, median, std, min, max, null counts)
  - [ ] Distribution charts for numeric columns (histograms, box plots)
  - [ ] Correlation heatmap for numeric features
  - [ ] Missing values analysis and visualization
  - [ ] Categorical column value counts
- [ ] Visual Query Builder (no SQL required):
  - [ ] Point-and-click interface to build filters
  - [ ] Column selector with drag-and-drop
  - [ ] Aggregation builder (GROUP BY, SUM, COUNT, AVG)
- [ ] Custom Chart Builder:
  - [ ] Select columns for X and Y axes
  - [ ] Choose chart type (bar, line, scatter, pie, heatmap)
  - [ ] Interactive Plotly visualizations
  - [ ] Export charts as PNG/SVG
- [ ] Data Profiling Report:
  - [ ] One-click comprehensive data profile
  - [ ] Data quality scores
  - [ ] Anomaly detection in data

---

## Technical Debt & Improvements

### Bug Fixes
- [x] Fix Reflex form field display errors (some fields not showing values correctly)
  - [x] Fixed Spark Streaming schemas to match Kafka producers exactly
  - [x] Added JSON parsing for nested fields (location, device_info, origin, destination)
  - [x] Added Pydantic validators in River to parse JSON strings from Delta Lake
  - [x] Fixed /sample and /initial_sample endpoints to validate all projects
  - [x] Added hot-reload (--reload flag) to River entrypoint.sh
  - [x] All three project pages (TFD, ETA, ECCI) now display values correctly

### Code Quality
- [ ] Add unit tests for River/Sklearn endpoints
- [ ] Add integration tests for Reflex pages
- [ ] Add type hints throughout codebase
- [ ] Set up pre-commit hooks (black, isort, mypy)
- [ ] **Secure Environment Variables Handling** (Priority: HIGH)
  - [ ] Transform all `os.environ.get(..., default)` to `os.environ[...]` (fail fast if missing)
  - [ ] Centralize all environment variables in dedicated config modules per service
  - [ ] Ensure no credentials are hardcoded in source code
  - [ ] Validate required env vars at application startup
  - [ ] Update Helm ConfigMaps/Secrets to provide all necessary env vars

### Cleanup Tasks
- [ ] Update historical documentation in `docs/` folder:
  - [ ] `docs/DEPLOYMENT.md` - Remove FastAPI/Streamlit references
  - [ ] `docs/ARGOCD_INTEGRATION_ROADMAP.md` - Update CI/CD examples
  - [ ] `docs/secrets_instructions_k3d.md` - Update port mappings
  - [ ] `docs/ROADMAP.md` - Update or archive
- [ ] Clean up old test notebooks in `tests/`:
  - [ ] Update paths from `../fastapi_app/` to current structure
  - [ ] Or archive/delete if no longer needed

### Documentation
- [ ] Create README.md with project overview
- [ ] Document architecture decisions (ADR)
- [ ] API documentation (FastAPI auto-docs)
- [ ] Deployment guide for k3d

### UI/UX Enhancements
- [ ] **Add info tooltips for all form fields and metrics** (Priority: MEDIUM)
  - [ ] Add small info icon (ℹ️) next to each form field label
  - [ ] Add small info icon next to each metric card label
  - [ ] On hover/click, show tooltip with detailed explanation of:
    - What the field/metric means
    - Expected value ranges (if applicable)
    - How the metric is calculated (for ML metrics)
    - Example values or use cases
  - [ ] Implement for all pages: TFD, ETA, ECCI
  - [ ] Use Reflex `rx.tooltip` or `rx.hover_card` component
  - [ ] Consider adding links to documentation for complex metrics (F1, ROC AUC, etc.)

### Kafka Producers Improvements
- [ ] **Data Generation Quality** (Priority: MEDIUM)
  - [ ] Add more realistic data patterns:
    - Seasonal trends (daily, weekly, monthly patterns)
    - Correlated features (e.g., high amount + specific merchant = higher fraud probability)
    - Time-based patterns (rush hour traffic, weekend shopping behavior)
  - [ ] Add configurable fraud rate / anomaly injection rate
  - [ ] Add data validation before producing (schema validation, range checks)

- [ ] **Concept Drift Simulation** (Priority: HIGH)
  - [ ] Implement multiple drift types:
    - Sudden drift (abrupt change in distribution)
    - Gradual drift (slow transition between concepts)
    - Incremental drift (continuous small changes)
    - Recurring drift (seasonal/cyclical patterns)
  - [ ] Add drift severity configuration (mild, moderate, severe)
  - [ ] Log drift events to enable model performance analysis

- [ ] **Producer Configuration & Control** (Priority: LOW)
  - [ ] Add configurable generation rates per topic (messages/second)
  - [ ] Add pause/resume capability via API or environment variable
  - [ ] Add batch size configuration for burst scenarios
  - [ ] Add Prometheus metrics for messages produced per topic

- [ ] **New Data Scenarios** (Priority: LOW)
  - [ ] Add "attack" scenarios for fraud detection (coordinated fraud patterns)
  - [ ] Add special events for ETA (accidents, road closures, weather events)
  - [ ] Add promotional events for E-Commerce (flash sales, holiday shopping)

### Performance
- [ ] Profile Reflex page load times
- [ ] Optimize Kafka consumer batch sizes
- [ ] Add caching for expensive computations
- [ ] Consider Redis for Reflex state (already configured)
- [x] **Optimize River/Sklearn Docker builds with UV** (COMPLETED)
  - Migrated from pip to UV package manager (10-100x faster installs)
  - Multi-stage Docker builds: builder stage installs deps, runtime copies venv
  - Benefits: Faster pod startup, no OOM during dependency installation
  - Hot-reload works with Skaffold file sync
- [x] **Static Dropdown Options for Instant Form Loading** (COMPLETED)
  - Problem: Polars queries to Delta Lake were slow (5-10s per request)
  - Solution: Static Python constants mirroring Kafka producer values
  - Implementation:
    - [x] Added `STATIC_DROPDOWN_OPTIONS` dict in `apps/river/functions.py`
    - [x] Created `get_static_dropdown_options()` function
    - [x] Updated River startup to use static options (instant, no I/O)
    - [x] Updated `/unique_values` endpoint with static → Polars fallback
    - [x] Updated `/initial_sample` to load on-demand from Delta Lake
  - Benefits: Instant form loading, works on fresh start (no Delta Lake data needed)
- [x] **Local Random Generation for "Randomize All Fields"** (COMPLETED)
  - Problem: Randomize button called `/sample` → Polars → Delta Lake (5-10s)
  - Solution: Generate random values locally in Reflex using loaded dropdown options
  - Implementation:
    - [x] Updated `randomize_tfd_form()` - local generation, no network call
    - [x] Updated `randomize_eta_form()` - local generation, no network call
    - [x] Updated `randomize_ecci_form()` - local generation, no network call
  - Benefits: Instant randomization (< 50ms), no network calls
- [x] **Polars instead of PySpark for Delta Lake queries** (COMPLETED)
  - River and Sklearn now use Polars for Delta Lake queries
  - Benefits achieved: No JVM overhead, faster startup, lower memory footprint
  - River memory limits reduced from 4Gi back to 2Gi
  - Using `pl.scan_delta()` with lazy evaluation for optimized queries

### Infrastructure
- [x] Update MinIO volume mount to use dynamic project path instead of hardcoded `/data/minio`
  - Implemented in `k3d/terraform/main.tf` using `locals.project_root = abspath("${path.module}/../..")`
  - Data directory: `data/minio/` in project root
  - Added `.gitkeep` file and `.gitignore` rules to track directory but ignore data
- [x] Add PostgreSQL for MLflow metadata persistence (instead of SQLite)
  - PostgreSQL is production-ready and supports concurrency
  - Required for Airflow in future projects (SQLite is dev-only for Airflow)
  - One PostgreSQL instance can serve multiple services (MLflow, Airflow, Superset, etc.)
  - Bitnami PostgreSQL Helm chart with hostPath persistence at `/data/postgresql`
  - MLflow configured to use `postgresql://mlflow:mlflow123@coelho-realtime-postgresql:5432/mlflow`
- [x] **Migrate Kafka to Bitnami Helm Chart** (COMPLETED)
  - Migrated from custom Docker build to Bitnami Kafka Helm chart v32.4.3
  - Kafka 4.0.0 with KRaft mode (no Zookeeper required)
  - Python data generators moved to separate deployment (`apps/kafka-producers/`)
  - **Known Issue:** JMX exporter disabled - incompatible with Kafka 4.0
    - Error: "The Prometheus metrics HTTPServer caught an Exception"
    - Waiting for Bitnami to update JMX exporter for Kafka 4.0 compatibility
    - Kafka dashboard will show limited data until JMX is re-enabled
  - Migration completed:
    - [x] Add Bitnami Kafka Helm dependency to Chart.yaml
    - [x] Move Python data generators to separate deployment (`apps/kafka-producers/`)
    - [x] Update values.yaml with Bitnami Kafka configuration (KRaft mode)
    - [x] Remove custom Kafka Docker build and templates
    - [x] Test Kafka connectivity with existing services
    - [ ] Enable JMX metrics (pending Bitnami update for Kafka 4.0)

---

## Architecture Diagram (Current State)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         k3d Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐                        ┌──────────┐    ┌──────────┐  │
│  │  Reflex  │◄──────────────────────►│  Kafka   │◄──►│ Producers│  │
│  │ Frontend │                        │          │    │ (Python) │  │
│  │  :3000   │                        └────┬─────┘    └──────────┘  │
│  └────┬─────┘                             │                         │
│       │                                   │                         │
│       │    ┌──────────┐    ┌──────────┐  │                         │
│       ├───►│  River   │◄───┤  Sklearn │  │                         │
│       │    │(Incr. ML)│    │(Batch ML)│  │                         │
│       │    │  :8002   │    │  :8003   │  │                         │
│       │    └────┬─────┘    └────┬─────┘  │                         │
│       │         │               │        │                          │
│       │         └───────┬───────┘        │                          │
│       │                 │                │                          │
│       │                 ▼                ▼                          │
│       │    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│       │    │  MLflow  │◄──►│PostgreSQL│    │  MinIO   │            │
│       │    │  :5000   │    │(metadata)│    │(artifacts)│            │
│       │    └──────────┘    └──────────┘    └────┬─────┘            │
│       │                                         │                   │
│       │                                         ▼                   │
│       │                                    ┌──────────┐            │
│       │                                    │  Delta   │            │
│       │                                    │  Lake    │            │
│       │                                    └──────────┘            │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                      │
│  │Prometheus│◄──►│ Grafana  │◄──►│Alertmgr  │                      │
│  │  :9090   │    │  :3001   │    │  :9094   │                      │
│  └──────────┘    └──────────┘    └──────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Services Removed:
- FastAPI Analytics (8001) → Merged into River/Sklearn
- Streamlit (8501) → Replaced by Reflex
```

---

## Priority Matrix

| Phase | Priority | Effort | Value | Dependencies |
|-------|----------|--------|-------|--------------|
| 4. MinIO + Storage | ~~HIGH~~ DONE | Medium | High | None |
| 5. MLflow Model Integration | HIGH | Medium | Very High | MinIO |
| 6. Scikit-Learn Service | ~~LOW~~ DONE | Medium | Medium | None |
| 6b. Consolidate FastAPI Services | ~~MEDIUM~~ DONE | Medium | High | Phase 6 |
| 7. Prometheus/Grafana | ~~HIGH~~ 95% DONE | Medium | High | None |
| 7a. Alertmanager Notifications | ~~HIGH~~ DONE | Low | High | Phase 7 |
| 7b. Custom App Dashboards | MEDIUM | Medium | Medium | Phase 7 |
| 9. Delta Lake + SQL Tab | ~~MEDIUM~~ DONE | High | High | MinIO |
| 10. Batch ML Studies | LOW | Medium | Low | Delta Lake |
| Kafka Migration | ~~MEDIUM~~ DONE | Medium | High | None |

---

## Big Wins Achieved 🎉

### MLflow-Only Storage Architecture (December 2025)
- **Removed all local disk storage** for models and encoders
- All artifacts now stored exclusively in MLflow (backed by MinIO S3)
- Training scripts use `tempfile.TemporaryDirectory()` for artifact logging
- Enabled **best model continuation** - new training loads best existing model

### Best Model Selection from MLflow
- Implemented `get_best_mlflow_run()` function with metric-based selection
- Classification models: Maximize F1 score
- Regression models: Minimize MAE
- Clustering models: Use latest run (pending metrics implementation)
- Encoders loaded from same run as model for consistency

### Reflex UI Improvements
- MLflow model availability checking with real-time validation
- Warning messages when no trained model is available
- MLflow button with logo to redirect to experiment page
- Standardized predict buttons across all forms
- Model badge showing current River ML model name

### Delta Lake + Polars Integration
- Replaced PySpark for data queries with Polars (lighter, faster)
- Lazy evaluation with `pl.scan_delta()` for optimized queries
- River memory limits reduced from 4Gi to 2Gi

### UV Package Manager + Multi-Stage Docker Builds (December 2025)
- Migrated River, Sklearn, Reflex, Kafka-producers from pip to UV
- Multi-stage builds: builder installs deps, runtime copies venv only
- 10-100x faster dependency installation
- Hot-reload preserved with Skaffold file sync

### Instant Form Loading with Static Options (December 2025)
- Form dropdowns now load instantly (was 5-10s with Polars/Delta Lake)
- Static Python constants mirror Kafka producer values
- Works on fresh start with no Delta Lake data
- "Randomize All Fields" now instant (local generation, no network calls)

### Delta Lake SQL Tab (December 2025)
- **New "Delta Lake SQL" tab** added to all pages (TFD, ETA, ECCI)
- Dual-engine support: Polars (faster) and DuckDB for benchmarking
- Features implemented:
  - SQL editor with query templates per project
  - Scrollable results table with cell borders/dividers
  - Column sorting (ascending/descending)
  - Search filter for results
  - Engine selector dropdown
  - Execution time and row count display
- Security: Read-only queries only, DDL/DML blocked, row limits enforced
- Tab icons added to all tabs (Incremental ML, Batch ML, Delta Lake SQL)

### Comprehensive River ML Metrics Configuration (January 2026)
- **Deep research** on optimal metrics arguments for Transaction Fraud Detection
- **New metrics structure** with three categories:
  - `class_metrics` (10): Recall, Precision, F1, FBeta, Accuracy, BalancedAccuracy, MCC, GeometricMean, CohenKappa, Jaccard
  - `proba_metrics` (3): ROCAUC, RollingROCAUC, LogLoss
  - `report_metrics` (2): ConfusionMatrix, ClassificationReport (pickle serialized to MLflow)
- **Optimal arguments based on research:**
  - `pos_val=1` for all metrics (fraud = positive class)
  - `FBeta.beta=2.0` - Industry standard for fraud detection (prioritize Recall)
  - `ROCAUC.n_thresholds=50` - Better accuracy than default 10
  - `RollingROCAUC.window_size=5000` - Stable AUC for rare fraud events
  - `ClassificationReport.decimals=4` - Precision for monitoring
- **Shared ConfusionMatrix** for efficiency (metrics share TP/TN/FP/FN counts)
- **predict_proba_one()** added for probability-based metrics (was using class labels)
- **MLflow experiments cleanup** - Deleted old TFD/ETA experiments for fresh start
- **Best model selection** updated to use FBeta instead of F1 for TFD
- **Research sources documented** in code comments:
  - River ML Documentation: https://riverml.xyz/dev/api/metrics/
  - Fraud Detection Best Practices: https://www.cesarsotovalero.net/blog/
  - F-Beta Score Guide: https://machinelearningmastery.com/fbeta-measure-for-ml

### TFD Metrics Dashboard Complete (January 2026)
- **All 15 metrics displayed** on Reflex TFD Incremental ML page
- **Dashboard layout:**
  - ROW 1: KPI Indicators with delta (FBeta, ROC AUC, Precision, Recall, Rolling AUC)
  - ROW 2: Gauge charts (MCC, Balanced Accuracy)
  - ROW 3: Confusion Matrix heatmap + YellowBrick-style Classification Report heatmap
  - ROW 4: Additional metric cards (F1, Accuracy, Geo Mean, Cohen κ, Jaccard, LogLoss)
- **Delta indicators** show change from baseline (previous best model)
- **Real-time updates** via refresh button during ML training
- **LIVE/FINISHED status** indicator shows training state
- **Report metrics endpoint** (`/report_metrics`) loads ConfusionMatrix and ClassificationReport from MLflow artifacts
- **Graceful handling** of fresh starts (shows "No data available yet" when no artifacts exist)

### Redis Live Model Cache for Real-Time Predictions (December 2025)
- **Problem:** Predictions used MLflow's saved model, not the live training model
- **Solution:** Redis cache for live model during training
- **Architecture:**
  ```
  Training Active:
  ┌─────────────────┐     ┌─────────┐     ┌─────────────┐
  │ Training Script │────►│  Redis  │◄────│  /predict   │
  │ (subprocess)    │     │ (cache) │     │  endpoint   │
  └─────────────────┘     └─────────┘     └─────────────┘
          │                                      │
          └──────► MLflow (periodic save) ◄──────┘
                   (for persistence)
  ```
- **Implementation:**
  - Training scripts save to Redis every 100 Kafka messages (`REDIS_CACHE_INTERVAL`)
  - Training scripts save to MLflow every 1000 messages (persistence)
  - `/predict` endpoint checks Redis first when training is active
  - `/training_status/{project_name}` endpoint for UI to check training state
  - `model_source` field in prediction response ("live" or "mlflow")
- **Benefits:**
  - Users see predictions improve in real-time during training
  - Instant model updates (~1-2 seconds vs waiting for MLflow save)
  - Automatic cleanup when training stops (TTL + explicit clear)
- **Files Modified:**
  - `apps/river/functions.py` - Redis helper functions
  - `apps/river/app.py` - /predict and /training_status endpoints
  - `apps/river/*_river.py` - All 4 training scripts
  - `apps/river/pyproject.toml` - Added redis dependency
  - `k3d/helm/templates/river/configmap.yaml` - Added REDIS_URL

### Service Architecture Simplification
- Consolidated from 3 services to 2 (River + Sklearn)
- FastAPI Analytics deprecated and merged into River
- Clear separation: Incremental ML (River) vs Batch ML (Sklearn)

---

## Notes

- **ML Training Service (DONE):** River service handles real-time ML training
- **Kafka Migration (DONE):** Bitnami Kafka 4.0.0 with KRaft mode, JMX disabled due to compatibility
- **Observability Stack (95% DONE):** kube-prometheus-stack v80.6.0 with dashboards and alerting rules
  - PostgreSQL, Redis, MinIO metrics exporters enabled and working
  - Alertmanager notifications configured (receivers commented out, routing rules active)
  - All microservices connected: River, Sklearn, Reflex, MLflow, Spark, Kafka Producers
  - Pending: custom app dashboards
- **Scikit-Learn Service (DONE):** Batch ML service with YellowBrick visualizations
  - Reflex TFD page now has Batch ML tab with XGBClassifier predictions
  - YellowBrick visualizations available (Classification, Target, Model Selection)
- **Service Consolidation (DONE - Phase 6b):** FastAPI Analytics deprecated
  - Data endpoints (`/sample`, `/unique_values`, `/initial_sample`) moved to River
  - Cluster endpoints (`/cluster_counts`, `/cluster_feature_counts`) moved to River
  - YellowBrick and batch ML remain in Sklearn
  - Result: 2 services (River + Sklearn) instead of 3
- **Streamlit Removal (DONE):** Streamlit app fully replaced by Reflex
  - All pages migrated: TFD, ETA, ECCI
  - `apps/streamlit/` and Helm templates deleted
- **MLflow Integration (PARTIALLY DONE):** Best model selection implemented
  - Models and encoders loaded from MLflow based on best metrics
  - New training runs continue from best existing model
  - Pending: UI model version display, hot-reloading, ECCI clustering metrics
- **Next Quick Wins:** ECCI clustering metrics (for best-model selection) or custom Grafana dashboards
- **Pending Metrics:** Kafka JMX (waiting for Bitnami update)

---

## Original Todo Items (from /todo file)
- [x] ~~Transfer saved models and encoders to MLflow or to MinIO~~ → Phase 4 (done)
- [x] ~~Configure MLflow to use MinIO as backend~~ → Phase 4 (done)
- [x] ~~Create a separated service to process ML training in real-time~~ → Phase 6 (done)
- [x] ~~Best model auto-selection from MLflow~~ → Phase 5 (done, replaced A/B testing)

---

*Last updated: 2026-01-14 (TFD Metrics Dashboard Complete - All 15 metrics with Plotly visualizations, delta indicators, and real-time updates)*
