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
- [ ] Add SQL query interface to Reflex (optional - future enhancement)

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
| 9. Delta Lake | ~~MEDIUM~~ DONE | High | Medium | MinIO |
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

*Last updated: 2025-12-29 (Phase 8 removed - auto best-model approach preferred over A/B testing)*
