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

### Phase 4: Storage & Persistence (Priority: HIGH)
**Goal:** Proper artifact storage with MinIO

- [x] Add MinIO Helm dependency (Bitnami chart)
- [ ] Configure MinIO buckets:
  - `mlflow-artifacts` - MLflow models and artifacts
  - `delta-lake` - Delta Lake tables (future)
  - `raw-data` - Raw Kafka data snapshots
- [x] Update MLflow to use MinIO as artifact store
- [x] Configure MLflow environment variables:
  ```
  MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  AWS_ACCESS_KEY_ID=<minio-access-key>
  AWS_SECRET_ACCESS_KEY=<minio-secret-key>
  ```
- [ ] **Transfer existing saved models and encoders to MLflow/MinIO**
- [ ] Test model logging and retrieval from MinIO
- [x] Add MinIO to Services dropdown in navbar

### Phase 5: MLflow Model Integration (Priority: HIGH)
**Goal:** Connect Reflex pages to latest models registered on MLflow instead of loading in-memory models

- [ ] Update FastAPI to load models from MLflow registry:
  - [ ] Query MLflow for latest model version per project
  - [ ] Load model artifacts from MinIO via MLflow
  - [ ] Cache loaded models to avoid repeated downloads
- [ ] Implement model hot-reloading:
  - [ ] Periodic check for new model versions
  - [ ] Graceful model swap without downtime
- [ ] Update prediction endpoints:
  - [ ] Use MLflow-loaded models for inference
  - [ ] Return model version in prediction response
- [ ] Add model info to Reflex UI:
  - [ ] Display current model version on each page
  - [ ] Show last model update timestamp

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
  - [ ] MLflow metrics (disabled - no /metrics endpoint, needs exporter)
  - [ ] Reflex backend metrics (disabled - no /metrics endpoint, needs instrumentation)
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

- [ ] **Set up Alertmanager notifications** (Priority: HIGH)
  - Currently alerts fire but notifications go nowhere
  - Configure receivers in `kube-prometheus-stack.alertmanager.config`:
    - [ ] Slack webhook integration
    - [ ] Email (SMTP) notifications
    - [ ] PagerDuty integration (optional)
  - Set up routing rules for different severity levels:
    - `critical` → immediate notification
    - `warning` → batched notifications
  - Test alert delivery with `amtool` or firing a test alert

- [ ] **Create custom application dashboards** (Priority: MEDIUM)
  - [ ] FastAPI Analytics Dashboard - request latency histograms, error rates by endpoint, throughput
  - [ ] River ML Training Dashboard - samples processed, prediction latency, model accuracy metrics
  - [ ] Reflex Frontend Dashboard - WebSocket connections, page load times, backend API calls
  - [ ] Kafka Producers Dashboard - messages produced per topic, producer lag, batch sizes
  - Consider using Grafana dashboard provisioning via ConfigMaps

- [ ] **Add centralized logging stack** (Priority: MEDIUM)
  - [ ] Deploy Grafana Loki for log aggregation
  - [ ] Deploy Promtail as DaemonSet for log collection
  - [ ] Configure Loki datasource in Grafana
  - [ ] Create log exploration dashboards
  - [ ] Set up log-based alerting for error patterns
  - Benefits: Correlate logs with metrics, search across all services, debug issues faster

### Phase 8: Model A/B Testing (Priority: MEDIUM)
**Goal:** Allow users to compare different MLflow models in production

- [ ] Design A/B testing UI in Reflex:
  - [ ] Model selector dropdown on each page
  - [ ] Side-by-side prediction comparison view
  - [ ] Model performance metrics display
- [ ] Backend implementation:
  - [ ] FastAPI endpoint to list available MLflow models per project
  - [ ] Dynamic model loading based on user selection
  - [ ] Caching strategy for loaded models
- [ ] MLflow integration:
  - [ ] Model versioning and staging (Staging → Production)
  - [ ] Model comparison metrics in MLflow UI
- [ ] Analytics:
  - [ ] Track which models users prefer
  - [ ] Log prediction accuracy per model version

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
- [ ] **Optimize River/Sklearn Docker builds** (Priority: HIGH)
  - Currently installing dependencies at runtime via `entrypoint.sh` (slow, OOM-prone)
  - Move `uv pip install -r requirements.txt` to Dockerfile build stage
  - Benefits: Faster pod startup, no OOM during dependency installation
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
| 4. MinIO | ~~HIGH~~ DONE | Medium | High | None |
| 5. MLflow Model Integration | HIGH | Medium | Very High | MinIO |
| 6. Scikit-Learn Service | ~~LOW~~ DONE | Medium | Medium | None |
| 6b. Consolidate FastAPI Services | MEDIUM | Medium | High | Phase 6 |
| 7. Prometheus/Grafana | ~~HIGH~~ 90% DONE | Medium | High | None |
| 7a. Alertmanager Notifications | HIGH | Low | High | Phase 7 |
| 7b. Custom App Dashboards | MEDIUM | Medium | Medium | Phase 7 |
| 7c. Logging Stack (Loki) | MEDIUM | Medium | High | Phase 7 |
| 8. A/B Testing | MEDIUM | Medium | High | MLflow, MinIO |
| 9. Delta Lake | MEDIUM | High | Medium | MinIO |
| 10. Batch ML | LOW | Medium | Low | Delta Lake |
| Kafka Migration | ~~MEDIUM~~ DONE | Medium | High | None |

---

## Notes

- **ML Training Service (DONE):** River service handles real-time ML training
- **Kafka Migration (DONE):** Bitnami Kafka 4.0.0 with KRaft mode, JMX disabled due to compatibility
- **Observability Stack (90% DONE):** kube-prometheus-stack v80.6.0 with dashboards and alerting rules
  - PostgreSQL, Redis, MinIO metrics exporters enabled and working
  - Pending: Alertmanager notifications, custom app dashboards, logging stack
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
- **MLflow Integration:** Phase 5 enables production-ready model serving from MLflow registry
- **Next Quick Wins:** Alertmanager notifications (low effort, high value) or Phase 5 (MLflow model serving)
- **Dependencies:** Delta Lake and A/B Testing require MinIO to be set up first
- **Pending Metrics:** Kafka JMX (waiting for Bitnami update), MLflow (needs exporter), Reflex (needs instrumentation)

---

## Original Todo Items (from /todo file)
- [ ] Transfer saved models and encoders to MLflow or to MinIO → Phase 4 (pending)
- [x] ~~Configure MLflow to use MinIO as backend~~ → Phase 4 (done)
- [x] ~~Create a separated service to process ML training in real-time~~ → Phase 6 (done)
- [ ] Give users options to choose different MLflow models (A/B test) → Phase 8 (pending)

---

*Last updated: 2025-12-27 (Fixed form field display errors - Spark schemas, JSON parsing, Pydantic validators)*
