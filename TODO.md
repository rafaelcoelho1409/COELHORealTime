# COELHO RealTime - Project Roadmap

## Project Overview
Real-time ML platform with incremental learning, featuring:
- Transaction Fraud Detection (River ML)
- Estimated Time of Arrival (River ML)
- E-Commerce Customer Interactions (DBSTREAM Clustering)

**Tech Stack:** Reflex, FastAPI, Kafka, MLflow, River ML, k3d/Kubernetes

---

## Completed Milestones

### Phase 1: Core Infrastructure
- [x] k3d Kubernetes cluster setup
- [x] Helm chart for deployment
- [x] Kafka for real-time data streaming
- [x] FastAPI backend with River ML models
- [x] MLflow for experiment tracking

### Phase 2: Streamlit MVP
- [x] Transaction Fraud Detection page
- [x] Estimated Time of Arrival page
- [x] E-Commerce Customer Interactions page
- [x] Real-time training toggle
- [x] Prediction forms and visualizations

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

### Phase 6: Scikit-Learn Service (Priority: LOW)
**Goal:** Create dedicated Scikit-Learn FastAPI service for batch ML

- [ ] Create Scikit-Learn FastAPI service with endpoints:
  - [ ] `/train` - Trigger batch model training
  - [ ] `/predict` - Batch ML predictions
  - [ ] `/status`, `/health` - Service monitoring
- [ ] Add Helm templates for Scikit-Learn service (deployment, service, configmap)
- [ ] Migrate Batch ML Streamlit pages to Reflex
- [ ] Update Reflex to call Scikit-Learn service for batch predictions
- [ ] Enable Scikit-Learn option in navbar (remove "Soon" badge)

### Phase 7: Observability Stack (COMPLETED)
**Goal:** Production-grade monitoring with Prometheus & Grafana

- [x] Add kube-prometheus-stack Helm dependency (v80.6.0)
- [x] Configure ServiceMonitors for:
  - [x] FastAPI metrics (prometheus-fastapi-instrumentator)
  - [x] River ML Training Service metrics (prometheus-fastapi-instrumentator)
  - [ ] Kafka metrics (disabled - pending Bitnami migration, see Infrastructure)
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
- **Note:** 19 Prometheus targets UP, 0 DOWN. Kafka/MLflow/Reflex ServiceMonitors disabled until metrics endpoints available.

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

### Phase 9: Data Lake & SQL Analytics (Priority: MEDIUM)
**Goal:** Enable SQL queries on streaming data with Delta Lake

- [ ] Add PySpark to project (separate container or sidecar)
- [ ] Configure Spark to use MinIO (S3A connector)
- [ ] Implement Delta Lake tables:
  - [ ] `transactions` - TFD historical data
  - [ ] `trips` - ETA historical data
  - [ ] `customer_interactions` - ECCI historical data
- [ ] Create Kafka → Delta Lake pipeline (batch or micro-batch)
- [ ] Add SQL query interface to Reflex (optional)
- [ ] Test Delta Lake ACID transactions on MinIO

**Delta Lake + MinIO Configuration:**
```python
spark.conf.set("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
spark.conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
spark.conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
```

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
- [ ] Fix Reflex form field display errors (some fields not showing values correctly)
  - Revisit 0 values, None handling, and edge cases
  - Test all three project pages (TFD, ETA, ECCI)

### Code Quality
- [ ] Add unit tests for FastAPI endpoints
- [ ] Add integration tests for Reflex pages
- [ ] Add type hints throughout codebase
- [ ] Set up pre-commit hooks (black, isort, mypy)

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
- [ ] **Migrate Kafka to Bitnami Helm Chart** (Priority: MEDIUM)
  - Current setup uses custom Docker build (`bitnamilegacy/kafka:3.5.1`) with bundled Python producers
  - Benefits of migration:
    - Up-to-date Kafka version (3.9.x vs 3.5.1)
    - Built-in JMX/Prometheus metrics exporter
    - No maintenance burden for security updates
    - Proper separation of concerns (broker vs producers)
    - Consistent with other Bitnami dependencies (PostgreSQL, Redis, MinIO)
  - **Monitoring benefit**: Bitnami Kafka has built-in metrics - just enable in values.yaml:
    ```yaml
    kafka:
      metrics:
        jmx:
          enabled: true
        serviceMonitor:
          enabled: true
    ```
    Current custom Kafka has no metrics endpoint, so Kafka dashboard shows "No data".
    ServiceMonitor is disabled in `helm/templates/kafka/servicemonitor.yaml` until migration.
  - Migration steps:
    - [ ] Add Bitnami Kafka Helm dependency to Chart.yaml
    - [ ] Move Python data generators to separate deployment (`apps/kafka-producers/`)
    - [ ] Update values.yaml with Bitnami Kafka configuration (KRaft mode)
    - [ ] Enable metrics and ServiceMonitor in values.yaml
    - [ ] Remove custom Kafka Docker build and templates
    - [ ] Test Kafka connectivity with existing services
    - [ ] Verify Kafka dashboard shows metrics
  - Note: Official Apache Kafka Helm Chart (KIP-1149) expected mid-2025, but Bitnami is mature now

---

## Architecture Diagram (Current State)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         k3d Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Reflex  │◄──►│  FastAPI │    │  Kafka   │◄──►│ Producers│      │
│  │ Frontend │    │(Analytics)    │          │    │ (Python) │      │
│  └────┬─────┘    └──────────┘    └────┬─────┘    └──────────┘      │
│       │                               │                              │
│       │    ┌──────────┐               │                              │
│       ├───►│  River   │◄──────────────┘                              │
│       │    │(ML Train)│                                              │
│       │    └────┬─────┘                                              │
│       │         │                                                    │
│       │         ▼                                                    │
│       │    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│       │    │  MLflow  │◄──►│PostgreSQL│    │  MinIO   │             │
│       │    │          │    │(metadata)│    │(artifacts)│             │
│       │    └──────────┘    └──────────┘    └────┬─────┘             │
│       │                                         │                    │
│       │                                         ▼                    │
│       │                                    ┌──────────┐             │
│       │                                    │  Delta   │ (future)    │
│       │                                    │  Lake    │             │
│       │                                    └──────────┘             │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│  │Prometheus│◄──►│ Grafana  │◄──►│Alertmgr  │                       │
│  │  :9090   │    │  :3001   │    │  :9094   │                       │
│  └──────────┘    └──────────┘    └──────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Priority Matrix

| Phase | Priority | Effort | Value | Dependencies |
|-------|----------|--------|-------|--------------|
| 4. MinIO | ~~HIGH~~ DONE | Medium | High | None |
| 5. MLflow Model Integration | HIGH | Medium | Very High | MinIO |
| 6. Scikit-Learn Service | LOW | Medium | Medium | Phase 10 |
| 7. Prometheus/Grafana | ~~HIGH~~ DONE | Medium | High | None |
| 8. A/B Testing | MEDIUM | Medium | High | MLflow, MinIO |
| 9. Delta Lake | MEDIUM | High | Medium | MinIO |
| 10. Batch ML | LOW | Medium | Low | Delta Lake |
| Kafka Migration | MEDIUM | Medium | High | None |

---

## Notes

- **ML Training Service (DONE):** River service now handles real-time ML training separately from FastAPI
- **Observability Stack (DONE):** kube-prometheus-stack v80.6.0 with 6 custom dashboards, 30+ alerting rules
- **MLflow Integration:** Phase 5 enables production-ready model serving from MLflow registry
- **Next Quick Wins:** Kafka Migration (enables dashboard metrics) or Phase 5 (MLflow model serving)
- **Dependencies:** Delta Lake and A/B Testing require MinIO to be set up first
- **Scikit-Learn Service:** Create when Batch ML studies (Phase 10) are ready
- **Pending Metrics:** Kafka (after Bitnami migration), MLflow (needs exporter), Reflex (needs instrumentation)

---

## Original Todo Items (from /todo file)
- [ ] Transfer saved models and encoders to MLflow or to MinIO → Phase 4 (pending)
- [x] ~~Configure MLflow to use MinIO as backend~~ → Phase 4 (done)
- [x] ~~Create a separated service to process ML training in real-time~~ → Phase 6 (done)
- [ ] Give users options to choose different MLflow models (A/B test) → Phase 8 (pending)

---

*Last updated: 2025-12-26*
