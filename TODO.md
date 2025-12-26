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

- **ML Training Service (DONE):** River service now handles real-time ML training separately from FastAPI
- **Kafka Migration (DONE):** Bitnami Kafka 4.0.0 with KRaft mode, JMX disabled due to compatibility
- **Observability Stack (90% DONE):** kube-prometheus-stack v80.6.0 with dashboards and alerting rules
  - PostgreSQL, Redis, MinIO metrics exporters enabled and working
  - Pending: Alertmanager notifications, custom app dashboards, logging stack
- **MLflow Integration:** Phase 5 enables production-ready model serving from MLflow registry
- **Next Quick Wins:** Alertmanager notifications (low effort, high value) or Phase 5 (MLflow model serving)
- **Dependencies:** Delta Lake and A/B Testing require MinIO to be set up first
- **Scikit-Learn Service:** Create when Batch ML studies (Phase 10) are ready
- **Pending Metrics:** Kafka JMX (waiting for Bitnami update), MLflow (needs exporter), Reflex (needs instrumentation)

---

## Original Todo Items (from /todo file)
- [ ] Transfer saved models and encoders to MLflow or to MinIO → Phase 4 (pending)
- [x] ~~Configure MLflow to use MinIO as backend~~ → Phase 4 (done)
- [x] ~~Create a separated service to process ML training in real-time~~ → Phase 6 (done)
- [ ] Give users options to choose different MLflow models (A/B test) → Phase 8 (pending)

---

*Last updated: 2025-12-26 (Observability enhancements added)*
