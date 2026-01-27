# COELHO RealTime - Project Roadmap

## Project Overview
Real-time ML platform with incremental learning, featuring:
- Transaction Fraud Detection (River ML)
- Estimated Time of Arrival (River ML)
- E-Commerce Customer Interactions (DBSTREAM Clustering)

**Tech Stack:** SvelteKit, FastAPI (unified), River ML, Scikit-Learn/CatBoost, Kafka, Spark, Delta Lake, MLflow, k3d/Kubernetes, Prometheus, Grafana

---

## Completed Milestones

### Recent Updates (January 2026)
- [x] **SvelteKit Frontend Migration** - Replaced Reflex with SvelteKit + TypeScript + Tailwind CSS
- [x] **Unified FastAPI Service** - Consolidated River (8002) + Sklearn (8003) into single FastAPI (8000) with versioned routers (`/api/v1/incremental`, `/api/v1/batch`, `/api/v1/sql`)
- [x] **Prometheus Metrics for Kafka Producers** - Custom metrics (messages_sent, errors, send_duration, connected, fraud_ratio, active_sessions) with dedicated metrics HTTP server
- [x] **Prometheus Metrics for FastAPI** - Custom metrics (training active/started/errors/duration, predictions, caching, model loading, MLflow operations, SQL queries, visualizations)
- [x] **10 Grafana Dashboards Completed** - All custom dashboards provisioned via ConfigMaps
- [x] **Total CPU/RAM Aggregate Panels** - Added to COELHORealTime Overview dashboard (stat panels with sparklines and color thresholds)
- [x] Notebook 020 regression visuals: added sklearn + scikit-plot visualizers with CatBoostRegressor wrappers and tuned kwargs
- [x] FastAPI batch utils: CatBoost sklearn wrappers for regression + scikit-plot regression now uses trained model and improved kwargs
- [x] FastAPI batch utils: scikit-plot detailed ROC/PR name normalization and sklearn/scikit-plot compatibility fixes
- [x] FastAPI batch utils: strip CatBoost params that break CV wrappers (`callbacks`, `callback`, `use_best_model`)
- [x] SvelteKit ETA Batch ML: split sklearn PredictionError into Actual vs Predicted + Residuals options
- [x] Batch ML info JSONs (TFD/ETA/ECCI): aligned sklearn/scikit-plot info windows with Yellowbrick structure + official docs URLs
- [x] ECCI Batch ML Overview: added training split hint to match TFD/ETA
- [x] Restart FastAPI pod and re-test sklearn PartialDependence + DecisionBoundary (TFD)
- [x] Restart FastAPI pod and re-test scikit-plot RocCurveDetailed + PrecisionRecallCurveDetailed

### Phase 1: Core Infrastructure
- [x] k3d Kubernetes cluster setup
- [x] Helm chart for deployment
- [x] Kafka for real-time data streaming
- [x] FastAPI backend with River ML models
- [x] MLflow for experiment tracking

### Phase 2: Streamlit MVP (DEPRECATED - Migrated to Reflex, then SvelteKit)
- [x] Transaction Fraud Detection page
- [x] Estimated Time of Arrival page
- [x] E-Commerce Customer Interactions page
- [x] **Streamlit app deleted** - fully replaced

### Phase 3: Reflex (DEPRECATED - Migrated to SvelteKit)
- [x] All Reflex pages migrated to SvelteKit
- [x] **Reflex app deleted** (`apps/reflex/` removed)

### Phase 3b: SvelteKit Migration (COMPLETED)
**Goal:** Replace Reflex with modern SvelteKit frontend for better performance and developer experience
- [x] Create SvelteKit app with TypeScript + Tailwind CSS
- [x] Implement modular component architecture:
  ```
  apps/sveltekit/src/lib/
  ├── api/           # API client modules (client.ts, batch.ts, incremental.ts, sql.ts)
  ├── components/    # Modular components
  │   ├── shared/    # Navbar, Tabs, Cards, Forms, etc.
  │   ├── tfd/       # Transaction Fraud Detection components
  │   ├── eta/       # Estimated Time of Arrival components
  │   └── ecci/      # E-Commerce Customer Interactions components
  ├── stores/        # Svelte stores for state management
  ├── types/         # TypeScript type definitions
  └── utils/         # Helper utilities
  ```
- [x] Implement nested routing with batch/incremental/sql tabs per project
- [x] Services dropdown navbar with 7 external services (FastAPI, Spark, MLflow, MinIO, Prometheus, Grafana, Alertmanager)
- [x] Applications dropdown with 3 ML projects (TFD, ETA, ECCI)
- [x] Docker build with Vite
- [x] Helm templates for SvelteKit deployment (port 5173, NodePort 30173)
- [x] Delete `apps/reflex/` directory

### Phase 4: Storage & Persistence (COMPLETED)
**Goal:** Proper artifact storage with MinIO

- [x] Add MinIO Helm dependency (Bitnami chart)
- [x] Configure MinIO buckets (`mlflow-artifacts`, `delta-lake`, `raw-data`)
- [x] Update MLflow to use MinIO as artifact store
- [x] Transfer existing saved models and encoders to MLflow/MinIO
- [x] Test model logging and retrieval from MinIO
- [x] Add MinIO to Services dropdown in navbar

### Phase 5: MLflow Model Integration (COMPLETED)
**Goal:** Connect frontend to latest/best models registered on MLflow

- [x] Query MLflow for best model per project (by metrics)
- [x] Load model artifacts from MinIO via MLflow
- [x] Implement `get_best_mlflow_run()` with metric-based selection:
  - Transaction Fraud Detection: Maximize FBeta (beta=2.0)
  - Estimated Time of Arrival: Minimize MAE
  - E-Commerce Customer Interactions: Maximize Silhouette
- [x] Best model continuation training
- [x] Model hot-reloading
- [x] MLflow Model Selector Dropdown - Batch ML (all pages)
- [x] MLflow Model Selector Dropdown - Incremental ML (all pages)
- [x] Persist River ML Metrics Across Training Runs
- [x] ECCI Clustering Metrics (Silhouette-based)

### Phase 6: Service Architecture Evolution (COMPLETED)
**Goal:** Evolve from multi-service to unified service architecture

**Evolution Timeline:**
1. Phase 6a: Created separate River (8002) + Sklearn (8003) services
2. Phase 6b: Deprecated FastAPI Analytics (8001), merged into River/Sklearn
3. Phase 6c: **Unified FastAPI (8000)** - Consolidated River + Sklearn into single service with versioned routers

**Current Architecture (Unified FastAPI):**
```
apps/fastapi/
├── app.py                    # Unified FastAPI service (port 8000)
├── config.py                 # Configuration constants
├── models.py                 # Pydantic models
├── metrics.py                # Prometheus metrics registry
├── routers/v1/
│   ├── incremental.py        # River ML endpoints (/api/v1/incremental/*)
│   ├── batch.py              # Sklearn/CatBoost endpoints (/api/v1/batch/*)
│   └── sql.py                # Delta Lake SQL endpoints (/api/v1/sql/*)
├── ml_training/
│   ├── river/                # River incremental training scripts
│   └── sklearn/              # Scikit-Learn batch training scripts
├── utils/
│   ├── incremental.py        # River ML helpers
│   └── batch.py              # Batch ML helpers
├── Dockerfile.fastapi
└── entrypoint.sh
```

**Deleted Services:**
- `apps/river/` - Merged into `apps/fastapi/routers/v1/incremental.py`
- `apps/sklearn/` - Merged into `apps/fastapi/routers/v1/batch.py`
- `apps/fastapi/` (old Analytics) - Rebuilt as unified service

**Key Endpoints:**
- `/api/v1/incremental/*` - 16 endpoints (health, status, switch-model, predict, mlflow-metrics, page-init, sample, report-metrics, cluster-counts, healthcheck, etc.)
- `/api/v1/batch/*` - 20 endpoints (health, status, switch-model, predict, mlflow-runs, init, visualization/metric/yellowbrick, visualization/metric/sklearn, visualization/metric/scikitplot, etc.)
- `/api/v1/sql/*` - 3 endpoints (query, schema, total-rows)

### Phase 7: Observability Stack (COMPLETED)
**Goal:** Production-grade monitoring with Prometheus & Grafana

- [x] Add kube-prometheus-stack Helm dependency (v80.6.0)
- [x] Configure ServiceMonitors for all services:
  - [x] FastAPI (unified) - prometheus-fastapi-instrumentator + custom metrics
  - [x] Kafka Producers - custom Prometheus metrics (messages, errors, latency, connections)
  - [x] PostgreSQL - Bitnami exporter
  - [x] Redis - Bitnami exporter
  - [x] MinIO - ServiceMonitor with cluster/node metrics
  - [x] MLflow - Bitnami chart metrics
  - [x] SvelteKit - ServiceMonitor
  - [x] Spark - ServiceMonitor
- [x] Create custom Grafana dashboards (10 total):
  1. [x] COELHORealTime Overview (service health, CPU, memory, network, total CPU/RAM aggregate panels)
  2. [x] ML Pipeline (training metrics, predictions, model performance)
  3. [x] FastAPI Detailed (request latency, error rates, throughput per endpoint)
  4. [x] Kafka Producers (messages produced, send latency, errors, connections)
  5. [x] Kafka (consumer lag, throughput, partitions)
  6. [x] PostgreSQL (connections, queries, replication)
  7. [x] Redis (memory, connections, ops/sec)
  8. [x] MinIO (S3 operations, storage, buckets)
  9. [x] Spark (performance metrics)
  10. [x] Spark Streaming (structured streaming metrics)
- [x] Configure alerting rules (30+ rules in PrometheusRule CRD):
  - fastapi.rules, kafka.rules, kafka-producers.rules, mlflow.rules, postgresql.rules, redis.rules, minio.rules, sveltekit.rules, spark.rules, application-general.rules
- [x] Alertmanager routing and inhibition rules configured
- [x] Grafana connected to PostgreSQL for persistence
- [x] Total CPU Usage + Total Memory Usage aggregate stat panels (COELHORealTime Overview)
- [x] Grafana dashboards provisioned via ConfigMaps with sidecar auto-discovery
- **Note:** Kafka JMX exporter disabled due to incompatibility with Kafka 4.0. Wait for Bitnami update.
- **Note:** Alertmanager notification receivers defined but commented out (Slack, Discord, Email, PagerDuty) - uncomment and add webhook URLs when needed.

### Phase 9: Data Lake & SQL Analytics (COMPLETED)
**Goal:** Enable SQL queries on streaming data with Delta Lake

- [x] PySpark with Delta Lake (Bitnami Spark 4.0.0)
- [x] Kafka -> Spark Structured Streaming -> Delta Lake (MinIO S3)
- [x] DuckDB SQL interface via `/api/v1/sql/` router
- [x] SQL editor in SvelteKit with query templates, results table, execution time
- [x] Safety: SELECT-only, DDL/DML blocked, row limits enforced

### Phase 10: Batch ML Studies (COMPLETED)
**Goal:** Complement incremental ML with batch ML techniques

- [x] Scikit-Learn experiments (CatBoostClassifier, DuckDB SQL preprocessing)
- [x] ECCI Batch ML - Event-Level Clustering (KMeans with YellowBrick)
- [x] YellowBrick visualizations (Classification, Feature, Target, Model Selection, Cluster, Text)
- [x] Scikit-plot visualizations (ROC, PR, Confusion Matrix, etc.)
- [x] Batch ML training status with tooltips and progress indicators
- [x] DuckDB SQL preprocessing (single-query transformations, ~5-10x faster than Pandas)

---

## Remaining Tasks

### Infrastructure
- [ ] **Activate Alertmanager Notifications** (Priority: LOW)
  - Receivers already configured (Slack, Discord, Email, PagerDuty) but commented out
  - To enable: uncomment receivers in `k3d/helm/values.yaml` and add webhook URLs
  - Routing rules and inhibition rules are already active

### Documentation
- [ ] **Update historical documentation in `docs/` folder** (Priority: MEDIUM)
  - [ ] `docs/DEPLOYMENT.md` - Update for SvelteKit + unified FastAPI architecture
  - [ ] `docs/ARGOCD_INTEGRATION_ROADMAP.md` - Update CI/CD examples
  - [ ] `docs/secrets_instructions_k3d.md` - Update port mappings (FastAPI 8000, SvelteKit 5173)
  - [ ] `docs/ROADMAP.md` - Update or archive (this TODO.md is the primary roadmap)
- [ ] **Create README.md with project overview** (Priority: MEDIUM)
- [ ] **Deployment guide for k3d** (Priority: LOW)

### Kafka Producers Improvements
- [ ] **Data Generation Quality** (Priority: LOW)
  - [ ] Add more realistic data patterns:
    - Seasonal trends (daily, weekly, monthly patterns)
    - Correlated features (e.g., high amount + specific merchant = higher fraud probability)
    - Time-based patterns (rush hour traffic, weekend shopping behavior)
  - [ ] Add configurable fraud rate / anomaly injection rate
  - [ ] Add data validation before producing (schema validation, range checks)

---

## Architecture Diagram (Current State)

```
+-----------------------------------------------------------------------+
|                         k3d Kubernetes Cluster                        |
+-----------------------------------------------------------------------+
|                                                                       |
|  +------------+                        +----------+    +----------+   |
|  | SvelteKit  |<---------------------->|  Kafka   |<-->| Producers|   |
|  | Frontend   |                        |          |    | (Python) |   |
|  |   :5173    |                        +----+-----+    +----------+   |
|  +------+-----+                             |                         |
|         |                                   |                         |
|         |         +-------------------+     |                         |
|         +-------->|    FastAPI         |     |                         |
|                   |  (Unified ML)     |     |                         |
|                   |     :8000         |     |                         |
|                   | /api/v1/incremental|     |                         |
|                   | /api/v1/batch     |     |                         |
|                   | /api/v1/sql       |     |                         |
|                   +--------+----------+     |                         |
|                            |                |                         |
|                            v                v                         |
|         +----------+   +----------+   +----------+                    |
|         |  MLflow  |<->|PostgreSQL|   |  MinIO   |                    |
|         |  :5000   |   |(metadata)|   |(artifacts)|                   |
|         +----------+   +----------+   +----+-----+                    |
|                                            |                          |
|                    +----------+            v                          |
|                    |  Spark   |       +----------+                    |
|                    | Master   |       | Delta    |                    |
|                    |  :4040   |       | Lake     |                    |
|                    +----+-----+       +----------+                    |
|                         |                                             |
|                    +----+-----+                                       |
|                    |  Spark   |                                       |
|                    | Worker   |                                       |
|                    +----------+                                       |
|                                                                       |
|  +----------+   +----------+   +----------+   +----------+           |
|  |Prometheus|<->| Grafana  |<->|Alertmgr  |   |  Redis   |           |
|  |  :9090   |   |  :3000   |   |  :9094   |   | (cache)  |           |
|  +----------+   +----------+   +----------+   +----------+           |
|                                                                       |
+-----------------------------------------------------------------------+

Services Removed (historical):
- Streamlit (8501) -> Replaced by Reflex -> Replaced by SvelteKit
- FastAPI Analytics (8001) -> Merged into unified FastAPI
- River (8002) -> Merged into unified FastAPI
- Sklearn (8003) -> Merged into unified FastAPI
```

---

## Port Mappings (Current)

| Service | Internal Port | NodePort | localhost |
|---------|--------------|----------|-----------|
| FastAPI | 8000 | 30006 | localhost:8000 |
| SvelteKit | 5173 | 30173 | localhost:5173 |
| Spark Master | 4040 | - | localhost:4040 |
| MLflow | 5000 | - | localhost:5001 |
| MinIO Console | 9001 | 30901 | localhost:9001 |
| Prometheus | 9090 | 30090 | localhost:9090 |
| Grafana | 3000 | 30031 | localhost:3000 |
| Alertmanager | 9094 | 30094 | localhost:9094 |

---

## Big Wins Achieved

### SvelteKit Migration (January 2026)
- **Replaced Reflex** with SvelteKit + TypeScript + Tailwind CSS
- Modern component architecture with nested routing
- Better performance, type safety, and developer experience
- Modular API client layer (`client.ts`, `batch.ts`, `incremental.ts`, `sql.ts`)

### Unified FastAPI Service (January 2026)
- **Consolidated 3 services into 1:** River (8002) + Sklearn (8003) + FastAPI Analytics (8001) -> FastAPI (8000)
- Versioned API routers: `/api/v1/incremental`, `/api/v1/batch`, `/api/v1/sql`
- Single Dockerfile, single deployment, shared metrics registry
- 39 total endpoints across 3 routers

### Complete Observability Stack (January 2026)
- **10 Grafana dashboards** covering all services
- **30+ Prometheus alerting rules** across 10 rule groups
- **Custom Prometheus metrics** for FastAPI and Kafka Producers
- **Alertmanager** with routing, inhibition, and receiver configuration
- **Total CPU/RAM aggregate panels** in overview dashboard

### MLflow-Only Storage Architecture (December 2025)
- Removed all local disk storage for models and encoders
- All artifacts stored in MLflow (backed by MinIO S3)
- Best model continuation training implemented
- Metric-based model selection (FBeta for TFD, MAE for ETA, Silhouette for ECCI)

### Delta Lake + DuckDB SQL (December 2025)
- Kafka -> Spark Structured Streaming -> Delta Lake (MinIO S3)
- DuckDB with Delta Lake extension for SQL queries
- SQL editor in frontend with query templates and safety features

### Architecture Diagrams (January 2026)
- Auto-generated architecture diagrams using Python (`diagrams/main.py`)
- PNG/SVG outputs for documentation and presentations

### UV Package Manager + Multi-Stage Docker Builds (December 2025)
- Migrated from pip to UV (10-100x faster installs)
- Multi-stage builds for smaller runtime images

### Redis Live Model Cache (December 2025)
- Real-time predictions use live training model via Redis cache
- Automatic cleanup when training stops (TTL + explicit clear)

### DuckDB SQL Preprocessing for Batch ML (January 2026)
- Pure DuckDB SQL preprocessing (single-query transformations)
- ~5-10x faster than Pandas/Sklearn for 1M+ rows
- CatBoost native categorical handling

### Performance Optimizations (January 2026)
- MLflow experiment cache (5-minute TTL)
- DuckDB connection optimization with retry
- Redis client optimization with retry_on_timeout
- Cluster and report metrics caching
- Overall: page initialization ~300-500ms faster

---

## Notes

- **Unified FastAPI Service (DONE):** Single service at port 8000 with versioned routers for incremental ML, batch ML, and SQL
- **SvelteKit Frontend (DONE):** Replaced Reflex with SvelteKit + TypeScript + Tailwind CSS
- **Kafka Migration (DONE):** Bitnami Kafka 4.0.0 with KRaft mode, JMX disabled due to compatibility
- **Observability Stack (DONE):** 10 Grafana dashboards, 30+ alerting rules, custom Prometheus metrics for all services
- **MLflow Integration (DONE):** Best model selection, continuation training, model selector dropdown
- **Pending:** Kafka JMX metrics (waiting for Bitnami update), documentation updates, data generation quality

---

*Last updated: 2026-01-27 (Architecture audit: SvelteKit migration, unified FastAPI, 10 Grafana dashboards, Prometheus metrics for all services)*
