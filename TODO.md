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

### Phase 3: Reflex Migration (Current - COMPLETED)
- [x] Migrate TFD page to Reflex
- [x] Migrate ETA page to Reflex
- [x] Migrate ECCI page to Reflex
- [x] Add cluster analytics tabs
- [x] Add "Randomize All Fields" buttons
- [x] Fix form field display issues (0 values, None handling)
- [x] Update navbar with official service logos
- [x] Improve page layouts (always-visible maps)

---

## Upcoming Phases

### Phase 4: Storage & Persistence (Priority: HIGH)
**Goal:** Proper artifact storage with MinIO

- [ ] Add MinIO Helm dependency (Bitnami chart)
- [ ] Configure MinIO buckets:
  - `mlflow-artifacts` - MLflow models and artifacts
  - `delta-lake` - Delta Lake tables (future)
  - `raw-data` - Raw Kafka data snapshots
- [ ] Update MLflow to use MinIO as artifact store
- [ ] Configure MLflow environment variables:
  ```
  MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  AWS_ACCESS_KEY_ID=<minio-access-key>
  AWS_SECRET_ACCESS_KEY=<minio-secret-key>
  ```
- [ ] **Transfer existing saved models and encoders to MLflow/MinIO**
- [ ] Test model logging and retrieval from MinIO
- [ ] Add MinIO to Services dropdown in navbar

### Phase 5: ML Training Service Separation (Priority: HIGH)
**Goal:** Isolate ML training from FastAPI to prevent crashes and improve stability

**Problem:** Currently, real-time ML training runs inside FastAPI, which can cause:
- Memory pressure and crashes under high load
- Training blocking API response times
- Difficulty scaling training independently

**Solution:** Create dedicated ML Training Service

- [ ] Design new service architecture:
  ```
  Kafka → ML Training Service → MLflow (models)
                ↓
            Redis/MinIO (state sync)
                ↓
          FastAPI (predictions only)
  ```
- [ ] Create new Python service for ML training:
  - [ ] Kafka consumer for each topic
  - [ ] River ML model training loop
  - [ ] Periodic model checkpointing to MLflow
  - [ ] Health endpoint for Kubernetes probes
- [ ] Add Helm templates for ML Training Service
- [ ] Update FastAPI to load models from MLflow (read-only)
- [ ] Implement model hot-reloading in FastAPI
- [ ] Add shared state mechanism (Redis or MinIO)
- [ ] Test failover scenarios

### Phase 6: Observability Stack (Priority: HIGH)
**Goal:** Production-grade monitoring with Prometheus & Grafana

- [ ] Add kube-prometheus-stack Helm dependency
- [ ] Configure ServiceMonitors for:
  - [ ] FastAPI metrics (add prometheus-fastapi-instrumentator)
  - [ ] Kafka metrics (JMX exporter or Kafka exporter)
  - [ ] ML Training Service metrics
  - [ ] Reflex backend metrics
- [ ] Create custom Grafana dashboards:
  - [ ] Kubernetes Resources Dashboard (CPU, Memory, Network)
  - [ ] Kafka Dashboard (consumer lag, throughput, partitions)
  - [ ] FastAPI Dashboard (request latency, error rates, predictions/sec)
  - [ ] ML Metrics Dashboard (model performance, training samples/sec)
- [ ] Configure alerting rules (optional)
- [ ] Update navbar Services dropdown (enable Prometheus & Grafana links)

### Phase 7: AI Assistant Integration (Priority: HIGH - 30-day focus)
**Goal:** Leverage Claude Opus 4.5 subscription with LangChain/LangGraph

- [ ] Design AI assistant architecture:
  - [ ] Decide: sidebar panel vs chat modal vs dedicated page
  - [ ] Define context injection strategy (page state, metrics, predictions)
- [ ] Backend implementation:
  - [ ] Add LangChain/LangGraph dependencies to FastAPI
  - [ ] Create `/chat` endpoint for AI interactions
  - [ ] Implement tools for AI to query:
    - Current page state/form data
    - MLflow metrics and model info
    - Cluster analytics data
    - Historical predictions
  - [ ] Add conversation memory (Redis or in-memory)
- [ ] Frontend implementation (Reflex):
  - [ ] Create AI chat component
  - [ ] Add to each application page
  - [ ] Stream responses for better UX
- [ ] Prompt engineering:
  - [ ] System prompt for page context explanation
  - [ ] Few-shot examples for common questions
  - [ ] RAG over codebase documentation (optional)

**Example interactions:**
- "What does this page show?"
- "Explain what Cluster 3 represents"
- "Why was this transaction flagged as fraud?"
- "What factors contributed to this ETA prediction?"

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

---

## Architecture Diagram (Target State)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         k3d Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Reflex  │◄──►│  FastAPI │◄──►│  Kafka   │◄──►│ Producers│      │
│  │ Frontend │    │ + River  │    │          │    │ (Python) │      │
│  └────┬─────┘    └────┬─────┘    └──────────┘    └──────────┘      │
│       │               │                                              │
│       │               ▼                                              │
│       │          ┌──────────┐    ┌──────────┐                       │
│       │          │  MLflow  │◄──►│  MinIO   │                       │
│       │          │          │    │ (S3)     │                       │
│       │          └──────────┘    └────┬─────┘                       │
│       │                               │                              │
│       │                               ▼                              │
│       │                          ┌──────────┐                       │
│       │                          │  Delta   │                       │
│       │                          │  Lake    │                       │
│       │                          └────┬─────┘                       │
│       │                               │                              │
│       ▼                               ▼                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│  │  Claude  │    │Prometheus│◄──►│ Grafana  │                       │
│  │   AI     │    │          │    │          │                       │
│  │ (LangG.) │    └──────────┘    └──────────┘                       │
│  └──────────┘                                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Priority Matrix

| Phase | Priority | Effort | Value | Dependencies |
|-------|----------|--------|-------|--------------|
| 4. MinIO | HIGH | Medium | High | None |
| 5. ML Training Service | HIGH | High | Very High | MinIO |
| 6. Prometheus/Grafana | HIGH | Medium | High | None |
| 7. AI Assistant | HIGH | High | Very High | Claude API |
| 8. A/B Testing | MEDIUM | Medium | High | MLflow, MinIO |
| 9. Delta Lake | MEDIUM | High | Medium | MinIO |
| 10. Batch ML | LOW | Medium | Low | Delta Lake |

---

## Notes

- **30-Day Focus:** AI Assistant integration (Phase 7) to maximize Claude Opus 4.5 subscription
- **Critical Architecture:** Phase 5 (ML Training Service) solves FastAPI crash issues - high priority
- **Quick Wins:** MinIO (Phase 4) and Prometheus (Phase 6) can be added in parallel
- **Dependencies:** Delta Lake and A/B Testing require MinIO to be set up first
- **Study vs Production:** Batch ML is study phase, defer until AI integration is complete
- **A/B Testing:** Great for comparing model versions and getting user feedback

---

## Original Todo Items (from /todo file)
- [x] ~~Transfer saved models and encoders to MLflow or to MinIO~~ → Phase 4
- [x] ~~Configure MLflow to use MinIO as backend~~ → Phase 4
- [x] ~~Create a separated service to process ML training in real-time~~ → Phase 5
- [x] ~~Give users options to choose different MLflow models (A/B test)~~ → Phase 8

---

*Last updated: 2025-12-15*
