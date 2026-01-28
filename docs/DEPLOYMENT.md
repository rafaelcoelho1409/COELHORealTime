# COELHO RealTime - Deployment Guide

Complete guide for deploying the COELHO RealTime MLOps platform using GitLab CI/CD and ArgoCD.

## Architecture Overview

```
Developer → GitLab → GitLab CI → GitLab Registry → ArgoCD → Kubernetes
            (git)    (build)     (docker images)   (deploy)  (pods)
```

**Components:**
- **GitLab**: Source control + CI/CD
- **GitLab Registry**: Docker image storage
- **ArgoCD**: GitOps continuous delivery
- **Kubernetes**: Container orchestration (k3d/k3s)

---

## Prerequisites

### System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Ubuntu 24.04 LTS (or similar Linux) |
| **RAM** | 32 GB minimum |
| **CPU** | 8+ cores recommended |
| **Storage** | 100+ GB free space |

### Required Services
- [x] Kubernetes cluster (k3d/k3s/k8s)
- [x] GitLab installed in `gitlab` namespace
- [x] ArgoCD installed in `argocd` namespace
- [x] kubectl configured

### Check Prerequisites
```bash
# Check cluster connection
kubectl cluster-info

# Check GitLab
kubectl get pods -n gitlab

# Check ArgoCD
kubectl get pods -n argocd
```

---

## Project Structure

```
COELHORealTime/
├── apps/
│   ├── fastapi/           # FastAPI backend (39 endpoints, 3 routers)
│   ├── kafka_producers/   # Kafka data producers
│   └── sveltekit/         # SvelteKit frontend
├── k3d/
│   ├── helm/              # Helm umbrella chart
│   │   ├── Chart.yaml     # 7 dependencies
│   │   ├── values.yaml
│   │   └── templates/
│   └── argocd/
│       └── application.yaml
├── scripts/
│   └── setup/
└── skaffold.yaml
```

### Docker Images Built

| Image | Description |
|-------|-------------|
| `coelho-realtime-fastapi` | FastAPI backend with 39 endpoints |
| `coelho-realtime-kafka-producers` | Kafka data generators for 3 ML use cases |
| `coelho-realtime-sveltekit` | SvelteKit frontend dashboard |

---

## Initial Setup (One-Time)

### Step 1: Run Setup Scripts

This creates the necessary namespace and registry credentials:

```bash
./scripts/setup/00-setup-all.sh
```

**What it does:**
- ✓ Creates `coelho-realtime` namespace
- ✓ Creates `gitlab-registry-secret` imagePullSecret
- ✓ Verifies prerequisites

### Step 2: Deploy ArgoCD Application

```bash
kubectl apply -f k3d/argocd/application.yaml
```

This tells ArgoCD to watch your Git repository and auto-deploy changes.

---

## Development Workflow

### Local Development (Skaffold)

For rapid local development with hot-reload:

```bash
skaffold dev
```

**Features:**
- Builds images locally (no registry push)
- Auto-deploys to local Kubernetes
- File sync for Python hot-reload
- Port forwarding automatically configured

**Access:**
- FastAPI: http://localhost:8000
- FastAPI Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- SvelteKit: http://localhost:3000
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

### Production Deployment (GitLab CI + ArgoCD)

Push code to trigger the automated pipeline:

```bash
git add .
git commit -m "Your changes"
git push origin master
```

**Automated Pipeline:**
1. **Build Stage**: Builds 3 Docker images (FastAPI, Kafka Producers, SvelteKit)
2. **Update Manifest**: Updates `k3d/helm/values.yaml` with new image tags
3. **Git Push**: Commits changes with `[skip ci]` (prevents infinite loop)
4. **ArgoCD Sync**: Detects Git change and deploys automatically

---

## Helm Dependencies

The umbrella chart includes these dependencies:

| Dependency | Version | Description |
|------------|---------|-------------|
| kafka | 31.0.0 | Apache Kafka (KRaft mode) |
| spark | 9.2.11 | Apache Spark for streaming |
| minio | 14.8.2 | S3-compatible storage for MLflow |
| mlflow | 2.2.0 | ML experiment tracking |
| redis | 20.2.1 | Caching for sub-ms inference |
| postgresql | 16.2.5 | Metadata storage |
| kube-prometheus-stack | 65.2.0 | Prometheus + Grafana |

---

## ML Use Cases

The platform runs 3 concurrent ML pipelines:

| Use Case | Type | Model | Key Metric |
|----------|------|-------|------------|
| **TFD** | Classification | River ML + CatBoost | F-Beta (β=2.0) |
| **ETA** | Regression | River ML + CatBoost | RMSE |
| **ECCI** | Clustering | River ML + KMeans | Silhouette Score |

**Learning Paradigm:**
- **Real-time**: River ML incremental learning from Kafka streams
- **Batch**: CatBoost/Scikit-Learn on Delta Lake data

---

## Monitoring Deployment

### Check GitLab CI Pipeline

```bash
# Via GitLab UI
open http://localhost:30082/public-projects/COELHORealTime/-/pipelines

# Or check runner logs
kubectl logs -n gitlab -l app=gitlab-runner -f
```

### Check ArgoCD Status

```bash
# Get application status
kubectl get application coelho-realtime -n argocd

# Watch sync progress
kubectl get application coelho-realtime -n argocd -w

# Get detailed info
kubectl describe application coelho-realtime -n argocd
```

### Check Pods

```bash
# Watch pods being created
kubectl get pods -n coelho-realtime -w

# Check specific pod logs
kubectl logs -n coelho-realtime -l app=coelho-realtime-fastapi-deployment -f

# Check all pod status
kubectl get pods -n coelho-realtime
```

---

## Observability Stack

### Prometheus Metrics

The platform exposes **50+ custom Prometheus metrics**:

| Category | Examples |
|----------|----------|
| API | `fastapi_requests_total`, `fastapi_request_duration_seconds` |
| ML Models | `ml_predictions_total`, `ml_inference_latency_seconds` |
| Kafka | `kafka_messages_produced_total`, `kafka_consumer_lag` |
| Business | `fraud_detected_total`, `delivery_prediction_accuracy` |

### Grafana Dashboards

**11 pre-configured dashboards:**

1. FastAPI Overview
2. ML Models Performance
3. Kafka Streaming
4. TFD Pipeline (Fraud Detection)
5. ETA Pipeline (Delivery Prediction)
6. ECCI Pipeline (Customer Segmentation)
7. Redis Cache Performance
8. MLflow Experiments
9. System Resources
10. Alerting Overview
11. Business Metrics

### Alerting Rules

**30+ alerting rules** covering:
- High API latency (>500ms)
- Model prediction errors
- Kafka consumer lag
- Resource exhaustion
- Service downtime

---

## Environment Configuration

### Local (Skaffold)
```yaml
# k3d/helm/values.yaml
environment: local
```
- Images: `coelho-realtime-fastapi` (no registry)
- Skaffold overrides with locally built images
- No imagePullSecret needed

### Production (ArgoCD)
```yaml
# k3d/argocd/application.yaml
helm:
  parameters:
    - name: environment
      value: production
```
- Images: `gitlab-registry.gitlab.svc.cluster.local:5000/coelho-realtime-fastapi:abc123`
- Uses imagePullSecret: `gitlab-registry-secret`
- Registry prefix added automatically by Helm

---

## Image Tagging Strategy

GitLab CI creates two tags for each image:

1. **Commit SHA** (e.g., `abc123def456`)
   - Immutable, traceable to code
   - Used by ArgoCD for production
   - Example: `coelho-realtime-fastapi:abc123def456`

2. **Latest**
   - Mutable, always points to newest
   - Convenient for testing
   - Example: `coelho-realtime-fastapi:latest`

---

## Troubleshooting

### ImagePullBackOff

**Symptom:** Pods stuck in `ImagePullBackOff` state

**Solution:**
```bash
# Check if secret exists
kubectl get secret gitlab-registry-secret -n coelho-realtime

# If missing, run setup again
./scripts/setup/02-create-registry-secret.sh

# Check pod events
kubectl describe pod <pod-name> -n coelho-realtime
```

### Pipeline Fails at Build Stage

**Symptom:** GitLab CI build fails with "docker: command not found"

**Solution:**
- Check GitLab Runner is running: `kubectl get pods -n gitlab -l app=gitlab-runner`
- Verify runner has Docker-in-Docker enabled
- Check runner logs: `kubectl logs -n gitlab -l app=gitlab-runner`

### ArgoCD Not Syncing

**Symptom:** ArgoCD shows "OutOfSync" but doesn't deploy

**Solution:**
```bash
# Check application status
kubectl get application coelho-realtime -n argocd

# Manually trigger sync
kubectl patch application coelho-realtime -n argocd --type merge -p '{"operation":{"initiatedBy":{"username":"admin"},"sync":{"syncStrategy":{"hook":{}}}}}'

# Or use ArgoCD UI
open http://localhost:8080
```

---

## Accessing Services

### Via Port Forward (Local)

```bash
# FastAPI
kubectl port-forward -n coelho-realtime svc/coelho-realtime-fastapi-service 8000:8000

# MLflow
kubectl port-forward -n coelho-realtime svc/coelho-realtime-mlflow-service 5000:5000

# SvelteKit
kubectl port-forward -n coelho-realtime svc/coelho-realtime-sveltekit-service 3000:3000

# Grafana
kubectl port-forward -n coelho-realtime svc/coelho-realtime-grafana 3001:80

# Prometheus
kubectl port-forward -n coelho-realtime svc/coelho-realtime-prometheus 9090:9090
```

---

## Rollback

### Via ArgoCD (Recommended)

```bash
# Get history
kubectl get application coelho-realtime -n argocd -o yaml

# Rollback via ArgoCD UI or CLI
# ArgoCD maintains full deployment history
```

### Via Git

```bash
# Revert the manifest commit
git revert HEAD
git push

# ArgoCD will automatically sync to the previous state
```

---

## Clean Up

### Remove Application

```bash
# Delete ArgoCD application (keeps namespace)
kubectl delete application coelho-realtime -n argocd

# Delete everything including namespace
kubectl delete namespace coelho-realtime
```

---

## CI/CD Pipeline Details

### Pipeline Stages

1. **build**
   - Image: `docker:24-cli`
   - Service: `docker:24-dind`
   - Builds 3 images (FastAPI, Kafka Producers, SvelteKit)
   - Pushes to GitLab Registry
   - Only runs on `master` branch

2. **update-manifest**
   - Image: `alpine:latest`
   - Updates `k3d/helm/values.yaml`
   - Commits with `[skip ci]`
   - Triggers ArgoCD sync

### Environment Variables (Auto-provided by GitLab)

- `CI_REGISTRY_USER`: GitLab CI token user
- `CI_REGISTRY_PASSWORD`: Temporary registry token
- `CI_COMMIT_SHA`: Git commit hash
- `CI_JOB_TOKEN`: Job-specific auth token

---

## References

- **Setup Scripts**: `scripts/setup/`
- **Helm Chart**: `k3d/helm/`
- **ArgoCD App**: `k3d/argocd/application.yaml`
- **CI Pipeline**: `.gitlab-ci.yml`
- **Skaffold Config**: `skaffold.yaml`
- **Portfolio Page**: https://rafaelcoelho1409.github.io/projects/COELHORealTime

---

*Last updated: 2026-01-28*
