# ArgoCD Integration - Implementation Guide

## Status: IMPLEMENTED

This document describes the ArgoCD GitOps integration for COELHO RealTime, which is now fully operational.

---

## Architecture Decision: Global ArgoCD Instance

**Chosen Approach**: Global ArgoCD instance managing multiple projects

**Benefits:**
- Single control plane for all projects
- Centralized RBAC, policies, and monitoring
- Lower resource overhead
- Better for multi-project patterns
- GitLab serves multiple projects to one ArgoCD

---

## Project Structure (Current)

```
COELHORealTime/
├── apps/
│   ├── fastapi/           # 39 endpoints, 3 routers
│   ├── kafka_producers/   # Data generators for TFD, ETA, ECCI
│   └── sveltekit/         # Frontend dashboard
├── k3d/
│   ├── helm/              # Helm umbrella chart
│   │   ├── Chart.yaml     # 7 dependencies
│   │   ├── values.yaml    # Image tags updated by CI
│   │   └── templates/     # Custom K8s resources
│   └── argocd/
│       └── application.yaml  # ArgoCD Application definition
├── scripts/
│   └── setup/
├── skaffold.yaml          # Local development
└── .gitlab-ci.yml         # CI/CD pipeline
```

---

## ArgoCD Application Manifest

**Location**: `k3d/argocd/application.yaml`

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: coelho-realtime
  namespace: argocd
spec:
  project: default
  source:
    repoURL: http://gitlab-webservice-default.gitlab.svc.cluster.local/public-projects/COELHORealTime.git
    targetRevision: HEAD
    path: k3d/helm
    helm:
      valueFiles:
        - values.yaml
      parameters:
        - name: environment
          value: production
  destination:
    server: https://kubernetes.default.svc
    namespace: coelho-realtime
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

---

## GitLab CI + ArgoCD Workflow

### Complete Pipeline Flow

```
┌─────────────────┐
│  Developer      │  → Code changes pushed to GitLab
└─────────────────┘
        ↓
┌─────────────────┐
│  GitLab CI      │  → Builds 3 Docker images:
│  (build stage)  │     - coelho-realtime-fastapi
│                 │     - coelho-realtime-kafka-producers
│                 │     - coelho-realtime-sveltekit
└─────────────────┘
        ↓
┌─────────────────┐
│  GitLab CI      │  → Updates k3d/helm/values.yaml
│  (update-       │     with new image tags (commit SHA)
│   manifest)     │  → Commits with [skip ci]
└─────────────────┘
        ↓
┌─────────────────┐
│  ArgoCD         │  → Detects Git change
│                 │  → Syncs Helm chart to cluster
│                 │  → Reports status in UI
└─────────────────┘
        ↓
┌─────────────────┐
│  Kubernetes     │  → New pods deployed
│                 │  → Old pods terminated
│                 │  → Zero-downtime rollout
└─────────────────┘
```

### GitLab CI Pipeline Stages

**Stage 1: Build**
```yaml
build:
  stage: build
  image: docker:24-cli
  services:
    - docker:24-dind
  script:
    - docker build -t $REGISTRY/fastapi:$CI_COMMIT_SHA ./apps/fastapi
    - docker build -t $REGISTRY/kafka-producers:$CI_COMMIT_SHA ./apps/kafka_producers
    - docker build -t $REGISTRY/sveltekit:$CI_COMMIT_SHA ./apps/sveltekit
    - docker push ...
  only:
    - master
```

**Stage 2: Update Manifest**
```yaml
update-manifest:
  stage: update-manifest
  image: alpine:latest
  script:
    - sed -i "s/tag:.*/tag: $CI_COMMIT_SHA/" k3d/helm/values.yaml
    - git commit -m "Update image tags to $CI_COMMIT_SHA [skip ci]"
    - git push
  only:
    - master
```

---

## Development vs Production Workflow

### Local Development (Skaffold)

```bash
skaffold dev
```

- Direct deployment to k3d cluster
- Fast iteration (hot-reload)
- No GitLab CI involved
- No ArgoCD involved
- Images built locally

### Production (GitLab CI + ArgoCD)

```bash
git add .
git commit -m "Feature: add new endpoint"
git push origin master
```

- GitLab CI builds images
- Pushes to GitLab Registry
- Updates values.yaml
- ArgoCD syncs automatically
- Zero-downtime deployment

---

## Commands Reference

### ArgoCD Status

```bash
# Check application status
kubectl get application coelho-realtime -n argocd

# Watch sync progress
kubectl get application coelho-realtime -n argocd -w

# Detailed info
kubectl describe application coelho-realtime -n argocd

# Force sync
argocd app sync coelho-realtime
```

### ArgoCD UI Access

```bash
# Port forward to ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### Application Deployment

```bash
# Initial deployment
kubectl apply -f k3d/argocd/application.yaml

# Check application in ArgoCD
argocd app get coelho-realtime

# Manual sync (if auto-sync disabled)
argocd app sync coelho-realtime
```

---

## Helm Dependencies Managed by ArgoCD

When ArgoCD syncs, it deploys these Helm dependencies:

| Dependency | Version | Namespace |
|------------|---------|-----------|
| kafka | 31.0.0 | coelho-realtime |
| spark | 9.2.11 | coelho-realtime |
| minio | 14.8.2 | coelho-realtime |
| mlflow | 2.2.0 | coelho-realtime |
| redis | 20.2.1 | coelho-realtime |
| postgresql | 16.2.5 | coelho-realtime |
| kube-prometheus-stack | 65.2.0 | coelho-realtime |

Plus custom templates for:
- FastAPI Deployment + Service
- Kafka Producers Deployment
- SvelteKit Deployment + Service
- Spark Streaming Job
- ConfigMaps and Secrets

---

## Sync Policies

### Automated Sync
```yaml
syncPolicy:
  automated:
    prune: true      # Delete resources removed from Git
    selfHeal: true   # Revert manual changes in cluster
```

### Sync Options
```yaml
syncOptions:
  - CreateNamespace=true  # Create namespace if not exists
```

---

## Troubleshooting

### Application OutOfSync

```bash
# Check sync status
argocd app get coelho-realtime

# Check sync diff
argocd app diff coelho-realtime

# Force sync
argocd app sync coelho-realtime --force
```

### Helm Rendering Issues

```bash
# Test Helm locally
helm template coelho-realtime k3d/helm/ --debug

# Check ArgoCD logs
kubectl logs -n argocd -l app.kubernetes.io/name=argocd-repo-server
```

### GitLab Connection Issues

```bash
# Test GitLab access from ArgoCD
kubectl exec -n argocd -it <argocd-repo-server-pod> -- \
  git ls-remote http://gitlab-webservice-default.gitlab.svc.cluster.local/public-projects/COELHORealTime.git
```

---

## Security Considerations

- ArgoCD uses cluster-internal GitLab URL
- Registry credentials stored as Kubernetes Secret
- Secrets are namespace-scoped
- CI uses temporary tokens (not permanent passwords)
- RBAC configured for application namespaces

---

## References

- **ArgoCD Docs**: https://argo-cd.readthedocs.io/
- **Helm Chart**: `k3d/helm/`
- **ArgoCD Manifest**: `k3d/argocd/application.yaml`
- **GitLab CI**: `.gitlab-ci.yml`
- **Deployment Guide**: `docs/DEPLOYMENT.md`

---

*Implementation Complete: January 2026*
*Last Updated: 2026-01-28*
