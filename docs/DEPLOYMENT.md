# COELHO RealTime - Deployment Guide

Complete guide for deploying the COELHO RealTime application using GitLab CI/CD and ArgoCD.

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

### Required
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
kubectl apply -f argocd_k3d/application.yaml
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
- MLflow: http://localhost:5000
- Streamlit: http://localhost:8501

### Production Deployment (GitLab CI + ArgoCD)

Push code to trigger the automated pipeline:

```bash
git add .
git commit -m "Your changes"
git push origin master
```

**Automated Pipeline:**
1. **Build Stage**: Builds 4 Docker images (FastAPI, Kafka, MLflow, Streamlit)
2. **Update Manifest**: Updates `helm_k3d/values.yaml` with new image tags
3. **Git Push**: Commits changes with `[skip ci]` (prevents infinite loop)
4. **ArgoCD Sync**: Detects Git change and deploys automatically

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

## Environment Configuration

The application uses environment-based configuration:

### Local (Skaffold)
```yaml
# helm_k3d/values.yaml
environment: local
```
- Images: `coelho-realtime-fastapi` (no registry)
- Skaffold overrides with locally built images
- No imagePullSecret needed

### Production (ArgoCD)
```yaml
# argocd_k3d/application.yaml
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
open http://localhost:8080 # (adjust port for your setup)
```

### Git Push Creates Infinite Loop

**Should not happen** - the pipeline uses `[skip ci]` to prevent this.

**If it happens:**
- Check commit messages contain `[skip ci]`
- Verify `.gitlab-ci.yml` line 73: `git commit -m "... [skip ci]"`

---

## Accessing Services

### Via Port Forward (Local)

```bash
# FastAPI
kubectl port-forward -n coelho-realtime svc/coelho-realtime-fastapi-service 8000:8000

# MLflow
kubectl port-forward -n coelho-realtime svc/coelho-realtime-mlflow-service 5000:5000

# Streamlit
kubectl port-forward -n coelho-realtime svc/coelho-realtime-streamlit-service 8501:8501
```

### Via LoadBalancer (Production)

Check service external IPs:
```bash
kubectl get svc -n coelho-realtime
```

---

## Rollback

### Via ArgoCD (Recommended)

```bash
# Get history
kubectl get application coelho-realtime -n argocd -o yaml

# Rollback to previous version via UI or CLI
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

### Remove Setup

```bash
# Just remove secrets (keep namespace)
kubectl delete secret gitlab-registry-secret -n coelho-realtime
```

---

## CI/CD Pipeline Details

### Pipeline Stages

1. **build**
   - Image: `docker:24-cli`
   - Service: `docker:24-dind`
   - Builds 4 images
   - Pushes to GitLab Registry
   - Only runs on `master` branch

2. **update-manifest**
   - Image: `alpine:latest`
   - Updates `helm_k3d/values.yaml`
   - Commits with `[skip ci]`
   - Triggers ArgoCD sync

### Environment Variables (Auto-provided by GitLab)

- `CI_REGISTRY_USER`: GitLab CI token user
- `CI_REGISTRY_PASSWORD`: Temporary registry token
- `CI_COMMIT_SHA`: Git commit hash
- `CI_JOB_TOKEN`: Job-specific auth token

---

## Security Notes

- Registry credentials stored as Kubernetes secret
- Secrets are namespace-scoped
- CI uses temporary tokens (not permanent passwords)
- For production: Consider using deploy tokens instead of root password

---

## References

- **Setup Scripts**: `scripts/setup/README.md`
- **Helm Chart**: `helm_k3d/`
- **ArgoCD App**: `argocd_k3d/application.yaml`
- **CI Pipeline**: `.gitlab-ci.yml`
- **Skaffold Config**: `skaffold.yaml`
