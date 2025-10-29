# ArgoCD Integration Roadmap

## 1. Global ArgoCD vs Per-Project ArgoCD

### Recommendation: Global ArgoCD Instance

**Pros of global ArgoCD:**
- Single control plane for all your projects (COELHORealTime and future ones)
- Centralized RBAC, policies, and monitoring
- Lower resource overhead (one ArgoCD instance vs many)
- Better for learning and experimentation with multi-project/multi-cluster patterns
- Easier to implement cross-project dependencies if needed
- Your local GitLab can serve multiple projects to one ArgoCD

**When per-project ArgoCD makes sense:**
- Large enterprise with strict project isolation requirements
- Different teams with completely separate infrastructure
- Projects on completely different clusters with no shared management

**Implementation:** Deploy ArgoCD in a dedicated management namespace (like `argocd` or `platform`) and use ArgoCD's Application and AppProject resources to manage different projects.

---

## 2. ArgoCD Integration Strategy

### Project Structure

```
COELHORealTime/
├── helm_k3d/              # Your existing Helm chart
├── argocd/                # NEW: ArgoCD manifests
│   └── application.yaml   # ArgoCD Application definition
└── skaffold.yaml          # Keep for dev workflow
```

### ArgoCD Application Manifest

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: coelho-realtime
  namespace: argocd  # Where ArgoCD is installed
spec:
  project: default
  source:
    repoURL: http://your-gitlab-url/your-repo.git
    targetRevision: HEAD  # or specific branch like 'main'
    path: helm_k3d
    helm:
      valueFiles:
        - values.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: coelho
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### Commands to Use

```bash
# Development (no ArgoCD, fast iteration)
skaffold dev

# Production-like (with ArgoCD)
kubectl apply -f argocd/application.yaml

# Check status
kubectl get applications -n argocd
argocd app get coelho-realtime
```

**Visibility:** You'll see it in both Rancher (as regular k8s resources in `coelho` namespace) and ArgoCD UI (application health/sync status).

---

## 3. GitLab CI + ArgoCD + Skaffold Integration

### Recommended Workflow

**Development (local):** `skaffold dev` - Direct deployment, fast feedback

**Production:** GitLab CI builds → pushes images → ArgoCD syncs → deploys

### GitLab CI Structure

Create `.gitlab-ci.yml`:

```yaml
stages:
  - build
  - deploy

variables:
  REGISTRY: registry.gitlab.com/your-namespace/coelho-realtime

build:
  stage: build
  script:
    # Build and push images to GitLab registry
    - docker build -t $REGISTRY/fastapi:$CI_COMMIT_SHA ./apps/fastapi -f ./apps/fastapi/Dockerfile.fastapi
    - docker push $REGISTRY/fastapi:$CI_COMMIT_SHA

    - docker build -t $REGISTRY/kafka:$CI_COMMIT_SHA ./apps/kafka -f ./apps/kafka/Dockerfile.kafka
    - docker push $REGISTRY/kafka:$CI_COMMIT_SHA

    - docker build -t $REGISTRY/mlflow:$CI_COMMIT_SHA ./apps/mlflow -f ./apps/mlflow/Dockerfile.mlflow
    - docker push $REGISTRY/mlflow:$CI_COMMIT_SHA

    - docker build -t $REGISTRY/streamlit:$CI_COMMIT_SHA ./apps/streamlit -f ./apps/streamlit/Dockerfile.streamlit
    - docker push $REGISTRY/streamlit:$CI_COMMIT_SHA
  only:
    - main
    - develop

deploy:
  stage: deploy
  script:
    # Update Helm values or use kustomize to update image tags
    - |
      kubectl patch application coelho-realtime -n argocd \
        --type merge \
        -p '{"spec":{"source":{"helm":{"parameters":[
          {"name":"fastapi.image","value":"'$REGISTRY/fastapi:$CI_COMMIT_SHA'"},
          {"name":"kafka.image","value":"'$REGISTRY/kafka:$CI_COMMIT_SHA'"},
          {"name":"mlflow.image","value":"'$REGISTRY/mlflow:$CI_COMMIT_SHA'"},
          {"name":"streamlit.image","value":"'$REGISTRY/streamlit:$CI_COMMIT_SHA'"}
        ]}}}}'
    # Or use ArgoCD CLI
    - argocd app sync coelho-realtime
  only:
    - main
```

### ArgoCD + Skaffold Integration

**Can you integrate ArgoCD with Skaffold?**

Yes, but it's typically not recommended because they serve different purposes:
- **Skaffold**: Dev-time tool for rapid iteration (build, deploy, tail logs, sync files)
- **ArgoCD**: GitOps tool for production declarative deployment

However, you can use Skaffold to render manifests that ArgoCD deploys:
```yaml
# skaffold.yaml can have a 'render' profile
profiles:
  - name: render
    build:
      artifacts: [...]
    deploy:
      kubectl: {}
```

Then: `skaffold render > rendered.yaml` and commit to Git for ArgoCD.

### Complete Workflow Diagram

```
┌─────────────────┐
│  Development    │  → skaffold dev (direct to k3d)
└─────────────────┘

┌─────────────────┐
│  Git Push       │  → GitLab CI builds images
└─────────────────┘
        ↓
┌─────────────────┐
│  GitLab CI      │  → Pushes to registry, updates Git
└─────────────────┘
        ↓
┌─────────────────┐
│  ArgoCD         │  → Detects Git change, syncs to cluster
└─────────────────┘
        ↓
┌─────────────────┐
│  Production     │  → Running in cluster (k3d/prod)
└─────────────────┘
```

**Summary:** Use Skaffold for development, GitLab CI + ArgoCD for production simulation.

---

## Next Steps

1. Install ArgoCD in your k3d cluster
2. Create `argocd/` directory with Application manifest
3. Create `.gitlab-ci.yml` for CI/CD pipeline
4. Configure GitLab registry access
5. Test the complete workflow

---

## References

- ArgoCD Documentation: https://argo-cd.readthedocs.io/
- Skaffold Documentation: https://skaffold.dev/
- GitLab CI/CD: https://docs.gitlab.com/ee/ci/
