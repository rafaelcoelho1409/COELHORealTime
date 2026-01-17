# Helm Dependencies Management

## Overview

Helm chart dependencies are **committed to the repository** to avoid downloading them on every Skaffold build/deploy cycle. This significantly speeds up development iterations.

## Configuration

Skaffold is configured with `skipBuildDependencies: true` in `skaffold.yaml` to skip running `helm dependency build` on every deploy.

## Location

- **Chart**: `k3d/helm/Chart.yaml`
- **Dependencies**: `k3d/helm/charts/`
- **Lock file**: `k3d/helm/Chart.lock`
- **Skaffold config**: `skaffold.yaml` (skipBuildDependencies: true)

## Current Dependencies

| Chart | Version | Repository |
|-------|---------|------------|
| mlflow | 1.8.1 | community-charts |
| redis | 24.0.8 | bitnami |
| minio | 5.4.0 | minio |
| postgresql | 18.1.14 | bitnami |
| kube-prometheus-stack | 80.6.0 | prometheus-community |
| kafka | 32.4.3 | bitnami |
| spark | 10.0.3 | bitnami |

## Updating Dependencies

When you modify `Chart.yaml` dependencies (add, remove, or change versions):

```bash
# Rebuild dependencies
cd k3d/helm
helm dependency build

# Commit the updated charts
git add charts/ Chart.lock
git commit -m "Update helm dependencies"
```

## Why Commit Dependencies?

1. **Speed**: No downloads during `skaffold dev` or `skaffold run`
2. **Reproducibility**: Everyone gets the exact same chart versions
3. **Offline**: Works without internet access
4. **CI/CD**: No external dependencies during pipeline runs

## Troubleshooting

### Dependencies not found
```bash
cd k3d/helm
helm dependency build
```

### Outdated dependencies
```bash
cd k3d/helm
helm dependency update  # Updates to latest matching versions
git add charts/ Chart.lock
git commit -m "Update helm dependencies"
```

### Verify dependencies match Chart.yaml
```bash
helm dependency list k3d/helm
```
