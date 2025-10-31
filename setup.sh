#!/bin/bash
set -e


NAMESPACE="coelho-realtime"
SECRET_NAME="gitlab-registry-secret"
REGISTRY_URL="gitlab-registry.gitlab.svc.cluster.local:5000"
GITLAB_PASSWORD=$(kubectl get secret -n gitlab gitlab-gitlab-initial-root-password -o jsonpath='{.data.password}' | base64 -d)


kubectl create namespace ${NAMESPACE}
# Delete existing secret if present
if kubectl get secret ${SECRET_NAME} -n ${NAMESPACE} >/dev/null 2>&1; then
    echo "→ Deleting existing secret..."
    kubectl delete secret ${SECRET_NAME} -n ${NAMESPACE}
fi
# Create the imagePullSecret
echo "→ Creating imagePullSecret..."
kubectl create secret docker-registry ${SECRET_NAME} \
    --docker-server=${REGISTRY_URL} \
    --docker-username=root \
    --docker-password=${GITLAB_PASSWORD} \
    -n ${NAMESPACE}
#ArgoCD
kubectl apply -f argocd_k3d/application.yaml



