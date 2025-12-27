#!/bin/bash

CLUSTER_NAME="coelho"

# Check if cluster exists
if k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "Cluster $CLUSTER_NAME already exists. Skipping creation."
else
    echo "Creating cluster $CLUSTER_NAME..."
    k3d cluster create "$CLUSTER_NAME" \
        --api-port 16443 \
        --port "7080:80@loadbalancer" \
        --port "7443:443@loadbalancer" \
        --port "8002:8002@loadbalancer" \
        --port "8003:8003@loadbalancer" \
        --port "9092:9092@loadbalancer" \
        --port "5000:5000@loadbalancer" \
        --port "3000:3000@loadbalancer" \
        --agents 1
    #7080: Rancher
    #7443: Rancher
    #8002: River (Incremental ML)
    #8003: Sklearn (Batch ML)
    #9092: Kafka
    #5000: MLflow
    #3000: Reflex
    #Rancher install
    helm repo add rancher-latest https://releases.rancher.com/server-charts/latest
    helm repo update
    kubectl create namespace cattle-system
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    helm install cert-manager jetstack/cert-manager \
    --namespace cert-manager \
    --create-namespace \
    --version v1.13.1 \
    --set installCRDs=true
    helm install rancher rancher-latest/rancher \
    --namespace cattle-system \
    --set hostname=localhost \
    --set bootstrapPassword=password \
    --set replicas=1
fi

sh upload_env_to_k3d.sh