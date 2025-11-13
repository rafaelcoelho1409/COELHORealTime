#!/bin/bash
# Start/restart K3D cluster with all services

mkdir -p modules/k3d/.generated
terraform init
terraform apply -target=module.k3d_cluster -auto-approve
terraform apply -auto-approve

# Verify registry mirrors
docker exec k3d-coelho-realtime-server-0 cat /etc/rancher/k3s/registries.yaml 2>/dev/null | grep -q "localhost:5000" && echo "✓ Registry mirrors configured" || echo "✗ Registry mirrors not configured"