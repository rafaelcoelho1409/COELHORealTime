# Complete Setup & Development Commands (for local development)
## 1. Initial Setup (One-Time)
k3d cluster create coelho \
--api-port 16443 \
--port "7080:80@loadbalancer" \
--port "7443:443@loadbalancer" \
--port "8000:8000@loadbalancer" \
--port "8501:8501@loadbalancer" \
--port "5000:5000@loadbalancer" \
--agents 1
#7080: Rancher
#7443: Rancher
#8000: FastAPI
#8501: Streamlit
#5000: MLflow
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
#access via http://localhost:7080
#To turn Rancher on again:
helm upgrade rancher rancher-latest/rancher \
  --namespace cattle-system \
  --set hostname=localhost \
  --set replicas=1

./upload_env_to_k3d.sh
./k3d.sh

## 3. Daily Development
skaffold dev --profile=dev
