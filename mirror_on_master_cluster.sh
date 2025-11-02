~/K3D/add-port.sh 8000 30000 loadbalancer "FastAPI"
~/K3D/add-port.sh 9092 30001 loadbalancer "Kafka"
~/K3D/add-port.sh 5001 30002 loadbalancer "MLflow"
~/K3D/add-port.sh 8501 30003 loadbalancer "Streamlit"

#Check later if these commands are really necessary, 
#and if these ports are naturally available when ArgoCD deploys each deployment and service