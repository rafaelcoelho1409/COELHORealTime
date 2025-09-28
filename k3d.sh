######## CONFIGMAP #######
## For Grafana
## Delete existing ConfigMaps (ignore errors if they don't exist)
#kubectl delete configmap grafana-provisioning --ignore-not-found=true -n xvia-demo
#kubectl delete configmap grafana-dashboards --ignore-not-found=true -n xvia-demo
## Recreate ConfigMaps with updated files
##kubectl create configmap grafana-provisioning --from-file=./apps/grafana/provisioning/ -n xvia-demo
#kubectl create configmap grafana-provisioning \
#--from-file=dashboards.yaml=./apps/grafana/provisioning/dashboards/dashboards.yaml \
#--from-file=datasources.yaml=./apps/grafana/provisioning/datasources/datasources.yaml \
#-n xvia-demo
#kubectl create configmap grafana-dashboards --from-file=./apps/grafana/dashboards/ -n xvia-demo
## For Prometheus
## Delete existing ConfigMaps (ignore errors if they don't exist)
#kubectl delete configmap prometheus-config --ignore-not-found=true -n xvia-demo
## Recreate ConfigMaps with updated files
#kubectl create configmap prometheus-config --from-file=prometheus.yml=./apps/prometheus/prometheus.yml -n xvia-demo
#