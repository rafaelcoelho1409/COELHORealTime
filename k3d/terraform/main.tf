# K3D coelho-realtime Cluster - Terraform Configuration
#
# This configuration replicates the K3D cluster setup with all services:
# - K3D cluster with 1 server + 3 agents
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply

# K3D Cluster Module
module "k3d_cluster" {
  source = "./modules/k3d"

  cluster_name  = var.cluster_name
  k3s_version   = var.k3s_version
  servers       = var.servers
  agents        = var.agents
  registry_port = var.registry_port

  # Port mappings for all services
  port_mappings = concat(
    var.install_rancher ? [
      {
        host_port      = var.rancher_http_host_port
        container_port = var.rancher_http_node_port
        node_filter    = "loadbalancer"
      },
      {
        host_port      = var.rancher_https_host_port
        container_port = var.rancher_https_node_port
        node_filter    = "loadbalancer"
      }
    ] : []
  )

  # Volume mounts for persistent storage (MinIO data)
  volume_mounts = var.volume_mounts
}


# Rancher Module
module "rancher" {
  count  = var.install_rancher ? 1 : 0
  source = "./modules/rancher"

  cluster_name       = var.cluster_name
  http_node_port     = var.rancher_http_node_port
  https_node_port    = var.rancher_https_node_port
  bootstrap_password = var.rancher_bootstrap_password

  # Use external values file from the module directory
  values_file = "${path.module}/modules/rancher/values.yaml"

  # Explicitly depend on cluster API being ready
  cluster_ready = module.k3d_cluster.cluster_ready
  depends_on    = [module.k3d_cluster]
}


#To destroy all resources
#k3d cluster delete coelho-realtime
#terraform state rm $(terraform state list)
#terraform destroy