# K3D coelho-realtime Cluster - Terraform Configuration
#
# This configuration replicates the K3D cluster setup with all services:
# - K3D cluster with 1 server + 3 agents
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply

# Compute project root from module path (k3d/terraform -> project root)
locals {
  project_root = abspath("${path.module}/../..")

  # Default volume mounts using project-relative paths
  default_volume_mounts = [
    {
      host_path      = "${local.project_root}/data/minio"
      container_path = "/data/minio"
      node_filters   = ["server:*", "agent:*"]
    },
    {
      host_path      = "${local.project_root}/data/postgresql"
      container_path = "/data/postgresql"
      node_filters   = ["server:*", "agent:*"]
    }
  ]

  # Use provided volume_mounts or fall back to defaults
  volume_mounts = var.volume_mounts != null ? var.volume_mounts : local.default_volume_mounts
}

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
    ] : [],
    # MinIO port mappings (API and Console)
    [
      {
        host_port      = 9000   # MinIO API
        container_port = 30900  # MinIO NodePort
        node_filter    = "loadbalancer"
      },
      {
        host_port      = 9001   # MinIO Console
        container_port = 30901  # MinIO Console NodePort
        node_filter    = "loadbalancer"
      }
    ]
  )

  # Volume mounts for persistent storage (MinIO and PostgreSQL data)
  # Uses project-relative paths computed in locals block
  volume_mounts = local.volume_mounts
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