variable "cluster_name" {
  description = "Name of the K3D cluster"
  type        = string
}

variable "k3s_version" {
  description = "K3s version to use"
  type        = string
}

variable "servers" {
  description = "Number of server nodes"
  type        = number
}

variable "agents" {
  description = "Number of agent nodes"
  type        = number
}

variable "registry_port" {
  description = "Port for the K3D registry"
  type        = number
}

variable "port_mappings" {
  description = "List of port mappings for the cluster"
  type = list(object({
    host_port      = number
    container_port = number
    node_filter    = string
  }))
  default = []
}
