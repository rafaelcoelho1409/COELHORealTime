terraform {
  required_providers {
    k3d = {
      source  = "pvotal-tech/k3d"
      version = "~> 0.0.7"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
}

# K3D Cluster
resource "k3d_cluster" "coelho-realtime" {
  name    = var.cluster_name
  servers = var.servers
  agents  = var.agents

  image = "rancher/k3s:${var.k3s_version}"

  # K3s server arguments
  k3s {
    extra_args {
      arg          = "--disable=traefik"
      node_filters = ["server:*"]
    }
  }

  # Registry configuration
  registries {
    create {
      name      = "${var.cluster_name}-registry"
      host      = "0.0.0.0"
      host_port = var.registry_port
    }
  }

  # Port mappings
  dynamic "port" {
    for_each = var.port_mappings
    content {
      host_port      = port.value.host_port
      container_port = port.value.container_port
      node_filters   = [port.value.node_filter]
    }
  }

  # Volume mounts for persistent data (survives cluster restarts)
  dynamic "volume" {
    for_each = var.volume_mounts
    content {
      source       = volume.value.host_path
      destination  = volume.value.container_path
      node_filters = volume.value.node_filters
    }
  }

  # K3D options
  k3d {
    disable_load_balancer = false
  }

  # Kubeconfig options
  kubeconfig {
    update_default_kubeconfig = true
    switch_current_context    = true
  }
}

# Wait for cluster to be ready
# This ensures API is accessible and kubeconfig is valid
resource "null_resource" "wait_for_cluster" {
  depends_on = [k3d_cluster.coelho-realtime]

  provisioner "local-exec" {
    command = <<-EOT
      # Wait for API to respond
      for i in {1..30}; do
        if kubectl cluster-info --context=k3d-${var.cluster_name} >/dev/null 2>&1; then
          echo "API ready, waiting for nodes..."
          kubectl wait --for=condition=Ready nodes --all --timeout=120s --context=k3d-${var.cluster_name}
          exit 0
        fi
        echo "Waiting for API... ($i/30)"
        sleep 2
      done
      exit 1
    EOT
  }
}

# Create alias for backward compatibility
resource "null_resource" "wait_for_api" {
  depends_on = [null_resource.wait_for_cluster]

  provisioner "local-exec" {
    command = "echo 'Cluster API ready'"
  }
}

# Configure auto-restart for cluster containers
resource "null_resource" "configure_auto_restart" {
  depends_on = [null_resource.wait_for_cluster]

  provisioner "local-exec" {
    command = "docker update --restart=unless-stopped $(docker ps -aq --filter 'name=k3d-${var.cluster_name}') 2>/dev/null || true"
  }

  # Trigger on cluster changes
  triggers = {
    cluster_id = k3d_cluster.coelho-realtime.id
  }
}

# Generate registries.yaml from template
resource "local_file" "registries_yaml" {
  content = templatefile("${path.module}/registries.yaml.tpl", {
    registry_port = var.registry_port
    registry_name = "${var.cluster_name}-registry"
  })
  filename = "${path.module}/.generated/registries.yaml"

  depends_on = [k3d_cluster.coelho-realtime]
}

# Configure registry mirrors in all cluster nodes
resource "null_resource" "configure_registry_mirrors" {
  depends_on = [null_resource.wait_for_cluster, local_file.registries_yaml]

  provisioner "local-exec" {
    command = <<-EOT
      # Copy registries.yaml to all server nodes
      for node in $(docker ps --filter "name=k3d-${var.cluster_name}-server" --format "{{.Names}}"); do
        echo "Configuring registry mirrors on $node..."
        docker cp ${local_file.registries_yaml.filename} $node:/etc/rancher/k3s/registries.yaml
      done

      # Copy registries.yaml to all agent nodes
      for node in $(docker ps --filter "name=k3d-${var.cluster_name}-agent" --format "{{.Names}}"); do
        echo "Configuring registry mirrors on $node..."
        docker cp ${local_file.registries_yaml.filename} $node:/etc/rancher/k3s/registries.yaml
      done

      # Restart all K3d cluster nodes to apply registry configuration
      echo "Restarting K3d cluster nodes to apply registry configuration..."
      docker restart $(docker ps --filter "name=k3d-${var.cluster_name}" --format "{{.Names}}" | grep -E "(server|agent)")

      # Wait for cluster to be ready again
      echo "Waiting for cluster to be ready..."
      sleep 10
      kubectl wait --for=condition=Ready nodes --all --timeout=120s --context=k3d-${var.cluster_name} || true
    EOT
  }

  # Trigger on registry configuration changes
  triggers = {
    registries_yaml_content = local_file.registries_yaml.content
    cluster_id              = k3d_cluster.coelho-realtime.id
  }
}
