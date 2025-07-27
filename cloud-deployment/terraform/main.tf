terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# Configure providers
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.jarvis_cluster.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.jarvis_cluster.master_auth.0.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.jarvis_cluster.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.jarvis_cluster.master_auth.0.cluster_ca_certificate)
  }
}

# Data sources
data "google_client_config" "default" {}

data "google_compute_zones" "available" {
  region = var.region
}

# Variables
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "The name of the GKE cluster"
  type        = string
  default     = "jarvis-v2-cluster"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "min_nodes" {
  description = "Minimum number of nodes in the cluster"
  type        = number
  default     = 3
}

variable "max_nodes" {
  description = "Maximum number of nodes in the cluster"
  type        = number
  default     = 20
}

variable "machine_type" {
  description = "Machine type for the cluster nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "disk_size" {
  description = "Disk size for cluster nodes"
  type        = number
  default     = 50
}

variable "enable_monitoring" {
  description = "Enable monitoring and logging"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable network policy"
  type        = bool
  default     = true
}

# Local values
locals {
  cluster_name = "${var.cluster_name}-${var.environment}"
  
  common_labels = {
    project     = "jarvis-v2"
    environment = var.environment
    managed_by  = "terraform"
  }
  
  node_pools = {
    general = {
      machine_type = "e2-standard-4"
      min_nodes    = 2
      max_nodes    = 10
      disk_size    = 50
      preemptible  = false
      labels = {
        node-type = "compute"
      }
      taints = []
    }
    
    compute_intensive = {
      machine_type = "c2-standard-8"
      min_nodes    = 1
      max_nodes    = 5
      disk_size    = 100
      preemptible  = false
      labels = {
        node-type = "compute-intensive"
      }
      taints = [{
        key    = "node-type"
        value  = "compute-intensive"
        effect = "NO_SCHEDULE"
      }]
    }
    
    memory_optimized = {
      machine_type = "n2-highmem-4"
      min_nodes    = 1
      max_nodes    = 3
      disk_size    = 50
      preemptible  = false
      labels = {
        node-type = "memory-optimized"
      }
      taints = [{
        key    = "node-type"
        value  = "memory-optimized"
        effect = "NO_SCHEDULE"
      }]
    }
    
    storage_optimized = {
      machine_type = "n2-standard-4"
      min_nodes    = 1
      max_nodes    = 3
      disk_size    = 200
      preemptible  = false
      labels = {
        node-type = "storage-optimized"
      }
      taints = [{
        key    = "node-type"
        value  = "storage-optimized"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# VPC Network
resource "google_compute_network" "jarvis_network" {
  name                    = "${local.cluster_name}-network"
  auto_create_subnetworks = false
  
  depends_on = [
    google_project_service.compute_api,
    google_project_service.container_api
  ]
}

# Subnet
resource "google_compute_subnetwork" "jarvis_subnet" {
  name          = "${local.cluster_name}-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.jarvis_network.name
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
  
  private_ip_google_access = true
}

# Firewall rules
resource "google_compute_firewall" "jarvis_firewall" {
  name    = "${local.cluster_name}-firewall"
  network = google_compute_network.jarvis_network.name
  
  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8000", "8001", "8002"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["jarvis-cluster"]
}

# Enable APIs
resource "google_project_service" "compute_api" {
  service = "compute.googleapis.com"
}

resource "google_project_service" "container_api" {
  service = "container.googleapis.com"
}

resource "google_project_service" "monitoring_api" {
  service = "monitoring.googleapis.com"
}

resource "google_project_service" "logging_api" {
  service = "logging.googleapis.com"
}

# GKE Cluster
resource "google_container_cluster" "jarvis_cluster" {
  name     = local.cluster_name
  location = var.region
  
  # Network configuration
  network    = google_compute_network.jarvis_network.name
  subnetwork = google_compute_subnetwork.jarvis_subnet.name
  
  # IP allocation policy
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1
  
  # Cluster configuration
  min_master_version = "1.27"
  
  # Network policy
  network_policy {
    enabled = var.enable_network_policy
  }
  
  # Monitoring and logging
  monitoring_config {
    enable_components = var.enable_monitoring ? ["SYSTEM_COMPONENTS", "WORKLOADS"] : []
  }
  
  logging_config {
    enable_components = var.enable_monitoring ? ["SYSTEM_COMPONENTS", "WORKLOADS"] : []
  }
  
  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    network_policy_config {
      disabled = !var.enable_network_policy
    }
    
    dns_cache_config {
      enabled = true
    }
  }
  
  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Maintenance policy
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }
  
  # Security
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  # Private cluster
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }
  
  # Resource labels
  resource_labels = local.common_labels
  
  depends_on = [
    google_project_service.compute_api,
    google_project_service.container_api,
    google_project_service.monitoring_api,
    google_project_service.logging_api
  ]
}

# Node pools
resource "google_container_node_pool" "jarvis_node_pools" {
  for_each = local.node_pools
  
  name       = each.key
  location   = var.region
  cluster    = google_container_cluster.jarvis_cluster.name
  node_count = each.value.min_nodes
  
  # Autoscaling
  autoscaling {
    min_node_count = each.value.min_nodes
    max_node_count = each.value.max_nodes
  }
  
  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  # Node configuration
  node_config {
    preemptible  = each.value.preemptible
    machine_type = each.value.machine_type
    disk_size_gb = each.value.disk_size
    disk_type    = "pd-ssd"
    
    # Service account
    service_account = google_service_account.jarvis_cluster_sa.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Labels
    labels = merge(local.common_labels, each.value.labels)
    
    # Taints
    dynamic "taint" {
      for_each = each.value.taints
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }
    
    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded instance
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
    
    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
    
    tags = ["jarvis-cluster"]
  }
  
  depends_on = [
    google_container_cluster.jarvis_cluster
  ]
}

# Service account for cluster
resource "google_service_account" "jarvis_cluster_sa" {
  account_id   = "${local.cluster_name}-sa"
  display_name = "Jarvis Cluster Service Account"
  description  = "Service account for Jarvis GKE cluster"
}

# IAM bindings for service account
resource "google_project_iam_member" "jarvis_cluster_sa_roles" {
  for_each = toset([
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.jarvis_cluster_sa.email}"
}

# Static IP for load balancer
resource "google_compute_global_address" "jarvis_ip" {
  name = "${local.cluster_name}-ip"
}

# Cloud SQL instance for production database
resource "google_sql_database_instance" "jarvis_postgres" {
  name             = "${local.cluster_name}-postgres"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier = "db-n1-standard-2"
    
    backup_configuration {
      enabled    = true
      start_time = "02:00"
    }
    
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.jarvis_network.id
      enable_private_path_for_google_cloud_services = true
    }
    
    database_flags {
      name  = "max_connections"
      value = "200"
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
  }
  
  depends_on = [
    google_service_networking_connection.jarvis_vpc_connection
  ]
}

# Database
resource "google_sql_database" "jarvis_db" {
  name     = "jarvis_v2"
  instance = google_sql_database_instance.jarvis_postgres.name
}

# Database user
resource "google_sql_user" "jarvis_db_user" {
  name     = "jarvis"
  instance = google_sql_database_instance.jarvis_postgres.name
  password = var.db_password
}

# Private service connection
resource "google_compute_global_address" "jarvis_private_ip_address" {
  name          = "${local.cluster_name}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.jarvis_network.id
}

resource "google_service_networking_connection" "jarvis_vpc_connection" {
  network                 = google_compute_network.jarvis_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.jarvis_private_ip_address.name]
}

# Memorystore Redis
resource "google_redis_instance" "jarvis_redis" {
  name           = "${local.cluster_name}-redis"
  memory_size_gb = 5
  region         = var.region
  
  authorized_network = google_compute_network.jarvis_network.id
  
  redis_version     = "REDIS_7_0"
  display_name      = "Jarvis Redis Instance"
  reserved_ip_range = "10.3.0.0/24"
  
  depends_on = [
    google_project_service.compute_api
  ]
}

# Cloud Storage bucket for backups
resource "google_storage_bucket" "jarvis_backups" {
  name     = "${var.project_id}-jarvis-backups"
  location = var.region
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
  
  labels = local.common_labels
}

# Outputs
output "cluster_name" {
  value = google_container_cluster.jarvis_cluster.name
}

output "cluster_endpoint" {
  value = google_container_cluster.jarvis_cluster.endpoint
}

output "cluster_ca_certificate" {
  value = google_container_cluster.jarvis_cluster.master_auth.0.cluster_ca_certificate
}

output "load_balancer_ip" {
  value = google_compute_global_address.jarvis_ip.address
}

output "postgres_connection_name" {
  value = google_sql_database_instance.jarvis_postgres.connection_name
}

output "redis_host" {
  value = google_redis_instance.jarvis_redis.host
}

output "backup_bucket" {
  value = google_storage_bucket.jarvis_backups.name
}

# Variables for sensitive data
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}