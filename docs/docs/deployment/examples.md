# Deployment Examples

This guide provides detailed examples for deploying Semblance Curation on various cloud platforms and environments.

## Deployment Requirements

### Local Deployment
- Minimum 32GB RAM
- 8+ CPU cores
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 500GB+ SSD storage
- Ubuntu 20.04+ or similar Linux distribution

## Cloud Deployment Examples

### AWS Deployment

```terraform
# aws/terraform/main.tf
provider "aws" {
  region = "us-west-2"
}

module "semblance_cluster" {
  source = "./modules/semblance"

  instance_type = "g4dn.2xlarge"  # 8 vCPUs, 32GB RAM, 1 NVIDIA T4 GPU
  volume_size   = 500             # GB
  
  # Networking
  vpc_id        = module.vpc.vpc_id
  subnet_ids    = module.vpc.private_subnets
  
  # Security
  ssh_key_name  = "semblance-key"
  allowed_ips   = ["your-ip-range"]
  
  # Monitoring
  enable_cloudwatch = true
  
  # Backup
  backup_retention_days = 30
}

# Optional: Add Elastic IP for stable access
resource "aws_eip" "semblance" {
  instance = module.semblance_cluster.instance_id
  vpc      = true
}
```

### Google Cloud Platform (GCP)

```terraform
# gcp/terraform/main.tf
provider "google" {
  project = var.project_id
  region  = "us-central1"
}

# VPC Configuration
resource "google_compute_network" "semblance_vpc" {
  name                    = "semblance-vpc"
  auto_create_subnetworks = false
}

# GPU-enabled Instance Template
resource "google_compute_instance_template" "semblance" {
  name        = "semblance-template"
  description = "Semblance Curation instance template with GPU"

  machine_type = "n1-standard-8"  # 8 vCPUs, 30 GB memory

  disk {
    source_image = "ubuntu-os-cloud/ubuntu-2004-lts"
    auto_delete  = true
    boot         = true
    disk_size_gb = 500
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  network_interface {
    network = google_compute_network.semblance_vpc.name
    access_config {}
  }

  metadata = {
    startup-script = file("${path.module}/startup-script.sh")
  }

  service_account {
    scopes = ["cloud-platform"]
  }
}

# Cloud Storage for Backups
resource "google_storage_bucket" "semblance_backup" {
  name     = "semblance-backup-${var.project_id}"
  location = "US"
  
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
}
```

### Microsoft Azure

```terraform
# azure/terraform/main.tf
provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "semblance" {
  name     = "semblance-resources"
  location = "eastus"
}

# Virtual Network
resource "azurerm_virtual_network" "semblance" {
  name                = "semblance-network"
  address_space       = ["10.0.0.0/16"]
  location           = azurerm_resource_group.semblance.location
  resource_group_name = azurerm_resource_group.semblance.name
}

# GPU-enabled Virtual Machine
resource "azurerm_virtual_machine" "semblance" {
  name                  = "semblance-vm"
  location             = azurerm_resource_group.semblance.location
  resource_group_name  = azurerm_resource_group.semblance.name
  network_interface_ids = [azurerm_network_interface.semblance.id]
  vm_size              = "Standard_NC6s_v3"  # GPU-enabled instance

  storage_image_reference {
    publisher = "Canonical"
    offer     = "UbuntuServer"
    sku       = "18.04-LTS"
    version   = "latest"
  }

  storage_os_disk {
    name              = "semblance-disk"
    caching           = "ReadWrite"
    create_option     = "FromImage"
    managed_disk_type = "Premium_LRS"
    disk_size_gb      = 500
  }

  os_profile {
    computer_name  = "semblance"
    admin_username = var.admin_username
  }

  os_profile_linux_config {
    disable_password_authentication = true
    ssh_keys {
      path     = "/home/${var.admin_username}/.ssh/authorized_keys"
      key_data = var.ssh_public_key
    }
  }
}

# Managed Disk for Data
resource "azurerm_managed_disk" "data" {
  name                 = "semblance-data"
  location            = azurerm_resource_group.semblance.location
  resource_group_name = azurerm_resource_group.semblance.name
  storage_account_type = "Premium_LRS"
  create_option       = "Empty"
  disk_size_gb        = 1024
}

# Storage Account for Backups
resource "azurerm_storage_account" "backup" {
  name                     = "semblancebackup"
  resource_group_name      = azurerm_resource_group.semblance.name
  location                 = azurerm_resource_group.semblance.location
  account_tier             = "Standard"
  account_replication_type = "GRS"

  blob_properties {
    delete_retention_policy {
      days = 30
    }
  }
}
```

## Post-Deployment Configuration

After deploying the infrastructure, you'll need to:

1. Configure DNS and SSL certificates
2. Set up monitoring and alerting
3. Configure backup schedules
4. Initialize the database
5. Set up authentication and access control

For detailed post-deployment steps, see:
- [Security Configuration](../configuration/security.md)
- [Monitoring Setup](../features/monitoring.md)
- [High Availability](../configuration/high-availability.md)

## Best Practices

1. **Security**
   - Use private subnets for compute resources
   - Implement proper IAM roles and permissions
   - Enable encryption at rest and in transit
   - Regular security updates and patches

2. **Monitoring**
   - Set up comprehensive logging
   - Configure alerting for critical metrics
   - Monitor resource utilization
   - Track application performance

3. **Backup and Recovery**
   - Regular automated backups
   - Test recovery procedures
   - Maintain backup retention policies
   - Cross-region backup replication

4. **Cost Optimization**
   - Use spot/preemptible instances where appropriate
   - Implement auto-scaling
   - Monitor and optimize resource usage
   - Regular cost analysis and optimization 
