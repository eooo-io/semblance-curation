# Semblance Curation

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg?style=for-the-badge)](https://docs.semblance-curation.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)

A comprehensive platform designed for organizations and researchers who need to build, maintain, and curate their own machine learning datasets. It provides an end-to-end solution for data collection, annotation, preprocessing, and quality control, with built-in support for multi-modal data types.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/semblance-curation.git
cd semblance-curation

# Copy and configure environment variables
cp env-example .env
# Edit .env with your settings

# Start the services
docker compose up -d
```

## Documentation

Comprehensive documentation is available at [docs.semblance-curation.io](https://docs.semblance-curation.io), including:

- Detailed installation instructions
- Configuration guides
- Deployment options (local and cloud)
- ML pipeline examples
- API reference
- Contributing guidelines

To run the documentation locally:

```bash
docker compose up docs
```

Then visit `http://localhost:8000`

## Features

- Multi-modal data handling (text, audio, images, video)
- Built-in annotation tools
- Local LLM inference capabilities
- Comprehensive monitoring and quality control
- Cloud-ready deployment options
- High availability configuration
- Extensive API support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [docs.semblance-curation.io](https://docs.semblance-curation.io)
- Issues: [GitHub Issues](https://github.com/yourusername/semblance-curation/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/semblance-curation/discussions)

## Overview

Semblance Curation is a comprehensive platform designed for organizations and researchers who need to build, maintain, and curate their own machine learning datasets. It provides an end-to-end solution for data collection, annotation, preprocessing, and quality control, with built-in support for multi-modal data types including text, audio, images, and video.

### Key Use Cases
- Build and maintain proprietary ML training datasets
- Curate and clean existing datasets
- Annotate data with custom labels and metadata
- Version and track data lineage
- Perform quality control on training data
- Deploy local LLM inference for data processing

### Deployment Options

#### Local Deployment Requirements
- Minimum 32GB RAM
- 8+ CPU cores
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 500GB+ SSD storage
- Ubuntu 20.04+ or similar Linux distribution

#### Cloud Deployment Example (AWS)

```yaml
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

### Cloud Deployment Examples

#### Google Cloud Platform (GCP)
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

#### Microsoft Azure
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
  resource_group_name = azurerm_resource_group.semblance.name
  location            = azurerm_resource_group.semblance.location
  address_space       = ["10.0.0.0/16"]
}

# GPU-enabled VM
resource "azurerm_linux_virtual_machine" "semblance" {
  name                = "semblance-machine"
  resource_group_name = azurerm_resource_group.semblance.name
  location            = azurerm_resource_group.semblance.location
  size                = "Standard_NC6s_v3"  # 6 vCPUs, 112 GB RAM, 1 NVIDIA Tesla V100
  admin_username      = "adminuser"

  network_interface_ids = [
    azurerm_network_interface.semblance.id,
  ]

  admin_ssh_key {
    username   = "adminuser"
    public_key = file("~/.ssh/id_rsa.pub")
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb        = 500
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "UbuntuServer"
    sku       = "18.04-LTS"
    version   = "latest"
  }
}

# Managed Kubernetes for Scaling
resource "azurerm_kubernetes_cluster" "semblance" {
  name                = "semblance-aks"
  location            = azurerm_resource_group.semblance.location
  resource_group_name = azurerm_resource_group.semblance.name
  dns_prefix          = "semblance"

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_DS2_v2"
  }

  identity {
    type = "SystemAssigned"
  }
}
```

### Additional ML Pipeline Examples

#### Text Classification Pipeline
```python
# pipelines/text_classification.py
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class TextClassificationPipeline:
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        batch_size: int = 16,
        num_epochs: int = 3,
    ):
        dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(num_epochs):
            self.model.train()
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        self.model.eval()
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return [
            {f"class_{i}": p for i, p in enumerate(prob)}
            for prob in probs.cpu().numpy()
        ]
```

#### Image Segmentation Pipeline
```python
# pipelines/image_segmentation.py
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import torchvision.transforms as T

class ImageSegmentationPipeline:
    def __init__(self, num_classes: int):
        self.model = deeplabv3_resnet50(num_classes=num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def process_image(self, image: Image.Image) -> torch.Tensor:
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        return output.argmax(0).cpu()
```

### High Availability Configuration

#### Kubernetes HA Setup
```yaml
# config/kubernetes/ha-config.yml
apiVersion: v1
kind: Service
metadata:
  name: semblance-lb
spec:
  type: LoadBalancer
  ports:
  - port: 80
  selector:
    app: semblance

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: semblance
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semblance
  serviceName: semblance
  template:
    metadata:
      labels:
        app: semblance
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - semblance
            topologyKey: "kubernetes.io/hostname"
      containers:
      - name: semblance
        image: semblance:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
```

### Enhanced Monitoring Dashboards

#### Data Quality Dashboard
```json
{
  "dashboard": {
    "title": "Data Quality Overview",
    "panels": [
      {
        "title": "Label Distribution",
        "type": "piechart",
        "metrics": ["label_counts"],
        "options": {
          "legend": {"show": true},
          "tooltip": {"show": true}
        }
      },
      {
        "title": "Feature Correlation Matrix",
        "type": "heatmap",
        "metrics": ["feature_correlations"],
        "options": {
          "colorScale": "RdYlBu"
        }
      },
      {
        "title": "Data Completeness",
        "type": "gauge",
        "metrics": ["completeness_score"],
        "thresholds": [
          {"value": 0, "color": "red"},
          {"value": 0.7, "color": "yellow"},
          {"value": 0.9, "color": "green"}
        ]
      }
    ]
  }
}
```

#### ML Pipeline Performance Dashboard
```json
{
  "dashboard": {
    "title": "ML Pipeline Performance",
    "panels": [
      {
        "title": "Training Progress",
        "type": "graph",
        "metrics": [
          "training_loss",
          "validation_loss",
          "learning_rate"
        ],
        "yaxes": [
          {"format": "short"},
          {"format": "scientific"}
        ]
      },
      {
        "title": "Model Metrics",
        "type": "stat",
        "metrics": [
          "accuracy",
          "f1_score",
          "precision",
          "recall"
        ]
      },
      {
        "title": "Resource Usage",
        "type": "graph",
        "metrics": [
          "gpu_memory_used",
          "cpu_usage",
          "memory_usage"
        ]
      }
    ]
  }
}
```

### Data Quality Monitoring Rules

```yaml
# config/monitoring/data-quality-rules.yml
rules:
  - name: missing_values
    description: "Check for missing values in critical fields"
    condition: "missing_value_rate > 0.1"
    severity: "high"
    actions:
      - notify_team
      - log_incident
      - pause_pipeline

  - name: label_imbalance
    description: "Monitor class distribution in training data"
    condition: "max_to_min_class_ratio > 10"
    severity: "medium"
    actions:
      - notify_team
      - trigger_resampling

  - name: feature_drift
    description: "Monitor feature distribution changes"
    condition: "ks_test_pvalue < 0.05"
    severity: "high"
    actions:
      - notify_team
      - trigger_retraining

  - name: data_freshness
    description: "Monitor data freshness"
    condition: "max_data_age > 7d"
    severity: "medium"
    actions:
      - notify_team
      - trigger_data_collection

metrics:
  collection_interval: 1h
  aggregation_window: 24h
  storage_retention: 30d

alerting:
  channels:
    - slack: "#data-quality-alerts"
    - email: "ml-team@company.com"
  throttling:
    min_interval: 1h
    max_alerts_per_day: 10
```

## Features

### Multi-modal Data Handling
- Process and store text, voice, and video data
- Scalable storage solutions for various data types
- Efficient data annotation and labeling workflows

### Machine Learning Operations
- Local LLM inference using Ollama
- GPU-accelerated processing
- Vector-based similarity search with Weaviate
- Comprehensive data annotation tools with Argilla

### Advanced Data Management
- Powerful text search with Elasticsearch
- Structured data storage in PostgreSQL
- High-performance caching with Redis
- Scalable object storage using MinIO
- Vector database for semantic search
- Data versioning and lineage tracking
- Automated data quality checks
- Real-time data pipeline monitoring

### Development Environment
- Interactive Jupyter Notebook environment
- Comprehensive data science toolkit
- Containerized architecture for consistency
- Full GPU support for ML workloads
- Distributed training support
- Experiment tracking and model versioning
- Automated ML pipeline orchestration

## Getting Started

### Prerequisites
- Docker and Docker Compose
- NVIDIA drivers (for GPU support)
- Git
- Minimum 16GB RAM recommended
- NVIDIA GPU with CUDA support (optional but recommended)
- 50GB+ available storage

### Optional Components

#### MLflow Integration
```bash
# Add to docker-compose.yml
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  ports:
    - "5000:5000"
  environment:
    - MLFLOW_TRACKING_URI=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
  depends_on:
    - postgres
```

#### Ray Cluster for Distributed Training
```bash
# Add to docker-compose.yml
ray-head:
  image: rayproject/ray:latest
  ports:
    - "8265:8265"  # Ray dashboard
    - "10001:10001"  # Ray client server
  command: ray start --head --dashboard-host=0.0.0.0
```

#### Weights & Biases Integration
```bash
# Add to your .env file
WANDB_API_KEY=your_key_here
```

### Advanced Configuration

#### GPU Memory Optimization
```bash
# Add to your .env file
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_VISIBLE_DEVICES=0,1  # Specify GPUs to use
```

#### Security Hardening
```bash
# Add to docker-compose.yml for each service
security_opt:
  - no-new-privileges:true
ulimits:
  nproc: 65535
  nofile:
    soft: 65535
    hard: 65535
```

#### Performance Tuning
```bash
# Add to docker-compose.yml for database services
command: >
  -c max_connections=200
  -c shared_buffers=2GB
  -c effective_cache_size=6GB
  -c maintenance_work_mem=512MB
  -c checkpoint_completion_target=0.9
  -c wal_buffers=16MB
  -c default_statistics_target=100
  -c random_page_cost=1.1
  -c effective_io_concurrency=200
  -c work_mem=6553kB
  -c min_wal_size=1GB
  -c max_wal_size=4GB
```

#### Prometheus Monitoring Configuration
Create `prometheus/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'semblance-services'
    static_configs:
      - targets: ['localhost:9090']
  - job_name: 'docker'
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
    relabel_configs:
      - source_labels: [__meta_docker_container_name]
        regex: '/(.*)'
        target_label: container_name

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

#### Grafana Dashboards
Create `grafana/provisioning/dashboards/ml-metrics.json`:
```json
{
  "dashboard": {
    "title": "ML Pipeline Metrics",
    "panels": [
      {
        "title": "Model Training Progress",
        "type": "graph",
        "metrics": ["training_loss", "validation_loss"]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "metrics": ["gpu_memory_used", "gpu_utilization"]
      },
      {
        "title": "Data Pipeline Throughput",
        "type": "stat",
        "metrics": ["records_processed_per_second"]
      }
    ]
  }
}
```

#### Enhanced Security Configuration
Create `security/security-policies.yml`:
```yaml
# Docker security options
security_opt:
  - seccomp:security/seccomp-profile.json
  - apparmor:security/apparmor-profile
  - no-new-privileges:true

# Network policies
networks:
  curation-net:
    driver: overlay
    attachable: true
    driver_opts:
      encrypted: "true"
    ipam:
      driver: default
      config:
        - subnet: 172.16.0.0/24

# Service-specific security
services:
  postgres:
    security_opt:
      - no-new-privileges:true
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password
    configs:
      - source: postgres_config
        target: /etc/postgresql/postgresql.conf

secrets:
  db_password:
    file: ./secrets/db_password.txt
  ssl_cert:
    file: ./secrets/ssl_cert.pem

configs:
  postgres_config:
    file: ./configs/postgresql.conf
```

### Monitoring Stack

Add these services to enable comprehensive monitoring:

```bash
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus:latest
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"

loki:
  image: grafana/loki:latest
  ports:
    - "3100:3100"

jaeger:
  image: jaegertracing/all-in-one:latest
  ports:
    - "16686:16686"
```

### Development Tools

#### VS Code Development Container
Create a `.devcontainer/devcontainer.json`:
```json
{
  "name": "Semblance Dev Environment",
  "dockerComposeFile": ["../docker-compose.yml"],
  "service": "jupyter",
  "workspaceFolder": "/home/jovyan/work",
  "extensions": [
    "ms-python.python",
    "ms-toolsai.jupyter",
    "ms-azuretools.vscode-docker"
  ]
}
```

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/semblance-curation.git
cd semblance-curation
\`\`\`

2. Copy the environment file:
\`\`\`bash
cp env-example .env
\`\`\`

3. Configure your environment variables in \`.env\`

4. Start the services:
\`\`\`bash
docker compose up -d
\`\`\`

### Default Ports
- Argilla: 6900
- Elasticsearch: 9200
- Weaviate: 8080
- PostgreSQL: 5432
- Redis: 6379
- MinIO: 9000 (API) / 9001 (Console)
- Ollama: 11434
- Jupyter: 8888

## Architecture

The platform consists of several containerized services:

- **Argilla**: Data annotation and curation
- **Elasticsearch**: Text search and analytics
- **Weaviate**: Vector database for ML features
- **PostgreSQL**: Structured data storage
- **Redis**: Caching and real-time features
- **MinIO**: Object storage for large files
- **Ollama**: Local LLM inference
- **Jupyter**: Interactive development environment

## Security

- All services run in isolated containers
- Configurable authentication for each service
- Secure data storage with volume persistence
- Environment-based configuration

## Documentation

Each component has its own documentation:

- [Argilla Documentation](https://docs.argilla.io/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Ollama Documentation](https://ollama.ai/docs)
- [MinIO Documentation](https://min.io/docs/minio/container/index.html)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Argilla](https://argilla.io/) for the data annotation platform
- [Weaviate](https://weaviate.io/) for the vector database
- [Ollama](https://ollama.ai/) for local LLM capabilities
- All other open-source projects that made this possible

## Best Practices

### Data Pipeline Optimization
- Use data streaming for large datasets
- Implement incremental processing
- Configure appropriate batch sizes
- Use parallel processing where possible
- Implement caching strategies

### Model Development
- Use experiment tracking (MLflow/W&B)
- Implement model versioning
- Set up automated testing
- Use distributed training for large models
- Implement model monitoring

### Production Deployment
- Use rolling updates
- Implement health checks
- Set up automated backups
- Configure auto-scaling
- Monitor resource usage

### ML Pipeline Examples

#### Basic Training Pipeline
Create `pipelines/training_pipeline.py`:
```python
import mlflow
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback

def train_model(config):
    mlflow.start_run(nested=True)
    
    # Data loading with versioning
    dataset = load_dataset(
        path=config["data_path"],
        version=config["data_version"]
    )
    
    # Model configuration
    model = create_model(
        architecture=config["model_arch"],
        params=config["model_params"]
    )
    
    # Training loop with metrics
    for epoch in range(config["num_epochs"]):
        metrics = train_epoch(model, dataset)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Report to Ray Tune
        tune.report(
            loss=metrics["loss"],
            accuracy=metrics["accuracy"]
        )

# Configure distributed training
training_config = {
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "model_arch": "transformer",
    "data_version": "v1.0"
}

# Launch training
analysis = tune.run(
    train_model,
    config=training_config,
    num_samples=10,
    callbacks=[MLflowLoggerCallback()]
)
```

### Backup and Disaster Recovery

#### Automated Backup Configuration
Create `backup/backup-config.yml`:
```yaml
backup_schedule:
  postgres:
    frequency: "0 2 * * *"  # Daily at 2 AM
    retention: "30d"
    command: |
      pg_dump -Fc -d ${POSTGRES_DB} -U ${POSTGRES_USER} > \
      /backups/postgres/$(date +%Y%m%d).dump

  elasticsearch:
    frequency: "0 3 * * *"  # Daily at 3 AM
    retention: "30d"
    command: |
      curator_cli snapshot --name $(date +%Y%m%d) \
      --repository es_backup_repo

  minio:
    frequency: "0 4 * * *"  # Daily at 4 AM
    retention: "30d"
    command: |
      mc mirror /data /backups/minio/$(date +%Y%m%d)
```

#### Disaster Recovery Procedures
Create `docs/disaster_recovery.md`:
```markdown
## Disaster Recovery Procedures

### 1. Database Recovery
```bash
# Restore PostgreSQL
pg_restore -d ${POSTGRES_DB} -U ${POSTGRES_USER} /backups/postgres/latest.dump

# Restore Elasticsearch
curl -X POST "localhost:9200/_snapshot/es_backup_repo/latest/_restore"

# Restore MinIO
mc mirror /backups/minio/latest /data
```

### 2. Service Recovery
```bash
# Stop affected services
docker compose stop affected_service

# Remove corrupted volumes
docker volume rm affected_volume

# Restore from backup
./scripts/restore.sh --service affected_service --backup latest

# Restart services
docker compose up -d
```

### 3. Verification Steps
```bash
# Verify database integrity
./scripts/verify_db.sh

# Check data consistency
./scripts/verify_data.sh

# Validate service health
./scripts/health_check.sh
```

### Performance Monitoring

#### Resource Monitoring
Create `monitoring/resource-alerts.yml`:
```yaml
alerts:
  high_memory:
    threshold: 85%
    duration: 5m
    action: "scale_up_memory"

  high_cpu:
    threshold: 90%
    duration: 5m
    action: "scale_up_cpu"

  disk_space:
    threshold: 80%
    duration: 10m
    action: "notify_admin"

metrics:
  collection_interval: 30s
  retention_period: 30d
  exporters:
    - prometheus
    - grafana
```
