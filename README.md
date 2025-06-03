# Semblance Curation

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg?style=for-the-badge)](https://docs.semblance-curation.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)

## Tech Stack

### Core Components
[![Label Studio](https://img.shields.io/badge/Label_Studio-222222.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJMMyA3djEwbDkgNSA5LTVWN2wtOS01em0wIDE4LjVMNSAxNy4yVjguOGw3LTMuOSA3IDMuOXY4LjRsLTcgMy45eiIvPjwvc3ZnPg==&logoColor=white)](https://labelstud.io/)
[![Argilla](https://img.shields.io/badge/Argilla-FF4B4B.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0OCIgaGVpZ2h0PSI0OCIgdmlld0JveD0iMCAwIDI0IDI0Ij48L3N2Zz4=&logoColor=white)](https://argilla.io/)
[![Weaviate](https://img.shields.io/badge/Weaviate-3b82f6.svg?style=for-the-badge&logo=weaviate&logoColor=white)](https://weaviate.io/)
[![Ollama](https://img.shields.io/badge/Ollama-black.svg?style=for-the-badge&logo=llama&logoColor=white)](https://ollama.ai/)

### Data Storage
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-DC382D.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-005571.svg?style=for-the-badge&logo=elasticsearch&logoColor=white)](https://www.elastic.co/)
[![MinIO](https://img.shields.io/badge/MinIO-C72E49.svg?style=for-the-badge&logo=minio&logoColor=white)](https://min.io/)

### Development & ML Tools
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Ray](https://img.shields.io/badge/Ray-028CF0.svg?style=for-the-badge&logo=ray&logoColor=white)](https://www.ray.io/)

### Monitoring & Observability
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C.svg?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-F46800.svg?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com/)
[![Loki](https://img.shields.io/badge/Loki-F5A800.svg?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com/oss/loki/)
[![Jaeger](https://img.shields.io/badge/Jaeger-66CFE3.svg?style=for-the-badge&logo=jaeger&logoColor=white)](https://www.jaegertracing.io/)

A comprehensive platform for building, maintaining, and curating machine learning datasets. It provides an end-to-end solution for data collection, annotation, preprocessing, and quality control, with built-in support for multi-modal data types.

## Quick Start

### Prerequisites
- Docker Engine 24.0.0+
- Docker Compose v2.20.0+
- NVIDIA GPU (recommended)
- 32GB RAM minimum
- 100GB storage minimum

### Installation

1. Clone the repository:
```bash
git clone https://github.com/eooo-io/semblance-curation.git
cd semblance-curation
```

2. Copy and configure environment variables:
```bash
cp env-example .env
# Edit .env with your preferred settings
```

3. Start the services:
```bash
# For production
docker compose up -d

# For development
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

4. Access the services:
- Label Studio: http://localhost:8080
- Jupyter Lab: http://localhost:8888
- MinIO Console: http://localhost:9001
- Grafana: http://localhost:3000

## Documentation

For detailed documentation, visit:
- [Installation Guide](docs/docs/getting-started/installation.md)
- [Architecture Overview](docs/docs/architecture/overview.md)
- [Configuration Guide](docs/docs/configuration/index.md)
- [Deployment Guide](docs/docs/deployment/index.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Features

- Multi-modal data handling (text, audio, images, video)
- Built-in annotation tools
- Local LLM inference capabilities
- Comprehensive monitoring and quality control
- Cloud-ready deployment options
- High availability configuration
- Extensive API support

## Overview

Semblance Curation is a comprehensive platform designed for organizations and researchers who need to build, maintain, and curate their own machine learning datasets. It provides an end-to-end solution for data collection, annotation, preprocessing, and quality control, with built-in support for multi-modal data types including text, audio, images, and video.

### Key Use Cases
- Build and maintain proprietary ML training datasets
- Curate and clean existing datasets
- Annotate data with custom labels and metadata
- Version and track data lineage
- Perform quality control on training data
- Deploy local LLM inference for data processing

### System Requirements

#### Minimum Requirements
- 32GB RAM
- 8+ CPU cores
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 500GB+ SSD storage
- Ubuntu 20.04+ or similar Linux distribution

For detailed deployment instructions and requirements, see our [deployment documentation](https://docs.semblance-curation.io/deployment/examples).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [docs.semblance-curation.io](https://docs.semblance-curation.io)
- Issues: [GitHub Issues](https://github.com/yourusername/semblance-curation/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/semblance-curation/discussions)

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
