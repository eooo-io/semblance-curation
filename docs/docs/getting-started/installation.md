# Installation Guide

This guide will help you set up Semblance Curation on your system.

## Prerequisites

- Docker Engine 24.0.0+
- Docker Compose v2.20.0+
- NVIDIA GPU (recommended)
- NVIDIA Container Toolkit (for GPU support)
- 32GB RAM minimum
- 100GB storage minimum

## Quick Start

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

For production:
```bash
docker compose up -d
```

For development:
```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

4. Access the services:
- Label Studio: http://localhost:8080
- Jupyter Lab: http://localhost:8888
- MinIO Console: http://localhost:9001
- Grafana: http://localhost:3000
- Documentation: http://localhost:8000 (development only)

## Configuration

### Environment Variables

Key environment variables you might want to customize:

```bash
# Label Studio settings
LABEL_STUDIO_USERNAME=admin@example.com
LABEL_STUDIO_PASSWORD=secure-password

# PostgreSQL settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure-password
POSTGRES_DB=curation_db

# MinIO settings
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=secure-password

# See env-example for all available options
```

### GPU Support

The stack is configured to use NVIDIA GPUs by default. Ensure you have the NVIDIA Container Toolkit installed:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Development Setup

For development, additional features are enabled:

- Hot reload for code changes
- Debug logging
- Development ports exposed
- Documentation server
- Source code mounting
- PIP cache mounting

To use the development configuration:

```bash
# Start with development settings
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Rebuild a specific service
docker compose -f docker-compose.yml -f docker-compose.override.yml build app

# View logs
docker compose -f docker-compose.yml -f docker-compose.override.yml logs -f
```

## Troubleshooting

### Common Issues

1. **Services not starting**:
   - Check if ports are already in use
   - Ensure sufficient system resources
   - Check logs: `docker compose logs -f [service_name]`

2. **GPU not detected**:
   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Check NVIDIA Container Toolkit installation
   - Ensure Docker is configured for GPU support

3. **Permission issues**:
   - Check volume permissions
   - Ensure proper user/group mappings
   - Verify environment variable permissions

### Health Checks

The stack includes comprehensive health checks. Monitor them with:

```bash
# Check service health
docker compose ps

# View health check logs
docker compose logs -f
```

## Next Steps

- [Configuration Guide](../configuration/index.md)
- [Deployment Guide](../deployment/index.md)
- [Feature Documentation](../features/index.md) 
