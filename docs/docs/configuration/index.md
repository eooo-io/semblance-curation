# Configuration Guide

This guide covers the detailed configuration options for Semblance Curation.

## Environment Variables

The system is configured primarily through environment variables. Copy the example file and customize it for your needs:

```bash
cp env-example .env
```

### Core Settings

```bash
# Application settings
APP_ENV=development
APP_DEBUG=true
APP_SECRET_KEY=your-secret-key-here

# Label Studio settings
LABEL_STUDIO_USERNAME=admin@example.com
LABEL_STUDIO_PASSWORD=secure-password
LABEL_STUDIO_BASE_DATA_DIR=/label-studio/data
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
```

### Database Settings

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure-password
POSTGRES_DB=curation_db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=secure-redis-password
```

### Storage Settings

```bash
# MinIO
MINIO_HOST=localhost
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=secure-password
MINIO_REGION=us-east-1

# Elasticsearch
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ES_JAVA_OPTS=-Xms512m -Xmx512m
ELASTIC_PASSWORD=secure-elastic-password
```

### ML Tools Settings

```bash
# Weaviate
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_AUTHENTICATION_APIKEY=your-api-key
WEAVIATE_QUERY_DEFAULTS_LIMIT=20

# Ollama
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# MLflow
MLFLOW_PORT=5000
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
MLFLOW_ARTIFACTS_PATH=/app/mlflow-artifacts
```

### Monitoring Settings

```bash
# Prometheus
PROMETHEUS_PORT=9090

# Grafana
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure-password

# Loki
LOKI_PORT=3100

# Jaeger
JAEGER_PORT=16686
```

## Service Configuration

### Label Studio

Label Studio is configured with the following defaults:

- Multi-user support enabled
- Local file serving enabled
- PostgreSQL backend
- Redis caching
- MinIO storage for large files

### Database Configuration

PostgreSQL is configured for:

- High availability
- Automatic backups
- Connection pooling
- SSL connections

### Storage Configuration

MinIO is configured for:

- S3-compatible API
- Multi-user access
- Bucket versioning
- Object lifecycle policies

### ML Tools Configuration

Weaviate is configured for:

- Vector similarity search
- Schema validation
- API key authentication
- Query rate limiting

### Monitoring Configuration

Prometheus is configured with:

- Custom metrics collection
- Alert rules
- Recording rules
- Long-term storage

## Development Configuration

For development environments:

```yaml
# docker-compose.override.yml
services:
  app:
    build:
      target: builder
    volumes:
      - .:/app
      - ~/.cache/pip:/root/.cache/pip
    environment:
      - APP_ENV=development
      - APP_DEBUG=true
```

## Production Configuration

For production environments:

```yaml
# docker-compose.yml
services:
  app:
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Security Configuration

Security best practices include:

1. **Authentication**:
   - Strong passwords
   - API key rotation
   - OAuth integration

2. **Authorization**:
   - Role-based access
   - Resource isolation
   - Least privilege principle

3. **Network Security**:
   - TLS encryption
   - Network isolation
   - Firewall rules

## Next Steps

- [Security Guide](security.md)
- [High Availability Configuration](high-availability.md)
- [Backup Configuration](backup.md)
- [Monitoring Configuration](../features/monitoring/index.md) 
