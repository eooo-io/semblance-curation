# syntax=docker/dockerfile:1.4

# Build stage
FROM python:3.9-slim-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.docs.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.docs.txt \
    && pip install --no-cache-dir label-studio psycopg2-binary

# Runtime stage
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3-pip \
    postgresql-client \
    redis-tools \
    curl \
    nginx \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Ollama (supports both amd64 and arm64)
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create necessary directories
WORKDIR /app
RUN mkdir -p \
    /label-studio/data \
    /app/mlflow-artifacts \
    /var/lib/postgresql/data \
    /var/lib/grafana \
    /var/log/grafana \
    /etc/prometheus \
    /etc/grafana \
    /etc/loki

# Copy application files
COPY . .

# Expose ports (using environment variables with defaults)
EXPOSE \
    ${LABEL_STUDIO_PORT:-8080} \
    ${ARGILLA_PORT:-6900} \
    ${ELASTICSEARCH_PORT:-9200} \
    ${WEAVIATE_PORT:-8080} \
    ${POSTGRES_PORT:-5432} \
    ${REDIS_PORT:-6379} \
    ${MINIO_PORT:-9000} \
    ${MINIO_CONSOLE_PORT:-9001} \
    ${OLLAMA_PORT:-11434} \
    ${JUPYTER_PORT:-8888} \
    ${MLFLOW_PORT:-5000} \
    ${PROMETHEUS_PORT:-9090} \
    ${GRAFANA_PORT:-3000} \
    ${LOKI_PORT:-3100} \
    ${JAEGER_PORT:-16686}

# Set default environment variables
ENV \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LABEL_STUDIO_BASE_DATA_DIR=/label-studio/data \
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data \
    MLFLOW_ARTIFACTS_PATH=/app/mlflow-artifacts

# Default command
CMD ["./scripts/entrypoint.sh"]
