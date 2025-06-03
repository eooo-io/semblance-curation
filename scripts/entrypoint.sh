#!/bin/bash
set -e

# Initialize services
init_services() {
    echo "Initializing services..."

    # Start PostgreSQL
    pg_ctl start -D /var/lib/postgresql/data \
        -o "-p ${POSTGRES_PORT:-5432}"

    # Start Redis
    redis-server --port ${REDIS_PORT:-6379} \
        --requirepass "${REDIS_PASSWORD}" \
        --daemonize yes

    # Start Elasticsearch
    ES_JAVA_OPTS="${ES_JAVA_OPTS:-"-Xms512m -Xmx512m"}" \
    elasticsearch -d \
        -Ehttp.port=${ELASTICSEARCH_PORT:-9200} \
        -Expack.security.enabled=true \
        -Expack.security.authc.api_key.enabled=true

    # Start MinIO
    MINIO_ROOT_USER=${MINIO_ROOT_USER:-admin} \
    MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minioadmin123} \
    minio server /data \
        --address ":${MINIO_PORT:-9000}" \
        --console-address ":${MINIO_CONSOLE_PORT:-9001}" &

    # Start Ollama
    OLLAMA_HOST=${OLLAMA_HOST:-localhost} \
    ollama serve --port ${OLLAMA_PORT:-11434} &

    # Start MLflow
    mlflow server \
        --port ${MLFLOW_PORT:-5000} \
        --host 0.0.0.0 \
        --backend-store-uri "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB}" \
        --default-artifact-root "${MLFLOW_ARTIFACTS_PATH:-/app/mlflow-artifacts}" &

    # Start Jupyter
    JUPYTER_TOKEN=${JUPYTER_TOKEN:-"your-secure-token"} \
    jupyter lab \
        --ip=0.0.0.0 \
        --port=${JUPYTER_PORT:-8888} \
        --no-browser \
        --allow-root &

    # Start monitoring stack
    prometheus \
        --config.file=/etc/prometheus/prometheus.yml \
        --web.listen-address=:${PROMETHEUS_PORT:-9090} &

    grafana-server \
        --config=/etc/grafana/grafana.ini \
        --homepath=/usr/share/grafana \
        --packaging=docker \
        cfg:default.paths.data=/var/lib/grafana \
        cfg:default.paths.logs=/var/log/grafana \
        cfg:default.paths.plugins=/var/lib/grafana/plugins \
        cfg:default.security.admin_user=${GRAFANA_ADMIN_USER:-admin} \
        cfg:default.security.admin_password=${GRAFANA_ADMIN_PASSWORD:-admin} &

    loki \
        -config.file=/etc/loki/loki.yml \
        -server.http-listen-port=${LOKI_PORT:-3100} &

    jaeger-all-in-one \
        --collector.http-port=${JAEGER_PORT:-16686} &

    # Start Label Studio
    label-studio start \
        --port ${LABEL_STUDIO_PORT:-8080} \
        --host 0.0.0.0 \
        --username ${LABEL_STUDIO_USERNAME:-admin@example.com} \
        --password ${LABEL_STUDIO_PASSWORD:-admin} \
        --database postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${LABEL_STUDIO_POSTGRES_DB:-label_studio} \
        --redis-location ${LABEL_STUDIO_REDIS_HOST:-localhost}:${LABEL_STUDIO_REDIS_PORT:-6379} \
        --no-browser &
}

# Wait for services to be ready
wait_for_services() {
    echo "Waiting for services to be ready..."

    # Add health checks for each service
    while ! pg_isready -h ${POSTGRES_HOST:-localhost} -p ${POSTGRES_PORT:-5432}; do sleep 1; done
    while ! redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} -a "${REDIS_PASSWORD}" ping; do sleep 1; done
    while ! curl -s ${ELASTICSEARCH_HOST:-localhost}:${ELASTICSEARCH_PORT:-9200}/_cluster/health > /dev/null; do sleep 1; done
    while ! curl -s ${MINIO_HOST:-localhost}:${MINIO_PORT:-9000}/minio/health/ready > /dev/null; do sleep 1; done
    while ! curl -s ${OLLAMA_HOST:-localhost}:${OLLAMA_PORT:-11434}/api/tags > /dev/null; do sleep 1; done
    while ! curl -s localhost:${LABEL_STUDIO_PORT:-8080}/health > /dev/null; do sleep 1; done
    echo "All services are ready!"
}

# Main execution
main() {
    init_services
    wait_for_services
    echo "All services are ready!"

    # Keep container running
    tail -f /dev/null
}

main "$@"
