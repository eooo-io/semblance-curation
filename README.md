# Semblance Data Curation

## Technologies

The Semblance Curation stack uses the following technologies for data curation and annotation:

[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Argilla](https://img.shields.io/badge/Argilla-FF6F61?style=for-the-badge&logo=argilla&logoColor=white)](https://argilla.io/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-005571?style=for-the-badge&logo=elasticsearch&logoColor=white)](https://www.elastic.co/elasticsearch/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

## Overview

Semblance Curation is an AI-driven data curation platform focused on cleaning, labeling, and persisting datasets for machine learning workflows. This repository provides a Docker Compose setup to deploy a streamlined stack for data annotation and metadata management, using:

- **Argilla**: A web-based platform for collaborative data annotation and labeling.
- **Elasticsearch**: A search engine for indexing and querying annotated data.
- **PostgreSQL**: A relational database for storing structured metadata (e.g., user roles, annotation tasks).

These services run on a custom bridge network (`curation-net`) with persistent volumes for data storage, optimized for reliability and security in data curation tasks.

## Prerequisites

- **Docker and Docker Compose**: Install Docker and Docker Compose on your system (e.g., Linux on an RTX 3090 or RunPod workspace).
- **Environment Variables**: Create a `.env` file in the repository root with the following variables:
  ```env
  # Argilla
  ARGILLA_PORT=6900
  ARGILLA_AUTH_SECRET_KEY=your-secret-key
  # Elasticsearch
  ELASTICSEARCH_PORT=9200
  ELASTIC_PASSWORD=secureelasticpass
  ES_JAVA_OPTS=-Xms512m -Xmx512m
  # PostgreSQL
  POSTGRES_PORT=5432
  POSTGRES_USER=admin
  POSTGRES_PASSWORD=securepassword
  POSTGRES_DB=semblance
  ```

## Services

### Argilla
- **Purpose**: Enables collaborative annotation of text datasets for NLP tasks (e.g., sentiment analysis, entity recognition), supporting data cleaning and labeling.
- **Image**: `argilla/argilla-server:latest`
- **Port**: `127.0.0.1:${ARGILLA_PORT}:6900` (default: 6900, localhost-only for security)
- **Dependencies**: Requires Elasticsearch for indexing and search.
- **Environment Variables**:
  - `ARGILLA_ELASTICSEARCH`: Points to Elasticsearch (e.g., `elasticsearch:9200`).
  - `ARGILLA_AUTH_SECRET_KEY`: Secret key for authentication.
  - `ARGILLA_AUTH_TYPE`: Set to `standard` (configure OAuth/LDAP for production).
- **Volumes**: `argilla-data` for persistent annotations.
- **Health Check**: Verifies UI availability via `curl`.
- **Access**: Open `http://localhost:${ARGILLA_PORT}` in a browser to use the Argilla UI.

### Elasticsearch
- **Purpose**: Indexes and searches annotated data, serving as Argilla’s backend for efficient querying and storage.
- **Image**: `docker.elastic.co/elasticsearch/elasticsearch:8.10.4`
- **Port**: `127.0.0.1:${ELASTICSEARCH_PORT}:9200` (default: 9200, localhost-only)
- **Environment Variables**:
  - `discovery.type=single-node`: Runs in single-node mode.
  - `xpack.security.enabled=true`: Enables authentication.
  - `ELASTIC_PASSWORD`: Password for the `elastic` user.
  - `ES_JAVA_OPTS`: Configures JVM memory (e.g., `-Xms512m -Xmx512m`).
- **Volumes**: `es-data` for persistent indices.
- **Health Check**: Verifies cluster health via `curl` with authentication.
- **Access**: Query via REST API at `http://localhost:${ELASTICSEARCH_PORT}` with credentials.

### PostgreSQL
- **Purpose**: Stores structured metadata (e.g., user roles, annotation tasks) to support curation workflows and ensure persistence.
- **Image**: `postgres:16`
- **Port**: `127.0.0.1:${POSTGRES_PORT}:5432` (default: 5432, localhost-only)
- **Environment Variables**:
  - `POSTGRES_USER`: Database user (e.g., `admin`).
  - `POSTGRES_PASSWORD`: Database password (e.g., `securepassword`).
  - `POSTGRES_DB`: Database name (e.g., `semblance`).
- **Volumes**: `postgres-data` for persistent storage.
- **Health Check**: Verifies database readiness via `pg_isready`.
- **Access**: Connect using a PostgreSQL client (e.g., `psql`).

## Data Curation Workflow

1. **Import Data**:
   - Prepare a dataset (e.g., CSV or JSON with text records).
   - Use a Python script to import data into Argilla via its API (example: `requests.post` to `http://localhost:${ARGILLA_PORT}/api/datasets`).
   - Configure dataset settings (e.g., labels, questions) in the Argilla UI.

2. **Annotate Data**:
   - Access Argilla at `http://localhost:${ARGILLA_PORT}`.
   - Create annotation tasks (e.g., text classification, token classification).
   - Assign tasks to users and apply validation rules (e.g., required labels) for quality control.

3. **Export Annotations**:
   - Export annotated data via Argilla’s API (e.g., `GET /api/datasets/{name}/records`).
   - Save results as JSONL or CSV for ML training.
   - Store metadata (e.g., task assignments) in PostgreSQL for tracking.

4. **Persist and Backup**:
   - Annotations are stored in Elasticsearch (`es-data`) and metadata in PostgreSQL (`postgres-data`).
   - Schedule Elasticsearch snapshots to S3 or local storage.
   - Use `pg_dump` for PostgreSQL backups (e.g., `pg_dump -U admin semblance > backup.sql`).

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/eooo-io/semblance-curation.git
   cd semblance-curation
   ```

2. **Create the `.env` File**:
   Copy the example environment variables above into a `.env` file, replacing placeholders with secure values:
   ```bash
   nano .env
   ```

3. **Start the Services**:
   Run the Docker Compose stack:
   ```bash
   docker-compose up -d
   ```

4. **Verify Services**:
   Check that all services are running:
   ```bash
   docker-compose ps
   ```

5. **Access Services**:
   Use the URLs or clients listed in each service’s “Access” section to interact with the stack.

## Usage

- **Data Curation**: Use Argilla to clean and annotate datasets, with Elasticsearch indexing results for fast querying.
- **Metadata Management**: Store user roles, task assignments, or curation metadata in PostgreSQL.
- **Persistence**: Ensure data durability with persistent volumes and regular backups.
- **Quality Control**: Configure Argilla with validation rules to maintain annotation quality.

## Notes

- **Security**: Replace default credentials in `.env` with secure values. Enable TLS and OAuth/LDAP for production.
- **Persistence**: Use Elasticsearch snapshots and `pg_dump` for backups (see “Data Curation Workflow”).
- **Scaling**: For large datasets, adjust `ES_JAVA_OPTS` (e.g., `-Xms4g -Xmx4g`) and consider multi-node Elasticsearch.
- **Future Enhancements**:
  - Add Prometheus/Grafana for monitoring annotation performance.
  - Use Nginx/Traefik as a reverse proxy for secure access.
  - Develop scripts for data import/export and quality metrics (e.g., inter-annotator agreement).

## Contributing

Contributions are welcome! Please submit issues or pull requests to the [GitHub repository](https://github.com/eooo-io/semblance-curation).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
