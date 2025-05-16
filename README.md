# Semblance Data Curation Workflow

> Data ingestion, document parsing, and indexing backend for the Semblance AI stack. Supports multi-format ingestion and both full-text and semantic (vector) search. Built for high-fidelity document curation pipelines.

[![Docker Compose](https://img.shields.io/badge/docker--compose-🛠️%20orchestrated-blue)](https://docs.docker.com/compose/)
[![PostgreSQL](https://img.shields.io/badge/postgres-💾%20metadata-informational)](https://www.postgresql.org/)
[![Elasticsearch](https://img.shields.io/badge/elasticsearch-🔍%20full--text--search-orange)](https://www.elastic.co/)
[![Weaviate](https://img.shields.io/badge/weaviate-🧠%20semantic--indexing-blueviolet)](https://weaviate.io/)
[![MinIO](https://img.shields.io/badge/minio-📦%20object--store-yellow)](https://min.io/)
[![License](https://img.shields.io/github/license/eooo-io/semblance-curation)](./LICENSE)

---

## Overview

This service is responsible for document ingestion and curation, transforming raw uploads into structured, searchable, and semantically indexed knowledge. It’s part of the larger [Semblance AI project](https://github.com/eooo-io).

Documents are:
- Uploaded to `MinIO`
- Parsed via a dedicated service
- Metadata tracked in `PostgreSQL`
- Indexed into both `Elasticsearch` (for keyword queries) and `Weaviate` (for vector-based retrieval)

---

## Services Overview

| Service             | Purpose |
|----------------------|---------|
| **MinIO**            | S3-compatible storage for raw document uploads |
| **document-parser**  | Converts PDFs, EPUB, HTML, etc. into plaintext |
| **PostgreSQL**       | Metadata and document registry |
| **pgvector**         | Vector similarity search plugin for Postgres |
| **Elasticsearch**    | Full-text keyword-based search |
| **Weaviate**         | Vector DB with semantic search support |
| **weaviate-setup**   | Bootstrap schema and configuration into Weaviate |
| **curation-engine**  | Handles orchestration and tracking (in development) |

---
