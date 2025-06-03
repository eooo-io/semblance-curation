# Installation Guide

## Prerequisites

Before installing Semblance Curation, ensure you have the following prerequisites:

- Python 3.8 or higher
- Docker and Docker Compose
- Git
- A Unix-like operating system (Linux, macOS) or Windows with WSL2

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
pip install semblance-curation
```

### Method 2: Using Docker

```bash
docker pull semblance/curation:latest
```

### Method 3: From Source

```bash
git clone https://github.com/yourusername/semblance-curation.git
cd semblance-curation
pip install -e .
```

## Configuration

After installation, you'll need to set up your configuration:

1. Copy the example environment file:
   ```bash
   cp env-example .env
   ```

2. Edit the `.env` file with your settings:
   ```bash
   SEMBLANCE_API_KEY=your_api_key
   SEMBLANCE_HOST=localhost
   SEMBLANCE_PORT=8000
   ```

## Verification

To verify your installation:

```bash
semblance --version
semblance verify
```

## Next Steps

- Continue to the [Quick Start Guide](quick-start.md)
- Read about [Configuration](configuration.md)
- Check out our [Examples](../examples/ml-pipelines.md) 
