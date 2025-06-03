# Data Management

Semblance Curation provides robust data management capabilities to help you organize, track, and version your data effectively.

## Data Sources

### Supported Formats

- CSV files
- JSON data
- Parquet files
- SQL databases
- NoSQL databases
- Streaming data sources

### Connecting to Data Sources

```python
from semblance.data import DataSource

# CSV Source
csv_source = DataSource.from_csv("data/input.csv")

# Database Source
db_source = DataSource.from_sql(
    "postgresql://user:pass@localhost:5432/db",
    query="SELECT * FROM table"
)
```

## Data Versioning

Semblance Curation integrates with DVC for data versioning:

```bash
# Initialize data versioning
semblance data init

# Add data to version control
semblance data add data/raw/

# Create a data version
semblance data tag v1.0

# Switch between versions
semblance data checkout v1.0
```

## Data Quality

### Schema Validation

```python
from semblance.quality import SchemaValidator

validator = SchemaValidator({
    "required_fields": ["id", "timestamp", "value"],
    "types": {
        "id": "string",
        "timestamp": "datetime",
        "value": "float"
    }
})

validation_report = validator.validate(data)
```

### Data Profiling

```python
from semblance.quality import DataProfiler

profiler = DataProfiler()
profile = profiler.profile(data)

# Generate HTML report
profile.to_html("profile_report.html")
```

## Data Lineage

Track the complete history of your data:

```python
from semblance.lineage import LineageTracker

tracker = LineageTracker()
tracker.track(data_source="raw.csv",
             transformations=["clean", "normalize"],
             output="processed.csv")

# View lineage graph
tracker.visualize()
```

## Best Practices

1. **Data Organization**
   - Use consistent naming conventions
   - Maintain clear directory structure
   - Document data sources and schemas

2. **Version Control**
   - Version both code and data
   - Use meaningful version tags
   - Document version changes

3. **Quality Assurance**
   - Implement automated validation
   - Regular data profiling
   - Monitor data drift

4. **Security**
   - Implement access controls
   - Encrypt sensitive data
   - Regular security audits

## Configuration Examples

### Data Source Configuration

```yaml
data_sources:
  raw_data:
    type: csv
    path: data/raw/
    pattern: "*.csv"
    schema: schemas/raw.json
  
  processed_data:
    type: parquet
    path: data/processed/
    compression: snappy
```

### Quality Check Configuration

```yaml
quality_checks:
  - name: completeness
    type: null_check
    threshold: 0.95
  
  - name: freshness
    type: timestamp_check
    max_age: 24h
  
  - name: accuracy
    type: range_check
    min: 0
    max: 100
```

## Next Steps

- Learn about [ML Pipelines](ml-pipelines.md)
- Explore [Monitoring](monitoring.md)
- Check out [Examples](../examples/data-quality.md) 
