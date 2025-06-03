# Quick Start Guide

This guide will help you get started with Semblance Curation in just a few minutes.

## Basic Usage

### 1. Initialize a Project

```bash
semblance init my-project
cd my-project
```

### 2. Configure Your Project

Create a basic configuration file:

```yaml
version: 1.0
project:
  name: my-project
  description: My first Semblance Curation project

data:
  source:
    type: csv
    path: data/input.csv
  
pipeline:
  steps:
    - name: validate
      type: schema_validation
    - name: clean
      type: data_cleaning
    - name: transform
      type: feature_engineering
```

### 3. Run Your First Pipeline

```bash
semblance run pipeline
```

## Example Use Cases

### Data Validation

```python
from semblance.validation import Schema

schema = Schema({
    "id": "integer",
    "name": "string",
    "age": "integer[0:120]",
    "email": "email"
})

validator = schema.validate("data/input.csv")
results = validator.run()
```

### Data Transformation

```python
from semblance.transform import Pipeline

pipeline = Pipeline([
    ("clean_text", TextCleaner()),
    ("normalize", Normalizer()),
    ("encode", LabelEncoder())
])

transformed_data = pipeline.fit_transform(data)
```

## Best Practices

1. Always version your data
2. Use meaningful names for pipelines and steps
3. Document your transformations
4. Test your pipelines with sample data first

## Next Steps

- Learn about [Data Management](../features/data-management.md)
- Explore [ML Pipelines](../features/ml-pipelines.md)
- Set up [Monitoring](../features/monitoring.md) 
