# Advanced Monitoring Metrics

## ML Model Metrics

### Model Performance Metrics

```python
from semblance.monitoring import MLMetrics

metrics = MLMetrics([
    # Classification Metrics
    ("accuracy", AccuracyMetric()),
    ("precision", PrecisionMetric(average="weighted")),
    ("recall", RecallMetric(average="weighted")),
    ("f1", F1Metric(average="weighted")),
    ("roc_auc", ROCAUCMetric(multi_class="ovr")),
    
    # Regression Metrics
    ("mse", MeanSquaredErrorMetric()),
    ("rmse", RootMeanSquaredErrorMetric()),
    ("mae", MeanAbsoluteErrorMetric()),
    ("r2", R2Metric()),
    
    # Custom Metrics
    ("business_impact", BusinessImpactMetric(
        revenue_weight=0.7,
        cost_weight=0.3
    ))
])

# Track metrics over time
metrics.track(model, X_test, y_test)
```

### Data Drift Detection

```python
from semblance.monitoring import DriftDetector

detector = DriftDetector([
    # Distribution Drift
    ("ks_test", KolmogorovSmirnovTest(
        threshold=0.05,
        correction=True
    )),
    ("chi2_test", ChiSquareTest(
        threshold=0.05,
        bins="auto"
    )),
    
    # Feature Drift
    ("feature_drift", FeatureDriftDetector(
        method="wasserstein",
        threshold=0.1
    )),
    
    # Concept Drift
    ("concept_drift", ConceptDriftDetector(
        window_size=1000,
        alpha=0.05
    ))
])

# Monitor drift
drift_report = detector.monitor(
    reference_data=X_train,
    current_data=X_new
)
```

## System Performance Metrics

### Resource Utilization

```python
from semblance.monitoring import SystemMetrics

system_metrics = SystemMetrics([
    # CPU Metrics
    ("cpu_usage", CPUMetric(
        per_core=True,
        include_iowait=True
    )),
    ("cpu_load", LoadAverageMetric(
        intervals=[1, 5, 15]
    )),
    
    # Memory Metrics
    ("memory_usage", MemoryMetric(
        include_swap=True,
        include_cached=True
    )),
    ("memory_fragmentation", MemoryFragmentationMetric()),
    
    # Disk Metrics
    ("disk_usage", DiskMetric(
        per_partition=True,
        include_inodes=True
    )),
    ("disk_io", DiskIOMetric(
        read_write_separate=True
    )),
    
    # Network Metrics
    ("network_throughput", NetworkThroughputMetric(
        per_interface=True
    )),
    ("network_latency", NetworkLatencyMetric(
        include_dns=True
    ))
])

# Start monitoring
system_metrics.start_monitoring(interval=60)
```

## Pipeline Performance Metrics

### Throughput and Latency

```python
from semblance.monitoring import PipelineMetrics

pipeline_metrics = PipelineMetrics([
    # Throughput Metrics
    ("requests_per_second", ThroughputMetric(
        window_size=60,
        include_failed=True
    )),
    ("batch_processing_rate", BatchThroughputMetric(
        batch_size=32
    )),
    
    # Latency Metrics
    ("processing_time", ProcessingTimeMetric(
        percentiles=[50, 90, 95, 99]
    )),
    ("queue_time", QueueTimeMetric()),
    
    # Error Metrics
    ("error_rate", ErrorRateMetric(
        error_types=["validation", "processing", "timeout"]
    )),
    ("retry_rate", RetryRateMetric())
])

# Monitor pipeline
pipeline_metrics.monitor(pipeline)
```

## Custom Business Metrics

### Business KPI Monitoring

```python
from semblance.monitoring import BusinessMetrics

business_metrics = BusinessMetrics([
    # Revenue Impact
    ("revenue_impact", RevenueImpactMetric(
        prediction_value_mapping={
            "high_value": 1000,
            "medium_value": 500,
            "low_value": 100
        }
    )),
    
    # Cost Metrics
    ("processing_cost", ProcessingCostMetric(
        compute_cost_per_hour=0.5,
        storage_cost_per_gb=0.1
    )),
    
    # Business SLAs
    ("sla_compliance", SLAComplianceMetric(
        thresholds={
            "processing_time": 100,  # ms
            "accuracy": 0.95,
            "availability": 0.999
        }
    ))
])

# Track business metrics
business_metrics.track()
```

## Advanced Alerting

### Alert Configuration

```yaml
alerts:
  # Model Performance Alerts
  model_performance:
    - name: accuracy_drop
      metric: accuracy
      condition: "< 0.95"
      window: 1h
      severity: high
      actions:
        - notify_team
        - trigger_retraining
    
    - name: prediction_latency
      metric: processing_time_p95
      condition: "> 100ms"
      window: 5m
      severity: critical
      actions:
        - scale_up_resources
        - notify_oncall
  
  # Data Quality Alerts
  data_quality:
    - name: missing_values
      metric: null_ratio
      condition: "> 0.01"
      window: 1h
      severity: medium
      actions:
        - log_incident
        - notify_data_team
    
    - name: feature_drift
      metric: drift_score
      condition: "> 0.2"
      window: 1d
      severity: high
      actions:
        - pause_pipeline
        - trigger_investigation
  
  # Resource Utilization Alerts
  resource_utilization:
    - name: high_memory
      metric: memory_usage
      condition: "> 90%"
      window: 5m
      severity: high
      actions:
        - scale_memory
        - cleanup_cache
    
    - name: disk_space
      metric: disk_usage
      condition: "> 85%"
      window: 30m
      severity: warning
      actions:
        - cleanup_old_data
        - notify_admin
  
  # Business Impact Alerts
  business_impact:
    - name: high_cost
      metric: processing_cost
      condition: "> 1000"
      window: 1d
      severity: high
      actions:
        - optimize_resources
        - notify_finance
    
    - name: sla_breach
      metric: sla_compliance
      condition: "< 0.99"
      window: 1h
      severity: critical
      actions:
        - escalate_incident
        - notify_stakeholders
```

### Alert Handling

```python
from semblance.monitoring import AlertHandler

handler = AlertHandler([
    # Notification Channels
    ("email", EmailNotifier(
        recipients=["team@example.com"],
        smtp_config={
            "host": "smtp.example.com",
            "port": 587,
            "use_tls": True
        }
    )),
    
    ("slack", SlackNotifier(
        channels=["#alerts", "#oncall"],
        thread_creation=True
    )),
    
    ("pagerduty", PagerDutyNotifier(
        service_key="YOUR_PD_KEY",
        escalation_policy="P123456"
    )),
    
    # Automated Actions
    ("auto_scale", AutoScaleAction(
        min_instances=2,
        max_instances=10,
        cooldown_period=300
    )),
    
    ("model_management", ModelManagementAction(
        retraining_trigger=True,
        fallback_model="v1.0.0"
    ))
])

# Register handler
alerts.set_handler(handler)
``` 
