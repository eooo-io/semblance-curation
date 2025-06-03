# Cloud Provider Optimizations

## AWS Optimizations

### EC2 Instance Selection

```yaml
compute:
  instance_types:
    cpu_optimized:
      - c6i.2xlarge  # Latest gen compute optimized
      - c6a.2xlarge  # AMD variant for cost savings
    memory_optimized:
      - r6i.2xlarge  # Latest gen memory optimized
      - r6a.2xlarge  # AMD variant for cost savings
    gpu_optimized:
      - g5.xlarge    # NVIDIA A10G GPUs
      - p4d.24xlarge # For distributed training
```

### Storage Configuration

```yaml
storage:
  efs:
    provisioned_throughput: 128 MiB/s
    lifecycle_policy:
      transition_to_ia: 30 days
  s3:
    bucket_configuration:
      lifecycle_rules:
        - prefix: "raw/"
          transition_to_intelligent_tiering: 30 days
        - prefix: "processed/"
          transition_to_standard_ia: 90 days
    transfer_acceleration: enabled
```

### Network Optimization

```yaml
network:
  vpc:
    placement_groups:
      - name: ml-cluster
        strategy: cluster
    endpoints:
      - s3
      - ecr
      - cloudwatch
  cloudfront:
    enabled: true
    price_class: PriceClass_200
```

## GCP Optimizations

### Compute Engine Configuration

```yaml
compute:
  instance_templates:
    - name: ml-training
      machine_type: c2-standard-16
      scheduling:
        preemptible: true
        automatic_restart: true
    - name: ml-inference
      machine_type: n2-standard-8
      scheduling:
        spot: true
```

### Storage Optimization

```yaml
storage:
  filestore:
    tier: PREMIUM
    capacity_gb: 2560
  cloud_storage:
    bucket_configuration:
      location_type: DUAL_REGION
      lifecycle_rules:
        - action: SetStorageClass
          storage_class: NEARLINE
          condition:
            age_in_days: 30
```

### Network Configuration

```yaml
network:
  vpc:
    network_tier: PREMIUM
    global_load_balancing: true
  cloud_cdn:
    enabled: true
    cache_mode: CACHE_ALL_STATIC
```

## Azure Optimizations

### VM Configuration

```yaml
compute:
  vm_sizes:
    cpu_optimized:
      - Standard_F16s_v2
      - Standard_F32s_v2
    memory_optimized:
      - Standard_E16s_v4
      - Standard_E32s_v4
    gpu_optimized:
      - Standard_NC24ads_A100_v4
```

### Storage Configuration

```yaml
storage:
  azure_files:
    tier: Premium_LRS
    quota: 5120
  blob_storage:
    account_kind: StorageV2
    access_tier: Hot
    lifecycle_management:
      - base_blob:
          tier_to_cool: 30
          tier_to_archive: 180
```

### Network Optimization

```yaml
network:
  vnet:
    accelerated_networking: true
    load_balancer:
      sku: Standard
      cross_region: true
  front_door:
    enabled: true
    backend_pools_settings:
      enforce_certificate_name_check: true
```

## Cost Optimization Strategies

### Auto-scaling Configuration

```yaml
autoscaling:
  metrics:
    - type: cpu
      target: 70
    - type: memory
      target: 80
    - type: gpu
      target: 85
  schedules:
    - name: training-schedule
      min_nodes: 0
      max_nodes: 10
      start_time: "00:00 UTC"
      end_time: "06:00 UTC"
```

### Resource Cleanup

```yaml
cleanup:
  unused_resources:
    - type: volumes
      age: 7d
    - type: snapshots
      age: 30d
    - type: images
      age: 90d
  cost_optimization:
    reserved_instances:
      coverage_target: 80
    savings_plans:
      commitment_term: 1yr
```

### Monitoring and Alerts

```yaml
monitoring:
  cost_alerts:
    - threshold: 1000
      period: monthly
      notification:
        type: email
        recipients: ["admin@example.com"]
  resource_alerts:
    - metric: unused_resources
      threshold: 20
      period: weekly
```

## Performance Optimization

### Caching Strategy

```yaml
caching:
  redis:
    instance_type: cache.r6g.xlarge
    multi_az: true
    cluster_mode: enabled
  cdn:
    rules:
      - pattern: "*.json"
        ttl: 3600
      - pattern: "*.png"
        ttl: 86400
```

### Data Transfer Optimization

```yaml
data_transfer:
  compression:
    enabled: true
    algorithms:
      - gzip
      - brotli
  batch_processing:
    batch_size: 1000
    parallel_threads: 4
```

### Security Optimization

```yaml
security:
  encryption:
    at_rest:
      enabled: true
      key_rotation: 90
    in_transit:
      enabled: true
      minimum_tls_version: "1.2"
  network_security:
    ddos_protection: true
    waf:
      enabled: true
      rules:
        - owasp_core
        - rate_limiting
