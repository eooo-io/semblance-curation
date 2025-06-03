# ML Pipeline Examples

## Text Classification Pipeline

```python
from semblance.pipeline import Pipeline
from semblance.preprocessing import TextPreprocessor
from semblance.features import TextFeatureExtractor
from sklearn.ensemble import RandomForestClassifier

# Define text preprocessing steps
text_preprocessor = TextPreprocessor([
    ("clean", TextCleaner(
        remove_urls=True,
        remove_emails=True,
        remove_numbers=False
    )),
    ("normalize", TextNormalizer(
        lowercase=True,
        remove_accents=True,
        remove_punctuation=True
    ))
])

# Define feature extraction
feature_extractor = TextFeatureExtractor([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )),
    ("sentiment", SentimentAnalyzer()),
    ("length", TextLengthFeatures())
])

# Create pipeline
pipeline = Pipeline([
    ("preprocess", text_preprocessor),
    ("features", feature_extractor),
    ("classifier", RandomForestClassifier())
])

# Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Time Series Forecasting Pipeline

```python
from semblance.pipeline import TimeSeriesPipeline
from semblance.preprocessing import TimeSeriesPreprocessor
from semblance.features import TimeSeriesFeatures
from semblance.models import Prophet

# Define preprocessing
ts_preprocessor = TimeSeriesPreprocessor([
    ("impute", TimeSeriesImputer(
        method="linear",
        max_gap="1D"
    )),
    ("resample", TimeSeriesResampler(
        freq="1H",
        aggregation="mean"
    ))
])

# Define feature engineering
ts_features = TimeSeriesFeatures([
    ("calendar", CalendarFeatures(
        include_holidays=True,
        country="US"
    )),
    ("lags", LagFeatures(
        lags=[1, 7, 30],
        rolling_stats=True
    )),
    ("fourier", FourierFeatures(
        period=24,
        order=4
    ))
])

# Create pipeline
pipeline = TimeSeriesPipeline([
    ("preprocess", ts_preprocessor),
    ("features", ts_features),
    ("model", Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative"
    ))
])

# Train and forecast
pipeline.fit(historical_data)
forecast = pipeline.predict(periods=30)
```

## Image Classification Pipeline

```python
from semblance.pipeline import ImagePipeline
from semblance.preprocessing import ImagePreprocessor
from semblance.features import ImageFeatureExtractor
from semblance.models import TransferLearning

# Define image preprocessing
img_preprocessor = ImagePreprocessor([
    ("resize", ImageResizer(
        target_size=(224, 224),
        keep_aspect_ratio=True
    )),
    ("augment", ImageAugmenter(
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2
    ))
])

# Define feature extraction
feature_extractor = ImageFeatureExtractor([
    ("base_model", PretrainedModel(
        model_name="ResNet50",
        weights="imagenet",
        include_top=False
    )),
    ("pooling", GlobalAveragePooling2D())
])

# Create pipeline
pipeline = ImagePipeline([
    ("preprocess", img_preprocessor),
    ("features", feature_extractor),
    ("classifier", Dense(num_classes, activation="softmax"))
])

# Train with GPU acceleration
pipeline.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    device="gpu"
)
pipeline.fit(train_data, epochs=10)
```

## Recommendation System Pipeline

```python
from semblance.pipeline import RecommendationPipeline
from semblance.preprocessing import UserItemPreprocessor
from semblance.features import InteractionFeatures
from semblance.models import MatrixFactorization

# Define preprocessing
interaction_preprocessor = UserItemPreprocessor([
    ("clean", InteractionCleaner(
        min_interactions=5,
        remove_duplicates=True
    )),
    ("encode", UserItemEncoder())
])

# Define feature engineering
interaction_features = InteractionFeatures([
    ("temporal", TemporalFeatures(
        time_decay=True,
        seasonal_patterns=True
    )),
    ("context", ContextFeatures(
        include_time=True,
        include_location=True
    ))
])

# Create pipeline
pipeline = RecommendationPipeline([
    ("preprocess", interaction_preprocessor),
    ("features", interaction_features),
    ("model", MatrixFactorization(
        n_factors=100,
        regularization=0.01,
        learn_rate=0.001
    ))
])

# Train and generate recommendations
pipeline.fit(interaction_data)
recommendations = pipeline.recommend(user_id, n_items=10)
```

## Automated Model Selection Pipeline

```python
from semblance.pipeline import AutoMLPipeline
from semblance.optimization import HyperparameterOptimizer
from semblance.evaluation import CrossValidator

# Define search space
search_space = {
    "preprocessor": [
        StandardScaler(),
        RobustScaler(),
        MinMaxScaler()
    ],
    "feature_selection": [
        PCA(n_components=0.95),
        SelectKBest(k=20),
        None
    ],
    "model": [
        RandomForestClassifier(),
        XGBClassifier(),
        LightGBMClassifier()
    ]
}

# Create AutoML pipeline
pipeline = AutoMLPipeline(
    search_space=search_space,
    optimizer=HyperparameterOptimizer(
        method="bayesian",
        n_trials=100
    ),
    validator=CrossValidator(
        cv=5,
        scoring="f1"
    )
)

# Find best pipeline
best_pipeline = pipeline.fit(X_train, y_train)
predictions = best_pipeline.predict(X_test)
```

## Deployment Configuration

```yaml
pipeline:
  name: text_classification
  version: 1.0
  
  resources:
    cpu: 4
    memory: 8Gi
    gpu: 1
  
  scaling:
    min_replicas: 2
    max_replicas: 10
    target_cpu_utilization: 70
  
  monitoring:
    metrics:
      - accuracy
      - latency
      - throughput
    alerts:
      - type: accuracy_drop
        threshold: 0.95
      - type: high_latency
        threshold: 100ms
  
  endpoints:
    - name: predict
      path: /v1/predict
      method: POST
      batch_size: 32
    - name: feedback
      path: /v1/feedback
      method: POST
  
  storage:
    model_artifacts: s3://models/
    predictions: s3://predictions/
    monitoring: s3://monitoring/
``` 
