# config/config.py

# Dataset configuration
DATASET_CONFIG = {
    'path': 'data/raw/',
    'test_size': 0.2,
    'random_state': 42,
    'time_window': '5T'  # 5 minutes
}

# Model configuration
MODEL_CONFIG = {
    'baseline_model': 'logistic_regression',
    'advanced_model': 'random_forest',
    'threshold': 0.7
}

# Feature engineering
FEATURE_CONFIG = {
    'aggregation_window': '5min',
    'min_failed_attempts': 3,
    'high_failure_threshold': 0.8  # 80% failure rate
}