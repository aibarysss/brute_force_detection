Project Overview

Project Title: Investigating Brute Force Attacks Using SIEM Logs (Splunk/ELK)

This project implements a complete machine learning pipeline for detecting HTTPS brute force attacks using network flow analysis. The system achieves 99.96% accuracy with Random Forest classification and provides full integration capabilities with SIEM systems like Splunk and ELK Stack.

Key Results

Model	ROC-AUC	F1-Score	Accuracy	Errors (out of 13,494)

Logistic Regression	0.9963	0.9323	98.23%	334 errors

Random Forest	1.0000	0.9989	99.96%	5 errors

Best Model: Random Forest with only 1 false positive and 4 false negatives.

Quick Start

1. Prerequisites

Python 3.8 or higher

4GB+ RAM

2GB+ free disk space

2. Installation
bash
# Clone the repository
git clone https://github.com/aibarysss/brute_force_detection.git
cd brute_force_detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
3. Download Dataset
Visit: https://zenodo.org/records/7227413

Download samples.csv

Place it in: data/raw/https_bruteforce_samples.csv

4. Run Complete Pipeline
bash
# Run the complete ML pipeline
python run_full_pipeline.py
The pipeline will:

Preprocess data (handle missing values, scale features)

Train both Logistic Regression and Random Forest models

Evaluate models with comprehensive metrics

Generate visualizations and reports

Create SIEM-compatible log formats

Dataset Information
Dataset: HTTPS Brute-force dataset with extended network flows
Source: Zenodo (https://zenodo.org/records/7227413)
Size: 67,469 samples, 61 features
Classes:

Benign (Class 0): 55,822 samples (82.7%)

Attack (Class 1): 11,647 samples (17.3%)

Attack Tools: Ncrack, Thc-hydra, Patator
Target Applications: WordPress, Joomla, MediaWiki, Apache, Nginx, and 6 more

Technical Implementation
Data Preprocessing Pipeline
Loading: Read CSV data with pandas

Cleaning: Handle missing values (TLS_SNI, TLS_JA3 - 16.5% missing)

Feature Selection: Remove irrelevant features (IPs, timestamps)

Scaling: StandardScaler for normalization

Splitting: 80/20 train/test split with stratification

Imbalance Handling: Class weighting and stratified sampling

Models Implemented
1. Baseline Model: Logistic Regression
Purpose: Establish performance baseline

Hyperparameters: C=10, solver='liblinear' (optimized via GridSearchCV)

Class Weighting: 'balanced' to handle 4.79:1 imbalance

2. Advanced Model: Random Forest
Purpose: High-accuracy production model

Hyperparameters: n_estimators=200, max_depth=None (optimized via GridSearchCV)

Advantages: Handles non-linear patterns, robust to noise

Evaluation Metrics
Primary: ROC-AUC, F1-Score (for imbalanced data)

Secondary: Precision, Recall, Accuracy

Visualizations: Confusion matrices, ROC curves, feature importance

SIEM Integration
ELK Stack Integration
Log Format: JSON Lines (JSONL) compatible with Elasticsearch

Index Template: Provided in config/elk_index_template.json

Visualization: Ready for Kibana dashboards

Fields: Timestamp, source IP, destination IP, ML predictions, confidence scores

Splunk Integration
Log Format: CSV with proper field extraction

Configuration: Provided in config/splunk_config.json

SPL Queries: Pre-built search queries for threat hunting

Alerts: Configurable alerting based on ML predictions

Sample Integration Code
python
# Real-time prediction and SIEM integration
from src.model_training import ModelTrainer
import json

# Load trained model
trainer = ModelTrainer()
model = trainer.load_model('models/advanced_model.pkl')

# Predict on new logs
predictions = model.predict(new_network_data)

# Send to SIEM
def send_to_siem(log_entry, prediction):
    siem_log = {
        **log_entry,
        'ml_prediction': prediction,
        'ml_confidence': model.predict_proba([log_entry])[0][1]
    }
    # Send to ELK or Splunk
Performance Analysis
Confusion Matrix (Random Forest)
text
              Predicted
              Benign   Attack
Actual Benign  11164      1   (99.99% correct)
Actual Attack     4    2325   (99.83% correct)
Error Analysis
Total Test Samples: 13,494

Random Forest Errors: 5 (0.037%)

False Positives: 1 (0.007%) - minimal false alarms

False Negatives: 4 (0.030%) - minimal missed attacks

Usage Examples
1. Data Exploration
bash
python notebooks/02_data_exploration_fixed.py
2. Train Models Only
python
from src.model_training import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_and_evaluate()
3. Generate SIEM Logs
bash
python src/siem_log_generator.py
4. Make Predictions on New Data
python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/advanced_model.pkl')
scaler = joblib.load('data/processed/scaler.pkl')

# Preprocess new data
new_data = pd.read_csv('new_network_logs.csv')
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)
probabilities = model.predict_proba(new_data_scaled)
🎓 Academic Requirements Met
This project fulfills all requirements from the "Machine Learning Semester Project – Student Instructions":


📄 Deliverables
Project Code: Complete, well-commented Python implementation

Cleaned Dataset: Processed training and test data

Final Report: 10+ page comprehensive report (reports/FINAL_REPORT.md)

Presentation Materials: Visualizations and summary slides

Trained Models: Serialized Logistic Regression and Random Forest

SIEM Integration: Configurations for ELK and Splunk

