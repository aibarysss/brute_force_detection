
# Brute Force Detection - Project Report

Generated: 2025-12-18 02:38:09

## Project Summary
- **Objective**: Detect HTTPS brute force attacks using ML
- **Dataset**: HTTPS Brute-force dataset (67,469 samples)
- **Models**: Logistic Regression (baseline) vs Random Forest

## Results

### Logistic Regression (Baseline)
- ROC-AUC: 0.9963
- F1-Score: 0.9323
- Precision: 0.8826
- Recall: 0.9880
- Errors: 334 (306 FP, 28 FN)

### Random Forest (Advanced)
- ROC-AUC: 1.0000
- F1-Score: 0.9989
- Precision: 0.9996
- Recall: 0.9983
- Errors: 5 (1 FP, 4 FN)

## Project Structure
    brute_force_detection/
    ├── data/ # Raw and processed data
    ├── src/ # Source code
    ├── models/ # Trained models
    ├── reports/ # Visualizations
    ├── notebooks/ # Analysis
    └── run_full_pipeline.py
## How to Reproduce
1. Install: `pip install -r requirements.txt`
2. Run: `python run_full_pipeline.py`
3. Check results in `reports/` folder

## Conclusion
Random Forest achieves near-perfect detection of brute force attacks
with only 5 errors out of 13,494 test samples.
