
# Brute Force Detection - Project Report

Generated: 2025-12-18 02:38:09

## Project Summary
- **Objective**: Detect HTTPS brute force attacks using ML
- **Dataset**: HTTPS Brute-force dataset (67,469 samples)
- **Models**: Logistic Regression (baseline) vs Random Forest

## Results
============================================================
DETAILED ATTACK ANALYSIS
============================================================

Total attacks in dataset: 11,647
Total benign traffic: 55,822

============================================================
ATTACKS BY TOOL:
============================================================
C:\Users\aibar\OneDrive\Desktop\brute_force_detection\analysis_attacks.py:37: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  attack_df['tool'] = attack_df['SCENARIO'].apply(extract_tool)
Patator: 5,671 attacks (48.7%)
Hydra: 5,565 attacks (47.8%)
Ncrack: 411 attacks (3.5%)

============================================================
ATTACKS BY TARGET APPLICATION:
============================================================
C:\Users\aibar\OneDrive\Desktop\brute_force_detection\analysis_attacks.py:59: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  attack_df['application'] = attack_df['SCENARIO'].apply(extract_app)
Mediawiki: 3,132 attacks (26.9%)
Ghost: 1,454 attacks (12.5%)
Wordpress: 1,422 attacks (12.2%)
Opencart: 1,325 attacks (11.4%)
Grafana: 957 attacks (8.2%)
Phpbb: 810 attacks (7.0%)
Apache: 796 attacks (6.8%)
Nginx: 786 attacks (6.7%)
Joomla: 527 attacks (4.5%)
Discourse: 266 attacks (2.3%)

============================================================
ATTACK STATISTICS:
============================================================

Average values comparison:
Metric               Attacks         Benign          Ratio
------------------------------------------------------------
DURATION             12.2            51.4            0.2x
BYTES                14977.9         279962.0        0.1x
PACKETS              84.0            381.0           0.2x
roundtrips           19.6            12.5            1.6x
bytes_per_sec        124837.9        46074.0         2.7x

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
