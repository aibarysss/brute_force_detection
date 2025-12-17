# notebooks/02_data_exploration_fixed.py

"""
Data Exploration Script for HTTPS Brute Force Detection
Fixed version for Windows encoding issues
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def print_separator(title=""):
    """Print separator with title"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"{title.center(60)}")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n{'=' * 60}\n")


def load_data():
    """Load the HTTPS brute force dataset"""
    print_separator("HTTPS Brute Force Detection - Data Exploration")

    # Get absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'raw', 'https_bruteforce_samples.csv')

    print(f"Loading data from: {data_path}")
    print("Dataset: HTTPS Brute-force dataset with extended network flows")
    print("Source: https://zenodo.org/records/7227413")

    try:
        # Try to load the data
        df = pd.read_csv(data_path)
        print("[OK] Data loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")

        return df

    except FileNotFoundError:
        print("[ERROR] File not found: {data_path}")
        print("\nPlease download the dataset:")
        print("1. Go to: https://zenodo.org/records/7227413")
        print("2. Download 'samples.csv'")
        print("3. Place it in: data/raw/https_bruteforce_samples.csv")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return None


def explore_data(df):
    """Perform basic data exploration"""
    if df is None:
        return

    print_separator("Basic Information")

    # Display basic info
    print("\n[INFO] Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}...")  # Show first 10 columns

    # Check for missing values
    print("\n[CHECK] Missing Values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_percent
    })
    missing_rows = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_rows) > 0:
        print(missing_rows)
    else:
        print("No missing values found!")

    # Target variable distribution
    if 'CLASS' in df.columns:
        print("\n[TARGET] Target Variable Distribution (CLASS):")
        target_dist = df['CLASS'].value_counts()
        target_percent = df['CLASS'].value_counts(normalize=True) * 100

        target_df = pd.DataFrame({
            'Count': target_dist,
            'Percentage': target_percent
        })
        print(target_df)

        # Plot target distribution
        plt.figure(figsize=(10, 6))
        colors = ['lightgreen', 'lightcoral']
        bars = plt.bar(['Benign (0)', 'Brute Force (1)'], target_dist.values,
                       color=colors, edgecolor='black', linewidth=2)

        # Add counts on bars
        for bar, count in zip(bars, target_dist.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{count}\n({count / len(df) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=12)

        plt.title('Distribution of Target Variable (CLASS)', fontsize=16, pad=20)
        plt.ylabel('Count', fontsize=14)
        plt.grid(axis='y', alpha=0.3)

        # Save plot
        os.makedirs('../reports', exist_ok=True)
        plt.savefig('../reports/target_distribution.png', dpi=300, bbox_inches='tight')
        print("\n[SAVE] Saved plot: reports/target_distribution.png")
        plt.show()

    # Display first few rows
    print("\n[DATA] First 3 rows:")
    print(df.head(3))

    # Basic statistics for numerical columns
    print("\n[STATS] Basic Statistics for numerical columns:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numerical columns: {len(numerical_cols)}")

    # Show stats for first 5 numerical columns
    for col in numerical_cols[:5]:
        print(f"\n  {col}:")
        print(f"    Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
        print(f"    Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")


def check_scenario_column(df):
    """Check SCENARIO column if it exists"""
    if 'SCENARIO' in df.columns:
        print_separator("SCENARIO Analysis")

        print("Unique scenarios:")
        scenarios = df['SCENARIO'].unique()
        print(f"Total unique scenarios: {len(scenarios)}")
        print(f"First 5 scenarios: {scenarios[:5]}")

        # Scenario distribution
        print("\nTop 10 scenario distribution:")
        scenario_counts = df['SCENARIO'].value_counts().head(10)
        print(scenario_counts)

        # Check scenario vs class
        if 'CLASS' in df.columns:
            print("\nScenario vs Class (benign vs attack):")
            # Group by scenario and class
            scenario_class = df.groupby(['SCENARIO', 'CLASS']).size().unstack(fill_value=0)
            print(scenario_class.head())


def check_data_balance(df):
    """Check if data is balanced"""
    if 'CLASS' in df.columns:
        print_separator("Data Balance Check")

        class_counts = df['CLASS'].value_counts()
        total = len(df)

        print(f"Class 0 (Benign): {class_counts[0]} samples ({class_counts[0] / total * 100:.1f}%)")
        print(f"Class 1 (Attack): {class_counts[1]} samples ({class_counts[1] / total * 100:.1f}%)")

        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 1.5:
            print("[WARNING] Data is imbalanced! Consider using techniques like:")
            print("  - Class weighting in models")
            print("  - SMOTE (Synthetic Minority Over-sampling)")
            print("  - Undersampling majority class")


def main():
    """Main function"""
    # Load data
    df = load_data()

    if df is not None:
        # Explore data
        explore_data(df)

        # Check scenario column
        check_scenario_column(df)

        # Check data balance
        check_data_balance(df)

        print_separator("Exploration Complete")
        print("\n[DONE] Data exploration completed successfully!")
        print(f"\nNext steps:")
        print("1. Check the saved plot: reports/target_distribution.png")
        print("2. Analyze feature distributions")
        print("3. Check correlations between features")
    else:
        print("\n[FAILED] Failed to load data. Please check the file path.")


if __name__ == "__main__":
    main()