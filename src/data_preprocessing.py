# src/data_preprocessing.py

"""
Data preprocessing for Brute Force Detection
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import joblib

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Class for preprocessing the brute force dataset"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.columns_to_drop = []

    def load_data(self, filepath, sample_size=None):
        """Load the dataset"""
        print(f"Loading data from: {filepath}")

        if sample_size:
            df = pd.read_csv(filepath, nrows=sample_size)
        else:
            df = pd.read_csv(filepath)

        print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def initial_cleanup(self, df):
        """Initial data cleanup"""
        print("\n=== Initial Data Cleanup ===")

        original_shape = df.shape
        print(f"Original shape: {original_shape}")

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows. Removing...")
            df = df.drop_duplicates()
            print(f"Removed duplicates. New shape: {df.shape}")

        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\n=== Handling Missing Values ===")

        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]

        if len(missing_cols) == 0:
            print("No missing values found!")
            return df

        print(f"Columns with missing values: {len(missing_cols)}")

        # For TLS_SNI and TLS_JA3 - fill with 'unknown'
        if 'TLS_SNI' in df.columns:
            df['TLS_SNI'] = df['TLS_SNI'].fillna('unknown_sni')
            print("  Filled TLS_SNI with 'unknown_sni'")

        if 'TLS_JA3' in df.columns:
            df['TLS_JA3'] = df['TLS_JA3'].fillna('unknown_ja3')
            print("  Filled TLS_JA3 with 'unknown_ja3'")

        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  Filled {col} with median: {median_val:.2f}")

        return df

    def drop_unnecessary_columns(self, df):
        """Drop columns that are not useful for modeling"""
        print("\n=== Dropping Unnecessary Columns ===")

        # Columns to drop (based on EDA)
        columns_to_drop = [
            'SRC_IP',  # Hashed IP - too many unique values
            'DST_IP',  # Hashed IP - too many unique values
            'TIME_FIRST',  # Timestamp
            'TIME_LAST',  # Timestamp
            'TLS_SNI',  # Too many unique values
            'TLS_JA3',  # Too many unique values
        ]

        # Keep only columns that exist
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]

        if existing_cols_to_drop:
            print(f"Dropping {len(existing_cols_to_drop)} columns: {existing_cols_to_drop}")
            df = df.drop(columns=existing_cols_to_drop)
            print(f"New shape: {df.shape}")
        else:
            print("No columns to drop")

        return df

    def prepare_features_target(self, df):
        """Prepare features (X) and target (y)"""
        print("\n=== Preparing Features and Target ===")

        if 'CLASS' not in df.columns:
            raise ValueError("Target column 'CLASS' not found")

        # Drop non-numeric columns except SCENARIO (for analysis)
        # We'll keep only numerical features for initial model
        X = df.select_dtypes(include=[np.number]).copy()

        # Remove CLASS from features if it's there
        if 'CLASS' in X.columns:
            X = X.drop(columns=['CLASS'])

        y = df['CLASS']

        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # Class distribution
        class_counts = y.value_counts()
        print(f"Class 0 (Benign): {class_counts[0]} ({class_counts[0] / len(y) * 100:.1f}%)")
        print(f"Class 1 (Attack): {class_counts[1]} ({class_counts[1] / len(y) * 100:.1f}%)")

        return X, y

    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets"""
        print("\n=== Splitting Data ===")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # Important for imbalanced data
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        print(f"Train - Class 0: {(y_train == 0).sum()} ({((y_train == 0).sum() / len(y_train)) * 100:.1f}%)")
        print(f"Train - Class 1: {(y_train == 1).sum()} ({((y_train == 1).sum() / len(y_train)) * 100:.1f}%)")
        print(f"Test - Class 0: {(y_test == 0).sum()} ({((y_test == 0).sum() / len(y_test)) * 100:.1f}%)")
        print(f"Test - Class 1: {(y_test == 1).sum()} ({((y_test == 1).sum() / len(y_test)) * 100:.1f}%)")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("\n=== Scaling Features ===")

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Scaled training data: {X_train_scaled.shape}")
        print(f"Scaled test data: {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled

    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """Save processed data to files"""
        print("\n=== Saving Processed Data ===")

        os.makedirs(output_dir, exist_ok=True)

        # Convert to DataFrames
        X_train_df = pd.DataFrame(X_train, columns=X_train.columns if hasattr(X_train, 'columns')
        else [f"feature_{i}" for i in range(X_train.shape[1])])
        X_test_df = pd.DataFrame(X_test, columns=X_test.columns if hasattr(X_test, 'columns')
        else [f"feature_{i}" for i in range(X_test.shape[1])])

        # Add target columns
        train_df = X_train_df.copy()
        train_df['CLASS'] = y_train.values

        test_df = X_test_df.copy()
        test_df['CLASS'] = y_test.values

        # Save to CSV
        train_path = os.path.join(output_dir, 'train_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Saved training data to: {train_path}")
        print(f"Saved test data to: {test_path}")

        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to: {scaler_path}")

        return train_path, test_path


def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("BRUTE FORCE DETECTION - DATA PREPROCESSING")
    print("=" * 60)

    # Get current directory and construct paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    data_path = os.path.join(project_root, 'data', 'raw', 'https_bruteforce_samples.csv')
    output_dir = os.path.join(project_root, 'data', 'processed')

    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"\n[ERROR] Data file not found: {data_path}")
        print("Please download the dataset and place it in data/raw/")
        return None

    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_state=42)

    try:
        # Step 1: Load data
        print("\n[STEP 1] Loading data...")
        df = preprocessor.load_data(data_path)

        # Step 2: Initial cleanup
        print("\n[STEP 2] Initial cleanup...")
        df = preprocessor.initial_cleanup(df)

        # Step 3: Handle missing values
        print("\n[STEP 3] Handling missing values...")
        df = preprocessor.handle_missing_values(df)

        # Step 4: Drop unnecessary columns
        print("\n[STEP 4] Dropping unnecessary columns...")
        df = preprocessor.drop_unnecessary_columns(df)

        # Step 5: Prepare features and target
        print("\n[STEP 5] Preparing features and target...")
        X, y = preprocessor.prepare_features_target(df)

        # Step 6: Split data
        print("\n[STEP 6] Splitting data...")
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)

        # Step 7: Scale features
        print("\n[STEP 7] Scaling features...")
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

        # Step 8: Save processed data
        print("\n[STEP 8] Saving processed data...")
        train_path, test_path = preprocessor.save_processed_data(
            X_train_scaled, X_test_scaled, y_train, y_test, output_dir
        )

        print("\n" + "=" * 60)
        print("[SUCCESS] PREPROCESSING COMPLETE!")
        print("=" * 60)

        print(f"\nSummary:")
        print(f"- Training samples: {len(y_train)}")
        print(f"- Test samples: {len(y_test)}")
        print(f"- Features: {X_train.shape[1]}")
        print(f"- Output saved to: {output_dir}")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_path': train_path,
            'test_path': test_path
        }

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()