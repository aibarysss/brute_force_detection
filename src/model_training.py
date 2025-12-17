# src/model_training.py

"""
Model training for Brute Force Detection
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             average_precision_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Class for training ML models for brute force detection"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def load_data(self, train_path, test_path):
        """Load processed data"""
        print(f"Loading training data from: {train_path}")
        train_df = pd.read_csv(train_path)

        print(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)

        # Separate features and target
        X_train = train_df.drop(columns=['CLASS'])
        y_train = train_df['CLASS']

        X_test = test_df.drop(columns=['CLASS'])
        y_test = test_df['CLASS']

        print(f"Training data: {X_train.shape}")
        print(f"Test data: {X_test.shape}")

        # Check class distribution
        print(f"\nClass distribution in training:")
        print(f"  Class 0 (Benign): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
        print(f"  Class 1 (Attack): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")

        print(f"\nClass distribution in test:")
        print(f"  Class 0 (Benign): {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)")
        print(f"  Class 1 (Attack): {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")

        return X_train, X_test, y_train, y_test

    def train_baseline_model(self, X_train, y_train):
        """Train baseline model (Logistic Regression)"""
        print("\n" + "=" * 60)
        print("TRAINING BASELINE MODEL: Logistic Regression")
        print("=" * 60)

        # Logistic Regression with class weighting for imbalance
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        )

        # Simple hyperparameter tuning
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }

        print("Performing grid search...")
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

        self.models['baseline'] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def train_advanced_model(self, X_train, y_train):
        """Train advanced model (Random Forest)"""
        print("\n" + "=" * 60)
        print("TRAINING ADVANCED MODEL: Random Forest")
        print("=" * 60)

        # Random Forest with class weighting
        model = RandomForestClassifier(
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        print("Performing grid search...")
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")

        self.models['advanced'] = grid_search.best_estimator_

        return grid_search.best_estimator_

    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        # Store results
        self.results[model_name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'f1_score': f1,
            'y_test': y_test  # Добавьте эту строку
        }

        # Print results
        print(f"\n{classification_report(y_test, y_pred)}")
        print(f"Confusion Matrix:")
        print(f"[[TN: {cm[0, 0]}  FP: {cm[0, 1]}]")
        print(f" [FN: {cm[1, 0]}  TP: {cm[1, 1]}]]")
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return self.results[model_name]

    def plot_confusion_matrix(self, cm, model_name, output_dir):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Benign', 'Predicted Attack'],
                    yticklabels=['Actual Benign', 'Actual Attack'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)

        # Save plot
        plot_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved confusion matrix to: {plot_path}")

        return plot_path

    def plot_roc_curves(self, results_dict, y_test, output_dir):
        """Plot ROC curves for all models"""
        from sklearn.metrics import roc_curve

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each model
        for model_name, result in results_dict.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            roc_auc = result['roc_auc']

            plt.plot(fpr, tpr, lw=2,
                     label=f'{model_name} (AUC = {roc_auc:.3f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves Comparison', fontsize=16, pad=20)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        # Save plot
        plot_path = os.path.join(output_dir, 'roc_curves_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved ROC curves to: {plot_path}")

        return plot_path

    def plot_feature_importance(self, model, feature_names, model_name, output_dir, top_n=20):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]

            plt.figure(figsize=(12, 8))
            plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=16, pad=20)
            bars = plt.barh(range(top_n), importances[indices][:top_n], align='center')
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
            plt.xlabel('Relative Importance', fontsize=14)
            plt.gca().invert_yaxis()

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, importances[indices][:top_n])):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', va='center', fontsize=10)

            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved feature importance plot to: {plot_path}")

            return plot_path, indices[:top_n]
        else:
            print(f"Model {model_name} doesn't have feature_importances_ attribute")
            return None, None

    def save_models(self, output_dir):
        """Save trained models"""
        print("\n" + "=" * 60)
        print("SAVING TRAINED MODELS")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        saved_models = {}
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            saved_models[model_name] = model_path
            print(f"Saved {model_name} model to: {model_path}")

        # Save results summary
        results_summary = {}
        for model_name, result in self.results.items():
            results_summary[model_name] = {
                'roc_auc': result['roc_auc'],
                'f1_score': result['f1_score'],
                'average_precision': result['average_precision']
            }

        summary_path = os.path.join(output_dir, 'model_results_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"Saved results summary to: {summary_path}")

        return saved_models

    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            print("No results to compare!")
            return

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': f"{result['roc_auc']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'Avg Precision': f"{result['average_precision']:.4f}",
                'Precision': f"{result['classification_report']['1']['precision']:.4f}",
                'Recall': f"{result['classification_report']['1']['recall']:.4f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))

        return comparison_df


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("BRUTE FORCE DETECTION - MODEL TRAINING")
    print("=" * 60)

    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    train_path = os.path.join(project_root, 'data', 'processed', 'train_data.csv')
    test_path = os.path.join(project_root, 'data', 'processed', 'test_data.csv')
    output_dir = os.path.join(project_root, 'models')

    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"Output directory: {output_dir}")

    # Initialize trainer
    trainer = ModelTrainer(random_state=42)

    try:
        # Step 1: Load data
        print("\n[STEP 1] Loading data...")
        X_train, X_test, y_train, y_test = trainer.load_data(train_path, test_path)

        # Step 2: Train baseline model
        print("\n[STEP 2] Training baseline model...")
        baseline_model = trainer.train_baseline_model(X_train, y_train)

        # Step 3: Train advanced model
        print("\n[STEP 3] Training advanced model...")
        advanced_model = trainer.train_advanced_model(X_train, y_train)

        # Step 4: Evaluate baseline model
        print("\n[STEP 4] Evaluating baseline model...")
        trainer.evaluate_model(baseline_model, 'Logistic Regression', X_test, y_test)

        # Step 5: Evaluate advanced model
        print("\n[STEP 5] Evaluating advanced model...")
        trainer.evaluate_model(advanced_model, 'Random Forest', X_test, y_test)

        # Step 6: Compare models
        print("\n[STEP 6] Comparing models...")
        comparison_df = trainer.compare_models()

        # Step 7: Create visualizations
        print("\n[STEP 7] Creating visualizations...")

        # Create reports directory
        reports_dir = os.path.join(project_root, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        # Plot confusion matrices
        for model_name in trainer.results.keys():
            cm = trainer.results[model_name]['confusion_matrix']
            trainer.plot_confusion_matrix(cm, model_name, reports_dir)

        # Plot ROC curves
        trainer.plot_roc_curves(trainer.results, y_test, reports_dir)

        # Plot feature importance for Random Forest
        if 'Random Forest' in trainer.models:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            trainer.plot_feature_importance(
                trainer.models['Random Forest'],
                feature_names,
                'Random Forest',
                reports_dir,
                top_n=15
            )

        # Step 8: Save models
        print("\n[STEP 8] Saving models...")
        saved_models = trainer.save_models(output_dir)

        print("\n" + "=" * 60)
        print("[SUCCESS] MODEL TRAINING COMPLETE!")
        print("=" * 60)

        print(f"\nSummary:")
        print(f"- Trained models: {list(trainer.models.keys())}")
        print(f"- Models saved to: {output_dir}")
        print(f"- Reports saved to: {reports_dir}")

        # Print best model
        best_model = max(trainer.results.items(), key=lambda x: x[1]['roc_auc'])
        print(f"- Best model: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")

        return trainer, comparison_df

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    trainer, comparison_df = main()