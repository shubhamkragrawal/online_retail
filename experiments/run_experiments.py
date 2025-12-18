"""
Experiment Runner for Customer Churn Classification
Runs 16 experiments: 4 algorithms × 4 configurations
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, 
    recall_score, confusion_matrix, classification_report
)
import optuna
from optuna.samplers import TPESampler

class ExperimentRunner:
    def __init__(self, data_path='data/processed/ml_dataset.csv', 
                 models_dir='models', results_dir='results'):
        self.data_path = data_path
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Results storage
        self.results = []
        
    def load_data(self):
        """Load and split dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        self.X = df.drop(['CustomerID', 'Churned'], axis=1)
        self.y = df['Churned']
        
        # Train-test split (stratified)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"✓ Dataset loaded: {len(self.X_train)} train, {len(self.X_test)} test samples")
        print(f"✓ Features: {self.X.shape[1]}")
        print(f"✓ Train churn rate: {self.y_train.mean():.2%}")
        print(f"✓ Test churn rate: {self.y_test.mean():.2%}")
        
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
        
    def apply_pca(self, X_train, X_test, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        print(f"  PCA: Reduced to {X_train_pca.shape[1]} components (from {X_train.shape[1]})")
        return X_train_pca, X_test_pca, pca
        
    def get_base_models(self):
        """Get base models with default parameters"""
        return {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True)
        }
        
    def tune_hyperparameters(self, model_name, X_train, y_train, n_trials=50):
        """Hyperparameter tuning using Optuna"""
        print(f"  Tuning {model_name} hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            if model_name == 'LogisticRegression':
                params = {
                    'C': trial.suggest_loguniform('C', 1e-3, 10.0),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                    'max_iter': 1000,
                    'random_state': 42
                }
                model = LogisticRegression(**params)
                
            elif model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
                
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': 42,
                    'eval_metric': 'logloss'
                }
                model = XGBClassifier(**params)
                
            elif model_name == 'SVM':
                params = {
                    'C': trial.suggest_loguniform('C', 1e-3, 10.0),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'random_state': 42,
                    'probability': True
                }
                model = SVC(**params)
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, 
                                    scoring='f1', n_jobs=-1)
            return scores.mean()
        
        # Run Optuna optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print(f"  Best F1: {study.best_value:.4f}")
        
        # Return best model
        if model_name == 'LogisticRegression':
            return LogisticRegression(**study.best_params, max_iter=1000, random_state=42)
        elif model_name == 'RandomForest':
            return RandomForestClassifier(**study.best_params, random_state=42)
        elif model_name == 'XGBoost':
            return XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss')
        elif model_name == 'SVM':
            return SVC(**study.best_params, random_state=42, probability=True)
            
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
        
    def run_single_experiment(self, exp_num, model_name, use_pca, use_tuning):
        """Run a single experiment"""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_num}: {model_name}")
        print(f"PCA: {use_pca}, Tuning: {use_tuning}")
        print(f"{'='*60}")
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = self.scale_features(
            self.X_train, self.X_test
        )
        
        # Apply PCA if needed
        pca = None
        if use_pca:
            X_train_proc, X_test_proc, pca = self.apply_pca(
                X_train_scaled, X_test_scaled
            )
        else:
            X_train_proc, X_test_proc = X_train_scaled, X_test_scaled
            
        # Get model
        if use_tuning:
            model = self.tune_hyperparameters(model_name, X_train_proc, self.y_train)
        else:
            model = self.get_base_models()[model_name]
            
        # Train model
        print("  Training model...")
        model.fit(X_train_proc, self.y_train)
        
        # Evaluate
        print("  Evaluating model...")
        metrics = self.evaluate_model(model, X_test_proc, self.y_test)
        
        # Print results
        print(f"\n  Results:")
        print(f"    F1-Score:  {metrics['f1_score']:.4f}")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        
        # Save model
        model_filename = f"exp{exp_num}_{model_name.lower()}_pca{use_pca}_tune{use_tuning}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        model_artifacts = {
            'model': model,
            'scaler': scaler,
            'pca': pca if use_pca else None,
            'feature_names': list(self.X.columns)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_artifacts, f)
        print(f"  ✓ Model saved: {model_path}")
        
        # Store results
        result = {
            'experiment_number': exp_num,
            'model_name': model_name,
            'use_pca': use_pca,
            'use_tuning': use_tuning,
            'model_file': model_filename,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.results.append(result)
        return result
        
    def run_all_experiments(self):
        """Run all 16 experiments"""
        print("\n" + "="*60)
        print("RUNNING ALL 16 EXPERIMENTS")
        print("="*60 + "\n")
        
        models = ['LogisticRegression', 'RandomForest', 'XGBoost', 'SVM']
        exp_num = 1
        
        for model_name in models:
            for use_pca in [False, True]:
                for use_tuning in [False, True]:
                    self.run_single_experiment(exp_num, model_name, use_pca, use_tuning)
                    exp_num += 1
                    
        # Save all results
        self.save_results()
        self.print_summary()
        
    def save_results(self):
        """Save experiment results to JSON"""
        results_file = os.path.join(self.results_dir, 'experiment_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved: {results_file}")
        
        # Also save as CSV for easy viewing
        df_results = pd.DataFrame(self.results)
        csv_file = os.path.join(self.results_dir, 'experiment_results.csv')
        df_results.to_csv(csv_file, index=False)
        print(f"✓ Results CSV saved: {csv_file}")
        
    def print_summary(self):
        """Print summary of all experiments"""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60 + "\n")
        
        df = pd.DataFrame(self.results)
        df_display = df[['experiment_number', 'model_name', 'use_pca', 
                        'use_tuning', 'f1_score', 'accuracy']].copy()
        
        print(df_display.to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"Best F1-Score: {df['f1_score'].max():.4f}")
        best_exp = df.loc[df['f1_score'].idxmax()]
        print(f"Best Model: Experiment {best_exp['experiment_number']} - {best_exp['model_name']}")
        print(f"  PCA: {best_exp['use_pca']}, Tuning: {best_exp['use_tuning']}")
        print(f"{'='*60}\n")

def main():
    """Main execution function"""
    runner = ExperimentRunner()
    runner.run_all_experiments()
    
if __name__ == "__main__":
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()