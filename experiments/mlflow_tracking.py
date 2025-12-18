"""
MLflow Experiment Tracking with DagsHub Integration
Logs all 16 experiments to DagsHub for visualization and comparison
"""

import mlflow
import mlflow.sklearn
import os
import json
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MLflowTracker:
    def __init__(self, experiment_name='retail_churn_classification'):
        """Initialize MLflow tracking"""
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Configure MLflow with DagsHub"""
        # DagsHub configuration (set these in .env file)
        dagshub_user = os.getenv('DAGSHUB_USER', 'your-username')
        dagshub_repo = os.getenv('DAGSHUB_REPO', 'retail-churn-classification')
        dagshub_token = os.getenv('DAGSHUB_TOKEN', '')
        
        # Set tracking URI
        if dagshub_token:
            tracking_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        else:
            # Local tracking if no DagsHub credentials
            tracking_uri = "file:./mlruns"
            print("⚠️  No DagsHub credentials found. Using local tracking.")
            print("   Set DAGSHUB_USER, DAGSHUB_REPO, and DAGSHUB_TOKEN in .env file")
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        print(f"✓ MLflow tracking configured")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Experiment: {self.experiment_name}")
        
    def log_experiment(self, result_dict, model_path):
        """Log a single experiment to MLflow"""
        with mlflow.start_run(run_name=f"Exp{result_dict['experiment_number']}_{result_dict['model_name']}"):
            # Log parameters
            mlflow.log_param("experiment_number", result_dict['experiment_number'])
            mlflow.log_param("model_name", result_dict['model_name'])
            mlflow.log_param("use_pca", result_dict['use_pca'])
            mlflow.log_param("use_tuning", result_dict['use_tuning'])
            
            # Log metrics
            mlflow.log_metric("f1_score", result_dict['f1_score'])
            mlflow.log_metric("accuracy", result_dict['accuracy'])
            mlflow.log_metric("precision", result_dict['precision'])
            mlflow.log_metric("recall", result_dict['recall'])
            
            # Log confusion matrix as JSON
            cm_dict = {
                "true_negative": result_dict['confusion_matrix'][0][0],
                "false_positive": result_dict['confusion_matrix'][0][1],
                "false_negative": result_dict['confusion_matrix'][1][0],
                "true_positive": result_dict['confusion_matrix'][1][1]
            }
            mlflow.log_dict(cm_dict, "confusion_matrix.json")
            
            # Log model artifact
            if os.path.exists(model_path):
                # Load model to log with MLflow
                with open(model_path, 'rb') as f:
                    model_artifacts = pickle.load(f)
                
                mlflow.sklearn.log_model(
                    model_artifacts['model'],
                    "model",
                    registered_model_name=None
                )
                
                # Log preprocessing artifacts
                mlflow.log_artifact(model_path, "full_pipeline")
            
            # Add tags
            mlflow.set_tag("model_type", result_dict['model_name'])
            mlflow.set_tag("preprocessing", "PCA" if result_dict['use_pca'] else "No_PCA")
            mlflow.set_tag("tuning", "Optuna" if result_dict['use_tuning'] else "Default")
            
    def log_all_experiments(self, results_file='results/experiment_results.json',
                           models_dir='models'):
        """Log all experiments from results file"""
        print("\n" + "="*60)
        print("LOGGING EXPERIMENTS TO MLFLOW/DAGSHUB")
        print("="*60 + "\n")
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"Found {len(results)} experiments to log\n")
        
        # Log each experiment
        for result in results:
            exp_num = result['experiment_number']
            model_file = result['model_file']
            model_path = os.path.join(models_dir, model_file)
            
            print(f"Logging Experiment {exp_num}: {result['model_name']} "
                  f"(PCA={result['use_pca']}, Tuning={result['use_tuning']})...")
            
            try:
                self.log_experiment(result, model_path)
                print(f"  ✓ Logged successfully")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\n{'='*60}")
        print("✓ All experiments logged!")
        print(f"{'='*60}\n")
        
    def create_comparison_report(self, results_file='results/experiment_results.json'):
        """Create a comparison report of all experiments"""
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        df = pd.DataFrame(results)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: F1 scores by model
        ax1 = axes[0, 0]
        df_pivot = df.pivot_table(
            values='f1_score',
            index='model_name',
            columns=['use_pca', 'use_tuning'],
            aggfunc='mean'
        )
        df_pivot.plot(kind='bar', ax=ax1)
        ax1.set_title('F1-Score by Model and Configuration', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('F1-Score')
        ax1.legend(title='(PCA, Tuning)', bbox_to_anchor=(1.05, 1))
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: All F1 scores
        ax2 = axes[0, 1]
        df_sorted = df.sort_values('f1_score', ascending=False)
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(df_sorted))]
        ax2.barh(range(len(df_sorted)), df_sorted['f1_score'], color=colors)
        ax2.set_yticks(range(len(df_sorted)))
        ax2.set_yticklabels([f"Exp{row['experiment_number']}" for _, row in df_sorted.iterrows()])
        ax2.set_xlabel('F1-Score')
        ax2.set_title('All Experiments Ranked by F1-Score', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Plot 3: Impact of PCA
        ax3 = axes[1, 0]
        pca_comparison = df.groupby('use_pca')['f1_score'].mean()
        ax3.bar(['Without PCA', 'With PCA'], pca_comparison.values, color=['coral', 'skyblue'])
        ax3.set_ylabel('Average F1-Score')
        ax3.set_title('Impact of PCA on Performance', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Impact of Tuning
        ax4 = axes[1, 1]
        tuning_comparison = df.groupby('use_tuning')['f1_score'].mean()
        ax4.bar(['No Tuning', 'With Tuning'], tuning_comparison.values, color=['lightcoral', 'lightgreen'])
        ax4.set_ylabel('Average F1-Score')
        ax4.set_title('Impact of Hyperparameter Tuning', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = 'results/experiment_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {plot_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("EXPERIMENT STATISTICS")
        print("="*60)
        print(f"\nBest F1-Score: {df['f1_score'].max():.4f}")
        best_exp = df.loc[df['f1_score'].idxmax()]
        print(f"Best Model: Experiment {best_exp['experiment_number']}")
        print(f"  Model: {best_exp['model_name']}")
        print(f"  PCA: {best_exp['use_pca']}")
        print(f"  Tuning: {best_exp['use_tuning']}")
        
        print(f"\nAverage F1-Score: {df['f1_score'].mean():.4f}")
        print(f"Std Dev F1-Score: {df['f1_score'].std():.4f}")
        
        print("\nAverage F1 by Model:")
        print(df.groupby('model_name')['f1_score'].mean().sort_values(ascending=False))
        
        print("\n" + "="*60 + "\n")

def main():
    """Main execution function"""
    tracker = MLflowTracker()
    
    # Log all experiments
    tracker.log_all_experiments()
    
    # Create comparison report
    tracker.create_comparison_report()
    
    print("\n✓ MLflow tracking complete!")
    print("\nTo view experiments:")
    print("  1. On DagsHub: Visit your repository's 'Experiments' tab")
    print("  2. Locally: Run 'mlflow ui' and visit http://localhost:5000")

if __name__ == "__main__":
    main()