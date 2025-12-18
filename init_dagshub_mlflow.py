# init_dagshub_mlflow.py
import dagshub
import mlflow
from dotenv import load_dotenv
import os

print("="*60)
print("INITIALIZING DAGSHUB MLFLOW")
print("="*60)

# Load credentials
load_dotenv()

user = os.getenv('DAGSHUB_USER')
repo = os.getenv('DAGSHUB_REPO')
token = os.getenv('DAGSHUB_TOKEN')

print(f"\nUser: {user}")
print(f"Repo: {repo}")
print(f"Token: {token[:5]}...{token[-3:]}")

# Initialize DagsHub
print("\nInitializing DagsHub connection...")
dagshub.init(repo_owner=user, repo_name=repo, mlflow=True)

print("✓ DagsHub initialized!")

# Now MLflow should work
print("\nSetting up MLflow experiment...")
mlflow.set_experiment("retail_churn_classification")

print("\nLogging test run...")
with mlflow.start_run(run_name="initialization_test"):
    mlflow.log_param("init", "success")
    mlflow.log_metric("test", 1.0)

print("\n" + "="*60)
print("✅ DAGSHUB MLFLOW READY!")
print("="*60)
print(f"\nView at: https://dagshub.com/{user}/{repo}/experiments")
print("\nNow run: python experiments/mlflow_tracking.py")
print("="*60)