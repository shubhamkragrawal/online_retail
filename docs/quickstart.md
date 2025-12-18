# 🚀 Quick Start Guide

## TL;DR - Get Running in 5 Minutes

```bash
# 1. Clone and setup
git clone <your-repo>
cd retail-churn-classification
chmod +x setup.sh
./setup.sh

# 2. Download dataset (if not done automatically)
# Place at: data/raw/online_retail.csv

# 3. Run everything
python experiments/run_experiments.py
python experiments/mlflow_tracking.py

# 4. Deploy
docker-compose up --build
```

## 📋 Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] Docker & Docker Compose installed (for deployment)
- [ ] Dataset downloaded (see below)
- [ ] DagsHub account created (optional)

## 📥 Dataset Download

**Option 1: UCI Repository**
```bash
# Download from: https://archive.ics.uci.edu/ml/datasets/online+retail
# Save as: data/raw/online_retail.csv
```

**Option 2: Kaggle**
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d lakshmi25npathi/online-retail-dataset
unzip online-retail-dataset.zip -d data/raw/
mv data/raw/OnlineRetail.csv data/raw/online_retail.csv
```

## 🎯 Step-by-Step Execution

### 1. Setup Environment

```bash
# Run automated setup
./setup.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Initialize Database

```bash
python database/init_db.py
```

**Expected Output:**
```
✓ Database schema created successfully
✓ Inserted 4,372 customers
✓ Inserted 3,958 products
✓ Inserted 22,190 invoices
✓ Inserted 397,884 invoice items
```

### 3. Generate ML Dataset

```bash
python data/feature_engineering.py
```

**Expected Output:**
```
✓ Loaded 397,884 transactions for 4,372 customers
✓ Churn rate: 42.15% (1,843/4,372 customers)
✓ Final dataset: 4,372 customers, 22 features
✓ Dataset saved to: data/processed/ml_dataset.csv
```

### 4. Run Experiments

```bash
python experiments/run_experiments.py
```

**Time:** ~15-30 minutes
**Output:**
- 16 trained models in `models/`
- Metrics in `results/experiment_results.json`

### 5. Track with DagsHub (Optional)

```bash
# Setup .env file first
cp .env.example .env
nano .env  # Add your DagsHub credentials

# Log experiments
python experiments/mlflow_tracking.py
```

### 6. Test API

```bash
# Terminal 1: Start API
cd api
uvicorn main:app --reload

# Terminal 2: Run tests
python test_inference.py
```

### 7. Test Streamlit UI

```bash
cd streamlit
streamlit run app.py
```

Visit: `http://localhost:8501`

### 8. Deploy with Docker

```bash
# Build and run
docker-compose up --build

# Background mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔧 Common Commands

### Database Operations

```bash
# View database contents
sqlite3 database/retail.db

# Run queries
sqlite3 database/retail.db "SELECT COUNT(*) FROM Customers;"
```

### Model Operations

```bash
# List all models
ls -lh models/

# Find best model
python -c "
import json
with open('results/experiment_results.json') as f:
    results = json.load(f)
best = max(results, key=lambda x: x['f1_score'])
print(f\"Best: {best['model_name']} - F1: {best['f1_score']:.4f}\")
"
```

### MLflow Operations

```bash
# Start local MLflow UI
mlflow ui

# View experiments
open http://localhost:5000
```

## 🐛 Troubleshooting

### Issue: "Dataset not found"

```bash
# Check if file exists
ls -la data/raw/online_retail.csv

# If missing, download from UCI or Kaggle
```

### Issue: "Module not found"

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Database locked"

```bash
# Close any open connections
# Delete and recreate database
rm database/retail.db
python database/init_db.py
```

### Issue: "Port already in use"

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn main:app --port 8001
```

### Issue: "Docker container won't start"

```bash
# Check logs
docker-compose logs api
docker-compose logs streamlit

# Rebuild
docker-compose down
docker-compose up --build
```

## 📊 Expected Results

### Database Stats
- ~4,300 customers
- ~4,000 products
- ~22,000 invoices
- ~400,000 invoice items

### ML Dataset
- ~4,300 samples
- 22 features
- ~42% churn rate (balanced)

### Model Performance
- Best F1-Score: 0.85-0.87
- Best Model: Usually XGBoost with tuning
- Training time: 15-30 minutes

## 🎯 Project Deliverables Checklist

- [x] 3NF normalized database (SQLite)
- [x] Classification dataset (customer churn)
- [x] 16 experiments (4 algorithms × 4 configs)
- [x] F1-scores for all experiments
- [x] MLflow/DagsHub tracking
- [x] FastAPI backend
- [x] Streamlit frontend
- [x] Docker deployment
- [x] Complete documentation

## 📝 Key Files Reference

| File | Purpose |
|------|---------|
| `database/schema.sql` | 3NF database schema |
| `database/init_db.py` | Initialize & populate database |
| `data/feature_engineering.py` | Create ML features |
| `experiments/run_experiments.py` | Run 16 experiments |
| `experiments/mlflow_tracking.py` | Log to DagsHub |
| `api/main.py` | FastAPI application |
| `streamlit/app.py` | Streamlit UI |
| `docker-compose.yml` | Deployment orchestration |
| `test_inference.py` | API testing |

## 🔗 Useful Links

- **Dataset**: https://archive.ics.uci.edu/ml/datasets/online+retail
- **DagsHub**: https://dagshub.com
- **FastAPI Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

## 💡 Tips

1. **Run experiments overnight** - They take 15-30 minutes
2. **Use Docker for deployment** - Much easier than manual setup
3. **Check DagsHub regularly** - Great for comparing experiments
4. **Start with sample data** - Test pipeline before full dataset
5. **Save best model separately** - Copy it to `models/best_model.pkl`

## 🆘 Getting Help

1. Check logs: `docker-compose logs -f`
2. Read error messages carefully
3. Ensure all dependencies installed
4. Verify dataset is in correct location
5. Check `.env` file configuration

## 🎉 Success Indicators

✅ Database created with 4 tables
✅ ML dataset has ~4,300 rows, 22 features
✅ All 16 experiments complete
✅ F1-scores between 0.75-0.87
✅ API returns predictions
✅ Streamlit UI loads
✅ Docker containers running

---

**Need more details?** See `README.md` for complete documentation.