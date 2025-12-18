# 🛒 Retail Customer Churn Classification

A complete end-to-end machine learning system for predicting customer churn in online retail, featuring database normalization (3NF), 16 experiments, MLflow tracking, and production deployment.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [Experiments](#experiments)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)

## 🎯 Project Overview

This project implements a **binary classification system** to predict customer churn in an online retail environment using the UCI Online Retail Dataset. The system includes:

- **3NF Normalized Database**: Proper database design with SQLite
- **16 Machine Learning Experiments**: 4 algorithms × 4 configurations
- **Experiment Tracking**: MLflow integration with DagsHub
- **Production API**: FastAPI backend for model serving
- **User Interface**: Streamlit frontend for predictions
- **Docker Deployment**: Containerized services with docker-compose

### Classification Problem

**Target Variable**: Customer Churn (Binary)
- **1 (Churned)**: Customer did not return within 90 days
- **0 (Retained)**: Customer made purchases within 90 days

## ✨ Features

- ✅ 3NF normalized SQLite database
- ✅ RFM (Recency, Frequency, Monetary) analysis
- ✅ 16 experiments with different configurations
- ✅ Hyperparameter tuning with Optuna
- ✅ PCA for dimensionality reduction
- ✅ MLflow/DagsHub experiment tracking
- ✅ FastAPI REST API for inference
- ✅ Interactive Streamlit dashboard
- ✅ Docker containerization
- ✅ Complete CI/CD ready

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐
│   Streamlit UI  │─────▶│   FastAPI API    │
│  (Port 8501)    │      │   (Port 8000)    │
└─────────────────┘      └──────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Trained Models  │
                         │   (Pickle files) │
                         └──────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for deployment)
- Git

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/retail-churn-classification.git
cd retail-churn-classification
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the **UCI Online Retail Dataset** from:
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
- [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)

Place the CSV file in: `data/raw/online_retail.csv`

### 5. Setup Environment Variables

```bash
cp .env.example .env
# Edit .env with your DagsHub credentials
```

## 📊 Usage

### Step 1: Initialize Database

Create and populate the 3NF normalized database:

```bash
python database/init_db.py
```

**Output:**
- `database/retail.db` - SQLite database with 4 normalized tables
- Customers, Products, Invoices, InvoiceItems

### Step 2: Generate ML Dataset

Create features and labels for machine learning:

```bash
python data/feature_engineering.py
```

**Output:**
- `data/processed/ml_dataset.csv` - ML-ready dataset with RFM features

### Step 3: Run Experiments

Execute all 16 experiments:

```bash
python experiments/run_experiments.py
```

**This will:**
- Train 16 models (4 algorithms × 4 configurations)
- Save all models to `models/` directory
- Save metrics to `results/experiment_results.json`
- Print comparison table

**Expected Runtime:** ~15-30 minutes depending on hardware

### Step 4: Track with MLflow/DagsHub

```bash
python experiments/mlflow_tracking.py
```

**This will:**
- Log all 16 experiments to DagsHub
- Create comparison visualizations
- Save charts to `results/`

### Step 5: Run API Locally

```bash
cd api
uvicorn main:app --reload
```

API will be available at: `http://localhost:8000`

### Step 6: Run Streamlit UI Locally

```bash
cd streamlit
streamlit run app.py
```

UI will be available at: `http://localhost:8501`

## 🗄️ Database Schema

### 3NF Normalized Design

```sql
Customers (CustomerID PK, Country, FirstPurchaseDate, LastPurchaseDate, TotalPurchases, TotalSpent)
Products (StockCode PK, Description, UnitPrice)
Invoices (InvoiceNo PK, CustomerID FK, InvoiceDate, Country)
InvoiceItems (ItemID PK, InvoiceNo FK, StockCode FK, Quantity, UnitPrice, TotalPrice)
```

**Benefits:**
- ✅ No data redundancy
- ✅ Easy to update customer/product info
- ✅ Maintains referential integrity
- ✅ Optimized queries with indexes

## 🔬 Experiments

### Algorithms Used

1. **Logistic Regression**: Fast, interpretable baseline
2. **Random Forest**: Ensemble method, handles non-linearity
3. **XGBoost**: Gradient boosting, typically best performer
4. **SVM**: Support Vector Machine with RBF kernel

### Hyperparameter Tuning

Using **Optuna** with 50 trials per model:
- Bayesian optimization (TPE sampler)
- 3-fold cross-validation
- F1-score as optimization metric

## 🐳 Deployment

### Local Deployment with Docker

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

### Cloud Deployment (DigitalOcean/Render)

#### Option 1: DigitalOcean

1. Create a Droplet (Ubuntu 22.04)
2. Install Docker and Docker Compose
3. Clone repository
4. Copy models to server
5. Run `docker-compose up -d`
6. Configure firewall (ports 8000, 8501)

#### Option 2: Render

1. Create new Web Service
2. Connect GitHub repository
3. Set Docker as runtime
4. Deploy `api` and `streamlit` as separate services
5. Configure environment variables

### Environment Variables

Required for deployment:

```bash
# API
MODEL_PATH=/app/models/best_model.pkl

# DagsHub (optional for tracking)
DAGSHUB_USER=your-username
DAGSHUB_REPO=retail-churn-classification
DAGSHUB_TOKEN=your-token
```

## 📡 API Documentation

### Endpoints

#### Health Check
```bash
GET /health
```

#### Model Info
```bash
GET /model/info
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "Recency": 30,
  "Frequency": 5,
  "Monetary": 500.0,
  "InvoiceNo_nunique": 5,
  "Quantity_sum": 50.0,
  "Quantity_mean": 10.0,
  "TotalPrice_sum": 500.0,
  "TotalPrice_mean": 100.0,
  "TotalPrice_std": 0.0,
  "StockCode_nunique": 10,
  "CustomerLifetime": 180,
  "AvgDaysBetweenPurchases": 30.0,
  "Country_United_Kingdom": 1,
  ...
}
```

**Response:**
```json
{
  "churn_probability": 0.25,
  "churn_prediction": 0,
  "risk_level": "Low",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### Interactive API Docs

Visit `http://localhost:8000/docs` for Swagger UI

