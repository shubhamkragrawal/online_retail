"""
FastAPI Application for Customer Churn Prediction
Serves model predictions via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pickle
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn in online retail",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and artifacts
model_artifacts = None
feature_names = None

# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    """Input features for a single customer"""
    Recency: float = Field(..., description="Days since last purchase", ge=0)
    Frequency: int = Field(..., description="Number of orders", ge=0)
    Monetary: float = Field(..., description="Total amount spent", ge=0)
    InvoiceNo_nunique: int = Field(..., description="Number of unique invoices", ge=0)
    Quantity_sum: float = Field(..., description="Total quantity purchased", ge=0)
    Quantity_mean: float = Field(..., description="Average quantity per order", ge=0)
    TotalPrice_sum: float = Field(..., description="Total revenue", ge=0)
    TotalPrice_mean: float = Field(..., description="Average order value", ge=0)
    TotalPrice_std: float = Field(0.0, description="Std dev of order values", ge=0)
    StockCode_nunique: int = Field(..., description="Number of unique products", ge=0)
    CustomerLifetime: float = Field(..., description="Days as customer", ge=0)
    AvgDaysBetweenPurchases: float = Field(..., description="Average days between orders", ge=0)
    Country_United_Kingdom: int = Field(0, description="Is from UK", ge=0, le=1)
    Country_Germany: int = Field(0, description="Is from Germany", ge=0, le=1)
    Country_France: int = Field(0, description="Is from France", ge=0, le=1)
    Country_EIRE: int = Field(0, description="Is from Ireland", ge=0, le=1)
    Country_Spain: int = Field(0, description="Is from Spain", ge=0, le=1)
    Country_Netherlands: int = Field(0, description="Is from Netherlands", ge=0, le=1)
    Country_Belgium: int = Field(0, description="Is from Belgium", ge=0, le=1)
    Country_Switzerland: int = Field(0, description="Is from Switzerland", ge=0, le=1)
    Country_Portugal: int = Field(0, description="Is from Portugal", ge=0, le=1)
    Country_Australia: int = Field(0, description="Is from Australia", ge=0, le=1)

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    customer_id: Optional[str] = None
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: int = Field(..., description="Binary prediction (0=retain, 1=churn)")
    risk_level: str = Field(..., description="Risk category (Low/Medium/High)")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerFeatures]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int

class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    model_file: str
    features_count: int
    uses_pca: bool
    model_loaded: bool
    timestamp: str

def load_model(model_path: str = "models/best_model.pkl"):
    """Load trained model and artifacts"""
    global model_artifacts, feature_names
    
    if not os.path.exists(model_path):
        # Try to find any model file
        models_dir = os.path.dirname(model_path)
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
                print(f"Using model: {model_path}")
            else:
                raise FileNotFoundError(f"No model files found in {models_dir}")
        else:
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    with open(model_path, 'rb') as f:
        model_artifacts = pickle.load(f)
    
    feature_names = model_artifacts['feature_names']
    print(f"✓ Model loaded successfully from {model_path}")
    print(f"✓ Features: {len(feature_names)}")
    return model_path

def prepare_features(customer: CustomerFeatures) -> np.ndarray:
    """Prepare features for prediction"""
    # Convert to dictionary
    features_dict = customer.dict()
    
    # Create DataFrame with correct feature order
    df = pd.DataFrame([features_dict])
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the features used in training
    df = df[feature_names]
    
    return df.values

def predict_single(features: np.ndarray) -> tuple:
    """Make prediction for single customer"""
    # Scale features
    features_scaled = model_artifacts['scaler'].transform(features)
    
    # Apply PCA if used during training
    if model_artifacts['pca'] is not None:
        features_processed = model_artifacts['pca'].transform(features_scaled)
    else:
        features_processed = features_scaled
    
    # Predict
    prediction = model_artifacts['model'].predict(features_processed)[0]
    
    # Get probability if available
    if hasattr(model_artifacts['model'], 'predict_proba'):
        probability = model_artifacts['model'].predict_proba(features_processed)[0][1]
    else:
        probability = float(prediction)
    
    return int(prediction), float(probability)

def get_risk_level(probability: float) -> str:
    """Determine risk level based on churn probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Warning: Could not load model on startup: {e}")
        print("   Model will be loaded on first prediction request")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_artifacts is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=type(model_artifacts['model']).__name__,
        model_file="Loaded",
        features_count=len(feature_names),
        uses_pca=model_artifacts['pca'] is not None,
        model_loaded=True,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures, customer_id: Optional[str] = None):
    """Predict churn for a single customer"""
    if model_artifacts is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    try:
        # Prepare features
        features = prepare_features(customer)
        
        # Make prediction
        prediction, probability = predict_single(features)
        
        # Get risk level
        risk_level = get_risk_level(probability)
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    if model_artifacts is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    try:
        predictions = []
        high_risk_count = 0
        
        for idx, customer in enumerate(request.customers):
            # Prepare features
            features = prepare_features(customer)
            
            # Make prediction
            prediction, probability = predict_single(features)
            
            # Get risk level
            risk_level = get_risk_level(probability)
            
            if risk_level == "High":
                high_risk_count += 1
            
            predictions.append(PredictionResponse(
                customer_id=f"customer_{idx+1}",
                churn_probability=round(probability, 4),
                churn_prediction=prediction,
                risk_level=risk_level,
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/features")
async def get_feature_names():
    """Get list of feature names"""
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": feature_names,
        "count": len(feature_names)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)