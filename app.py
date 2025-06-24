from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime

from src.models.fraud_detector import FraudDetector
from src.preprocessing.data_preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Fraud Detection API",
    description="API for detecting fraudulent financial transactions using machine learning",
    version="1.0.0"
)

# Load configuration
with open('config/model_config.json', 'r') as f:
    model_config = json.load(f)

with open('config/preprocessing_config.json', 'r') as f:
    preprocessing_config = json.load(f)

# Initialize models and preprocessor
fraud_detector = FraudDetector(config=model_config)
data_preprocessor = DataPreprocessor(config=preprocessing_config)

# Load the trained model
model_path = os.getenv('MODEL_PATH', 'models/fraud_detector.joblib')
fraud_detector.load_model(model_path)

class Transaction(BaseModel):
    """
    Pydantic model for transaction data validation.
    """
    amount: float = Field(..., description="Transaction amount")
    time: float = Field(..., ge=0, le=24, description="Time of day (0-24)")
    n_transactions_day: int = Field(..., ge=0, description="Number of transactions in the day")
    merchant_category: Optional[str] = Field(None, description="Merchant category code")
    merchant_name: Optional[str] = Field(None, description="Name of the merchant")
    customer_id: Optional[str] = Field(None, description="Unique customer identifier")
    transaction_id: str = Field(..., description="Unique transaction identifier")

class TransactionBatch(BaseModel):
    """
    Pydantic model for batch transaction processing.
    """
    transactions: List[Transaction]

class FraudPrediction(BaseModel):
    """
    Pydantic model for fraud prediction response.
    """
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    explanation: Optional[Dict[str, float]]
    timestamp: str

@app.get("/")
async def root():
    """
    Root endpoint returning API information.
    """
    return {
        "message": "Financial Fraud Detection API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: Transaction):
    """
    Endpoint for single transaction fraud prediction.
    """
    try:
        # Convert transaction to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Preprocess data
        X = data_preprocessor.prepare_data(df)[0]  # Get only the features
        
        # Make prediction
        fraud_probability = fraud_detector.predict_proba(X)[0]
        is_fraud = fraud_probability > 0.5
        
        # Generate explanation if using XGBoost
        explanation = None
        if fraud_detector.config['model_type'] == 'xgboost':
            shap_values = fraud_detector.explain_predictions(X)
            if shap_values is not None:
                explanation = dict(zip(
                    df.columns,
                    shap_values[0].tolist()
                ))
        
        return FraudPrediction(
            transaction_id=transaction.transaction_id,
            fraud_probability=float(fraud_probability),
            is_fraud=bool(is_fraud),
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error processing transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[FraudPrediction])
async def predict_fraud_batch(transactions: TransactionBatch):
    """
    Endpoint for batch transaction fraud prediction.
    """
    try:
        # Convert transactions to DataFrame
        df = pd.DataFrame([t.dict() for t in transactions.transactions])
        
        # Preprocess data
        X = data_preprocessor.prepare_data(df)[0]  # Get only the features
        
        # Make predictions
        fraud_probabilities = fraud_detector.predict_proba(X)
        is_fraud = fraud_probabilities > 0.5
        
        # Generate explanations if using XGBoost
        explanations = None
        if fraud_detector.config['model_type'] == 'xgboost':
            shap_values = fraud_detector.explain_predictions(X)
            if shap_values is not None:
                explanations = [
                    dict(zip(df.columns, shap_values[i].tolist()))
                    for i in range(len(shap_values))
                ]
        
        # Prepare response
        predictions = []
        for i, transaction in enumerate(transactions.transactions):
            predictions.append(
                FraudPrediction(
                    transaction_id=transaction.transaction_id,
                    fraud_probability=float(fraud_probabilities[i]),
                    is_fraud=bool(is_fraud[i]),
                    explanation=explanations[i] if explanations else None,
                    timestamp=datetime.now().isoformat()
                )
            )
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error processing batch transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_type": fraud_detector.config['model_type'],
        "timestamp": datetime.now().isoformat()
    } 