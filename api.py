from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import pandas as pd
from model_training import FraudDetectionModel
from database import DatabaseConnection

app = FastAPI(title="Fraud Detection API")
model = FraudDetectionModel()
db = DatabaseConnection()

class Transaction(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    transaction_datetime: datetime
    merchant_id: str
    merchant_category: str

class TransactionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    shap_values: dict

class FraudStatistics(BaseModel):
    date: datetime
    total_transactions: int
    fraud_count: int
    avg_fraud_amount: float

@app.on_event("startup")
async def startup_event():
    """Initialize database connection and load model"""
    # Connect to database
    if not db.connect():
        raise Exception("Failed to connect to database")
    
    # Create tables if they don't exist
    db.create_tables()
    
    # Load model
    try:
        model.load_models("models")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise Exception("Failed to load models")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection"""
    db.disconnect()

@app.post("/predict", response_model=TransactionResponse)
async def predict_fraud(transaction: Transaction):
    """Predict fraud for a single transaction"""
    try:
        # Convert transaction to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Make prediction
        prediction, score = model.predict(df)
        
        # Get SHAP values
        shap_values = model.explain_predictions(df)
        
        # Store transaction in database
        db.store_transaction(
            transaction.dict(),
            bool(prediction[0]),
            float(score[0])
        )
        
        return {
            "transaction_id": transaction.transaction_id,
            "is_fraud": bool(prediction[0]),
            "fraud_probability": float(score[0]),
            "shap_values": {
                "values": shap_values.tolist(),
                "feature_names": df.columns.tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customer/{customer_id}/transactions")
async def get_customer_transactions(customer_id: str):
    """Get transaction history for a customer"""
    try:
        transactions = db.get_customer_transactions(customer_id)
        return transactions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics", response_model=List[FraudStatistics])
async def get_fraud_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get fraud statistics for reporting"""
    try:
        statistics = db.get_fraud_statistics(start_date, end_date)
        return statistics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 