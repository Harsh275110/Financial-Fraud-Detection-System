from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime, timedelta
import uvicorn
import sys
import jwt
import redis
import aioredis
from prometheus_client import Counter, Histogram

# Setup proper path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(current_dir)

from models.fraud_detector import EnhancedFraudDetector
from preprocessing.data_preprocessor import DataPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup metrics
FRAUD_PREDICTIONS = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions made'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time taken for fraud predictions'
)

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Setup Redis for rate limiting
redis_url = os.getenv("REDIS_URL", "redis://localhost")

app = FastAPI(
    title="Enhanced Financial Fraud Detection API",
    description="""
    Advanced fraud detection system with real-time analysis and monitoring.
    
    Key Features:
    - Real-time transaction analysis
    - Ensemble ML models (XGBoost, LSTM, Isolation Forest)
    - Detailed fraud probability scores
    - SHAP-based explanations
    - Prometheus metrics
    - Rate limiting
    - JWT authentication
    """,
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus monitoring
Instrumentator().instrument(app).expose(app)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    return token_data

@app.on_event("startup")
async def startup():
    # Initialize rate limiter
    redis = await aioredis.create_redis_pool(redis_url)
    await FastAPILimiter.init(redis)
    
    # Load configurations
    config_dir = os.path.join(project_root, 'config')
    with open(os.path.join(config_dir, 'model_config.json'), 'r') as f:
        model_config = json.load(f)
    
    with open(os.path.join(config_dir, 'preprocessing_config.json'), 'r') as f:
        preprocessing_config = json.load(f)
    
    # Initialize models
    app.state.fraud_detector = EnhancedFraudDetector(config=model_config)
    app.state.data_preprocessor = DataPreprocessor(config=preprocessing_config)
    
    # Load models
    model_path = os.getenv('MODEL_PATH', os.path.join(project_root, 'models'))
    app.state.fraud_detector.load_models(model_path)

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # In production, validate against a database
    if form_data.username != "demo" or form_data.password != "demo123":
        raise HTTPException(
            status_code=400,
            detail="Incorrect username or password"
        )
    access_token = create_access_token(
        data={"sub": form_data.username}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
async def root():
    """Serve the main HTML interface."""
    return FileResponse(os.path.join(static_path, "index.html"))

@app.post("/predict")
@PREDICTION_LATENCY.time()
async def predict_fraud(
    transaction: Transaction,
    current_user: User = Depends(get_current_user),
    rate_limit: None = Depends(RateLimiter(times=10, seconds=60))
) -> Dict[str, Any]:
    """
    Analyze a single transaction for potential fraud.
    
    Protected endpoint with:
    - JWT authentication required
    - Rate limited to 10 requests per minute
    - Latency monitoring
    - Fraud prediction counting
    """
    try:
        # Convert transaction to DataFrame for preprocessing
        transaction_dict = transaction.dict()
        
        # Get prediction and probability
        fraud_prob = app.state.fraud_detector.predict_proba(transaction_dict)
        is_fraud = fraud_prob > 0.5
        
        # Get SHAP-based explanation
        feature_importance = app.state.fraud_detector.explain_prediction(transaction_dict)
        
        # Increment prediction counter
        FRAUD_PREDICTIONS.inc()
        
        return {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": float(fraud_prob),
            "is_fraud": bool(is_fraud),
            "risk_factors": feature_importance,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/fraud")
async def get_fraud_metrics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get fraud detection metrics"""
    return {
        "total_predictions": FRAUD_PREDICTIONS._value.get(),
        "average_latency": PREDICTION_LATENCY.describe()["mean"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint with detailed metrics.
    """
    return {
        "status": "healthy",
        "model_type": "ensemble",
        "models": {
            "xgboost": "loaded",
            "lstm": "loaded",
            "isolation_forest": "loaded"
        },
        "metrics": {
            "total_predictions": FRAUD_PREDICTIONS._value.get(),
            "average_latency": PREDICTION_LATENCY.describe()["mean"]
        },
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 