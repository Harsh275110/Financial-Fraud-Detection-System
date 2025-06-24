# AI-Powered Financial Fraud Detection System

A sophisticated machine learning system that detects fraudulent financial transactions in real-time using multiple AI models and advanced analytics.

## Project Overview

This project implements a comprehensive fraud detection system that combines multiple machine learning models to identify fraudulent financial transactions. The system uses a combination of supervised and unsupervised learning techniques, including:

- XGBoost for primary classification
- Isolation Forest for anomaly detection
- Autoencoder for deep learning-based fraud detection
- Logistic Regression as a baseline model
- SHAP (SHapley Additive exPlanations) for model interpretability

## Tech Stack

- **Python**: Primary programming language
- **Scikit-learn**: For traditional machine learning models
- **XGBoost**: For gradient boosting implementation
- **PyCaret**: For automated machine learning workflows
- **AWS Lambda**: For serverless deployment
- **PostgreSQL**: For transaction data storage
- **Power BI**: For visualization and reporting
- **FastAPI**: For API development
- **TensorFlow**: For deep learning models

## Project Structure

```
financial_fraud_detection/
├── data/                  # Data directory
├── models/               # Saved model files
├── src/                  # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── database.py
│   ├── api.py
│   ├── lambda_handler.py
│   └── train.py
├── notebooks/           # Jupyter notebooks
├── dashboard/          # Power BI dashboard files
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Features

1. **Real-time Fraud Detection**
   - Process transactions in real-time
   - Multiple model ensemble for robust predictions
   - API endpoints for integration

2. **Advanced Analytics**
   - Anomaly detection using Isolation Forest
   - Deep learning with Autoencoders
   - Feature importance analysis using SHAP

3. **Model Interpretability**
   - SHAP values for prediction explanation
   - Feature importance visualization
   - Compliance with financial regulations

4. **Scalable Architecture**
   - Serverless deployment on AWS Lambda
   - PostgreSQL database for transaction storage
   - RESTful API using FastAPI

5. **Monitoring and Reporting**
   - Power BI dashboard for fraud patterns
   - Real-time monitoring capabilities
   - Custom reporting functionality

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd financial_fraud_detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file with the following variables
DB_HOST=your_db_host
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_PORT=5432
MODEL_BUCKET=your_s3_bucket_name
```

## Usage

1. **Train Models**
```bash
python src/train.py
```

2. **Run API Server**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

3. **Make Predictions**
```python
import requests

# Example transaction
transaction = {
    "transaction_id": "123456",
    "customer_id": "C789",
    "amount": 1000.00,
    "transaction_datetime": "2024-03-15T10:30:00",
    "merchant_id": "M456",
    "merchant_category": "retail"
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=transaction)
result = response.json()
```

## API Endpoints

- `POST /predict`: Predict fraud for a single transaction
- `GET /customer/{customer_id}/transactions`: Get customer transaction history
- `GET /statistics`: Get fraud statistics for reporting

## Model Training

The system uses a comprehensive training pipeline that includes:

1. Data preprocessing and feature engineering
2. Training multiple models (XGBoost, Isolation Forest, Autoencoder, Logistic Regression)
3. Model evaluation and validation
4. SHAP value calculation for interpretability
5. Model persistence and versioning

## AWS Lambda Deployment

1. Package the application:
```bash
zip -r function.zip .
```

2. Deploy to AWS Lambda:
- Create a new Lambda function
- Upload the function.zip file
- Configure environment variables
- Set up API Gateway trigger

## Database Schema

The PostgreSQL database includes tables for:
- Transactions
- Fraud predictions
- Model metadata
- Audit logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 