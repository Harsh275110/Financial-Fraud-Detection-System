import json
import boto3
import os
import pandas as pd
import numpy as np
from model_training import FraudDetectionModel

# Initialize the model
model = FraudDetectionModel()

# Load the model from S3
def load_model_from_s3():
    """Load model artifacts from S3"""
    s3 = boto3.client('s3')
    bucket_name = os.environ['MODEL_BUCKET']
    model_path = '/tmp/models'
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Download model files
    model_files = ['xgboost_model.joblib', 'isolation_forest.joblib', 
                  'logistic_regression.joblib', 'autoencoder_model']
    
    for file in model_files:
        s3.download_file(bucket_name, f'models/{file}', f'{model_path}/{file}')
    
    model.load_models(model_path)

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        # Load model if not loaded
        if model.xgb_model is None:
            load_model_from_s3()
        
        # Parse input data
        if 'body' in event:
            data = json.loads(event['body'])
        else:
            data = event
            
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction, score = model.predict(df)
        
        # Generate SHAP values for explainability
        shap_values = model.explain_predictions(df)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'fraud_probability': float(score[0]),
            'shap_values': {
                'values': shap_values.tolist(),
                'feature_names': df.columns.tolist()
            }
        }
        
        return {
            'statusCode': 200,
            'body': json.dumps(response),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json'
            }
        } 