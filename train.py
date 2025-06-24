import numpy as np
import pandas as pd
import json
import os
from models.fraud_detector import FraudDetector
from preprocessing.data_preprocessor import DataPreprocessor

def generate_sample_data(n_samples=10000):
    """Generate synthetic transaction data for demonstration."""
    np.random.seed(42)
    
    data = {
        'amount': np.random.exponential(100, n_samples),
        'time': np.random.uniform(0, 24, n_samples),
        'n_transactions_day': np.random.poisson(3, n_samples),
        'merchant_category': np.random.choice(['retail', 'travel', 'food', 'entertainment'], n_samples),
        'merchant_name': np.random.choice(['Amazon', 'Walmart', 'Target', 'Best Buy', 'Starbucks'], n_samples),
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        'transaction_id': [f'TXN_{i:07d}' for i in range(n_samples)]
    }
    
    # Generate fraud labels (about 5% fraud rate)
    fraud_mask = (
        (data['amount'] > np.percentile(data['amount'], 95)) & 
        (np.random.random(n_samples) < 0.3)
    ) | (np.random.random(n_samples) < 0.02)
    
    data['is_fraud'] = fraud_mask.astype(int)
    
    return pd.DataFrame(data)

def main():
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load configurations
    with open('config/model_config.json', 'r') as f:
        model_config = json.load(f)
    
    with open('config/preprocessing_config.json', 'r') as f:
        preprocessing_config = json.load(f)
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    df.to_csv('data/sample_transactions.csv', index=False)
    print(f"Generated {len(df)} transactions with {df['is_fraud'].sum()} fraudulent cases")
    
    # Initialize preprocessor and model
    print("\nInitializing preprocessor and model...")
    preprocessor = DataPreprocessor(preprocessing_config)
    model = FraudDetector(model_config)
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Train model
    print("\nTraining model...")
    model.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save model
    print("\nSaving model...")
    model.save_model('models/fraud_detector.joblib')
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 