import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.model_training import FraudDetectionModel

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'transaction_id': [f'T{i}' for i in range(n_samples)],
        'customer_id': [f'C{i%100}' for i in range(n_samples)],
        'amount': np.random.lognormal(mean=4, sigma=1, size=n_samples),
        'transaction_datetime': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
        'merchant_id': [f'M{i%50}' for i in range(n_samples)],
        'merchant_category': np.random.choice(['retail', 'online', 'travel', 'food'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance"""
    return DataPreprocessor()

@pytest.fixture
def model():
    """Create FraudDetectionModel instance"""
    return FraudDetectionModel()

def test_data_preprocessing(preprocessor, sample_data):
    """Test data preprocessing pipeline"""
    # Test missing value handling
    data_with_missing = sample_data.copy()
    data_with_missing.loc[0:10, 'amount'] = np.nan
    cleaned_data = preprocessor.handle_missing_values(data_with_missing)
    assert cleaned_data['amount'].isna().sum() == 0
    
    # Test feature engineering
    engineered_data = preprocessor.feature_engineering(sample_data)
    assert 'hour' in engineered_data.columns
    assert 'day_of_week' in engineered_data.columns
    assert 'amount_log' in engineered_data.columns
    
    # Test outlier handling
    numeric_columns = ['amount']
    cleaned_data = preprocessor.handle_outliers(sample_data, numeric_columns)
    assert cleaned_data['amount'].max() < sample_data['amount'].max()

def test_model_training(model, preprocessor, sample_data):
    """Test model training pipeline"""
    # Prepare data
    df = preprocessor.handle_missing_values(sample_data)
    df = preprocessor.feature_engineering(df)
    
    # Split features and target
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Test XGBoost training
    model.train_xgboost(X, y)
    assert model.xgb_model is not None
    
    # Test Isolation Forest training
    model.train_isolation_forest(X)
    assert model.isolation_forest is not None
    
    # Test Autoencoder training
    model.train_autoencoder(X, epochs=2)  # Small number of epochs for testing
    assert model.autoencoder is not None
    
    # Test Logistic Regression training
    model.train_logistic_regression(X, y)
    assert model.logistic_regression is not None

def test_model_prediction(model, preprocessor, sample_data):
    """Test model prediction pipeline"""
    # Prepare data
    df = preprocessor.handle_missing_values(sample_data)
    df = preprocessor.feature_engineering(df)
    
    # Split features and target
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Train models
    model.train_xgboost(X, y)
    model.train_isolation_forest(X)
    model.train_autoencoder(X, epochs=2)
    model.train_logistic_regression(X, y)
    
    # Make predictions
    predictions, scores = model.predict(X)
    
    # Test predictions
    assert len(predictions) == len(X)
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
    assert all(0 <= score <= 1 for score in scores)

def test_model_explanation(model, preprocessor, sample_data):
    """Test model explanation using SHAP"""
    # Prepare data
    df = preprocessor.handle_missing_values(sample_data)
    df = preprocessor.feature_engineering(df)
    
    # Split features and target
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Train XGBoost model
    model.train_xgboost(X, y)
    
    # Get SHAP values
    shap_values = model.explain_predictions(X)
    
    # Test SHAP values
    assert shap_values is not None
    assert len(shap_values) == len(X)

def test_model_save_load(model, preprocessor, sample_data, tmp_path):
    """Test model saving and loading"""
    # Prepare data
    df = preprocessor.handle_missing_values(sample_data)
    df = preprocessor.feature_engineering(df)
    
    # Split features and target
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Train models
    model.train_xgboost(X, y)
    model.train_isolation_forest(X)
    model.train_autoencoder(X, epochs=2)
    model.train_logistic_regression(X, y)
    
    # Save models
    model_path = tmp_path / "models"
    model.save_models(str(model_path))
    
    # Create new model instance
    new_model = FraudDetectionModel()
    new_model.load_models(str(model_path))
    
    # Test predictions with loaded model
    predictions1, scores1 = model.predict(X)
    predictions2, scores2 = new_model.predict(X)
    
    assert np.array_equal(predictions1, predictions2)
    assert np.allclose(scores1, scores2) 