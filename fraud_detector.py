import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import optuna
import shap
import joblib
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFraudDetector:
    """Enhanced fraud detection using ensemble of XGBoost, LSTM, and Isolation Forest"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'random_state': 42,
            'threshold': 0.5,
            'feature_importance': True
        }
        self.models = {}
        self.explainer = None
        self.feature_names = None
        
    def optimize_xgboost(self, trial: optuna.Trial) -> float:
        """Optimize XGBoost hyperparameters using Optuna"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'objective': 'binary:logistic',
            'random_state': self.config['random_state']
        }
        return params

    def create_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Create LSTM model for sequence-based fraud detection"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train ensemble of models"""
        self.feature_names = X.columns.tolist()
        
        # Optimize and train XGBoost
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.optimize_xgboost(trial), n_trials=50)
        best_params = study.best_params
        
        self.models['xgboost'] = xgb.XGBClassifier(**best_params)
        self.models['xgboost'].fit(X, y)
        
        # Train Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=self.config['random_state']
        )
        self.models['isolation_forest'].fit(X)
        
        # Prepare and train LSTM
        sequence_length = 10
        X_lstm = self.prepare_sequences(X, sequence_length)
        self.models['lstm'] = self.create_lstm_model((sequence_length, X.shape[1]))
        self.models['lstm'].fit(X_lstm, y, epochs=10, batch_size=32, validation_split=0.2)
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.models['xgboost'])
        
        logger.info("Successfully trained all models")

    def prepare_sequences(self, X: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """Prepare sequences for LSTM"""
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X.iloc[i:i+sequence_length].values)
        return np.array(sequences)

    def predict_proba(self, transaction: Dict[str, Any]) -> float:
        """Get ensemble prediction probability"""
        try:
            # Prepare input data
            X = pd.DataFrame([transaction])
            
            # Get predictions from each model
            xgb_pred = self.models['xgboost'].predict_proba(X)[:, 1]
            iso_pred = -self.models['isolation_forest'].score_samples(X)
            iso_pred = (iso_pred - iso_pred.min()) / (iso_pred.max() - iso_pred.min())
            
            # Prepare sequence for LSTM
            X_lstm = self.prepare_sequences(X, 10)
            lstm_pred = self.models['lstm'].predict(X_lstm)
            
            # Weighted ensemble
            ensemble_pred = (0.5 * xgb_pred + 0.3 * lstm_pred + 0.2 * iso_pred)
            return float(ensemble_pred[0])
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def explain_prediction(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Generate SHAP-based explanation for prediction"""
        try:
            X = pd.DataFrame([transaction])
            shap_values = self.explainer.shap_values(X)
            
            # Get feature importance
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                importance = abs(shap_values[0][i])
                feature_importance[feature] = float(importance)
            
            # Normalize importance values
            total_importance = sum(feature_importance.values())
            feature_importance = {
                k: v/total_importance 
                for k, v in feature_importance.items()
            }
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error in explanation: {str(e)}")
            raise

    def save_models(self, path: str):
        """Save all models"""
        try:
            # Save XGBoost
            self.models['xgboost'].save_model(f"{path}/xgboost_model.json")
            
            # Save Isolation Forest
            joblib.dump(self.models['isolation_forest'], f"{path}/isolation_forest.joblib")
            
            # Save LSTM
            self.models['lstm'].save(f"{path}/lstm_model")
            
            logger.info(f"Successfully saved models to {path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    def load_models(self, path: str):
        """Load all models"""
        try:
            # Load XGBoost
            self.models['xgboost'] = xgb.XGBClassifier()
            self.models['xgboost'].load_model(f"{path}/xgboost_model.json")
            
            # Load Isolation Forest
            self.models['isolation_forest'] = joblib.load(f"{path}/isolation_forest.joblib")
            
            # Load LSTM
            self.models['lstm'] = load_model(f"{path}/lstm_model")
            
            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.models['xgboost'])
            
            logger.info(f"Successfully loaded models from {path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise 