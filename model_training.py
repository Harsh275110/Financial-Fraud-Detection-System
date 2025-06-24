import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import shap
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model

class FraudDetectionModel:
    def __init__(self):
        self.xgb_model = None
        self.isolation_forest = None
        self.autoencoder = None
        self.logistic_regression = None
        self.shap_explainer = None

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        self.xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.01,
            n_estimators=200,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)

    def train_isolation_forest(self, X_train):
        """Train Isolation Forest for anomaly detection"""
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.isolation_forest.fit(X_train)

    def create_autoencoder(self, input_dim):
        """Create and compile autoencoder model"""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)

        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

        # Autoencoder model
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train_autoencoder(self, X_train, epochs=50, batch_size=32):
        """Train autoencoder model"""
        if self.autoencoder is None:
            self.create_autoencoder(X_train.shape[1])
        
        self.autoencoder.fit(
            X_train, 
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1
        )

    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        self.logistic_regression = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.logistic_regression.fit(X_train, y_train)

    def get_anomaly_scores(self, X):
        """Get anomaly scores from autoencoder"""
        predictions = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - predictions, 2), axis=1)
        return mse

    def predict(self, X):
        """Combine predictions from all models"""
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        if_pred = self.isolation_forest.predict(X)
        lr_pred = self.logistic_regression.predict_proba(X)[:, 1]
        ae_scores = self.get_anomaly_scores(X)

        # Normalize anomaly scores
        ae_scores = (ae_scores - np.min(ae_scores)) / (np.max(ae_scores) - np.min(ae_scores))

        # Combine predictions (weighted average)
        combined_scores = (0.4 * xgb_pred + 
                         0.2 * (if_pred == -1).astype(float) +
                         0.2 * lr_pred +
                         0.2 * ae_scores)

        return (combined_scores > 0.5).astype(int), combined_scores

    def explain_predictions(self, X):
        """Generate SHAP values for model interpretability"""
        shap_values = self.shap_explainer.shap_values(X)
        return shap_values

    def save_models(self, path):
        """Save all trained models"""
        joblib.dump(self.xgb_model, f"{path}/xgboost_model.joblib")
        joblib.dump(self.isolation_forest, f"{path}/isolation_forest.joblib")
        joblib.dump(self.logistic_regression, f"{path}/logistic_regression.joblib")
        self.autoencoder.save(f"{path}/autoencoder_model")

    def load_models(self, path):
        """Load all saved models"""
        self.xgb_model = joblib.load(f"{path}/xgboost_model.joblib")
        self.isolation_forest = joblib.load(f"{path}/isolation_forest.joblib")
        self.logistic_regression = joblib.load(f"{path}/logistic_regression.joblib")
        self.autoencoder = tf.keras.models.load_model(f"{path}/autoencoder_model")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions, _ = self.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions)) 