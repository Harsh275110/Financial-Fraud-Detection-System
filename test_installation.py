import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import shap
import plotly.express as px
from sklearn.metrics import classification_report
import tensorflow as tf
import os

# Generate sample fraud data
np.random.seed(42)
n_samples = 1000

# Create normal transactions
normal_data = {
    'amount': np.random.normal(100, 50, int(n_samples * 0.95)),
    'time': np.random.uniform(0, 24, int(n_samples * 0.95)),
    'n_transactions_day': np.random.poisson(3, int(n_samples * 0.95)),
    'is_fraud': np.zeros(int(n_samples * 0.95))
}

# Create fraudulent transactions
fraud_data = {
    'amount': np.random.normal(500, 100, int(n_samples * 0.05)),
    'time': np.random.uniform(0, 24, int(n_samples * 0.05)),
    'n_transactions_day': np.random.poisson(10, int(n_samples * 0.05)),
    'is_fraud': np.ones(int(n_samples * 0.05))
}

# Combine normal and fraudulent transactions
df_normal = pd.DataFrame(normal_data)
df_fraud = pd.DataFrame(fraud_data)
df = pd.concat([df_normal, df_fraud], ignore_index=True)

# Split features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTesting Isolation Forest...")
# Train Isolation Forest
iso_forest = IsolationForest(random_state=42, contamination=0.05)
iso_forest.fit(X_train_scaled)
iso_pred = iso_forest.predict(X_test_scaled)
iso_pred = np.where(iso_pred == 1, 0, 1)  # Convert predictions to binary (0: normal, 1: fraud)

print("\nIsolation Forest Results:")
print(classification_report(y_test, iso_pred))

print("\nTesting XGBoost...")
# Train XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

print("\nXGBoost Results:")
print(classification_report(y_test, xgb_pred))

print("\nTesting SHAP...")
# Calculate SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

print("\nSHAP values shape:", shap_values.shape)

print("\nTesting TensorFlow...")
# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nNeural Network Summary:")
model.summary()

print("\nAll package tests completed successfully!")

# Create visualization
fig = px.scatter(df, x='amount', y='n_transactions_day', color='is_fraud',
                title='Transaction Amount vs Number of Daily Transactions',
                labels={'is_fraud': 'Is Fraud', 'amount': 'Amount', 
                       'n_transactions_day': 'Number of Transactions per Day'})

# Ensure the directory exists
os.makedirs('tests', exist_ok=True)
fig.write_html("tests/fraud_visualization.html") 