import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_engine.outliers import OutlierTrimmer
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.outlier_trimmer = OutlierTrimmer(capping_method='iqr', tail='both', fold=1.5)
        self.smote = SMOTE(random_state=42)

    def load_data(self, filepath):
        """Load data from CSV file"""
        return pd.read_csv(filepath)

    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        return df

    def handle_outliers(self, df, columns):
        """Handle outliers using IQR method"""
        self.outlier_trimmer.fit(df[columns])
        df[columns] = self.outlier_trimmer.transform(df[columns])
        return df

    def feature_engineering(self, df):
        """Create new features from existing ones"""
        # Time-based features
        df['hour'] = pd.to_datetime(df['transaction_datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['transaction_datetime']).dt.dayofweek
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_rounded'] = df['amount'].round(-1)
        
        # Transaction frequency features
        df['transaction_freq'] = df.groupby('customer_id')['transaction_id'].transform('count')
        
        return df

    def encode_categorical_features(self, df):
        """Encode categorical features using one-hot encoding"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_columns)
        return df

    def scale_features(self, df, columns):
        """Scale numerical features"""
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def balance_dataset(self, X, y):
        """Balance the dataset using SMOTE"""
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def prepare_data(self, df, target_column='is_fraud'):
        """Complete data preparation pipeline"""
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Handle outliers in numerical columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df = self.handle_outliers(df, numerical_columns)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale features
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        # Balance dataset
        X_resampled, y_resampled = self.balance_dataset(X, y)
        
        return X_resampled, y_resampled 