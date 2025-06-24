import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from feature_engine.outliers import OutlierTrimmer
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class to handle all data preprocessing tasks for fraud detection.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the DataPreprocessor with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing preprocessing parameters
        """
        self.config = config or {
            'test_size': 0.2,
            'random_state': 42,
            'outlier_threshold': 3,
            'categorical_features': [],
            'numerical_features': ['amount', 'time', 'n_transactions_day'],
            'target_column': 'is_fraud'
        }
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.outlier_detector = OutlierTrimmer(
            capping_method='iqr',
            tail='both',
            fold=self.config['outlier_threshold']
        )

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats (CSV, JSON, etc.).
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        # For numerical features, fill with median
        for col in self.config['numerical_features']:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical features, fill with mode
        for col in self.config['categorical_features']:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info("Successfully handled missing values")
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in self.config['categorical_features']:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        logger.info("Successfully encoded categorical features")
        return df_encoded

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in numerical features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled outliers
        """
        df_clean = df.copy()
        
        # Apply outlier detection and handling only to numerical features
        self.outlier_detector.fit(df_clean[self.config['numerical_features']])
        df_clean[self.config['numerical_features']] = self.outlier_detector.transform(
            df_clean[self.config['numerical_features']]
        )
        
        logger.info("Successfully handled outliers")
        return df_clean

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        df_scaled = df.copy()
        df_scaled[self.config['numerical_features']] = self.scaler.fit_transform(
            df[self.config['numerical_features']]
        )
        
        logger.info("Successfully scaled numerical features")
        return df_scaled

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features for fraud detection.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        df_features = df.copy()
        
        # Add time-based features
        if 'time' in df.columns:
            # Fill missing values with median before converting to int
            df_features['time'] = df_features['time'].fillna(df_features['time'].median())
            df_features['hour_of_day'] = df_features['time'].round().clip(0, 23).astype(int)
            df_features['is_night'] = (df_features['hour_of_day'] >= 22) | (df_features['hour_of_day'] <= 5)
        
        # Add amount-based features
        if 'amount' in df.columns:
            df_features['amount'] = df_features['amount'].fillna(df_features['amount'].median())
            df_features['amount_log'] = np.log1p(df_features['amount'])
        
        # Add transaction frequency features
        if 'n_transactions_day' in df.columns:
            df_features['n_transactions_day'] = df_features['n_transactions_day'].fillna(df_features['n_transactions_day'].median())
            df_features['high_frequency'] = df_features['n_transactions_day'] > df_features['n_transactions_day'].quantile(0.95)
        
        logger.info("Successfully created additional features")
        return df_features

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training by applying all preprocessing steps.
        
        Args:
            df (pd.DataFrame): Raw input DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        try:
            # Remove ID columns that shouldn't be used as features
            df_processed = df.drop(['customer_id', 'transaction_id'], axis=1, errors='ignore')
            
            # Apply preprocessing steps
            df_processed = self.handle_missing_values(df_processed)
            df_processed = self.handle_outliers(df_processed)
            df_processed = self.encode_categorical_features(df_processed)
            df_processed = self.create_features(df_processed)
            
            # Separate features and target
            X = df_processed.drop(self.config['target_column'], axis=1)
            y = df_processed[self.config['target_column']]
            
            # Scale features
            X_scaled = self.scale_features(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y
            )
            
            logger.info("Successfully prepared data for model training")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise 