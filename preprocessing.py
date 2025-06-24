"""
Data preprocessing module for financial fraud detection.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionPreprocessor:
    """
    Class to preprocess financial transaction data for fraud detection.
    """
    def __init__(self):
        self.numerical_features = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 
            'oldbalanceDest', 'newbalanceDest'
        ]
        self.categorical_features = ['type']
        self.time_features = ['step']
        self.target = 'isFraud'
        self.preprocessor = None
        
    def fit(self, data):
        """
        Fit the preprocessing pipeline on training data.
        
        Args:
            data (pd.DataFrame): Raw transaction data
            
        Returns:
            self: The fitted preprocessor
        """
        logger.info("Fitting preprocessing pipeline")
        
        # Create preprocessing steps for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessing steps for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop other columns not specified
        )
        
        # Fit the preprocessor on the data
        self.preprocessor.fit(data)
        
        logger.info("Preprocessing pipeline fitted successfully")
        return self
    
    def transform(self, data):
        """
        Transform data using the fitted preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info(f"Transforming data with shape {data.shape}")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
        
        # Transform the data
        transformed_data = self.preprocessor.transform(data)
        
        # Get feature names from one-hot encoding
        cat_feature_names = []
        for i, transformer in enumerate(self.preprocessor.transformers_):
            if transformer[0] == 'cat':
                cat_feature_names = list(
                    transformer[1].named_steps['onehot'].get_feature_names_out(self.categorical_features)
                )
        
        # Create DataFrame with appropriate column names
        feature_names = self.numerical_features + cat_feature_names
        transformed_df = pd.DataFrame(
            transformed_data, 
            columns=feature_names, 
            index=data.index
        )
        
        logger.info(f"Data transformed successfully, shape: {transformed_df.shape}")
        return transformed_df
    
    def fit_transform(self, data):
        """
        Fit the preprocessing pipeline and transform the data in one step.
        
        Args:
            data (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        return self.fit(data).transform(data)
    
    def create_features(self, data):
        """
        Create additional features from transaction data.
        
        Args:
            data (pd.DataFrame): Original transaction data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        logger.info("Creating additional features")
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Create transaction velocity features
        df['transaction_velocity'] = df.groupby('nameOrig')['step'].diff().fillna(0)
        
        # Create amount ratio features
        df['amount_to_oldbalanceOrg_ratio'] = df['amount'] / df['oldbalanceOrg'].replace(0, 0.01)
        df['amount_to_newbalanceOrig_ratio'] = df['amount'] / df['newbalanceOrig'].replace(0, 0.01)
        
        # Create balance change features
        df['orig_balance_delta'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['dest_balance_delta'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Flag suspicious cases (amount doesn't match balance change)
        df['suspicious_transfer'] = ((df['type'] == 'TRANSFER') & 
                                   (abs(df['amount'] - abs(df['orig_balance_delta'])) > 0.1)).astype(int)
        
        logger.info(f"Created features, new shape: {df.shape}")
        return df
    
    def handle_class_imbalance(self, X, y, method='smote', random_state=42):
        """
        Handle class imbalance in the fraud detection dataset.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Method to use ('smote', 'adasyn', 'random_oversample')
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        logger.info(f"Handling class imbalance using {method}")
        
        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
        
        if method == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        elif method == 'random_oversample':
            sampler = RandomOverSampler(random_state=random_state)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logger.info(f"Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        logger.info(f"Resampled class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        
        return X_resampled, y_resampled


def load_and_preprocess(file_path, preprocess=True, create_features=True, handle_imbalance=False):
    """
    Load and preprocess financial transaction data.
    
    Args:
        file_path (str): Path to the raw data file
        preprocess (bool): Whether to apply preprocessing
        create_features (bool): Whether to create additional features
        handle_imbalance (bool): Whether to handle class imbalance
        
    Returns:
        tuple: (X, y, preprocessor) where X is the feature matrix,
               y is the target vector, and preprocessor is the fitted preprocessor
    """
    logger.info(f"Loading data from {file_path}")
    
    # Load the data
    data = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape {data.shape}")
    
    # Initialize preprocessor
    preprocessor = TransactionPreprocessor()
    
    # Extract target variable
    if preprocessor.target in data.columns:
        y = data[preprocessor.target]
        X = data.drop(preprocessor.target, axis=1)
    else:
        logger.warning(f"Target column '{preprocessor.target}' not found in data")
        y = None
        X = data
    
    # Create additional features
    if create_features:
        X = preprocessor.create_features(X)
    
    # Apply preprocessing
    if preprocess:
        X = preprocessor.fit_transform(X)
    
    # Handle class imbalance
    if handle_imbalance and y is not None:
        X, y = preprocessor.handle_class_imbalance(X, y)
    
    logger.info(f"Data processing completed. X shape: {X.shape}")
    if y is not None:
        logger.info(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, preprocessor 