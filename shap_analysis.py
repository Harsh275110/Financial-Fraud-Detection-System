"""
SHAP analysis module for model interpretability.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShapAnalyzer:
    """
    Class for analyzing and visualizing SHAP values.
    """
    def __init__(self, model, feature_names=None):
        """
        Initialize the SHAP analyzer.
        
        Args:
            model: Trained model (XGBoost, sklearn, etc.)
            feature_names (list): List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.base_values = None
        self.data = None
        
    def fit(self, data, background_data=None, nsamples=100):
        """
        Create a SHAP explainer for the model.
        
        Args:
            data (pd.DataFrame): Data to explain
            background_data (pd.DataFrame): Background data for explainer (if None, use data sample)
            nsamples (int): Number of samples for background data
            
        Returns:
            self: The fitted explainer
        """
        logger.info("Creating SHAP explainer")
        
        # Store data
        self.data = data
        
        # Set feature names if not provided
        if self.feature_names is None:
            self.feature_names = data.columns.tolist()
        
        # Sample background data if not provided
        if background_data is None:
            background_data = shap.sample(data, nsamples)
        
        # Create explainer based on model type
        try:
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.Explainer(self.model)
            else:
                # For raw XGBoost models
                self.explainer = shap.TreeExplainer(self.model)
            
            # Calculate SHAP values
            self.shap_values = self.explainer(data)
            
            # Store expected value
            if hasattr(self.explainer, 'expected_value'):
                self.expected_value = self.explainer.expected_value
                
            # If explainer has base_values attribute
            if hasattr(self.shap_values, 'base_values'):
                self.base_values = self.shap_values.base_values
                
            logger.info("SHAP explainer created successfully")
            
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {str(e)}")
            raise
            
        return self
    
    def get_feature_importance(self):
        """
        Get global feature importance based on SHAP values.
        
        Returns:
            pd.DataFrame: DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        # Calculate mean absolute SHAP values for each feature
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.shap_values.values).mean(axis=0)
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def plot_summary(self, max_display=20, plot_type="bar", save_path=None):
        """
        Plot SHAP summary plot.
        
        Args:
            max_display (int): Maximum number of features to display
            plot_type (str): Type of plot ('bar', 'dot', 'violin')
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The summary plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        logger.info(f"Creating SHAP summary plot with {max_display} features")
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == "bar":
            shap.plots.bar(self.shap_values, max_display=max_display, show=False)
        elif plot_type == "dot":
            shap.plots.dot(self.shap_values, max_display=max_display, show=False)
        else:  # violin plot
            shap.summary_plot(self.shap_values.values, 
                             self.data, 
                             feature_names=self.feature_names, 
                             max_display=max_display,
                             show=False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_dependence(self, feature, interaction_feature=None, save_path=None):
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            feature (str): Feature to plot
            interaction_feature (str): Interaction feature to color by
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The dependence plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        logger.info(f"Creating SHAP dependence plot for feature '{feature}'")
        
        plt.figure(figsize=(10, 6))
        
        try:
            # Get feature index
            feature_idx = self.feature_names.index(feature)
            
            if interaction_feature:
                interaction_idx = self.feature_names.index(interaction_feature)
                shap.dependence_plot(
                    feature_idx, 
                    self.shap_values.values, 
                    self.data,
                    interaction_index=interaction_idx,
                    feature_names=self.feature_names,
                    show=False
                )
            else:
                shap.dependence_plot(
                    feature_idx, 
                    self.shap_values.values, 
                    self.data,
                    feature_names=self.feature_names,
                    show=False
                )
        except Exception as e:
            logger.error(f"Error creating dependence plot: {str(e)}")
            raise
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dependence plot saved to {save_path}")
        
        return plt.gcf()
    
    def get_explanation_for_instance(self, instance_idx):
        """
        Get explanation for a specific instance.
        
        Args:
            instance_idx (int): Index of the instance to explain
            
        Returns:
            dict: Explanation for the instance
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        # Get SHAP values for the instance
        instance_shap = self.shap_values[instance_idx]
        
        # Create dataframe with feature name, value, and SHAP value
        explanation = pd.DataFrame({
            'feature': self.feature_names,
            'value': self.data.iloc[instance_idx].values,
            'shap_value': instance_shap.values,
            'abs_shap_value': np.abs(instance_shap.values)
        })
        
        # Sort by absolute SHAP value
        explanation = explanation.sort_values('abs_shap_value', ascending=False)
        
        # Add base value
        base_value = instance_shap.base_values if hasattr(instance_shap, 'base_values') else self.expected_value
        
        return {
            'base_value': base_value,
            'features': explanation.to_dict(orient='records'),
            'prediction': float(base_value + np.sum(instance_shap.values))
        }
    
    def export_shap_values(self, output_dir):
        """
        Export SHAP values to CSV files.
        
        Args:
            output_dir (str): Directory to save the output files
            
        Returns:
            tuple: Paths to saved files
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export SHAP values
        shap_values_path = os.path.join(output_dir, "shap_values.csv")
        shap_df = pd.DataFrame(
            self.shap_values.values,
            columns=self.feature_names
        )
        shap_df.to_csv(shap_values_path, index=False)
        
        # Export feature importance
        importance_path = os.path.join(output_dir, "shap_feature_importance.csv")
        self.get_feature_importance().to_csv(importance_path, index=False)
        
        logger.info(f"SHAP values exported to {shap_values_path}")
        logger.info(f"Feature importance exported to {importance_path}")
        
        return shap_values_path, importance_path
    
    def plot_force(self, instance_idx, save_path=None):
        """
        Create a force plot for a specific instance.
        
        Args:
            instance_idx (int): Index of the instance to explain
            save_path (str): Path to save the figure
            
        Returns:
            shap.plots.force: The force plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        # Get SHAP values for the instance
        instance_shap = self.shap_values[instance_idx]
        
        # Create force plot
        plt.figure(figsize=(20, 3))
        force_plot = shap.plots.force(
            instance_shap, 
            matplotlib=True,
            show=False
        )
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        
        return force_plot
    
    def get_shap_values_as_dict(self):
        """
        Get SHAP values as a dictionary suitable for API responses.
        
        Returns:
            dict: SHAP values and metadata
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        # Create a dictionary of feature importance
        importance = self.get_feature_importance()
        feature_importance = {
            row['feature']: float(row['importance']) 
            for _, row in importance.iterrows()
        }
        
        # Get expected value
        expected_value = float(self.expected_value) if self.expected_value is not None else 0.0
        
        # Create response dictionary
        response = {
            'expected_value': expected_value,
            'feature_importance': feature_importance,
            'features': self.feature_names,
            'shap_values': self.shap_values.values.tolist()
        }
        
        return response


def create_shap_analysis(model, X, output_dir=None):
    """
    Create SHAP analysis for a trained model.
    
    Args:
        model: Trained model (XGBoost, sklearn, etc.)
        X (pd.DataFrame): Feature matrix
        output_dir (str): Directory to save outputs
        
    Returns:
        ShapAnalyzer: Fitted SHAP analyzer
    """
    # Initialize and fit SHAP analyzer
    analyzer = ShapAnalyzer(model, feature_names=X.columns.tolist())
    analyzer.fit(X)
    
    # If output directory provided, create visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Summary plot
        summary_path = os.path.join(output_dir, "shap_summary.png")
        analyzer.plot_summary(save_path=summary_path)
        
        # Top 5 feature dependence plots
        top_features = analyzer.get_feature_importance()['feature'].head(5).tolist()
        for feature in top_features:
            dependence_path = os.path.join(output_dir, f"shap_dependence_{feature}.png")
            analyzer.plot_dependence(feature, save_path=dependence_path)
        
        # Export SHAP values
        analyzer.export_shap_values(output_dir)
        
        # Example force plot for the first instance
        force_path = os.path.join(output_dir, "shap_force_example.png")
        analyzer.plot_force(0, save_path=force_path)
    
    return analyzer 