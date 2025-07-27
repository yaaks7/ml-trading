"""
Random Forest model for trading predictions
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base import BaseMLModel


class RandomForestModel(BaseMLModel):
    """Random Forest classifier for trading predictions"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            random_state: Random state for reproducibility
        """
        super().__init__("Random Forest", **kwargs)
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state
        }
        
        self.model = self.create_model(**self.params)
    
    def create_model(self, **kwargs):
        """Create Random Forest classifier"""
        return RandomForestClassifier(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit Random Forest model"""
        return super().fit(X, y, **kwargs)
    
    def get_model_info(self) -> dict:
        """Get detailed model information"""
        info = {
            'model_type': 'Random Forest',
            'n_estimators': self.params['n_estimators'],
            'max_depth': self.params['max_depth'],
            'min_samples_split': self.params['min_samples_split'],
            'min_samples_leaf': self.params['min_samples_leaf'],
            'max_features': self.params['max_features']
        }
        
        if self.is_fitted:
            info.update({
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names[:10] if self.feature_names else None,  # First 10 features
                'oob_score': getattr(self.model, 'oob_score_', None)
            })
        
        return info
