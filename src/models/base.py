"""
Base classes for ML models
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib
import os


class BaseMLModel(ABC):
    """Base class for all ML models"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def create_model(self, **kwargs):
        """Create the underlying model with given parameters"""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseMLModel':
        """Fit the model to training data"""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} must be fitted before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[predictions == 0, 0] = 1.0
            proba[predictions == 1, 1] = 1.0
            return proba
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_names, abs(self.model.coef_[0])))
        else:
            return None
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> 'BaseMLModel':
        """Load a trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.name = model_data['name']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def set_params(self, **params) -> 'BaseMLModel':
        """Set model parameters"""
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self
    
    def __str__(self) -> str:
        return f"{self.name} (fitted: {self.is_fitted})"
    
    def __repr__(self) -> str:
        return self.__str__()
