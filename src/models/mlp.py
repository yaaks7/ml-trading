"""
Multi-Layer Perceptron (Neural Network) model for trading predictions
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .base import BaseMLModel


class MLPModel(BaseMLModel):
    """Multi-Layer Perceptron classifier for trading predictions"""
    
    def __init__(self, 
                 hidden_layer_sizes: tuple = (100, 50),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 learning_rate: str = 'constant',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 1000,
                 random_state: int = 42,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 **kwargs):
        super().__init__("MLP", **kwargs)
        
        self.params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'learning_rate_init': learning_rate_init,
            'max_iter': max_iter,
            'random_state': random_state,
            'early_stopping': early_stopping,
            'validation_fraction': validation_fraction
        }
        
        self.model = self.create_model(**self.params)
        self.scaler = StandardScaler()

    def create_model(self, **kwargs):
        """Create MLP classifier"""
        return MLPClassifier(**kwargs)

    def _scale(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Standardize features before they hit the network.

        The raw feature matrix mixes wildly different scales — Volume in the
        billions next to ratio features around 1.0 — which starves gradient
        descent: pre-activations blow up, the loss diverges, and the model
        converges to a degenerate constant-ish prediction instead of learning
        anything. Random Forest doesn't need this (it's scale-invariant), so
        this scaling lives here rather than in DataFetcher.
        """
        values = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return pd.DataFrame(values, columns=X.columns, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit MLP model"""
        return super().fit(self._scale(X, fit=True), y, **kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return super().predict(self._scale(X, fit=False))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return super().predict_proba(self._scale(X, fit=False))

    def get_model_info(self) -> dict:
        """Get detailed model information"""
        info = {
            'model_type': 'MLP',
            'hidden_layer_sizes': self.params['hidden_layer_sizes'],
            'activation': self.params['activation'],
            'solver': self.params['solver'],
            'alpha': self.params['alpha'],
            'learning_rate': self.params['learning_rate'],
            'learning_rate_init': self.params['learning_rate_init'],
            'max_iter': self.params['max_iter'],
            'early_stopping': self.params['early_stopping']
        }
        
        if self.is_fitted:
            info.update({
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names[:10] if self.feature_names else None,  # First 10 features
                'n_layers': len(self.model.coefs_),
                'n_iter': getattr(self.model, 'n_iter_', None),
                'loss': getattr(self.model, 'loss_', None)
            })
        
        return info
