"""
Multi-Layer Perceptron (Neural Network) model for trading predictions
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
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
        """
        Initialize MLP model
        
        Args:
            hidden_layer_sizes: Tuple defining the number of neurons in each hidden layer
            activation: Activation function ('relu', 'tanh', 'logistic')
            solver: Solver for weight optimization ('adam', 'lbfgs', 'sgd')
            alpha: L2 penalty (regularization term) parameter
            learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive')
            learning_rate_init: Initial learning rate
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
            early_stopping: Whether to use early stopping
            validation_fraction: Fraction of training data to use for validation
        """
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
    
    def create_model(self, **kwargs):
        """Create MLP classifier"""
        return MLPClassifier(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit MLP model"""
        return super().fit(X, y, **kwargs)
    
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
