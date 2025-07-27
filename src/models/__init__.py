"""
ML Models for Trading Prediction System
"""

from .base import BaseMLModel
from .random_forest import RandomForestModel
from .mlp import MLPModel


def get_all_ml_models(**kwargs) -> dict:
    """
    Get all available ML model classes
    
    Returns:
        dict: Dictionary mapping model names to model classes
    """
    return {
        'Random Forest': RandomForestModel,
        'MLP': MLPModel
    }


def get_model_descriptions() -> dict:
    """
    Get human-readable descriptions for each model
    
    Returns:
        dict: Dictionary mapping model keys to descriptions
    """
    return {
        'Random Forest': 'Random Forest',
        'MLP': 'MLP'
    }


__all__ = [
    'BaseMLModel', 
    'RandomForestModel', 
    'MLPModel',
    'get_all_ml_models',
    'get_model_descriptions'
]
