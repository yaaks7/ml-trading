from .base import BaseMLModel
from .random_forest import RandomForestModel
from .mlp import MLPModel


def get_all_ml_models(**kwargs) -> dict:
    """Model classes keyed by display name."""
    return {
        'Random Forest': RandomForestModel,
        'MLP': MLPModel
    }


def get_model_descriptions() -> dict:
    """Short display name for each model key."""
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
