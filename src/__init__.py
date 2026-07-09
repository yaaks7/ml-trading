"""Data fetching, ML models, and naive baseline strategies for next-day direction prediction."""

__version__ = "1.0.0"
__author__ = "Yanis Aksas"
__email__ = "yanis.aksas@gmail.com"

# Import main components for easy access
from .strategies import get_all_naive_strategies, get_strategy_descriptions

__all__ = [
    'get_all_naive_strategies',
    'get_strategy_descriptions'
]
