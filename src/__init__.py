"""
ML Trading Prediction System - Source Package

A modular system for predicting financial market direction using machine learning
with comprehensive benchmarking against naive strategies.
"""

__version__ = "1.0.0"
__author__ = "Yanis"
__email__ = "your.email@example.com"

# Import main components for easy access
from .strategies import get_all_naive_strategies, get_strategy_descriptions

__all__ = [
    'get_all_naive_strategies',
    'get_strategy_descriptions'
]
