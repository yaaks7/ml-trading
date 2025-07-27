"""
Strategy modules for ML Trading Prediction System
"""

from .naive import (
    BaseNaiveStrategy,
    AlwaysUpStrategy,
    AlwaysDownStrategy,
    RandomStrategy,
    HistoricalFrequencyStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    get_all_naive_strategies,
    get_strategy_descriptions
)

__all__ = [
    'BaseNaiveStrategy',
    'AlwaysUpStrategy',
    'AlwaysDownStrategy',
    'RandomStrategy',
    'HistoricalFrequencyStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'get_all_naive_strategies',
    'get_strategy_descriptions'
]
