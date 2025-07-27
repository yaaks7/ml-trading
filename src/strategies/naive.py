"""
Naive benchmark strategies for ML Trading Prediction System

This module implements simple baseline strategies that serve as benchmarks
for comparing the performance of more sophisticated ML models.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseNaiveStrategy(ABC):
    """
    Abstract base class for all naive strategies
    """
    
    def __init__(self, name: str, random_state: Optional[int] = None):
        """
        Initialize base naive strategy
        
        Args:
            name: Strategy name for identification
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self.is_fitted = False
        self.strategy_info = {}
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseNaiveStrategy':
        """
        Fit the strategy on training data
        
        Args:
            X: Feature matrix (not used by naive strategies but kept for consistency)
            y: Target vector (direction: 1 for up, 0 for down)
            
        Returns:
            self: Fitted strategy
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Feature matrix (shape: n_samples, n_features)
            
        Returns:
            Array of predictions (1 for up, 0 for down)
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for [down, up]
        """
        predictions = self.predict(X)
        # Convert binary predictions to probabilities
        proba = np.zeros((len(predictions), 2))
        proba[:, 1] = predictions  # Probability of up
        proba[:, 0] = 1 - predictions  # Probability of down
        return proba
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and parameters"""
        return {
            'name': self.name,
            'type': 'naive_strategy',
            'is_fitted': self.is_fitted,
            **self.strategy_info
        }


class AlwaysUpStrategy(BaseNaiveStrategy):
    """
    Naive strategy that always predicts market will go up
    
    This represents the simplest possible bullish bias and serves as
    a baseline for any market with a long-term upward trend.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Bullish", random_state)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AlwaysUpStrategy':
        """
        Fit the strategy (no-op for this strategy)
        
        Args:
            X: Feature matrix (unused)
            y: Target vector (used only to compute training statistics)
            
        Returns:
            self: Fitted strategy
        """
        # Calculate some statistics for information
        up_ratio = y.mean()
        self.strategy_info = {
            'description': 'Always predicts market will go up (bullish bias)',
            'training_up_ratio': up_ratio,
            'expected_accuracy': up_ratio,
            'prediction_value': 1
        }
        
        self.is_fitted = True
        logger.info(f"AlwaysUpStrategy fitted. Training up ratio: {up_ratio:.3f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict always up (1) for all samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of ones (all up predictions)
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        return np.ones(len(X), dtype=int)


class AlwaysDownStrategy(BaseNaiveStrategy):
    """
    Naive strategy that always predicts market will go down
    
    This represents a pessimistic baseline, useful for markets
    in downtrends or as a contrarian benchmark.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Bearish", random_state)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AlwaysDownStrategy':
        """
        Fit the strategy (no-op for this strategy)
        
        Args:
            X: Feature matrix (unused)
            y: Target vector (used only to compute training statistics)
            
        Returns:
            self: Fitted strategy
        """
        # Calculate some statistics for information
        down_ratio = 1 - y.mean()
        self.strategy_info = {
            'description': 'Always predicts market will go down (bearish bias)',
            'training_down_ratio': down_ratio,
            'expected_accuracy': down_ratio,
            'prediction_value': 0
        }
        
        self.is_fitted = True
        logger.info(f"AlwaysDownStrategy fitted. Training down ratio: {down_ratio:.3f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict always down (0) for all samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of zeros (all down predictions)
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        return np.zeros(len(X), dtype=int)


class RandomStrategy(BaseNaiveStrategy):
    """
    Naive strategy that makes random predictions with 50/50 probability
    
    This represents the null hypothesis and provides a baseline
    that any meaningful strategy should beat.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Random", random_state)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomStrategy':
        """
        Fit the strategy (no-op for this strategy)
        
        Args:
            X: Feature matrix (unused)
            y: Target vector (used only for statistics)
            
        Returns:
            self: Fitted strategy
        """
        self.strategy_info = {
            'description': 'Random predictions with 50% probability each direction',
            'up_probability': 0.5,
            'down_probability': 0.5,
            'expected_accuracy': 0.5,
            'random_state': self.random_state
        }
        
        self.is_fitted = True
        logger.info("RandomStrategy fitted with 50/50 probability")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate random predictions with 50% probability for up/down
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of random predictions (1 for up, 0 for down)
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        return np.random.choice([0, 1], size=len(X))
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities (always 50/50)
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with 0.5 probability for each class
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        proba = np.full((len(X), 2), 0.5)
        return proba


class HistoricalFrequencyStrategy(BaseNaiveStrategy):
    """
    Naive strategy based on historical frequency of up/down movements
    
    This strategy predicts based on empirical probabilities from training data.
    For example, if 57% of historical days were "up", it predicts "up" with 57% probability.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Frequency", random_state)
        self.up_probability = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HistoricalFrequencyStrategy':
        """
        Fit the strategy by calculating historical up/down frequencies
        
        Args:
            X: Feature matrix (unused)
            y: Target vector (1 for up, 0 for down)
            
        Returns:
            self: Fitted strategy
        """
        # Calculate empirical probability of up movements
        self.up_probability = y.mean()
        
        self.strategy_info = {
            'description': f'Predicts based on historical frequency: {self.up_probability:.1%} up',
            'up_probability': self.up_probability,
            'down_probability': 1 - self.up_probability,
            'expected_accuracy': max(self.up_probability, 1 - self.up_probability),
            'training_samples': len(y),
            'training_up_count': y.sum(),
            'training_down_count': len(y) - y.sum()
        }
        
        self.is_fitted = True
        logger.info(f"HistoricalFrequencyStrategy fitted. Up probability: {self.up_probability:.3f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions based on historical frequency
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions based on historical probabilities
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        # Generate random predictions with historical probability
        return np.random.choice([0, 1], size=len(X), p=[1 - self.up_probability, self.up_probability])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate prediction probabilities based on historical frequency
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with historical probabilities
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        # All samples get the same probability distribution
        proba = np.full((len(X), 2), [1 - self.up_probability, self.up_probability])
        return proba


class MomentumStrategy(BaseNaiveStrategy):
    """
    Naive momentum strategy that predicts based on the last movement
    
    This strategy assumes that the market will continue in the same direction
    as the most recent movement (persistence of momentum).
    """
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Momentum (Last Direction)", random_state)
        self.last_direction = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MomentumStrategy':
        """
        Fit the strategy by storing the last direction
        
        Args:
            X: Feature matrix (unused)
            y: Target vector (1 for up, 0 for down)
            
        Returns:
            self: Fitted strategy
        """
        # Store the last observed direction
        self.last_direction = int(y.iloc[-1])
        
        # Calculate momentum persistence in training data
        momentum_accuracy = (y.iloc[1:].values == y.iloc[:-1].values).mean()
        
        self.strategy_info = {
            'description': 'Predicts market will continue in the last observed direction',
            'last_training_direction': 'up' if self.last_direction == 1 else 'down',
            'momentum_persistence': momentum_accuracy,
            'expected_accuracy': momentum_accuracy,
        }
        
        self.is_fitted = True
        logger.info(f"MomentumStrategy fitted. Last direction: {'up' if self.last_direction else 'down'}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict based on last observed direction
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (all same as last direction)
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        return np.full(len(X), self.last_direction, dtype=int)


class MeanReversionStrategy(BaseNaiveStrategy):
    """
    Naive mean reversion strategy that predicts opposite of last movement
    
    This strategy assumes that the market will reverse direction
    (contrarian approach based on mean reversion).
    """
    
    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Mean Reversion (Contrarian)", random_state)
        self.last_direction = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MeanReversionStrategy':
        """
        Fit the strategy by storing the last direction
        
        Args:
            X: Feature matrix (unused)
            y: Target vector (1 for up, 0 for down)
            
        Returns:
            self: Fitted strategy
        """
        # Store the last observed direction
        self.last_direction = int(y.iloc[-1])
        
        # Calculate mean reversion accuracy in training data
        reversion_accuracy = (y.iloc[1:].values != y.iloc[:-1].values).mean()
        
        self.strategy_info = {
            'description': 'Predicts market will reverse from the last observed direction',
            'last_training_direction': 'up' if self.last_direction == 1 else 'down',
            'predicted_direction': 'down' if self.last_direction == 1 else 'up',
            'reversion_tendency': reversion_accuracy,
            'expected_accuracy': reversion_accuracy,
        }
        
        self.is_fitted = True
        logger.info(f"MeanReversionStrategy fitted. Predicting opposite of: {'up' if self.last_direction else 'down'}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict opposite of last observed direction
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (opposite of last direction)
        """
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        # Predict opposite direction
        opposite_direction = 1 - self.last_direction
        return np.full(len(X), opposite_direction, dtype=int)


def get_all_naive_strategies(random_state: Optional[int] = None) -> Dict[str, type]:
    """
    Get all available naive strategies
    
    Args:
        random_state: Random seed for reproducibility (not used when returning classes)
        
    Returns:
        Dictionary mapping strategy names to strategy classes
    """
    strategies = {
        'Bullish': AlwaysUpStrategy,
        'Bearish': AlwaysDownStrategy,
        'Random': RandomStrategy,
        'Frequency': HistoricalFrequencyStrategy,
        'Momentum (Last Direction)': MomentumStrategy,
        'Mean Reversion (Contrarian)': MeanReversionStrategy
    }
    
    return strategies


def get_strategy_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all naive strategies
    
    Returns:
        Dictionary mapping strategy names to their descriptions
    """
    return {
        'Bullish': 'Bullish',
        'Bearish': 'Bearish',
        'Random': 'Random',
        'Frequency': 'Frequency',
        'Momentum (Last Direction)': 'Momentum',
        'Mean Reversion (Contrarian)': 'Mean Reversion'
    }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample target data (60% up days)
    y_train = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    y_train = pd.Series(y_train, name='direction')
    
    # Generate dummy features
    X_train = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'feature_{i}' for i in range(5)])
    X_test = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    
    # Test all strategies
    strategies = get_all_naive_strategies(random_state=42)
    
    print("Testing Naive Strategies:")
    print("=" * 50)
    
    for name, strategy in strategies.items():
        # Fit and predict
        strategy.fit(X_train, y_train)
        predictions = strategy.predict(X_test)
        probabilities = strategy.predict_proba(X_test)
        
        # Get strategy info
        info = strategy.get_strategy_info()
        
        print(f"\n{strategy.name}:")
        print(f"  Description: {info['description']}")
        print(f"  Predictions: {predictions[:10]}...")  # First 10 predictions
        print(f"  Up probability: {probabilities[0, 1]:.3f}")
        if 'expected_accuracy' in info:
            print(f"  Expected accuracy: {info['expected_accuracy']:.3f}")
