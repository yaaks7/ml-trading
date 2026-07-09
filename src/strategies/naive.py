"""
Naive baseline strategies — see docs/BENCHMARK_STRATEGIES.md for why these six
and not something like Buy & Hold or a Sharpe-ratio comparison.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseNaiveStrategy(ABC):
    """Shared interface for the naive strategies below (fit/predict, sklearn-style)."""

    def __init__(self, name: str, random_state: Optional[int] = None):
        self.name = name
        self.random_state = random_state
        self.is_fitted = False
        self.strategy_info = {}

        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseNaiveStrategy':
        """X is ignored — these strategies only look at y (or nothing at all)."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions (1 = up, 0 = down), one per row of X."""
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Hard 0/1 predictions recast as (n_samples, 2) probabilities."""
        predictions = self.predict(X)
        proba = np.zeros((len(predictions), 2))
        proba[:, 1] = predictions
        proba[:, 0] = 1 - predictions
        return proba
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Name, fit state, and whatever stats `fit()` recorded in `strategy_info`."""
        return {
            'name': self.name,
            'type': 'naive_strategy',
            'is_fitted': self.is_fitted,
            **self.strategy_info
        }


class AlwaysUpStrategy(BaseNaiveStrategy):
    """Always predicts up. Baseline for an asset with a long-term upward drift."""

    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Bullish", random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AlwaysUpStrategy':
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
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        return np.ones(len(X), dtype=int)


class AlwaysDownStrategy(BaseNaiveStrategy):
    """Always predicts down. Mirror of AlwaysUpStrategy."""

    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Bearish", random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AlwaysDownStrategy':
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
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        return np.zeros(len(X), dtype=int)


class RandomStrategy(BaseNaiveStrategy):
    """50/50 coin flip. The null hypothesis — anything should beat this."""

    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Random", random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomStrategy':
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
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        return np.random.choice([0, 1], size=len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        return np.full((len(X), 2), 0.5)


class HistoricalFrequencyStrategy(BaseNaiveStrategy):
    """Samples from the training set's up/down ratio (e.g. predicts up 57% of the time if 57% of training days were up)."""

    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Frequency", random_state)
        self.up_probability = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'HistoricalFrequencyStrategy':
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
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        return np.random.choice([0, 1], size=len(X), p=[1 - self.up_probability, self.up_probability])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        return np.full((len(X), 2), [1 - self.up_probability, self.up_probability])


class MomentumStrategy(BaseNaiveStrategy):
    """Bets that the most recent direction continues, re-evaluated at every row.

    Unlike the other naive strategies, this one does look at X — specifically its
    'Returns' column (added by DataFetcher for every real feature matrix). That's
    what makes it walk-forward: predict(X_test) uses each row's own most-recent
    price move, not a single value frozen at the end of training. Without this it
    degenerates into a Bullish/Bearish clone — one constant prediction for the
    whole test set, whatever the last training-set day happened to be.
    """

    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Momentum (Last Direction)", random_state)
        self.last_direction = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MomentumStrategy':
        self.last_direction = int(y.iloc[-1])
        momentum_accuracy = (y.iloc[1:].values == y.iloc[:-1].values).mean()
        self.strategy_info = {
            'description': 'Predicts market will continue in the most recently observed direction',
            'last_training_direction': 'up' if self.last_direction == 1 else 'down',
            'momentum_persistence': momentum_accuracy,
            'expected_accuracy': momentum_accuracy,
        }
        self.is_fitted = True
        logger.info(f"MomentumStrategy fitted. Last training-set direction: {'up' if self.last_direction else 'down'}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        if 'Returns' not in X.columns:
            raise ValueError("MomentumStrategy needs a 'Returns' column in X (see DataFetcher)")
        return (X['Returns'] > 0).astype(int).values


class MeanReversionStrategy(BaseNaiveStrategy):
    """Bets that the most recent direction reverses, re-evaluated at every row.

    Mirror of MomentumStrategy — same walk-forward dependency on X['Returns'],
    same reason for it (see MomentumStrategy's docstring).
    """

    def __init__(self, random_state: Optional[int] = None):
        super().__init__("Mean Reversion (Contrarian)", random_state)
        self.last_direction = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MeanReversionStrategy':
        self.last_direction = int(y.iloc[-1])
        reversion_accuracy = (y.iloc[1:].values != y.iloc[:-1].values).mean()
        self.strategy_info = {
            'description': 'Predicts market will reverse from the most recently observed direction',
            'last_training_direction': 'up' if self.last_direction == 1 else 'down',
            'predicted_direction': 'down' if self.last_direction == 1 else 'up',
            'reversion_tendency': reversion_accuracy,
            'expected_accuracy': reversion_accuracy,
        }
        self.is_fitted = True
        logger.info(f"MeanReversionStrategy fitted. Last training-set direction: {'up' if self.last_direction else 'down'}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        if 'Returns' not in X.columns:
            raise ValueError("MeanReversionStrategy needs a 'Returns' column in X (see DataFetcher)")
        return (X['Returns'] <= 0).astype(int).values


def get_all_naive_strategies(random_state: Optional[int] = None) -> Dict[str, type]:
    """Strategy classes keyed by name. `random_state` is accepted for API symmetry with
    get_all_ml_models() but unused here — pass it to the class constructor instead."""
    return {
        'Bullish': AlwaysUpStrategy,
        'Bearish': AlwaysDownStrategy,
        'Random': RandomStrategy,
        'Frequency': HistoricalFrequencyStrategy,
        'Momentum (Last Direction)': MomentumStrategy,
        'Mean Reversion (Contrarian)': MeanReversionStrategy
    }


def get_strategy_descriptions() -> Dict[str, str]:
    """Short display name for each strategy key."""
    return {
        'Bullish': 'Bullish',
        'Bearish': 'Bearish',
        'Random': 'Random',
        'Frequency': 'Frequency',
        'Momentum (Last Direction)': 'Momentum',
        'Mean Reversion (Contrarian)': 'Mean Reversion'
    }
