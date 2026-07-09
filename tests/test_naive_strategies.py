"""Tests for the naive benchmark strategies, using synthetic data (no network)."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.strategies import get_all_naive_strategies


@pytest.fixture
def synthetic_data():
    """A mildly bullish synthetic market (60% up days) with unrelated features —
    the strategies below don't use the features at all, only the target."""
    rng = np.random.RandomState(42)

    y_train = pd.Series(rng.choice([0, 1], size=1000, p=[0.4, 0.6]))
    y_test = pd.Series(rng.choice([0, 1], size=200, p=[0.4, 0.6]))
    X_train = pd.DataFrame(rng.randn(1000, 10))
    X_test = pd.DataFrame(rng.randn(200, 10))

    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize("name", list(get_all_naive_strategies().keys()))
def test_strategy_fits_and_predicts(name, synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    strategy_class = get_all_naive_strategies()[name]

    strategy = strategy_class(random_state=42)
    strategy.fit(X_train, y_train)
    assert strategy.is_fitted

    predictions = strategy.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(np.unique(predictions)).issubset({0, 1})

    accuracy = accuracy_score(y_test, predictions)
    assert 0.0 <= accuracy <= 1.0


@pytest.mark.parametrize("name", list(get_all_naive_strategies().keys()))
def test_predict_proba_shape(name, synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    strategy = get_all_naive_strategies()[name](random_state=42)
    strategy.fit(X_train, y_train)

    proba = strategy.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bullish_always_predicts_up(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    strategy = get_all_naive_strategies()['Bullish']()
    strategy.fit(X_train, y_train)
    assert (strategy.predict(X_test) == 1).all()


def test_bearish_always_predicts_down(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    strategy = get_all_naive_strategies()['Bearish']()
    strategy.fit(X_train, y_train)
    assert (strategy.predict(X_test) == 0).all()


def test_predict_before_fit_raises(synthetic_data):
    _, X_test, _, _ = synthetic_data
    strategy = get_all_naive_strategies()['Random']()
    with pytest.raises(ValueError):
        strategy.predict(X_test)
