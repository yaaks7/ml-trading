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
    """A mildly bullish synthetic market (60% up days) with unrelated filler
    features — most strategies below ignore X entirely and only look at the
    target. The exception is Momentum/MeanReversion, which read the 'Returns'
    column (as DataFetcher always provides on real data), so it's included here
    too even though its values are independent of y in this fixture."""
    rng = np.random.RandomState(42)

    y_train = pd.Series(rng.choice([0, 1], size=1000, p=[0.4, 0.6]))
    y_test = pd.Series(rng.choice([0, 1], size=200, p=[0.4, 0.6]))
    X_train = pd.DataFrame(rng.randn(1000, 10))
    X_test = pd.DataFrame(rng.randn(200, 10))
    X_train['Returns'] = rng.randn(1000) * 0.01
    X_test['Returns'] = rng.randn(200) * 0.01

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


def test_momentum_is_walk_forward_not_frozen(synthetic_data):
    """Regression test: Momentum used to store a single direction from the last
    training row and repeat it for the whole test set (indistinguishable from
    Bullish/Bearish). It should now track X['Returns'] row by row instead."""
    X_train, X_test, y_train, y_test = synthetic_data
    strategy = get_all_naive_strategies()['Momentum (Last Direction)']()
    strategy.fit(X_train, y_train)

    predictions = strategy.predict(X_test)
    expected = (X_test['Returns'] > 0).astype(int).values

    np.testing.assert_array_equal(predictions, expected)
    # With 200 independent random Returns values, a frozen single-direction
    # prediction would be a near-impossible coincidence.
    assert 0 < predictions.sum() < len(predictions)


def test_mean_reversion_is_walk_forward_not_frozen(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    strategy = get_all_naive_strategies()['Mean Reversion (Contrarian)']()
    strategy.fit(X_train, y_train)

    predictions = strategy.predict(X_test)
    expected = (X_test['Returns'] <= 0).astype(int).values

    np.testing.assert_array_equal(predictions, expected)
    assert 0 < predictions.sum() < len(predictions)


def test_momentum_and_mean_reversion_are_opposites(synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    momentum = get_all_naive_strategies()['Momentum (Last Direction)']()
    reversion = get_all_naive_strategies()['Mean Reversion (Contrarian)']()
    momentum.fit(X_train, y_train)
    reversion.fit(X_train, y_train)

    assert (momentum.predict(X_test) != reversion.predict(X_test)).all()


@pytest.mark.parametrize("name", ['Momentum (Last Direction)', 'Mean Reversion (Contrarian)'])
def test_momentum_family_requires_returns_column(name, synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    strategy = get_all_naive_strategies()[name]()
    strategy.fit(X_train, y_train)

    with pytest.raises(ValueError):
        strategy.predict(X_test.drop(columns=['Returns']))
