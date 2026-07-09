"""Tests for the Random Forest and MLP wrappers, using synthetic data (no network)."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.models import get_all_ml_models


@pytest.fixture
def synthetic_data():
    """Features with a weak-but-real linear relationship to the target, so a
    reasonable model should beat 50% without needing real market data."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 1000, 10

    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series((X.sum(axis=1) + rng.randn(n_samples) * 0.5 > 0).astype(int))

    split = int(0.8 * n_samples)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


@pytest.mark.parametrize("model_name", list(get_all_ml_models().keys()))
def test_model_fits_and_predicts(model_name, synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    model_class = get_all_ml_models()[model_name]

    model = model_class(random_state=42)
    model.fit(X_train, y_train)
    assert model.is_fitted

    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(np.unique(predictions)).issubset({0, 1})

    accuracy = accuracy_score(y_test, predictions)
    assert 0.0 <= accuracy <= 1.0


@pytest.mark.parametrize("model_name", list(get_all_ml_models().keys()))
def test_model_beats_random_guessing(model_name, synthetic_data):
    """With a real signal in the features, both models should clear 50% test accuracy."""
    X_train, X_test, y_train, y_test = synthetic_data
    model_class = get_all_ml_models()[model_name]

    model = model_class(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    assert accuracy > 0.5


def test_predict_before_fit_raises():
    from src.models.random_forest import RandomForestModel

    model = RandomForestModel()
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame({'a': [1, 2, 3]}))
