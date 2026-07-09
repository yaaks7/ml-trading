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


def test_mlp_robust_to_feature_scale_disparity(synthetic_data):
    """Regression test: the real feature matrix mixes wildly different scales
    (Volume ~1e9 next to ratio features ~1.0). Fed unscaled into MLPClassifier,
    this used to blow up the training loss and collapse predictions to near-noise
    (measured: 92% -> 49% accuracy, loss 0.2 -> 18.2, on this exact fixture with
    one huge-scale column added). MLPModel now standardizes internally, so this
    should no longer degrade performance."""
    from src.models.mlp import MLPModel

    X_train, X_test, y_train, y_test = synthetic_data
    rng = np.random.RandomState(0)

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['huge_scale_feature'] = rng.randn(len(X_train)) * 1e9
    X_test['huge_scale_feature'] = rng.randn(len(X_test)) * 1e9

    model = MLPModel(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Same bar as test_model_beats_random_guessing, but with a scale-disparate
    # column mixed in — this is what actually failed before the fix.
    assert accuracy > 0.7


def test_random_forest_min_samples_leaf_reduces_overfitting(synthetic_data):
    """Regression test: the Streamlit UI used to leave min_samples_leaf at
    sklearn's default of 1, letting trees memorize individual training rows
    (measured 99.9% train / 43% test accuracy on real data). Raising it should
    meaningfully shrink the train-test gap, even if it doesn't raise test
    accuracy — there's little real signal to find, regularizing just stops the
    model from confidently fitting noise."""
    from src.models.random_forest import RandomForestModel

    rng = np.random.RandomState(42)
    n_samples, n_features = 300, 20
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
    # Weak signal buried in noise, mirroring "most of ~35 real features carry
    # little information about next-day direction."
    y = pd.Series((X['f0'] * 0.3 + X['f1'] * 0.3 + rng.randn(n_samples) * 1.5 > 0).astype(int))

    split = int(0.7 * n_samples)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    def train_test_gap(min_samples_leaf):
        model = RandomForestModel(n_estimators=200, max_depth=10, min_samples_leaf=min_samples_leaf, random_state=42)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        return train_acc - test_acc

    unregularized_gap = train_test_gap(min_samples_leaf=1)
    regularized_gap = train_test_gap(min_samples_leaf=20)

    assert regularized_gap < unregularized_gap - 0.1
