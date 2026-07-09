# ML models

Two scikit-learn classifiers, both trained on the same ~40-feature matrix to predict
next-day direction (binary). Implementation: [src/models/random_forest.py](../src/models/random_forest.py),
[src/models/mlp.py](../src/models/mlp.py).

## Random Forest

`RandomForestClassifier`, defaults: 200 trees, max depth 10. Handles the non-linear,
noisy feature set reasonably well without needing scaling, and `get_feature_importance()`
gives some visibility into which indicators the model actually relies on.

## MLP

`MLPClassifier`, defaults: one hidden layer of 100 units, Adam solver, up to 1000
iterations. More capacity for non-linear interactions between features, but slower to
train and more prone to overfitting on a dataset this size. Note: features are fed in
unscaled — no `StandardScaler` — which isn't ideal for a gradient-based model like
this; something to fix if you want to push MLP performance further.

## Picking between them

Both are configurable from the Streamlit sidebar (tree count/depth for RF, hidden layer
size/iterations for MLP). In practice, check the train/test accuracy gap for each —
a large gap means overfitting regardless of which model produced it. See
[BENCHMARK_STRATEGIES.md](BENCHMARK_STRATEGIES.md) for what "good" looks like relative
to the naive baselines.
