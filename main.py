#!/usr/bin/env python3
"""
CLI for evaluating the naive benchmark strategies against real or synthetic data.

Note: this only runs the naive strategies (see src/strategies/naive.py). Training
the Random Forest / MLP models is done through the Streamlit app.
"""

import argparse
import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import SUPPORTED_ASSETS, DataConfig
from src.data import DataFetcher
from src.strategies import get_all_naive_strategies, get_strategy_descriptions

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate naive benchmark strategies')

    parser.add_argument(
        '--asset',
        type=str,
        default='^GSPC',
        choices=list(SUPPORTED_ASSETS.keys()),
        help='Asset symbol to analyze'
    )
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategies', type=str, default='all', help='Comma-separated list of strategies, or "all"')
    parser.add_argument('--test', action='store_true', help='Run with synthetic data, no network required')

    return parser.parse_args()


STRATEGY_NAME_MAPPING = {
    'bullish': 'Bullish',
    'bearish': 'Bearish',
    'random': 'Random',
    'frequency': 'Frequency',
    'momentum': 'Momentum (Last Direction)',
    'mean_reversion': 'Mean Reversion (Contrarian)'
}


def run_analysis(asset: str, start_date: str, end_date: str, strategies: list):
    """Fetch data for `asset`, fit each requested strategy, and print accuracy on the test split."""
    logger.info(f"Starting analysis for {asset}")

    try:
        config = DataConfig()
        config.start_date = start_date
        config.end_date = end_date

        fetcher = DataFetcher(config)
        X, y = fetcher.process_symbol(asset)

        asset_info = SUPPORTED_ASSETS[asset]
        print(f"\n{asset_info['name']} ({asset})")
        print(f"Period: {start_date} to {end_date}")
        print(f"Samples: {len(X)}")
        print(f"Up-day rate: {y.mean():.1%}")

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        all_strategy_classes = get_all_naive_strategies(random_state=42)
        strategy_descriptions = get_strategy_descriptions()

        if 'all' in strategies:
            selected_strategy_classes = all_strategy_classes
        else:
            selected_strategy_classes = {}
            for strategy in strategies:
                class_name = STRATEGY_NAME_MAPPING.get(strategy.lower(), strategy)
                if class_name in all_strategy_classes:
                    selected_strategy_classes[class_name] = all_strategy_classes[class_name]
                else:
                    print(f"Unknown strategy: {strategy}")
                    print(f"Available: {', '.join(STRATEGY_NAME_MAPPING.keys())}")

        print(f"\nEvaluating {len(selected_strategy_classes)} strategies:")
        print("-" * 50)

        results = []

        for name, strategy_class in selected_strategy_classes.items():
            strategy = strategy_class()
            strategy.fit(X_train, y_train)
            predictions = strategy.predict(X_test)

            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, predictions)

            strategy_name = strategy_descriptions.get(name, name)
            results.append((strategy_name, accuracy))
            print(f"{strategy_name:<40}: {accuracy:.3f}")

        results.sort(key=lambda x: x[1], reverse=True)

        print("\nTop strategies:")
        print("-" * 30)
        for i, (name, accuracy) in enumerate(results[:3], 1):
            print(f"{i}. {name}: {accuracy:.3f}")

        print(f"\nBaseline (actual up-day rate): {y_test.mean():.3f}")

        return results

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return None


def run_quick_test():
    """Run the strategy comparison on synthetic data — no network access needed."""
    print("Quick test with synthetic data")
    print("=" * 40)

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score

    np.random.seed(42)
    n_train, n_test = 1000, 200

    # Simulate a mildly bullish market (60% up days)
    y_train = pd.Series(np.random.choice([0, 1], n_train, p=[0.4, 0.6]))
    y_test = pd.Series(np.random.choice([0, 1], n_test, p=[0.4, 0.6]))

    X_train = pd.DataFrame(np.random.randn(n_train, 5))
    X_test = pd.DataFrame(np.random.randn(n_test, 5))

    print(f"Synthetic data: {n_train} train, {n_test} test")
    print(f"Up-day rate: {y_test.mean():.3f}")

    strategy_classes = get_all_naive_strategies(random_state=42)
    strategy_descriptions = get_strategy_descriptions()

    print("\nResults:")
    print("-" * 25)

    for name, strategy_class in strategy_classes.items():
        strategy = strategy_class()
        strategy.fit(X_train, y_train)
        predictions = strategy.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        strategy_name = strategy_descriptions.get(name, name)
        print(f"{strategy_name:<40}: {accuracy:.3f}")

    print("\nDone.")


def main():
    args = parse_arguments()

    if args.test:
        run_quick_test()
        return 0

    if args.strategies == 'all':
        strategies = ['all']
    else:
        strategies = [s.strip() for s in args.strategies.split(',')]

    if args.asset not in SUPPORTED_ASSETS:
        print(f"Unsupported asset: {args.asset}")
        print(f"Available: {', '.join(list(SUPPORTED_ASSETS.keys())[:10])}...")
        return 1

    results = run_analysis(args.asset, args.start_date, args.end_date, strategies)

    if results is None:
        print("Analysis failed")
        return 1

    print("\nFor the full interface with ML models: streamlit run streamlit_app/app.py")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
