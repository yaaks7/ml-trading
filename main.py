#!/usr/bin/env python3
"""
Main CLI interface for ML Trading Prediction System
"""

import argparse
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import SUPPORTED_ASSETS, DataConfig
from src.data import DataFetcher
from src.strategies import get_all_naive_strategies, get_strategy_descriptions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ML Trading Prediction System - CLI Interface'
    )
    
    parser.add_argument(
        '--asset', 
        type=str, 
        default='^GSPC',
        choices=list(SUPPORTED_ASSETS.keys()),
        help='Asset symbol to analyze'
    )
    
    parser.add_argument(
        '--start-date', 
        type=str, 
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str, 
        default='2024-12-31',
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--strategies', 
        type=str, 
        default='all',
        help='Comma-separated list of strategies or "all"'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick test with sample data'
    )
    
    return parser.parse_args()


def run_analysis(asset: str, start_date: str, end_date: str, strategies: list):
    """Run analysis for specified parameters"""
    
    logger.info(f"Starting analysis for {asset}")
    
    try:
        # Configure and fetch data
        config = DataConfig()
        config.start_date = start_date
        config.end_date = end_date
        
        fetcher = DataFetcher(config)
        X, y = fetcher.process_symbol(asset)
        
        asset_info = SUPPORTED_ASSETS[asset]
        print(f"\n📈 {asset_info['name']} ({asset})")
        print(f"📅 Période: {start_date} à {end_date}")
        print(f"📊 Données: {len(X)} échantillons")
        print(f"📈 Taux de hausse: {y.mean():.1%}")
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"🎓 Entraînement: {len(X_train)} échantillons")
        print(f"🧪 Test: {len(X_test)} échantillons")
        
        # Get and evaluate strategies
        all_strategy_classes = get_all_naive_strategies(random_state=42)
        strategy_descriptions = get_strategy_descriptions()
        
        if 'all' in strategies:
            selected_strategy_classes = all_strategy_classes
        else:
            # Map strategy names to class names
            strategy_name_mapping = {
                'bullish': 'Bullish',
                'bearish': 'Bearish', 
                'random': 'Random',
                'frequency': 'Frequency',
                'momentum': 'Momentum (Last Direction)',
                'mean_reversion': 'Mean Reversion (Contrarian)'
            }
            
            selected_strategy_classes = {}
            for strategy in strategies:
                class_name = strategy_name_mapping.get(strategy.lower(), strategy)
                if class_name in all_strategy_classes:
                    selected_strategy_classes[class_name] = all_strategy_classes[class_name]
                else:
                    print(f"⚠️ Stratégie inconnue: {strategy}")
                    print(f"💡 Stratégies disponibles: {', '.join(strategy_name_mapping.keys())}")
        
        print(f"\n🤖 Évaluation de {len(selected_strategy_classes)} stratégies:")
        print("-" * 50)
        
        results = []
        
        for name, strategy_class in selected_strategy_classes.items():
            # Instantiate, fit and predict
            strategy = strategy_class()
            strategy.fit(X_train, y_train)
            predictions = strategy.predict(X_test)
            
            # Calculate accuracy
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, predictions)
            
            strategy_name = strategy_descriptions.get(name, name)
            results.append((strategy_name, accuracy))
            print(f"📈 {strategy_name:<40}: {accuracy:.3f}")
        
        # Sort and display best strategies
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Meilleures Stratégies:")
        print("-" * 30)
        for i, (name, accuracy) in enumerate(results[:3], 1):
            print(f"{i}. {name}: {accuracy:.3f}")
        
        print(f"\n🎯 Baseline (% hausse réel): {y_test.mean():.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse: {str(e)}")
        return None


def run_quick_test():
    """Run quick test with synthetic data"""
    
    print("🧪 Test Rapide des Stratégies Naïves")
    print("=" * 40)
    
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    
    # Create synthetic data
    np.random.seed(42)
    n_train, n_test = 1000, 200
    
    # Simulate bullish market (60% up days)
    y_train = pd.Series(np.random.choice([0, 1], n_train, p=[0.4, 0.6]))
    y_test = pd.Series(np.random.choice([0, 1], n_test, p=[0.4, 0.6]))
    
    X_train = pd.DataFrame(np.random.randn(n_train, 5))
    X_test = pd.DataFrame(np.random.randn(n_test, 5))
    
    print(f"📊 Données synthétiques: {n_train} train, {n_test} test")
    print(f"📈 Taux de hausse: {y_test.mean():.3f}")
    
    # Test strategies
    strategy_classes = get_all_naive_strategies(random_state=42)
    strategy_descriptions = get_strategy_descriptions()
    
    print("\n🤖 Résultats:")
    print("-" * 25)
    
    for name, strategy_class in strategy_classes.items():
        strategy = strategy_class()  # Instantiate the strategy
        strategy.fit(X_train, y_train)
        predictions = strategy.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        strategy_name = strategy_descriptions.get(name, name)
        print(f"📈 {strategy_name:<40}: {accuracy:.3f}")
    
    print("\n✅ Test rapide terminé!")


def main():
    """Main execution function"""
    
    print("🚀 ML Trading Prediction System")
    print("=" * 50)
    
    args = parse_arguments()
    
    if args.test:
        run_quick_test()
        return 0
    
    # Parse strategies
    if args.strategies == 'all':
        strategies = ['all']
    else:
        strategies = [s.strip() for s in args.strategies.split(',')]
    
    # Validate asset
    if args.asset not in SUPPORTED_ASSETS:
        print(f"❌ Actif non supporté: {args.asset}")
        print(f"💡 Actifs disponibles: {', '.join(list(SUPPORTED_ASSETS.keys())[:10])}...")
        return 1
    
    # Run analysis
    results = run_analysis(args.asset, args.start_date, args.end_date, strategies)
    
    if results is None:
        print("❌ Analyse échouée")
        return 1
    
    print("\n🎉 Analyse terminée!")
    print("💡 Pour une interface graphique, lancez: streamlit run streamlit_app/app.py")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

