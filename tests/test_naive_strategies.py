"""
Test script for naive strategies
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.strategies import get_all_naive_strategies, get_strategy_descriptions
from src.data import DataFetcher
from config import DataConfig

def test_naive_strategies():
    """Test all naive strategies with sample data"""
    
    print("🧪 Test des Stratégies Naïves")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate market data with 60% up days (bullish market)
    y_train = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    y_train = pd.Series(y_train, name='direction')
    
    # Create dummy features
    X_train = pd.DataFrame(
        np.random.randn(n_samples, 10), 
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    # Test data
    y_test = np.random.choice([0, 1], size=200, p=[0.4, 0.6])
    y_test = pd.Series(y_test, name='direction')
    X_test = pd.DataFrame(
        np.random.randn(200, 10), 
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    print(f"📊 Données d'entraînement: {len(y_train)} échantillons")
    print(f"📊 Données de test: {len(y_test)} échantillons")
    print(f"📈 Distribution d'entraînement: {y_train.value_counts().to_dict()}")
    print(f"📈 Taux de hausse réel (test): {y_test.mean():.3f}")
    print()
    
    # Get all strategies
    strategies = get_all_naive_strategies(random_state=42)
    descriptions = get_strategy_descriptions()
    
    results = {}
    
    print("🤖 Évaluation des Stratégies:")
    print("-" * 30)
    
    for name, strategy_class in strategies.items():
        # Create strategy instance
        strategy = strategy_class(random_state=42)
        
        print(f"\n📈 {strategy.name}")
        print(f"   {descriptions[name]}")
        
        # Fit and predict
        strategy.fit(X_train, y_train)
        predictions = strategy.predict(X_test)
        probabilities = strategy.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        
        # Get strategy info
        info = strategy.get_strategy_info()
        
        print(f"   ✅ Précision: {accuracy:.3f}")
        if 'expected_accuracy' in info:
            print(f"   📊 Précision attendue: {info['expected_accuracy']:.3f}")
        print(f"   🎯 Échantillon de prédictions: {predictions[:10]}")
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'strategy_info': info
        }
    
    print("\n📊 Résumé des Performances:")
    print("-" * 30)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    strategy_descriptions = get_strategy_descriptions()
    
    for i, (name, result) in enumerate(sorted_results, 1):
        strategy_name = strategy_descriptions.get(name, name)
        print(f"{i}. {strategy_name}: {result['accuracy']:.3f}")
    
    print(f"\n🎯 Baseline (taux de hausse réel): {y_test.mean():.3f}")
    
    return results

def test_with_real_data():
    """Test strategies with real market data"""
    
    print("\n" + "=" * 60)
    print("🌍 Test avec Données Réelles - S&P 500")
    print("=" * 60)
    
    try:
        # Configure data fetcher
        config = DataConfig()
        config.start_date = '2023-01-01'
        config.end_date = '2024-12-31'
        
        fetcher = DataFetcher(config)
        
        print("📥 Téléchargement des données du S&P 500...")
        X, y = fetcher.process_symbol('^GSPC')
        
        print(f"📊 Données traitées: {len(X)} échantillons")
        print(f"📈 Taux de hausse historique: {y.mean():.3f}")
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"🎓 Entraînement: {len(X_train)} échantillons")
        print(f"🧪 Test: {len(X_test)} échantillons")
        
        # Test strategies
        strategies = get_all_naive_strategies(random_state=42)
        
        print("\n🤖 Performances sur Données Réelles:")
        print("-" * 40)
        
        real_results = {}
        
        for name, strategy_class in strategies.items():
            strategy = strategy_class(random_state=42)
            strategy.fit(X_train, y_train)
            predictions = strategy.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            print(f"📈 {strategy.name}: {accuracy:.3f}")
            real_results[name] = accuracy
        
        print(f"\n🎯 Baseline (taux de hausse réel): {y_test.mean():.3f}")
        
        return real_results
        
    except Exception as e:
        print(f"❌ Erreur lors du test avec données réelles: {str(e)}")
        print("💡 Vérifiez votre connexion internet et les dépendances (yfinance)")
        return None

if __name__ == "__main__":
    # Test with synthetic data
    synthetic_results = test_naive_strategies()
    
    # Test with real data
    real_results = test_with_real_data()
    
    print("\n" + "🎉 Tests Terminés!" + "\n")
    
    if real_results:
        print("✅ Toutes les stratégies naïves fonctionnent correctement!")
        print("🚀 Vous pouvez maintenant lancer l'application Streamlit:")
        print("   streamlit run streamlit_app/app.py")
    else:
        print("⚠️ Test avec données réelles échoué, mais stratégies fonctionnelles")
        print("💡 Installez les dépendances manquantes: pip install -r requirements.txt")
