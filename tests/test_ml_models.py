"""
Test script for ML models
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.models import get_all_ml_models, get_model_descriptions
from src.data import DataFetcher
from config import DataConfig


def test_ml_models():
    """Test all ML models with sample data"""
    
    print("🤖 Test des Modèles ML")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create synthetic features
    X_synthetic = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create synthetic target with some pattern
    # Make it somewhat predictable based on features
    y_synthetic = ((X_synthetic.sum(axis=1) + np.random.randn(n_samples) * 0.5) > 0).astype(int)
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X_synthetic[:train_size]
    X_val = X_synthetic[train_size:train_size+val_size]
    X_test = X_synthetic[train_size+val_size:]
    
    y_train = y_synthetic[:train_size]
    y_val = y_synthetic[train_size:train_size+val_size]
    y_test = y_synthetic[train_size+val_size:]
    
    print(f"📊 Données d'entraînement: {len(X_train)} échantillons")
    print(f"📊 Données de validation: {len(X_val)} échantillons")
    print(f"📊 Données de test: {len(X_test)} échantillons")
    print(f"📈 Taux de hausse réel (test): {y_test.mean():.3f}")
    
    # Get ML models (classes, not instances)
    model_classes = get_all_ml_models()
    descriptions = get_model_descriptions()
    
    print("\\n🤖 Évaluation des Modèles ML:")
    print("-" * 40)
    
    results = {}
    
    for model_name, model_class in model_classes.items():
        try:
            print(f"\\n📈 {model_name}")
            print(f"   {descriptions.get(model_name, 'Aucune description')}")
            
            # Instantiate model
            model = model_class(random_state=42)
            
            # Train model (new API: only X_train, y_train)
            model.fit(X_train, y_train)
            
            # Evaluate (simple predictions and accuracy)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            results[model_name] = accuracy
            
            print(f"   ✅ Précision: {accuracy:.3f}")
            
            # Show feature importance for Random Forest
            if hasattr(model, 'get_feature_importance'):
                try:
                    importance = model.get_feature_importance(top_n=5)
                    print(f"   🎯 Top 5 features importantes:")
                    for _, row in importance.iterrows():
                        print(f"      {row['feature']}: {row['importance']:.3f}")
                except:
                    pass
            
        except Exception as e:
            print(f"   ❌ Erreur: {str(e)}")
            results[model_name] = 0.0
    
    # Display results summary
    print(f"\\n📊 Résumé des Performances:")
    print("-" * 30)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (model_name, accuracy) in enumerate(sorted_results, 1):
        print(f"{i}. {model_name}: {accuracy:.3f}")
    
    print(f"🎯 Baseline (taux de hausse réel): {y_test.mean():.3f}")


def test_with_real_data():
    """Test ML models with real financial data"""
    
    print("\\n" + "=" * 60)
    print("🌍 Test avec Données Réelles - S&P 500")
    print("=" * 60)
    
    try:
        # Configure data fetcher
        config = DataConfig()
        config.start_date = "2023-01-01"
        config.end_date = "2024-01-01"
        
        fetcher = DataFetcher(config)
        
        print("📥 Téléchargement des données du S&P 500...")
        X, y = fetcher.process_symbol("^GSPC")
        
        print(f"📊 Données traitées: {len(X)} échantillons")
        print(f"📈 Taux de hausse historique: {y.mean():.3f}")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = fetcher.split_data(X, y)
        
        print(f"🎓 Entraînement: {len(X_train)} échantillons")
        print(f"📝 Validation: {len(X_val)} échantillons")
        print(f"🧪 Test: {len(X_test)} échantillons")
        
        # Get ML models (classes)
        model_classes = get_all_ml_models()
        
        print("\\n🤖 Performances sur Données Réelles:")
        print("-" * 40)
        
        real_results = {}
        
        for model_name, model_class in model_classes.items():
            try:
                # Instantiate model
                model = model_class(random_state=42)
                # Train model (new API)
                model.fit(X_train, y_train)
                
                # Evaluate (simple predictions)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                real_results[model_name] = accuracy
                print(f"📈 {model_name}: {accuracy:.3f}")
                
            except Exception as e:
                print(f"❌ {model_name}: Erreur - {str(e)}")
                real_results[model_name] = 0.0
        
        print(f"🎯 Baseline (taux de hausse réel): {y_test.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Erreur lors du test avec données réelles: {str(e)}")


if __name__ == "__main__":
    test_ml_models()
    test_with_real_data()
    
    print("\\n🎉 Tests Terminés!")
    print("✅ Tous les modèles ML ont été testés!")
    print("🚀 Vous pouvez maintenant utiliser les modèles dans l'application Streamlit")
