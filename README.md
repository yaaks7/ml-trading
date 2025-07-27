# 🤖 ML Trading Prediction System

> A modular machine learning system for financial market directional prediction with Streamlit interface and benchmark strategies.

## 🎯 Overview

This project implements **machine learning models** to predict the direction (up/down) of financial markets with a modular architecture. The system supports multiple asset classes and provides both CLI and web interfaces for evaluation against benchmark strategies.

### ✨ Key Features

- 🤖 **Multiple ML Models**: Random Forest, MLP (Multi-Layer Perceptron)
- 🎯 **Benchmark Strategies**: Comparison strategies (Bullish, Bearish, Random, Frequency, Momentum, Mean Reversion)
- � **Multi-Asset Support**: Stocks, Crypto, Commodities
- 📈 **Interactive Interface**: 3-step Streamlit application with technical indicators
- 🔍 **Comprehensive Evaluation**: Complete metrics and performance analysis

### 🎯 **Asset Portfolio Coverage**
- **� Stock Indices**: S&P 500, NASDAQ, Dow Jones
- **🏢 Individual Stocks**: AAPL, MSFT, GOOGL, TSLA, NVDA, META, AMZN
- **₿ Cryptocurrencies**: BTC, ETH, SOL
- **🥇 Commodities**: Gold, Silver, Crude Oil

## 📱 Interface Preview

<div align="center">

![ML Trading Interface](data/img/Capture%20d'écran%202025-07-27%20012443.png)

*Streamlit interface with interactive charts and  ML analytics*

</div>

## �🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yaaks7/ml-trading.git
cd ml-trading

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Web Interface (Recommended)

```bash
streamlit run streamlit_app/app.py
```

➡️ **Open [http://localhost:8501](http://localhost:8501)** in your browser

### 3. Command Line Interface

```bash
# Default analysis (S&P 500, 2023)
python main.py

# Custom asset and strategies
python main.py --asset BTC-USD --start-date 2023-01-01 --end-date 2024-12-31 --strategies bullish,random

# All strategies with verbose output
python main.py --asset ^GSPC --strategies all --verbose
```

### 4. Quick Test

```bash
# Run system test
python main.py --test

```

## 💻 Usage Guide

### 🌐 Streamlit Interface (3-Step Workflow)

The web interface follows a logical, pedagogical approach with an intuitive three-step workflow:

#### **Step 1: Data Configuration & Features** 📊

Configure your analysis setup and visualize technical indicators before model training.

![Data Configuration](data/img/Capture%20d'écran%202025-07-27%20012452.png)
*Data configuration showing asset selection, date range, and technical indicator setup*

1. **Select Asset**: Choose from S&P 500, BTC-USD, etc.
2. **Define Period**: Start and end dates for analysis
3. **Configure Technical Indicators**: 
   - Moving Averages (MA 5, 10, 20, 50, 100, 200)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
4. **Load & Process Data**: View price charts and generated features

#### **Step 2: Model Configuration** 🤖

Configure your ML models and benchmark strategies for comprehensive evaluation.

![Model Configuration](data/img/Capture%20d'écran%202025-07-27%20012832.png)
*Model configuration panel with ML models and benchmark strategy selection*

1. **Select ML Models**: Random Forest, MLP
2. **Choose Benchmarks**: Comparison strategies for rigorous evaluation
3. **Configure Parameters**: Model hyperparameters and training settings
4. **Train Models**: Automated training process with progress tracking

#### **Step 3: Results & Evaluation** 📈

Analyze model performance with comprehensive metrics and visualizations.

![Results Analysis](data/img/Capture%20d'écran%202025-07-27%20012856.png)
*Complete results with performance metrics, model comparison, and detailed analysis*

1. **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
2. **Model Comparison**: Ranking and detailed analysis
3. **Visualizations**: Confusion matrices, feature importance
4. **Export Results**: Download reports in CSV/JSON format

### � CLI Interface

```bash
python main.py [OPTIONS]

Options:
  --asset {^GSPC,BTC-USD,AAPL,...}     Asset to analyze (default: ^GSPC)
  --start-date YYYY-MM-DD              Start date (default: 2020-01-01)
  --end-date YYYY-MM-DD                End date (default: 2024-12-31)
  --strategies LIST                    Comma-separated strategies or "all"
  --test                               Run quick test with synthetic data
  --verbose, -v                        Enable detailed logging
```

### 🐍 Programmatic Usage

```python
from src.data.fetcher import DataFetcher
from src.models.ml_models import get_all_ml_models
from src.strategies.naive import get_all_naive_strategies
from config.settings import DataConfig

# Configure and fetch data
config = DataConfig()
config.start_date = "2023-01-01"
config.end_date = "2024-12-31"

fetcher = DataFetcher(config)
X, y = fetcher.process_symbol("^GSPC")

# Split data
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train and evaluate models
ml_models = get_all_ml_models()
for name, model_class in ml_models.items():
    model = model_class()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name}: {accuracy:.3f}")
```

## �📊 Supported Assets & Technical Indicators

### 📈 **Stock Indices**
| Symbol | Name | Description |
|--------|------|-------------|
| ^GSPC | S&P 500 | US large-cap index |
| ^IXIC | NASDAQ Composite | Tech-heavy index |
| ^DJI | Dow Jones | 30 large US companies |

### 🏢 **Individual Stocks**
| Symbol | Name | Sector |
|--------|------|---------|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corporation | Technology |
| GOOGL | Alphabet Inc. | Technology |
| TSLA | Tesla Inc. | Automotive |
| NVDA | NVIDIA Corporation | Semiconductors |
| META | Meta Platforms Inc. | Social Media |
| AMZN | Amazon.com Inc. | E-commerce |

### ₿ **Cryptocurrencies**
| Symbol | Name | Market Cap Rank |
|--------|------|-----------------|
| BTC-USD | Bitcoin | #1 |
| ETH-USD | Ethereum | #2 |
| SOL-USD | Solana | Top 10 |

### 🥇 **Commodities**
| Symbol | Name | Category |
|--------|------|----------|
| GC=F | Gold Futures | Precious Metals |
| CL=F | Crude Oil Futures | Energy |

### 📊 **Technical Indicators**

**Basic Features**
- OHLCV prices (Open, High, Low, Close, Volume)
- Returns (daily price changes)
- Log returns for volatility analysis

**Technical Indicators**
- **Moving Averages**: 5, 10, 20, 50, 100, 200 periods
- **RSI**: Relative Strength Index (14 periods)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands

**Derived Features**
- Price/MA ratios for trend analysis
- Multi-horizon trends (2, 5, 10, 20 days)
- Momentum indicators
- Volatility measures

## 🤖 Implemented Models & Strategies

### **Machine Learning Models**
- **Random Forest**: Ensemble of decision trees with feature importance analysis
- **MLP (Multi-Layer Perceptron)**: Neural network for non-linear pattern recognition

### **Benchmark Strategies**
- **Bullish**: Always predicts upward movement
- **Bearish**: Always predicts downward movement
- **Random**: Random predictions (50/50)
- **Frequency**: Based on historical up/down frequency
- **Momentum**: Follows the last price direction
- **Mean Reversion**: Contrarian approach

## Technology Stack
- **Python 3.8+**: Core language
- **pandas/numpy**: Data manipulation and analysis
- **scikit-learn**: Machine learning models and metrics
- **yfinance**: Financial data retrieval
- **pandas-ta**: Technical indicator library
- **plotly**: Interactive data visualization
- **streamlit**: Web application framework
- **joblib**: Model persistence and caching

## 📈 Evaluation Metrics

### **Classification Metrics**
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to upward movements
- **F1-Score**: Harmonic mean of precision/recall
- **Confusion Matrix**: Detailed error analysis

### **Performance Analysis**
- **Overfitting Detection**: Train vs Test performance comparison
- **Model Ranking**: Comparative performance evaluation
- **Feature Importance**: Model interpretability and feature contribution
- **Benchmark Comparison**: ML models vs naive strategies

## 🧪 Testing & Validation

```bash

# Quick functionality test
python main.py --test

# Performance validation with real data
python main.py --asset BTC-USD --strategies all --verbose
```

## 🔧 Configuration & Customization

### Adding New Assets

Edit `config/settings.py`:
```python
SUPPORTED_ASSETS = {
    'YOUR_SYMBOL': {
        'name': 'Your Asset Name',
        'type': 'stock',  # or 'crypto', 'forex', 'commodity'
        'description': 'Asset description',
        'currency': 'USD',
        'sector': 'Technology'
    }
}
```

### Custom Technical Indicators

Extend `src/data/indicators.py`:
```python
def add_custom_indicator(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add your custom technical indicator"""
    df = df.copy()
    df['CUSTOM_INDICATOR'] = your_calculation(df)
    return df
```

### Custom ML Models

Extend `src/models/`:
```python
from src.models.base import BaseMLModel

class YourCustomModel(BaseMLModel):
    def __init__(self, **params):
        super().__init__("Your Model", **params)
        self.model = YourModelClass(**params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```


## 🚀 Performance Tips

- Use longer time periods (2+ years) for better model training
- Select appropriate technical indicators for your asset class
- Compare multiple models to avoid overfitting
- Always validate against benchmark strategies
- Monitor for data leakage in feature engineering


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Yanis** - Machine Learning & Quantitative Finance Portfolio

- GitHub: [github.com/yaaks7](https://github.com/yaaks7)
- LinkedIn: [linkedin.com/in/yanisaks](https://linkedin.com/in/yanisaks)

---


