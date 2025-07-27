"""
Configuration settings for ML Trading Prediction System
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    # Random Forest
    rf_n_estimators: int = 200
    rf_min_samples_split: int = 50
    rf_max_depth: int = 10
    rf_random_state: int = 42
    
    # MLP
    mlp_hidden_layer_sizes: tuple = (100, 50)
    mlp_activation: str = 'relu'
    mlp_solver: str = 'adam'
    mlp_max_iter: int = 1000
    mlp_random_state: int = 42
    
    # LSTM
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_sequence_length: int = 60


@dataclass
class DataConfig:
    """Configuration for data fetching and preprocessing"""
    interval: str = '1d'
    start_date: str = '2020-01-01'
    end_date: str = '2024-12-31'
    
    # Features configuration
    ma_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Train/validation/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 200]


@dataclass
class BacktestConfig:
    """Configuration for backtesting and evaluation"""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.001    # 0.1%
    
    # Risk management
    max_position_size: float = 0.95  # 95% of capital
    min_confidence: float = 0.6      # Minimum prediction confidence
    
    # Performance metrics
    risk_free_rate: float = 0.02     # 2% annual risk-free rate


# Supported assets with their configurations
SUPPORTED_ASSETS = {
    # Indices
    '^GSPC': {
        'name': 'S&P 500',
        'type': 'index',
        'description': 'US Stock Market Index',
        'currency': 'USD',
        'sector': 'Market Index'
    },
    '^IXIC': {
        'name': 'NASDAQ Composite',
        'type': 'index',
        'description': 'NASDAQ Stock Market Index',
        'currency': 'USD',
        'sector': 'Market Index'
    },
    '^DJI': {
        'name': 'Dow Jones Industrial Average',
        'type': 'index',
        'description': 'US Industrial Average Index',
        'currency': 'USD',
        'sector': 'Market Index'
    },
    
    # Individual Stocks - Technology
    'AAPL': {
        'name': 'Apple Inc.',
        'type': 'stock',
        'description': 'Technology Hardware & Equipment',
        'currency': 'USD',
        'sector': 'Technology'
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'type': 'stock',
        'description': 'Software & Technology Services',
        'currency': 'USD',
        'sector': 'Technology'
    },
    'GOOGL': {
        'name': 'Alphabet Inc.',
        'type': 'stock',
        'description': 'Internet Software & Services',
        'currency': 'USD',
        'sector': 'Technology'
    },
    'TSLA': {
        'name': 'Tesla Inc.',
        'type': 'stock',
        'description': 'Electric Vehicles & Clean Energy',
        'currency': 'USD',
        'sector': 'Automotive'
    },
    'NVDA': {
        'name': 'NVIDIA Corporation',
        'type': 'stock',
        'description': 'Semiconductors & AI Computing',
        'currency': 'USD',
        'sector': 'Technology'
    },
    'META': {
        'name': 'Meta Platforms Inc.',
        'type': 'stock',
        'description': 'Social Media & Metaverse',
        'currency': 'USD',
        'sector': 'Technology'
    },
    'AMZN': {
        'name': 'Amazon.com Inc.',
        'type': 'stock',
        'description': 'E-commerce & Cloud Computing',
        'currency': 'USD',
        'sector': 'Technology'
    },
    
    # Cryptocurrencies
    'BTC-USD': {
        'name': 'Bitcoin',
        'type': 'crypto',
        'description': 'Digital Currency Cryptocurrency',
        'currency': 'USD',
        'sector': 'Cryptocurrency'
    },
    'ETH-USD': {
        'name': 'Ethereum',
        'type': 'crypto',
        'description': 'Smart Contract Platform',
        'currency': 'USD',
        'sector': 'Cryptocurrency'
    },
    'SOL-USD': {
        'name': 'Solana',
        'type': 'crypto',
        'description': 'High-Performance Blockchain',
        'currency': 'USD',
        'sector': 'Cryptocurrency'
    },
    
    
    # Commodities
    'GC=F': {
        'name': 'Gold Futures',
        'type': 'commodity',
        'description': 'Precious Metal Commodity',
        'currency': 'USD',
        'sector': 'Commodities'
    },
    'CL=F': {
        'name': 'Crude Oil Futures',
        'type': 'commodity',
        'description': 'Energy Commodity',
        'currency': 'USD',
        'sector': 'Commodities'
    }
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'ML Trading Prediction System',
    'page_icon': '🤖',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'data/logs/ml_trading.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}
