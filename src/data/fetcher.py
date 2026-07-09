"""
Fetches OHLCV data from Yahoo Finance and turns it into a feature matrix + target.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging

from config import DataConfig

logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp_fast = prices.ewm(span=fast).mean()
    exp_slow = prices.ewm(span=slow).mean()
    macd = exp_fast - exp_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    
    return {
        'macd': macd,
        'signal': macd_signal,
        'histogram': macd_histogram
    }


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    
    return {
        'upper': upper_band,
        'middle': rolling_mean,
        'lower': lower_band
    }


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    
    return atr


class DataFetcher:
    """Handles data fetching and preprocessing for trading strategies"""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.data = None
        self.processed_data = None
        
    def fetch_data(self, 
                   symbol: str, 
                   start_date: Optional[str] = None, 
                   end_date: Optional[str] = None, 
                   interval: Optional[str] = None) -> pd.DataFrame:
        """Fetch OHLCV data for `symbol` from Yahoo Finance. Falls back to config defaults for any date/interval left unset."""
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        interval = interval or self.config.interval
        
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} (interval: {interval})")
            
            # Download data from Yahoo Finance
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # yfinance returns a MultiIndex when the ticker list has one entry wrapped oddly
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            data.columns = [col.title() for col in data.columns]
            data = data.dropna()
            data = data.sort_index()

            self.data = data
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages, RSI, MACD, Bollinger Bands, ATR, and momentum/microstructure features to an OHLCV frame."""
        df = data.copy()
        
        try:
            logger.info("Adding technical indicators...")
            
            # Price-based features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Moving averages
            if self.config.ma_periods:
                for period in self.config.ma_periods:
                    df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
                    df[f'MA_Ratio_{period}'] = df['Close'] / df[f'MA_{period}']
            
            # RSI (Relative Strength Index)
            if self.config.rsi_period:
                df['RSI'] = calculate_rsi(df['Close'], self.config.rsi_period)
            
            # MACD
            if (self.config.macd_fast and self.config.macd_slow and self.config.macd_signal):
                macd_data = calculate_macd(df['Close'], 
                                         fast=self.config.macd_fast, 
                                         slow=self.config.macd_slow, 
                                         signal=self.config.macd_signal)
                df['MACD'] = macd_data['macd']
                df['MACD_Signal'] = macd_data['signal']
                df['MACD_Histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            if (self.config.bb_period and self.config.bb_std):
                bb_data = calculate_bollinger_bands(df['Close'], 
                                                  period=self.config.bb_period, 
                                                  std_dev=self.config.bb_std)
                df['BB_Upper'] = bb_data['upper']
                df['BB_Middle'] = bb_data['middle']
                df['BB_Lower'] = bb_data['lower']
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
                df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Average True Range (ATR)
            df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], period=14)
            
            # Volume indicators
            if 'Volume' in df.columns:
                df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Volatility measures
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            
            # Trend indicators
            for horizon in [2, 5, 10, 20]:
                df[f'Trend_{horizon}'] = (df['Close'] > df['Close'].shift(horizon)).astype(int)
                df[f'Price_Change_{horizon}'] = df['Close'].pct_change(periods=horizon)
            
            # Momentum indicators
            df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
            df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
            
            # Market microstructure
            df['Open_Close_Ratio'] = df['Open'] / df['Close']
            df['High_Close_Ratio'] = df['High'] / df['Close']
            df['Low_Close_Ratio'] = df['Low'] / df['Close']
            
            logger.info(f"Added technical indicators. Dataset now has {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
        
        return df
    
    def create_target_variable(self, data: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """Label each row with whether `target_column` closes higher the next day."""
        df = data.copy()

        df['Tomorrow_Price'] = df[target_column].shift(-1)
        df['Target'] = (df['Tomorrow_Price'] > df[target_column]).astype(int)
        df = df[:-1]  # last row has no next-day price to compare against
        
        logger.info(f"Created target variable. Target distribution: {df['Target'].value_counts().to_dict()}")
        
        return df
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split into (X, y), dropping the columns that would leak the target."""
        df = data.copy()

        exclude_columns = [
            'Target', 'Tomorrow_Price',
            'Adj Close', 'Dividends', 'Stock Splits'
        ]
        feature_columns = [col for col in df.columns if col not in exclude_columns]

        X = df[feature_columns]
        y = df['Target']

        # Rolling-window indicators (MA_200, Trend_20, ...) are NaN for their lookback
        # period. Rather than dropping every row with any NaN, skip past the longest
        # lookback window and forward-fill the rest.
        max_lookback = 0
        for col in X.columns:
            if 'MA_' in col or 'Trend_' in col or 'Price_Change_' in col:
                try:
                    period = int(col.split('_')[1])
                    max_lookback = max(max_lookback, period)
                except ValueError:
                    pass

        start_idx = max(max_lookback, 20)

        if start_idx < len(X):
            X = X.iloc[start_idx:]
            y = y.iloc[start_idx:]

            critical_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Returns']
            critical_features = [col for col in critical_features if col in X.columns]

            valid_indices = ~(y.isnull() | X[critical_features].isnull().any(axis=1))
            X = X[valid_indices]
            y = y[valid_indices]
            X = X.ffill().bfill()
        else:
            # Not enough rows to skip past the lookback window; fall back to dropping
            # any row with a NaN.
            valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_indices]
            y = y[valid_indices]
        
        logger.info(f"Prepared {len(X.columns)} features and {len(y)} samples")
        logger.info(f"Feature columns: {list(X.columns)[:10]}..." if len(X.columns) > 10 else list(X.columns))
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Chronological train/val/test split — no shuffling, this is a time series."""
        n_samples = len(X)

        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.val_ratio))

        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train period: {X_train.index[0]} to {X_train.index[-1]}")
        logger.info(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict:
        """Row/column counts, price range, and target distribution for a processed frame."""
        summary = {
            'rows': len(data),
            'columns': len(data.columns),
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'missing_values': data.isnull().sum().sum(),
        }
        
        if 'Close' in data.columns:
            summary.update({
                'price_range': {
                    'min': data['Close'].min(),
                    'max': data['Close'].max(),
                    'current': data['Close'].iloc[-1]
                }
            })
        
        if 'Target' in data.columns:
            summary.update({
                'target_distribution': data['Target'].value_counts().to_dict(),
                'up_ratio': data['Target'].mean()
            })
        
        return summary
    
    def process_symbol(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch, add indicators, label, and clean — end to end for one symbol."""
        logger.info(f"Starting complete processing for {symbol}")

        raw_data = self.fetch_data(symbol, start_date, end_date)
        data_with_indicators = self.add_technical_indicators(raw_data)
        data_with_target = self.create_target_variable(data_with_indicators)
        X, y = self.prepare_features(data_with_target)

        self.processed_data = data_with_target

        logger.info(f"Processing complete for {symbol}")
        return X, y

    def fetch_and_prepare(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Alias for process_symbol; the Streamlit app calls it under this name."""
        return self.process_symbol(symbol, start_date, end_date)

    def get_raw_data(self):
        """Processed data with indicators if available, otherwise the raw OHLCV frame."""
        if self.processed_data is not None:
            return self.processed_data
        return self.data
