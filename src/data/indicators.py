"""
Professional Technical Indicators Module
Clean, efficient implementations of common technical analysis indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class TechnicalIndicators:
    """Professional implementation of technical indicators"""
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            prices: Price series
            period: RSI period (default: 14)
            
        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with 'upper', 'middle', and 'lower' bands
        """
        middle = TechnicalIndicators.sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
            
        Returns:
            ATR values
        """
        # Calculate True Range components
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        # True Range is the maximum of the three
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR is the moving average of True Range
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with '%K' and '%D' series
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        k_percent = np.where(range_hl != 0, 
                           100 * (close - lowest_low) / range_hl, 
                           50)
        k_percent = pd.Series(k_percent, index=close.index)
        
        # %D is the moving average of %K
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            '%K': k_percent,
            '%D': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Williams %R
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Lookback period
            
        Returns:
            Williams %R values (-100 to 0)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        williams_r = np.where(range_hl != 0,
                             -100 * (highest_high - close) / range_hl,
                             -50)
        
        return pd.Series(williams_r, index=close.index)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: CCI period
            
        Returns:
            CCI values
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        # Avoid division by zero
        cci = np.where(mean_deviation != 0,
                      (typical_price - sma_tp) / (0.015 * mean_deviation),
                      0)
        
        return pd.Series(cci, index=close.index)


def add_all_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Add all technical indicators to a dataframe
    
    Args:
        df: OHLCV dataframe
        config: DataConfig object with indicator parameters
        
    Returns:
        DataFrame with added technical indicators
    """
    result_df = df.copy()
    ta = TechnicalIndicators()
    
    try:
        # Price-based features
        result_df['Returns'] = result_df['Close'].pct_change()
        result_df['Log_Returns'] = np.log(result_df['Close'] / result_df['Close'].shift(1))
        
        # Moving averages
        for period in config.ma_periods:
            result_df[f'SMA_{period}'] = ta.sma(result_df['Close'], period)
            result_df[f'EMA_{period}'] = ta.ema(result_df['Close'], period)
            result_df[f'Price_SMA_Ratio_{period}'] = result_df['Close'] / result_df[f'SMA_{period}']
            result_df[f'Price_EMA_Ratio_{period}'] = result_df['Close'] / result_df[f'EMA_{period}']
        
        # RSI
        result_df['RSI'] = ta.rsi(result_df['Close'], config.rsi_period)
        
        # MACD
        macd_data = ta.macd(result_df['Close'], 
                           fast=config.macd_fast, 
                           slow=config.macd_slow, 
                           signal=config.macd_signal)
        result_df['MACD'] = macd_data['macd']
        result_df['MACD_Signal'] = macd_data['signal']
        result_df['MACD_Histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = ta.bollinger_bands(result_df['Close'], 
                                   period=config.bb_period, 
                                   std_dev=config.bb_std)
        result_df['BB_Upper'] = bb_data['upper']
        result_df['BB_Middle'] = bb_data['middle']
        result_df['BB_Lower'] = bb_data['lower']
        result_df['BB_Width'] = (result_df['BB_Upper'] - result_df['BB_Lower']) / result_df['BB_Middle']
        result_df['BB_Position'] = (result_df['Close'] - result_df['BB_Lower']) / (result_df['BB_Upper'] - result_df['BB_Lower'])
        
        # ATR
        result_df['ATR'] = ta.atr(result_df['High'], result_df['Low'], result_df['Close'], period=14)
        
        # Stochastic
        stoch_data = ta.stochastic(result_df['High'], result_df['Low'], result_df['Close'])
        result_df['Stoch_K'] = stoch_data['%K']
        result_df['Stoch_D'] = stoch_data['%D']
        
        # Williams %R
        result_df['Williams_R'] = ta.williams_r(result_df['High'], result_df['Low'], result_df['Close'])
        
        # CCI
        result_df['CCI'] = ta.cci(result_df['High'], result_df['Low'], result_df['Close'])
        
        # Volume indicators (if Volume column exists)
        if 'Volume' in result_df.columns:
            result_df['Volume_SMA'] = ta.sma(result_df['Volume'], 20)
            result_df['Volume_Ratio'] = result_df['Volume'] / result_df['Volume_SMA']
            result_df['Price_Volume'] = result_df['Close'] * result_df['Volume']
        
        # Volatility measures
        result_df['Volatility'] = result_df['Returns'].rolling(window=20).std()
        result_df['High_Low_Ratio'] = result_df['High'] / result_df['Low']
        result_df['Range_Pct'] = (result_df['High'] - result_df['Low']) / result_df['Close']
        
        # Trend indicators
        for horizon in [2, 5, 10, 20]:
            result_df[f'Trend_{horizon}'] = (result_df['Close'] > result_df['Close'].shift(horizon)).astype(int)
            result_df[f'Price_Change_{horizon}'] = result_df['Close'].pct_change(periods=horizon)
            result_df[f'Price_Change_Abs_{horizon}'] = result_df[f'Price_Change_{horizon}'].abs()
        
        # Momentum indicators
        result_df['Momentum_5'] = result_df['Close'] - result_df['Close'].shift(5)
        result_df['Momentum_10'] = result_df['Close'] - result_df['Close'].shift(10)
        result_df['ROC_5'] = result_df['Close'].pct_change(periods=5)
        result_df['ROC_10'] = result_df['Close'].pct_change(periods=10)
        
        # Market microstructure
        result_df['Open_Close_Ratio'] = result_df['Open'] / result_df['Close']
        result_df['High_Close_Ratio'] = result_df['High'] / result_df['Close']
        result_df['Low_Close_Ratio'] = result_df['Low'] / result_df['Close']
        
        # Intraday ranges
        result_df['Body_Size'] = np.abs(result_df['Close'] - result_df['Open']) / result_df['Close']
        result_df['Upper_Shadow'] = (result_df['High'] - np.maximum(result_df['Open'], result_df['Close'])) / result_df['Close']
        result_df['Lower_Shadow'] = (np.minimum(result_df['Open'], result_df['Close']) - result_df['Low']) / result_df['Close']
        
        return result_df
        
    except Exception as e:
        raise ValueError(f"Error adding technical indicators: {str(e)}")
