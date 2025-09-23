# 📈 Benchmark Strategies Guide

> Guide to the benchmark strategies used to evaluate ML model performance: Buy & Hold and Moving Average strategies.

## 📋 Overview

Benchmark strategies serve as **performance baselines** to evaluate whether machine learning models add value over simple, traditional approaches. The system implements two fundamental strategies:

- **Buy & Hold**: The simplest long-term investment strategy
- **Moving Average**: Classic technical analysis approach

These benchmarks help answer: *"Is the ML model actually better than doing nothing or using basic technical analysis?"*

---

## 💼 Buy & Hold Strategy

### **Concept**
Buy & Hold is the simplest investment strategy: buy an asset and hold it for the entire period, regardless of market fluctuations.

### **How It Works**
1. **Initial Investment**: Invest 100% of capital at the beginning
2. **Hold Period**: Maintain position throughout the entire timeframe
3. **No Trading**: Zero transactions after initial purchase
4. **Final Return**: Calculate total return at the end

### **Mathematical Formula**
```
Return = (Final_Price - Initial_Price) / Initial_Price
```

---

## 📊 Moving Average Strategy

### **Concept**
Moving Average strategy uses the relationship between current price and its historical average to generate buy/sell signals.

### **How It Works**
1. **Calculate Moving Average**: Average price over N periods
2. **Generate Signals**: 
   - **BUY** when price > moving average (uptrend)
   - **SELL** when price < moving average (downtrend)
3. **Position Management**: Switch between long and cash positions
4. **Trend Following**: Captures sustained price movements

### **Mathematical Formula**
```
MA(n) = (P₁ + P₂ + ... + Pₙ) / n
Signal = 1 if Price > MA(n), else 0
```

### **Common Variations**
- **SMA (Simple Moving Average)**: Equal weight to all periods
- **EMA (Exponential Moving Average)**: More weight to recent prices
- **Different Periods**: 20, 50, 100, 200 days are common

### **Key Parameters**
```python
ma_period = 20  # Number of periods for moving average
# Shorter periods: More responsive, more signals
# Longer periods: Smoother, fewer signals
```

---

## 📊 Performance Metrics for Benchmarks

### **Buy & Hold Metrics**
- **Total Return**: Same as underlying asset
- **Volatility**: Same as underlying asset
- **Sharpe Ratio**: (Return - Risk-free) / Volatility
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Calmar Ratio**: Return / Maximum Drawdown

### **Moving Average Metrics**
- **Total Return**: Strategy return vs buy & hold
- **Win Rate**: Percentage of profitable trades
- **Average Trade**: Mean profit/loss per trade
- **Profit Factor**: Gross profit / Gross loss
- **Time in Market**: Percentage of time holding positions

---
## **Key Questions to Answer:**
1. Does ML model beat buy & hold on risk-adjusted basis?
2. Does ML model reduce maximum drawdowns significantly?
4. Is outperformance consistent across different periods?
5. Does model add value in different market conditions?

---

