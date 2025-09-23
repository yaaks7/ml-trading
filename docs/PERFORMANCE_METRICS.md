# 📊 Performance Metrics & Evaluation Guide

> Guide to understanding the metrics used to evaluate trading strategies and ML models, with interpretation guidelines for financial performance analysis.

## 📋 Overview

Performance evaluation in algorithmic trading requires multiple metrics to capture different aspects of strategy performance. This guide covers:

- **Return Metrics**: How much money was made
- **Risk Metrics**: How much risk was taken
- **Risk-Adjusted Metrics**: Return per unit of risk
- **Drawdown Metrics**: Worst-case scenarios
- **Trade-Based Metrics**: Individual transaction analysis

---

## 💰 Return Metrics

### **Total Return**
The overall percentage gain or loss over the entire period.

```
Total Return = (Final Value - Initial Value) / Initial Value × 100%
```

**Interpretation:**
- **Positive**: Strategy made money
- **Negative**: Strategy lost money
- **Comparison**: Must compare to benchmark over same period

**Example:**
- Initial: $10,000 → Final: $12,000
- Total Return = (12,000 - 10,000) / 10,000 = 20%

### **Annualized Return**
Return adjusted to a yearly basis for comparison across different time periods.

```
Annualized Return = (1 + Total Return)^(365/Days) - 1
```

**Interpretation:**
- **8-12%**: Reasonable for stock market strategies
- **>15%**: Excellent performance (but check risk)
- **<5%**: May not justify complexity vs buy & hold

### **Compound Annual Growth Rate (CAGR)**
The constant annual return that would achieve the same final result.

```
CAGR = (Final Value / Initial Value)^(1/Years) - 1
```

**Interpretation:**
- **Standard for comparing strategies** over different periods

---

## ⚠️ Risk Metrics

### **Volatility (Standard Deviation)**
Measures the variability of returns - higher volatility means more unpredictable returns.

```
Volatility = √(Σ(Return - Mean Return)² / (N-1))
```

**Interpretation:**
- **<10%**: Low volatility (conservative strategies)
- **10-20%**: Moderate volatility (typical for diversified portfolios)
- **>20%**: High volatility (aggressive strategies)

### **Downside Deviation**
Like volatility, but only considers negative returns - focuses on downside risk.

```
Downside Deviation = √(Σ(min(Return - Target, 0)² / N))
```

**Interpretation:**
- **Better measure than volatility** for risk-averse investors
- **Lower is always better** (only measures bad outcomes)
- **Useful for asymmetric return distributions**

### **Value at Risk (VaR)**
The maximum expected loss at a given confidence level (e.g., 95%) over a specific time period.

**Interpretation:**
- **VaR 95% = 2%**: 95% chance daily loss won't exceed 2%
- **Higher confidence = larger VaR**: 99% VaR > 95% VaR

---

## 🎯 Risk-Adjusted Metrics

### **Sharpe Ratio**
The most important metric - measures return per unit of total risk.

```
Sharpe Ratio = (Strategy Return - Risk-Free Rate) / Strategy Volatility
```

**Interpretation:**
- **>1.0**: Good risk-adjusted performance
- **>1.5**: Very good performance
- **>2.0**: Exceptional performance (rare in practice)
- **<0**: Strategy loses money or has worse risk-adjusted returns than risk-free rate

**Example:**
- Strategy: 12% return, 15% volatility
- Risk-free rate: 3%
- Sharpe = (12% - 3%) / 15% = 0.6

### **Sortino Ratio**
Like Sharpe ratio, but uses downside deviation instead of total volatility.

```
Sortino Ratio = (Strategy Return - Target Return) / Downside Deviation
```

**Interpretation:**
- **Better than Sharpe** for strategies with asymmetric returns

### **Calmar Ratio**
Compares annualized return to maximum drawdown.

```
Calmar Ratio = Annualized Return / Maximum Drawdown
```

**Interpretation:**
- **>1.0**: Return exceeds worst drawdown
- **>2.0**: Very strong performance
- **Focuses on worst-case scenarios**

---

## 📉 Drawdown Metrics

### **Maximum Drawdown (MDD)**
The largest peak-to-trough decline in portfolio value.

```
Drawdown = (Peak Value - Trough Value) / Peak Value
Maximum Drawdown = max(All Drawdowns)
```

**Interpretation:**
- **<10%**: Conservative strategy
- **10-20%**: Moderate risk
- **>30%**: High risk

### **Drawdown Duration**
How long it takes to recover from drawdowns.

**Interpretation:**
- **<6 months**: Quick recovery
- **1-2 years**: Acceptable for most investors
- **>3 years**: May test investor patience

### **Pain Index**
Measures the severity and duration of drawdowns combined.

```
Pain Index = Σ(Drawdown²) / Number of Periods
```

**Interpretation:**
- **Lower is better** (less cumulative pain)
- **Accounts for both depth and duration** of drawdowns

---

## 📈 Trade-Based Metrics

### **Win Rate**
Percentage of profitable trades.

```
Win Rate = Number of Winning Trades / Total Trades
```

**Interpretation:**
- **>50%**: More winners than losers
- **40-60%**: Typical for most strategies
- **>70%**: Excellent

**Important:** *High win rate doesn't guarantee profitability if losses are large*

### **Profit Factor**
Ratio of gross profits to gross losses.

```
Profit Factor = Total Winning Amount / Total Losing Amount
```

**Interpretation:**
- **>1.0**: Strategy is profitable
- **1.5-2.0**: Good performance
- **>2.5**: Excellent performance

### **Average Trade**
Mean profit/loss per trade.

```
Average Trade = Total Profit / Number of Trades
```

**Interpretation:**
- **Positive**: Profitable on average
- **Must exceed transaction costs** to be viable
- **Compare to buy & hold** equivalent per trade

### **Risk-Reward Ratio**
Average winning trade size compared to average losing trade size.

```
Risk-Reward Ratio = Average Win / Average Loss
```

**Interpretation:**
- **>1.0**: Winners larger than losers
- **>2.0**: Strong risk management
- **Can compensate for lower win rate**

---

## 📊 Interpretation Framework

### **🟢 Strong Performance Indicators**
- **Sharpe Ratio > 1.5**: Excellent risk-adjusted returns
- **Max Drawdown < 15%**: Manageable downside risk
- **Positive vs All Benchmarks**: Consistent outperformance
- **Calmar Ratio > 1.0**: Good return vs worst case
- **Win Rate > 50%**: More winners than losers

### **🟡 Warning Signs**
- **High Return, High Volatility**: May be taking excessive risk
- **Good Sharpe, Poor Calmar**: Hidden tail risks
- **Beats Moving Avg, Loses to Buy & Hold**: Limited value add
- **Inconsistent Performance**: May be overfitted

### **🔴 Red Flags**
- **Negative Sharpe Ratio**: Worse than risk-free investment
- **Max Drawdown > 30%**: Unacceptable downside risk
- **Very High Win Rate (>90%)**: Likely overfitted or using future data
- **Performance Too Good**: May indicate data snooping or errors

---
