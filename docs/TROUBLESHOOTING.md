# 🛠️ Troubleshooting Guide

> Comprehensive troubleshooting guide for common issues in ML trading applications, from data problems to performance issues and deployment challenges.

## 📋 Overview

This guide covers the most common issues you'll encounter when building and running ML trading systems, with practical solutions and prevention strategies.

---

## 📊 Data-Related Issues

### **🚨 Data Loading Errors**

#### **Problem: "No data downloaded for symbol XXX"**
```
Error: No data downloaded for INVALID_SYMBOL
```

**Causes:**
- Invalid stock ticker symbol
- Market closed during fetch attempt  
- Network connectivity issues
- API rate limiting from yfinance

**Solutions:**
```python
# Verify symbol exists
import yfinance as yf
ticker = yf.Ticker("AAPL")
info = ticker.info  # Check if data exists

# Add error handling
try:
    data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
    if data.empty:
        print("No data available for this symbol")
except Exception as e:
    print(f"Data fetch error: {e}")
```

**Prevention:**
- Use a list of known valid symbols
- Implement retry logic with delays
- Cache successful downloads

#### **Problem: "Insufficient data for analysis"**
```
Error: Need at least 100 data points, got 50
```

**Causes:**
- Date range too short
- Weekends/holidays in date range
- Recently listed company
- Data gaps from source

**Solutions:**
```python
# Check data length before analysis
if len(data) < required_periods:
    # Extend date range automatically
    start_date = pd.Timestamp.now() - pd.Timedelta(days=365*2)
    data = yf.download(symbol, start=start_date)

# Handle insufficient data gracefully
def safe_analysis(data, min_periods=100):
    if len(data) < min_periods:
        return {"error": f"Insufficient data: {len(data)} < {min_periods}"}
    return perform_analysis(data)
```

### **🚨 Date Range Issues**

#### **Problem: "Start date after end date"**
**Solution:**
```python
# Validate date inputs
def validate_dates(start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    if start >= end:
        # Auto-correct: set end date to 1 year after start
        end = start + pd.Timedelta(days=365)
        st.warning(f"End date adjusted to {end.date()}")
    
    return start, end
```

#### **Problem: "Future dates in data request"**
**Solution:**
```python
# Prevent future dates
def safe_date_range(start_date, end_date):
    today = pd.Timestamp.now().normalize()
    
    # Cap end date to today
    end_date = min(pd.to_datetime(end_date), today)
    
    # Ensure reasonable range
    if pd.to_datetime(start_date) >= end_date:
        start_date = end_date - pd.Timedelta(days=365)
    
    return start_date, end_date
```

---

## 🤖 Model Training Issues

### **🚨 Model Training Failures**

#### **Problem: "Model fails to converge"**
```
ConvergenceWarning: lbfgs failed to converge (status=1)
```

**Causes:**
- Learning rate too high/low
- Insufficient training iterations
- Poor feature scaling
- Inadequate training data

**Solutions:**
```python
# For MLP models
mlp = MLPClassifier(
    max_iter=2000,        # Increase iterations
    learning_rate_init=0.001,  # Lower learning rate
    early_stopping=True,  # Stop when no improvement
    validation_fraction=0.1
)

# For features scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### **Problem: "Perfect Training Accuracy (Overfitting)"**
```
Training Accuracy: 100%
Test Accuracy: 52%
```

**Causes:**
- Model too complex for data size
- Data leakage (future information)
- Insufficient regularization

**Solutions:**
```python
# Reduce model complexity
rf = RandomForestClassifier(
    n_estimators=50,      # Fewer trees
    max_depth=5,          # Limit depth
    min_samples_split=20, # Require more samples
    min_samples_leaf=10
)

# Add regularization for MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(50,),  # Smaller network
    alpha=0.01,               # L2 regularization
    early_stopping=True
)

# Check for data leakage
def check_data_leakage(features, target, dates):
    # Ensure features only use past data
    for i in range(1, len(features)):
        assert dates[i] > dates[i-1], "Features must be chronological"
```

### **🚨 Feature Engineering Problems**

#### **Problem: "Technical indicators return NaN values"**
```
Error: All technical indicator values are NaN
```

**Causes:**
- Insufficient data for indicator calculation
- Wrong parameter settings
- Data quality issues

**Solutions:**
```python
# Safe indicator calculation
def safe_technical_indicators(data, periods=20):
    try:
        # Ensure sufficient data
        if len(data) < periods * 2:
            st.warning(f"Insufficient data for {periods}-period indicators")
            return pd.DataFrame()
        
        # Calculate with error handling
        data['SMA'] = data['Close'].rolling(periods).mean()
        data['RSI'] = ta.rsi(data['Close'], length=periods)
        
        # Drop NaN values
        return data.dropna()
    
    except Exception as e:
        st.error(f"Technical indicator error: {e}")
        return pd.DataFrame()
```

#### **Problem: "Features have different scales"**
**Solution:**
```python
# Feature scaling pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

def prepare_features(X):
    # Use RobustScaler for financial data (handles outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check for infinite values
    X_scaled = np.where(np.isinf(X_scaled), 0, X_scaled)
    
    return X_scaled, scaler
```

---

## 📈 Performance Issues

### **🚨 Poor Model Performance**

#### **Problem: "ML model worse than buy & hold"**

**Diagnostic Steps:**
```python
# Performance comparison
def diagnose_performance(results):
    print(f"ML Return: {results['ml_return']:.2%}")
    print(f"Buy&Hold Return: {results['buy_hold_return']:.2%}")
    print(f"ML Sharpe: {results['ml_sharpe']:.2f}")
    print(f"Buy&Hold Sharpe: {results['buy_hold_sharpe']:.2f}")
    
    # Check transaction costs
    print(f"Number of trades: {results['num_trades']}")
    print(f"Transaction cost impact: {results['transaction_costs']:.2%}")
```

**Solutions:**
1. **Reduce Trading Frequency**: Lower transaction costs
2. **Improve Features**: Add more predictive indicators
3. **Better Risk Management**: Position sizing and stop-losses
4. **Ensemble Methods**: Combine multiple models

#### **Problem: "High volatility in ML predictions"**
**Solutions:**
```python
# Smooth predictions
def smooth_predictions(predictions, window=5):
    return pd.Series(predictions).rolling(window).mean().fillna(predictions)

# Confidence-based trading
def confidence_based_signals(predictions, probabilities, threshold=0.6):
    # Only trade when model is confident
    confident_predictions = predictions.copy()
    low_confidence = probabilities.max(axis=1) < threshold
    confident_predictions[low_confidence] = 0  # No trade
    return confident_predictions
```

### **🚨 Memory and Performance Issues**

#### **Problem: "Application runs slowly"**
**Solutions:**
```python
# Cache expensive calculations
@st.cache_data
def load_and_process_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return calculate_indicators(data)

# Optimize data processing
def optimize_dataframe(df):
    # Use appropriate dtypes
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```

#### **Problem: "Memory usage too high"**
**Solutions:**
```python
# Process data in chunks
def process_large_dataset(data, chunk_size=1000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result)
    return pd.concat(results)

# Clean up memory
import gc
gc.collect()  # Force garbage collection
```

---

## 🖥️ Streamlit Application Issues

### **🚨 UI/UX Problems**

#### **Problem: "Streamlit app crashes on user input"**
**Solutions:**
```python
# Input validation
def validate_inputs():
    try:
        if 'symbol' not in st.session_state or not st.session_state.symbol:
            st.error("Please enter a stock symbol")
            return False
        
        if st.session_state.start_date >= st.session_state.end_date:
            st.error("Start date must be before end date")
            return False
        
        return True
    
    except Exception as e:
        st.error(f"Input validation error: {e}")
        return False

# Wrap main logic in try-catch
def safe_main():
    try:
        if validate_inputs():
            run_analysis()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again")
```

#### **Problem: "Charts not displaying properly"**
**Solutions:**
```python
# Safe plotting
def safe_plot(data, title="Chart"):
    try:
        if data.empty:
            st.warning("No data to plot")
            return
        
        fig = px.line(data, title=title)
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Plotting error: {e}")
        st.write("Raw data:", data.head())
```

### **🚨 Session State Issues**

#### **Problem: "App resets unexpectedly"**
**Solutions:**
```python
# Initialize session state properly
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = ""
    if 'results' not in st.session_state:
        st.session_state.results = {}

# Persistent data storage
def save_results(results):
    st.session_state.results = results
    # Optionally save to file for persistence
    with open('last_results.pkl', 'wb') as f:
        pickle.dump(results, f)
```

---

## 🔧 Environment and Dependencies

### **🚨 Package Import Errors**

#### **Problem: "ModuleNotFoundError"**
```
ModuleNotFoundError: No module named 'pandas_ta'
```

**Solutions:**
```bash
# Install missing packages
pip install pandas-ta
pip install yfinance
pip install plotly
pip install streamlit

# Or install from requirements.txt
pip install -r requirements.txt
```

#### **Problem: "Version conflicts"**
**Solutions:**
```bash
# Create clean environment
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
# trading_env\Scripts\activate  # Windows

# Install specific versions
pip install pandas==1.5.3
pip install scikit-learn==1.3.0
```

### **🚨 Configuration Issues**

#### **Problem: "Streamlit configuration errors"**
**Solution: Create `.streamlit/config.toml`:**
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[server]
port = 8501
maxUploadSize = 200
```

---

## 🐛 Debugging Strategies

### **🔍 Systematic Debugging Approach**

#### **1. Enable Logging**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_function(data):
    logger.info(f"Input data shape: {data.shape}")
    logger.info(f"Data range: {data.index[0]} to {data.index[-1]}")
    # Your function logic here
    logger.info("Function completed successfully")
```

#### **2. Add Checkpoints**
```python
def analyze_with_checkpoints(data):
    st.write("✅ Data loaded successfully")
    
    features = create_features(data)
    st.write(f"✅ Features created: {features.shape}")
    
    model = train_model(features)
    st.write("✅ Model trained successfully")
    
    results = generate_results(model, features)
    st.write("✅ Results generated")
    
    return results
```

#### **3. Data Validation**
```python
def validate_data_quality(data):
    issues = []
    
    if data.isnull().any().any():
        issues.append("Contains NaN values")
    
    if (data == 0).all().any():
        issues.append("Contains columns of all zeros")
    
    if len(data) < 100:
        issues.append(f"Insufficient data: {len(data)} rows")
    
    if issues:
        st.error("Data quality issues found:")
        for issue in issues:
            st.write(f"- {issue}")
        return False
    
    return True
```

---

## 📋 Troubleshooting Checklist

### **Before Running Analysis:**
- [ ] Valid stock symbol entered
- [ ] Reasonable date range selected
- [ ] Sufficient data available (>100 points)
- [ ] All required packages installed
- [ ] No conflicting package versions

### **During Model Training:**
- [ ] Features properly scaled
- [ ] No data leakage in feature creation
- [ ] Adequate training data size
- [ ] Model parameters reasonable
- [ ] Training converges successfully

### **After Getting Results:**
- [ ] Results make economic sense
- [ ] Performance metrics reasonable
- [ ] No obvious overfitting signs
- [ ] Transaction costs considered
- [ ] Compared to benchmarks

### **Before Deployment:**
- [ ] Out-of-sample testing performed
- [ ] Error handling implemented
- [ ] User input validation added
- [ ] Performance acceptable
- [ ] Documentation complete

---

## 🆘 Emergency Fixes

### **Quick Fixes for Common Crashes:**

#### **App Won't Start:**
```python
# Minimal working version
import streamlit as st

st.title("ML Trading App")
st.write("If you see this, Streamlit is working")

# Add components one by one to identify the problem
```

#### **Data Loading Fails:**
```python
# Fallback to sample data
try:
    data = yf.download(symbol, start=start_date, end=end_date)
except:
    st.warning("Using sample data due to download failure")
    data = create_sample_data()
```

#### **Model Training Fails:**
```python
# Fallback to simple model
try:
    model = MLPClassifier().fit(X, y)
except:
    st.warning("Using simple model due to training failure")
    model = DummyClassifier(strategy='most_frequent').fit(X, y)
```

---

## 📞 Getting Help

### **When to Seek Help:**
- Persistent errors after trying documented solutions
- Performance significantly worse than expected
- Data quality issues you can't resolve
- Deployment problems in production

### **How to Ask for Help:**
1. **Describe the problem clearly**
2. **Include error messages** (full stack trace)
3. **Share relevant code** that reproduces the issue
4. **Specify your environment** (Python version, OS, package versions)
5. **Explain what you've already tried**

### **Useful Resources:**
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Stack Overflow**: Tag questions with specific libraries

---

*Remember: Most issues in ML trading applications are data-related. Always start by validating your data quality and checking for common data problems before investigating model or application issues.*
