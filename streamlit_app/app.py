"""
Streamlit application for ML Trading Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Try to import yfinance for data fetching
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    st.warning("⚠️ yfinance not available. Using synthetic data.")

# Add src and config to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'config'))

try:
    from config.settings import SUPPORTED_ASSETS, DataConfig, ModelConfig, BacktestConfig, STREAMLIT_CONFIG
    from src.data.fetcher import DataFetcher
    from src.models.ml_models import get_all_ml_models, get_model_descriptions
    from src.strategies.naive import get_all_naive_strategies, get_strategy_descriptions
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    HAS_ALL_IMPORTS = True
except ImportError as e:
    HAS_ALL_IMPORTS = False
    st.warning(f"⚠️ Some imports failed: {e}")
    st.info("🔄 Using fallback mode with demo data")
    
    # Fallback configuration - demo data
    SUPPORTED_ASSETS = {
        '^GSPC': {
            'name': 'S&P 500',
            'type': 'index',
            'description': 'US Stock Market Index',
            'currency': 'USD',
            'sector': 'Market Index'
        },
        'AAPL': {
            'name': 'Apple Inc.',
            'type': 'stock',
            'description': 'Technology Hardware & Equipment',
            'currency': 'USD',
            'sector': 'Technology'
        },
        'BTC-USD': {
            'name': 'Bitcoin',
            'type': 'crypto',
            'description': 'Digital Currency',
            'currency': 'USD',
            'sector': 'Cryptocurrency'
        }
    }
    
    # Fallback DataConfig
    class DataConfig:
        def __init__(self, **kwargs):
            self.start_date = kwargs.get('start_date', '2020-01-01')
            self.end_date = kwargs.get('end_date', '2024-12-31')
            self.ma_periods = kwargs.get('ma_periods', [5, 10, 20, 50, 200])
            self.rsi_period = kwargs.get('rsi_period', 14)
            self.macd_fast = kwargs.get('macd_fast', 12)
            self.macd_slow = kwargs.get('macd_slow', 26)
            self.macd_signal = kwargs.get('macd_signal', 9)
            self.bb_period = kwargs.get('bb_period', 20)
            self.bb_std = kwargs.get('bb_std', 2.0)
            self.train_ratio = kwargs.get('train_ratio', 0.7)
            self.val_ratio = kwargs.get('val_ratio', 0.15)
            self.test_ratio = kwargs.get('test_ratio', 0.15)
    
    # Fallback DataFetcher
    class DataFetcher:
        def __init__(self, config=None):
            self.config = config or DataConfig()
            
        def fetch_and_prepare(self, symbol, start_date, end_date):
            # Créer des données synthétiques pour la démonstration
            if HAS_YFINANCE:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    
                    if data.empty:
                        raise ValueError("No data available")
                    
                    # Create simple features
                    X = pd.DataFrame(index=data.index)
                    X['Close'] = data['Close']
                    X['Volume'] = data['Volume'] if 'Volume' in data.columns else 1000
                    X['High'] = data['High']
                    X['Low'] = data['Low']
                    X['Open'] = data['Open']
                    
                    # Add simple technical features
                    X['Returns'] = X['Close'].pct_change()
                    X['MA_5'] = X['Close'].rolling(5).mean()
                    X['MA_20'] = X['Close'].rolling(20).mean()
                    X['RSI'] = 50  # Simplified RSI
                    
                    # Remove NaN
                    X = X.dropna()
                    
                    # Create target (1 if up next day, 0 otherwise)
                    y = (X['Close'].shift(-1) > X['Close']).astype(int)
                    y = y[:-1]  # Remove last element
                    X = X[:-1]  # Remove last element
                    
                    return X, y
                except Exception as e:
                    st.warning(f"yfinance error: {e}. Using synthetic data.")
            
            # Create completely synthetic data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            n = len(dates)
            
            X = pd.DataFrame(index=dates[:n-1])
            X['Close'] = 100 + np.cumsum(np.random.randn(n-1) * 0.02)
            X['Volume'] = 1000 + np.random.randint(0, 500, n-1)
            X['Returns'] = np.random.randn(n-1) * 0.02
            X['MA_5'] = X['Close']
            X['MA_20'] = X['Close']
            X['RSI'] = 50
            
            y = pd.Series(np.random.choice([0, 1], n-1, p=[0.45, 0.55]), index=dates[:n-1])
            
            return X, y
    
    # Fallback model functions
    def get_all_ml_models():
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        return {
            'Random Forest': RandomForestClassifier,
            'MLP': MLPClassifier
        }
    
    def get_model_descriptions():
        return {
            'Random Forest': 'Random Forest',
            'MLP': 'MLP'
        }
    
    def get_all_naive_strategies():
        return {
            'Bullish': AlwaysUpStrategy,
            'Bearish': AlwaysDownStrategy,
            'Random': RandomStrategy,
            'Frequency': FrequencyStrategy
        }
    
    def get_strategy_descriptions():
        return {
            'Bullish': 'Bullish',
            'Bearish': 'Bearish',
            'Random': 'Random',
            'Frequency': 'Frequency'
        }

# Fallback strategy classes for demo mode
class AlwaysUpStrategy:
    def __init__(self, random_state=None):
        self.random_state = random_state
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return np.ones(len(X))

class AlwaysDownStrategy:
    def __init__(self, random_state=None):
        self.random_state = random_state
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return np.zeros(len(X))

class RandomStrategy:
    def __init__(self, random_state=None):
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
            
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return np.random.choice([0, 1], len(X))

class FrequencyStrategy:
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.up_frequency = 0.5
        
    def fit(self, X, y):
        self.up_frequency = y.mean()
        return self
        
    def predict(self, X):
        return np.random.choice([0, 1], len(X), p=[1-self.up_frequency, self.up_frequency])


# Function to get short model names for display
def get_short_name(long_name):
    """Convert long model names to short display names"""
    name_mapping = {
        'Random Forest': 'Random Forest',
        'Multi-Layer Perceptron': 'MLP',
        'MLP': 'MLP',
        'Always Up': 'Bullish',
        'Bullish': 'Bullish',
        'Always Down': 'Bearish',
        'Bearish': 'Bearish',
        'Random (50/50)': 'Random',
        'Random': 'Random',
        'Historical Frequency': 'Frequency',
        'Frequency': 'Frequency',
        'Momentum (Last Direction)': 'Momentum',
        'Mean Reversion (Contrarian)': 'Mean Reversion',
        # Fallback pour les descriptions longues en français
        'Réseau de neurones feed-forward. Capable de capturer des relations complexes non-linéaires entre les features.': 'MLP',
        'Ensemble de arbres de décision. Robuste, gère bien les features non-linéaires et fournit l\'importance des variables.': 'Random Forest'
    }
    return name_mapping.get(long_name, long_name)


# ==================== HELPER FUNCTIONS ====================

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    /* Main container */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    
    /* Success boxes */
    .element-container div[data-testid="stAlert"][data-baseweb="notification"] {
        border-radius: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Progress indicator styling */
    .stProgress > div > div > div {
        background-color: #28a745;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 0.5rem;
    }
    
    /* Chart containers */
    .stPlotlyChart {
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def validate_date_range(start_date, end_date):
    """Validate date range and return warnings if any"""
    warnings = []
    
    if start_date >= end_date:
        warnings.append("Start date must be before end date")
    
    days_diff = (end_date - start_date).days
    if days_diff < 30:
        warnings.append("Period too short (< 1 month)")
    elif days_diff < 365:
        warnings.append("Short period (< 1 year). Recommended: at least 2 years")
    
    if end_date > datetime.now().date():
        warnings.append("End date cannot be in the future")
    
    return warnings


def create_price_chart(data, title="Asset Price"):
    """Create price chart with optional indicators"""
    
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add moving averages if available
    for col in data.columns:
        if 'MA_' in col and len(data.columns) <= 10:  # Limit to avoid clutter
            period = col.split('_')[1]
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines',
                name=f'MA {period}',
                line=dict(width=1),
                opacity=0.7
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    
    return fig


def display_metric_cards(metrics_dict):
    """Display metrics in a card format"""
    # Determine number of columns based on number of metrics
    n_metrics = len(metrics_dict)
    n_cols = min(4, n_metrics)
    
    if n_cols > 0:
        cols = st.columns(n_cols)
        
        for i, (metric_name, metric_value) in enumerate(metrics_dict.items()):
            with cols[i % n_cols]:
                if isinstance(metric_value, float):
                    st.metric(metric_name, f"{metric_value:.3f}")
                else:
                    st.metric(metric_name, str(metric_value))


def create_info_box(title, content, box_type="info"):
    """Create styled info box"""
    colors = {
        "info": ("#d1ecf1", "#0c5460"),
        "success": ("#d4edda", "#155724"),
        "warning": ("#fff3cd", "#856404"),
        "error": ("#f8d7da", "#721c24")
    }
    
    bg_color, text_color = colors.get(box_type, colors["info"])
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        color: {text_color};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {text_color};
        margin: 1rem 0;
    ">
        <h4 style="margin-top: 0; color: {text_color};">{title}</h4>
        <p style="margin-bottom: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


# ==================== MAIN APPLICATION ====================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ML Trading Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main Streamlit application"""
    
    # Apply custom styling
    try:
        apply_custom_css()
    except:
        pass
    
    st.title("📈 ML Trading Prediction")
    st.markdown("*Financial market directional prediction with Machine Learning*")
    st.markdown("---")
    
    # Initialize session state for workflow
    init_session_state()
    
    # Progress indicator
    display_progress_indicator()
    
    st.markdown("---")
    
    # Navigation between steps
    if st.session_state.step == 1:
        render_step_1_data_configuration()
    elif st.session_state.step == 2:
        render_step_2_model_configuration()
    elif st.session_state.step == 3:
        render_step_3_results_evaluation()


def init_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = {}
    if 'selected_benchmarks' not in st.session_state:
        st.session_state.selected_benchmarks = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}


def display_progress_indicator():
    """Display progress indicator for the 3-step workflow"""
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        if st.session_state.step >= 1:
            st.success("✅ 1. Data Configuration")
        else:
            st.info("⏳ 1. Data Configuration")
    
    with progress_col2:
        if st.session_state.step >= 2:
            st.success("✅ 2. Model Configuration")
        else:
            st.info("⏳ 2. Model Configuration")
    
    with progress_col3:
        if st.session_state.step >= 3:
            st.success("✅ 3. Results & Evaluation")
        else:
            st.info("⏳ 3. Results & Evaluation")


def render_step_1_data_configuration():
    """Step 1: Data Configuration & Feature Engineering"""
    
    st.header("📊 Step 1: Data Configuration")
    st.markdown("Configure technical indicators and advanced parameters for data processing.")
    
    # Sidebar for Asset Selection, Time Period, and Load button
    with st.sidebar:
        st.header("📈 Asset & Period")
        
        # Asset selection
        selected_symbol = st.selectbox(
            "Choose asset:",
            options=list(SUPPORTED_ASSETS.keys()),
            format_func=lambda x: f"{SUPPORTED_ASSETS[x]['name']} ({x})",
            help="Select the financial asset to analyze"
        )
        
        asset_info = SUPPORTED_ASSETS[selected_symbol]
        
        # Date range
        st.subheader("✂️ Data Split")
        start_date = st.date_input(
            "Start date:",
            value=datetime.strptime("2020-01-01", "%Y-%m-%d").date(),
            max_value=datetime.now().date() - timedelta(days=30)
        )
        end_date = st.date_input(
            "End date:",
            value=datetime.strptime("2024-12-31", "%Y-%m-%d").date(),
            max_value=datetime.now().date()
        )
        
        # Data validation
        date_warnings = validate_date_range(start_date, end_date)
        if date_warnings:
            st.warning("⚠️ Date issues:")
            for warning in date_warnings:
                st.caption(f"• {warning}")
        else:
            st.success("✅ Valid period")
        
        # Load data button in sidebar
        st.markdown("---")
        load_data_btn = st.button("📊 Load & Process Data", type="primary", use_container_width=True)
    
    # Main screen for advanced parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset info display
        st.subheader("ℹ️ Selected Asset")
        st.info(f"**{asset_info['name']}** ({selected_symbol})")
        st.write(f"**Type:** {asset_info['type'].title()}")
        st.write(f"**Sector:** {asset_info['sector']}")
        st.write(f"**Currency:** {asset_info['currency']}")
        
        # Technical indicators configuration
        st.subheader("⚙️ Technical Indicators")
        
        # Helper button for indicator explanations
        with st.expander("ℹ️ Indicator Explanations"):
            st.markdown("""
            **📈 Moving Averages (MA)**: Average price over a specified period. Shows trend direction and support/resistance levels.
            
            **⚡ RSI (Relative Strength Index)**: Momentum oscillator (0-100) that measures speed and change of price movements. Values above 70 indicate overbought conditions, below 30 indicate oversold.
            
            **📉 MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator that shows the relationship between two moving averages. Helps identify trend changes and momentum shifts.
            
            **📊 Bollinger Bands**: Price volatility bands around a moving average. The bands widen during volatile periods and contract during stable periods.
            
            *Note: Only Moving Averages and Bollinger Bands appear on the main price chart. RSI and MACD are calculated as features for model training.*
            """)
        
        indicator_col1, indicator_col2 = st.columns(2)
        with indicator_col1:
            use_ma = st.checkbox("Moving Averages", value=True)
            use_rsi = st.checkbox("RSI (Relative Strength Index)", value=True)
        with indicator_col2:
            use_macd = st.checkbox("MACD", value=True)
            use_bb = st.checkbox("Bollinger Bands", value=False)
        
        # Moving Averages parameters
        if use_ma:
            st.write("**Moving Average Periods:**")
            ma_periods = st.multiselect(
                "Select periods:",
                options=[5, 10, 20, 50, 100, 200],
                default=[20, 50, 100],
                help="Choose moving average periods (max 3)",
                max_selections=3
            )
        else:
            ma_periods = []
    
    with col2:
        # RSI parameters
        if use_rsi:
            st.subheader("📈 RSI Configuration")
            rsi_period = st.slider("RSI Period:", 5, 30, 14, help="Period for RSI calculation")
        else:
            rsi_period = None
        
        # MACD parameters
        if use_macd:
            st.subheader("📉 MACD Configuration")
            macd_col1, macd_col2, macd_col3 = st.columns(3)
            with macd_col1:
                macd_fast = st.number_input("Fast:", 5, 20, 12)
            with macd_col2:
                macd_slow = st.number_input("Slow:", 20, 40, 26)
            with macd_col3:
                macd_signal = st.number_input("Signal:", 5, 15, 9)
        else:
            macd_fast = macd_slow = macd_signal = None
        
        # Bollinger Bands parameters
        if use_bb:
            st.subheader("📈 Bollinger Bands Configuration")
            bb_col1, bb_col2 = st.columns(2)
            with bb_col1:
                bb_period = st.slider("BB Period:", 10, 30, 20)
            with bb_col2:
                bb_std = st.slider("Standard Deviation:", 1.0, 3.0, 2.0, 0.1)
        else:
            bb_period = bb_std = None
        
        # Data split configuration
        st.subheader("✂️ Data Split")
        train_ratio = st.slider("Training Ratio:", 0.6, 0.8, 0.7, 0.05)
        test_ratio = 1.0 - train_ratio
        val_ratio = 0.0  # Simplified
        st.metric("Test Ratio:", f"{test_ratio:.2f}")
    
    # Process data loading
    if load_data_btn:
        if start_date >= end_date:
            st.error("❌ Please correct the dates in the sidebar")
            return
            
        with st.spinner("Loading and processing data..."):
            try:
                # Create data configuration
                data_config = DataConfig(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    ma_periods=ma_periods if use_ma else [],
                    rsi_period=rsi_period if use_rsi else None,
                    macd_fast=macd_fast if use_macd else None,
                    macd_slow=macd_slow if use_macd else None,
                    macd_signal=macd_signal if use_macd else None,
                    bb_period=bb_period if use_bb else None,
                    bb_std=bb_std if use_bb else None,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio
                )
                
                # Initialize data fetcher
                fetcher = DataFetcher(config=data_config)
                
                # Fetch and process data
                st.info(f"Fetching data for {asset_info['name']}...")
                try:
                    X, y = fetcher.fetch_and_prepare(
                        symbol=selected_symbol,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d")
                    )
                except AttributeError:
                    X, y = fetcher.process_symbol(selected_symbol)
                
                # Check if we have enough data
                if len(X) == 0 or len(y) == 0:
                    st.error(f"❌ No valid data available for {asset_info['name']} after processing. This can happen with assets that have insufficient data or too many technical indicators for the time period.")
                    st.info("💡 Try:\n- Selecting a longer time period\n- Choosing fewer technical indicators\n- Selecting a different asset")
                    return
                
                if len(X) < 100:
                    st.warning(f"⚠️ Only {len(X)} samples available. Consider a longer time period for better model performance.")
                
                # Split data
                split_idx = max(1, int(len(X) * train_ratio))  # Ensure at least 1 sample in training
                
                st.session_state.X_train = X.iloc[:split_idx]
                st.session_state.X_test = X.iloc[split_idx:]
                st.session_state.y_train = y.iloc[:split_idx]
                st.session_state.y_test = y.iloc[split_idx:]
                
                # Store raw data for visualization
                try:
                    st.session_state.raw_data = fetcher.get_raw_data()
                except AttributeError:
                    st.session_state.raw_data = X
                st.session_state.data_loaded = True
                
                st.success("✅ Data loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return
    
    # Display results if data is loaded
    if st.session_state.data_loaded and st.session_state.X_train is not None:
        
        st.markdown("---")
        st.subheader("📈 Data Summary")
        
        # Clean metrics display
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Total Samples", len(st.session_state.X_train) + len(st.session_state.X_test))
        with metric_col2:
            st.metric("Features", st.session_state.X_train.shape[1])
        with metric_col3:
            st.metric("Training", len(st.session_state.X_train))
        with metric_col4:
            st.metric("Testing", len(st.session_state.X_test))
        
        # Target distribution
        if len(st.session_state.y_train) > 0 and len(st.session_state.y_test) > 0:
            total_up_moves = st.session_state.y_train.sum() + st.session_state.y_test.sum()
            total_samples = len(st.session_state.y_train) + len(st.session_state.y_test)
            up_ratio = total_up_moves / total_samples if total_samples > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Up Days", f"{up_ratio:.1%}")
            with col2:
                st.metric("Down Days", f"{1-up_ratio:.1%}")
        else:
            st.warning("⚠️ No valid target data available")
        # Price chart
        if st.session_state.raw_data is not None:
            st.subheader("📈 Price Evolution")
            
            fig = go.Figure()
            raw_data = st.session_state.raw_data
            fig.add_trace(go.Scatter(
                x=raw_data.index,
                y=raw_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add main MAs for clarity
            if use_ma and ma_periods:
                for period in ma_periods[:3]:  # Show up to 3 MAs
                    ma_col = f'MA_{period}'
                    if ma_col in raw_data.columns:
                        fig.add_trace(go.Scatter(
                            x=raw_data.index,
                            y=raw_data[ma_col],
                            mode='lines',
                            name=f'MA {period}',
                            line=dict(width=1),
                            opacity=0.7
                        ))
            
            # Add Bollinger Bands if available
            if use_bb and 'BB_Upper' in raw_data.columns and 'BB_Lower' in raw_data.columns:
                fig.add_trace(go.Scatter(
                    x=raw_data.index,
                    y=raw_data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='red', width=1, dash='dash'),
                    opacity=0.5
                ))
                fig.add_trace(go.Scatter(
                    x=raw_data.index,
                    y=raw_data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='red', width=1, dash='dash'),
                    opacity=0.5,
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)'
                ))
            
            fig.update_layout(
                title=f"Price Evolution - {asset_info['name']}",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="price_evolution_chart")
        
        # Next step button on main screen
        st.success("✅ Data ready for model training!")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("➡️ Configure Models", type="primary"):
                st.session_state.step = 2
                st.rerun()


def render_step_2_model_configuration():
    """Step 2: Model Configuration"""
    
    st.header("🤖 Step 2: Model Configuration")
    st.markdown("Select ML models and benchmark strategies, configure their parameters.")
    
    if not st.session_state.data_loaded:
        st.error("❌ No data loaded. Return to step 1.")
        if st.button("⬅️ Back to Data Configuration"):
            st.session_state.step = 1
            st.rerun()
        return
    
    # Sidebar for model selection and training
    with st.sidebar:
        st.header("🤖 Model Selection")
        
        # ML Models selection (simple names only)
        available_models = get_all_ml_models()
        model_descriptions = get_model_descriptions()
        
        selected_ml_models = st.multiselect(
            "ML Models:",
            options=list(available_models.keys()),
            default=[],  # No default selection
            help="Select ML models to train"
        )
        
        # Benchmark strategies selection (simple names only)
        available_benchmarks = get_all_naive_strategies()
        benchmark_descriptions = get_strategy_descriptions()
        
        selected_benchmarks = st.multiselect(
            "Benchmarks:",
            options=list(available_benchmarks.keys()),
            default=[],  # No default selection
            help="Select benchmark strategies"
        )
        
        # Model parameters for selected models
        st.subheader("⚙️ Parameters")
        model_configs = {}
        
        for model_name in selected_ml_models:
            if model_name == 'Random Forest':
                with st.expander("Random Forest"):
                    n_estimators = st.slider("Trees:", 50, 500, 200, key=f"{model_name}_n_est")
                    max_depth = st.slider("Max Depth:", 5, 50, 10, key=f"{model_name}_depth")
                    model_configs[model_name] = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'random_state': 42
                    }
            
            elif model_name == 'MLP':
                with st.expander("MLP"):
                    hidden_size = st.slider("Hidden Size:", 50, 200, 100, key=f"{model_name}_hidden")
                    max_iter = st.slider("Max Iterations:", 200, 2000, 1000, key=f"{model_name}_iter")
                    model_configs[model_name] = {
                        'hidden_layer_sizes': (hidden_size,),
                        'max_iter': max_iter,
                        'random_state': 42
                    }
        
        # Train models button in sidebar
        st.markdown("---")
        train_models_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)
    
    # Main screen for detailed descriptions and data summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Selected ML Models with descriptions
        st.subheader("🤖 Selected ML Models")
        if selected_ml_models:
            for model_name in selected_ml_models:
                model_display_name = model_descriptions.get(model_name, model_name)
                
                if model_name == 'Random Forest':
                    description = "**Random Forest** - Ensemble of decision trees. Robust algorithm that handles non-linear features well and provides feature importance."
                elif model_name == 'MLP':
                    description = "**MLP** - Neural network with fully connected layers. Capable of learning complex non-linear patterns."
                else:
                    description = f"**{model_display_name}** - Machine learning model for classification."
                
                st.info(description)
        else:
            st.warning("No ML models selected")
        
        # Selected Benchmarks with descriptions
        st.subheader("🎯 Selected Benchmark Strategies")
        if selected_benchmarks:
            for bench_name in selected_benchmarks:
                bench_display_name = benchmark_descriptions.get(bench_name, bench_name)
                
                if bench_name == 'Bullish':
                    description = "**Bullish Strategy** - Always predicts upward movement. Simple baseline that assumes the market generally trends upward over time."
                elif bench_name == 'Bearish':
                    description = "**Bearish Strategy** - Always predicts downward movement. Simple baseline that assumes the market generally trends downward over time."
                elif bench_name == 'Random':
                    description = "**Random Strategy** - Makes random predictions (50/50). Represents pure chance and helps establish a baseline performance level."
                elif bench_name == 'Frequency':
                    description = "**Frequency Strategy** - Predicts based on historical frequency of up/down movements. Uses past market behavior to inform predictions."
                elif bench_name == 'Momentum (Last Direction)':
                    description = "**Momentum Strategy** - Assumes that trends persist and the market will continue in the same direction as the most recent movement."
                elif bench_name == 'Mean Reversion (Contrarian)':
                    description = "**Mean Reversion Strategy** - Assumes that prices will reverse direction from recent movements."
                else:
                    description = f"**{bench_display_name}** - Baseline strategy for comparison."
                
                st.info(description)
        else:
            st.warning("No benchmark strategies selected")
    
    with col2:
        # Data summary from Step 1
        st.subheader("📋 Data Summary")
        st.metric("Total Samples", len(st.session_state.X_train) + len(st.session_state.X_test))
        st.metric("Features", st.session_state.X_train.shape[1])
        st.metric("Training", len(st.session_state.X_train))
        st.metric("Testing", len(st.session_state.X_test))
        
        # Configuration summary
        st.subheader("⚙️ Selection Summary")
        st.write(f"**ML Models:** {len(selected_ml_models)}")
        st.write(f"**Benchmarks:** {len(selected_benchmarks)}")
        st.write(f"**Total:** {len(selected_ml_models) + len(selected_benchmarks)}")
    
    # Validation warnings
    if not selected_ml_models and not selected_benchmarks:
        st.error("⚠️ Please select at least one model or benchmark strategy in the sidebar")
    
    # Train models
    if train_models_btn:
        if not selected_ml_models and not selected_benchmarks:
            st.error("❌ Please select at least one model")
            return
        
        with st.spinner("Training models..."):
            try:
                results = {}
                progress_bar = st.progress(0)
                total_models = len(selected_ml_models) + len(selected_benchmarks)
                current_model = 0
                
                # Train ML models
                for model_name in selected_ml_models:
                    st.info(f"Training: {model_descriptions.get(model_name, model_name)}")
                    
                    model_class = available_models[model_name]
                    config = model_configs.get(model_name, {})
                    
                    model = model_class(**config)
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                    
                    train_pred = model.predict(st.session_state.X_train)
                    test_pred = model.predict(st.session_state.X_test)
                    
                    train_acc = accuracy_score(st.session_state.y_train, train_pred)
                    test_acc = accuracy_score(st.session_state.y_test, test_pred)
                    
                    results[model_name] = {
                        'type': 'ml_model',
                        'name': model_descriptions.get(model_name, model_name),
                        'model': model,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'train_predictions': train_pred,
                        'test_predictions': test_pred,
                        'config': config
                    }
                    
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                
                # Train benchmark strategies
                for bench_name in selected_benchmarks:
                    st.info(f"Training: {benchmark_descriptions.get(bench_name, bench_name)}")
                    
                    strategy_class = available_benchmarks[bench_name]
                    strategy = strategy_class(random_state=42)
                    strategy.fit(st.session_state.X_train, st.session_state.y_train)
                    
                    train_pred = strategy.predict(st.session_state.X_train)
                    test_pred = strategy.predict(st.session_state.X_test)
                    
                    train_acc = accuracy_score(st.session_state.y_train, train_pred)
                    test_acc = accuracy_score(st.session_state.y_test, test_pred)
                    
                    results[bench_name] = {
                        'type': 'benchmark',
                        'name': benchmark_descriptions.get(bench_name, bench_name),
                        'model': strategy,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'train_predictions': train_pred,
                        'test_predictions': test_pred
                    }
                    
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                
                st.session_state.results = results
                st.session_state.selected_models = {k: v for k, v in results.items() if v['type'] == 'ml_model'}
                st.session_state.selected_benchmarks = {k: v for k, v in results.items() if v['type'] == 'benchmark'}
                
                progress_bar.progress(1.0)
                st.success("✅ All models trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
                logger.error(f"Training error: {e}")
    
    # Navigation buttons on main screen
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Data"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.session_state.results:
            if st.button("➡️ View Results", type="primary"):
                st.session_state.step = 3
                st.rerun()


def render_step_3_results_evaluation():
    """Step 3: Results & Evaluation"""
    
    st.header("📊 Step 3: Results & Evaluation")
    st.markdown("Detailed performance analysis of models and benchmark strategies.")
    
    if not st.session_state.results:
        st.error("❌ No results available. Return to previous steps.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Models"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("🔄 New Analysis"):
                reset_session_state()
                st.rerun()
        return
    
    # Results summary
    st.subheader("🏆 Model Rankings")
    
    # Sort results by test accuracy
    sorted_results = sorted(st.session_state.results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    # Create clean results dataframe
    results_data = []
    for model_name, result in sorted_results:
        # Use short names for display
        short_name = get_short_name(result['name'])
        results_data.append({
            'Model': short_name,
            'Type': '🤖 ML' if result['type'] == 'ml_model' else '🎯 Benchmark',
            'Train Accuracy': f"{result['train_accuracy']:.3f}",
            'Test Accuracy': f"{result['test_accuracy']:.3f}",
            'Overfitting': f"{result['train_accuracy'] - result['test_accuracy']:.3f}"
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Display results table
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Performance metrics for top 3 models
    st.subheader("📈 Detailed Analysis")
    
    top_models = sorted_results[:3]
    
    for i, (model_name, result) in enumerate(top_models):
        rank_emoji = ["🥇", "🥈", "🥉"][i]
        model_type = "🤖" if result['type'] == 'ml_model' else "🎯"
        short_name = get_short_name(result['name'])
        
        with st.expander(f"{rank_emoji} {model_type} {short_name} - Detailed Metrics"):
            
            # Calculate additional metrics
            test_pred = result['test_predictions']
            y_test = st.session_state.y_test
            
            precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{result['test_accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1-Score", f"{f1:.3f}")
            
            # Confusion matrix and performance over time
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, test_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion matrix heatmap
                fig = px.imshow(
                    cm,
                    labels=dict(x="Prediction", y="Actual", color="Count"),
                    x=['Down', 'Up'],
                    y=['Down', 'Up'],
                    title="Confusion Matrix"
                )
                fig.update_traces(text=cm, texttemplate="%{text}")
                st.plotly_chart(fig, use_container_width=True, key=f"confusion_matrix_{model_name}_{i}")
            
            with col2:
                # Performance over time
                if len(test_pred) > 0:
                    correct_predictions = (test_pred == y_test).astype(int)
                    cumulative_accuracy = np.cumsum(correct_predictions) / np.arange(1, len(correct_predictions) + 1)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=cumulative_accuracy,
                        mode='lines',
                        name='Cumulative Accuracy',
                        line=dict(color='#1f77b4')
                    ))
                    fig.add_hline(y=result['test_accuracy'], line_dash="dash", 
                                line_color="red", annotation_text=f"Average: {result['test_accuracy']:.3f}")
                    fig.update_layout(
                        title="Accuracy Evolution",
                        xaxis_title="Sample",
                        yaxis_title="Cumulative Accuracy"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"performance_time_{model_name}_{i}")
    
    # Comparative analysis
        st.subheader("📊 Comparative Analysis")    # Performance comparison chart
    model_names = [get_short_name(result['name']) for _, result in sorted_results]  # Use short names
    train_accuracies = [result['train_accuracy'] for _, result in sorted_results]
    test_accuracies = [result['test_accuracy'] for _, result in sorted_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Training',
        x=model_names,
        y=train_accuracies,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Testing',
        x=model_names,
        y=test_accuracies,
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title="Performance Comparison - Training vs Testing",
        xaxis_title="Models",
        yaxis_title="Accuracy",
        barmode='group',
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True, key="performance_comparison_chart")
    
    # Overfitting analysis
    overfitting_scores = [result['train_accuracy'] - result['test_accuracy'] for _, result in sorted_results]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overfitting chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_names,
            y=overfitting_scores,
            marker_color=['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in overfitting_scores]
        ))
        fig.update_layout(
            title="Overfitting Analysis",
            xaxis_title="Models",
            yaxis_title="Train-Test Difference",
            xaxis_tickangle=-45
        )
        fig.add_hline(y=0.1, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        st.plotly_chart(fig, use_container_width=True, key="overfitting_analysis_chart")
    
    with col2:
        # Model type performance
        ml_models = [result for _, result in sorted_results if result['type'] == 'ml_model']
        benchmarks = [result for _, result in sorted_results if result['type'] == 'benchmark']
        
        if ml_models and benchmarks:
            ml_avg = np.mean([r['test_accuracy'] for r in ml_models])
            bench_avg = np.mean([r['test_accuracy'] for r in benchmarks])
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['ML Models', 'Benchmarks'],
                y=[ml_avg, bench_avg],
                marker_color=['#1f77b4', '#ff7f0e']
            ))
            fig.update_layout(
                title="Average Performance by Type",
                yaxis_title="Average Accuracy"
            )
            st.plotly_chart(fig, use_container_width=True, key="model_type_performance_chart")
    
    # Insights and recommendations
    st.subheader("💡 Insights & Recommendations")
    
    best_model = sorted_results[0]
    worst_model = sorted_results[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"🏆 **Best Model**: {get_short_name(best_model[1]['name'])}")
        st.write(f"• Accuracy: {best_model[1]['test_accuracy']:.3f}")
        st.write(f"• Type: {best_model[1]['type'].replace('_', ' ').title()}")
        
        if best_model[1]['type'] == 'ml_model':
            overfitting = best_model[1]['train_accuracy'] - best_model[1]['test_accuracy']
            if overfitting > 0.1:
                st.warning("⚠️ Warning: overfitting detected")
            else:
                st.info("✅ Good bias-variance balance")
    
    with col2:
        st.info(f"📉 **Worst Performing**: {get_short_name(worst_model[1]['name'])}")
        st.write(f"• Accuracy: {worst_model[1]['test_accuracy']:.3f}")
        st.write(f"• Type: {worst_model[1]['type'].replace('_', ' ').title()}")
    
    # Download results
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        results_csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Results (CSV)",
            data=results_csv,
            file_name=f"ml_trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'samples': len(st.session_state.X_train) + len(st.session_state.X_test),
                'features': st.session_state.X_train.shape[1],
                'train_samples': len(st.session_state.X_train),
                'test_samples': len(st.session_state.X_test)
            },
            'results': {name: {
                'name': result['name'],
                'type': result['type'],
                'train_accuracy': result['train_accuracy'],
                'test_accuracy': result['test_accuracy']
            } for name, result in st.session_state.results.items()}
        }
        
        import json
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥 Report (JSON)",
            data=report_json,
            file_name=f"ml_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Models"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🔄 New Analysis"):
            reset_session_state()
            st.rerun()


def reset_session_state():
    """Reset all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


if __name__ == "__main__":
    main()
