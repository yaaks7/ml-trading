"""
Streamlit app: a 3-step wizard (data -> models -> results) for the ML trading
direction predictor. See README.md for what this project does and why.
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

# Visual palette — see the "intentional restyle" notes; kept as a single source
# of truth so charts and CSS agree with .streamlit/config.toml.
INK = "#14171C"
PANEL = "#1B1F27"
LINE = "#2A2F3A"
PAPER = "#E8E6DF"
BRASS = "#C9973F"
UP_COLOR = "#5B9279"
DOWN_COLOR = "#B3563F"

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    st.warning("yfinance is not installed — falling back to synthetic data.")

# Add src and config to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'config'))

try:
    from config.settings import SUPPORTED_ASSETS, DataConfig, STREAMLIT_CONFIG
    from src.data.fetcher import DataFetcher
    from src.models import get_all_ml_models, get_model_descriptions
    from src.strategies.naive import get_all_naive_strategies, get_strategy_descriptions
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    HAS_ALL_IMPORTS = True
except ImportError as e:
    HAS_ALL_IMPORTS = False
    st.warning(f"Some imports failed ({e}) — running in fallback mode with demo data.")

    # Small fallback asset list, used only if the real config/src imports fail
    # (e.g. missing dependency on a fresh deployment).
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

    class DataFetcher:
        """Minimal stand-in for src.data.fetcher.DataFetcher — enough to demo the
        UI flow with real prices when possible, synthetic data otherwise."""

        def __init__(self, config=None):
            self.config = config or DataConfig()

        def fetch_and_prepare(self, symbol, start_date, end_date):
            if HAS_YFINANCE:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)

                    if data.empty:
                        raise ValueError("No data available")

                    X = pd.DataFrame(index=data.index)
                    X['Close'] = data['Close']
                    X['Volume'] = data['Volume'] if 'Volume' in data.columns else 1000
                    X['High'] = data['High']
                    X['Low'] = data['Low']
                    X['Open'] = data['Open']

                    X['Returns'] = X['Close'].pct_change()
                    X['MA_5'] = X['Close'].rolling(5).mean()
                    X['MA_20'] = X['Close'].rolling(20).mean()
                    X['RSI'] = 50  # placeholder — the real RSI lives in src/data/fetcher.py

                    X = X.dropna()

                    y = (X['Close'].shift(-1) > X['Close']).astype(int)
                    y = y[:-1]
                    X = X[:-1]

                    return X, y
                except Exception as e:
                    st.warning(f"yfinance error ({e}) — using synthetic data.")

            # No yfinance, or the request above failed: synthesize a random walk
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

# Fallback strategy classes, only used if the real src.strategies import fails
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


def get_short_name(long_name):
    """Collapse the longer strategy/model names down to a compact label for tables."""
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
    }
    return name_mapping.get(long_name, long_name)


# ==================== HELPERS ====================

def apply_custom_css():
    """Small, deliberately narrow CSS pass on top of .streamlit/config.toml's theme:
    tabular monospace figures for metrics/tables (a ticker-board touch), a thin
    brass rule in place of markdown dividers, and flatter, sharper-edged controls.
    Targets stable data-testid selectors rather than Streamlit's hashed CSS classes,
    which change between versions.
    """
    st.markdown(f"""
    <style>
    div[data-testid="stMetricValue"] {{
        font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
        letter-spacing: -0.02em;
    }}

    div[data-testid="stDataFrame"] * {{
        font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    }}

    hr {{
        border: none;
        border-top: 2px solid {BRASS};
        opacity: 0.6;
        margin: 1.25rem 0;
    }}

    .stButton > button {{
        border-radius: 4px;
        border: 1px solid {LINE};
        font-weight: 600;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {BRASS};
        border-color: {BRASS};
        color: {INK};
    }}

    div[data-testid="stMetric"] {{
        background-color: {PANEL};
        border: 1px solid {LINE};
        border-radius: 4px;
        padding: 0.6rem 0.8rem;
    }}

    .streamlit-expanderHeader {{
        border-radius: 4px;
    }}
    </style>
    """, unsafe_allow_html=True)


def validate_date_range(start_date, end_date):
    """Return a list of human-readable warnings for a start/end date pair, empty if fine."""
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


# ==================== MAIN APPLICATION ====================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ML Trading Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    apply_custom_css()

    st.title("ML Trading Prediction")
    st.markdown("*Next-day direction prediction — Random Forest and MLP evaluated against six naive baselines.*")
    st.markdown("---")

    init_session_state()
    display_progress_indicator()

    st.markdown("---")

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
    """Quiet single-line step tracker — current step in brass, the rest muted."""
    steps = ["Data Configuration", "Model Configuration", "Results & Evaluation"]
    current = st.session_state.step

    parts = []
    for i, label in enumerate(steps, start=1):
        if i == current:
            style = f"color:{BRASS}; font-weight:600;"
        elif i < current:
            style = f"color:{PAPER}; opacity:0.6;"
        else:
            style = "opacity:0.35;"
        parts.append(f"<span style='{style}'>{i}. {label}</span>")

    st.markdown(
        f"<div style='font-size:0.95rem;'>{'&nbsp;&nbsp;/&nbsp;&nbsp;'.join(parts)}</div>",
        unsafe_allow_html=True
    )


def render_step_1_data_configuration():
    """Step 1: pick an asset, a date range, and which technical indicators to compute."""

    st.header("Step 1 — Data Configuration")

    with st.sidebar:
        st.header("Asset & Period")

        selected_symbol = st.selectbox(
            "Choose asset:",
            options=list(SUPPORTED_ASSETS.keys()),
            format_func=lambda x: f"{SUPPORTED_ASSETS[x]['name']} ({x})"
        )

        asset_info = SUPPORTED_ASSETS[selected_symbol]

        st.subheader("Date Range")
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

        date_warnings = validate_date_range(start_date, end_date)
        for warning in date_warnings:
            st.caption(f"⚠ {warning}")

        st.markdown("---")
        load_data_btn = st.button("Load & Process Data", type="primary", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selected Asset")
        st.write(f"**{asset_info['name']}** ({selected_symbol})")
        st.caption(f"{asset_info['type'].title()} · {asset_info['sector']} · {asset_info['currency']}")

        st.subheader("Technical Indicators")

        with st.expander("What these indicators mean"):
            st.markdown(
                "**Moving Averages** — average price over N days; shows trend direction.\n\n"
                "**RSI** — momentum oscillator (0–100); above 70 is often read as overbought, below 30 as oversold.\n\n"
                "**MACD** — difference between two moving averages; used to spot momentum shifts.\n\n"
                "**Bollinger Bands** — volatility bands around a moving average.\n\n"
                "Only Moving Averages and Bollinger Bands are drawn on the chart below — "
                "RSI and MACD are computed as model features only."
            )

        indicator_col1, indicator_col2 = st.columns(2)
        with indicator_col1:
            use_ma = st.checkbox("Moving Averages", value=True)
            use_rsi = st.checkbox("RSI", value=True)
        with indicator_col2:
            use_macd = st.checkbox("MACD", value=True)
            use_bb = st.checkbox("Bollinger Bands", value=False)

        if use_ma:
            ma_periods = st.multiselect(
                "Moving average periods (max 3):",
                options=[5, 10, 20, 50, 100, 200],
                default=[20, 50, 100],
                max_selections=3
            )
        else:
            ma_periods = []

    with col2:
        if use_rsi:
            st.subheader("RSI")
            rsi_period = st.slider("Period:", 5, 30, 14)
        else:
            rsi_period = None

        if use_macd:
            st.subheader("MACD")
            macd_col1, macd_col2, macd_col3 = st.columns(3)
            with macd_col1:
                macd_fast = st.number_input("Fast:", 5, 20, 12)
            with macd_col2:
                macd_slow = st.number_input("Slow:", 20, 40, 26)
            with macd_col3:
                macd_signal = st.number_input("Signal:", 5, 15, 9)
        else:
            macd_fast = macd_slow = macd_signal = None

        if use_bb:
            st.subheader("Bollinger Bands")
            bb_col1, bb_col2 = st.columns(2)
            with bb_col1:
                bb_period = st.slider("Period:", 10, 30, 20)
            with bb_col2:
                bb_std = st.slider("Std. deviations:", 1.0, 3.0, 2.0, 0.1)
        else:
            bb_period = bb_std = None

        st.subheader("Train / Test Split")
        train_ratio = st.slider("Training ratio:", 0.6, 0.8, 0.7, 0.05)
        test_ratio = 1.0 - train_ratio
        val_ratio = 0.0
        st.metric("Test ratio", f"{test_ratio:.2f}")

    if load_data_btn:
        if start_date >= end_date:
            st.error("Fix the dates in the sidebar — start must be before end.")
            return

        with st.spinner("Loading and processing data..."):
            try:
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

                fetcher = DataFetcher(config=data_config)

                try:
                    X, y = fetcher.fetch_and_prepare(
                        symbol=selected_symbol,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d")
                    )
                except AttributeError:
                    X, y = fetcher.process_symbol(selected_symbol)

                if len(X) == 0 or len(y) == 0:
                    st.error(
                        f"No usable data for {asset_info['name']} after processing — the indicator "
                        "lookback windows may be longer than the selected period."
                    )
                    st.caption("Try a longer time period, fewer indicators, or a different asset.")
                    return

                if len(X) < 100:
                    st.warning(f"Only {len(X)} samples available — results will be noisy. Consider a longer period.")

                split_idx = max(1, int(len(X) * train_ratio))

                st.session_state.X_train = X.iloc[:split_idx]
                st.session_state.X_test = X.iloc[split_idx:]
                st.session_state.y_train = y.iloc[:split_idx]
                st.session_state.y_test = y.iloc[split_idx:]

                try:
                    st.session_state.raw_data = fetcher.get_raw_data()
                except AttributeError:
                    st.session_state.raw_data = X
                st.session_state.data_loaded = True

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return

    if st.session_state.data_loaded and st.session_state.X_train is not None:

        st.markdown("---")
        st.subheader("Data Summary")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Total Samples", len(st.session_state.X_train) + len(st.session_state.X_test))
        with metric_col2:
            st.metric("Features", st.session_state.X_train.shape[1])
        with metric_col3:
            st.metric("Training", len(st.session_state.X_train))
        with metric_col4:
            st.metric("Testing", len(st.session_state.X_test))

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
            st.warning("No valid target data available.")

        if st.session_state.raw_data is not None:
            st.subheader("Price Evolution")

            fig = go.Figure()
            raw_data = st.session_state.raw_data
            fig.add_trace(go.Scatter(
                x=raw_data.index,
                y=raw_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=BRASS, width=2)
            ))

            if use_ma and ma_periods:
                for period in ma_periods[:3]:
                    ma_col = f'MA_{period}'
                    if ma_col in raw_data.columns:
                        fig.add_trace(go.Scatter(
                            x=raw_data.index,
                            y=raw_data[ma_col],
                            mode='lines',
                            name=f'MA {period}',
                            line=dict(width=1, color=PAPER),
                            opacity=0.5
                        ))

            if use_bb and 'BB_Upper' in raw_data.columns and 'BB_Lower' in raw_data.columns:
                fig.add_trace(go.Scatter(
                    x=raw_data.index, y=raw_data['BB_Upper'], mode='lines', name='BB Upper',
                    line=dict(color=DOWN_COLOR, width=1, dash='dash'), opacity=0.6
                ))
                fig.add_trace(go.Scatter(
                    x=raw_data.index, y=raw_data['BB_Lower'], mode='lines', name='BB Lower',
                    line=dict(color=DOWN_COLOR, width=1, dash='dash'), opacity=0.6,
                    fill='tonexty', fillcolor='rgba(179,86,63,0.12)'
                ))

            fig.update_layout(
                title=f"{asset_info['name']} — Price",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified',
                showlegend=True,
                height=400,
                paper_bgcolor=PANEL,
                plot_bgcolor=PANEL,
                font=dict(color=PAPER),
            )

            st.plotly_chart(fig, use_container_width=True, key="price_evolution_chart")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Configure Models →", type="primary"):
                st.session_state.step = 2
                st.rerun()


def render_step_2_model_configuration():
    """Step 2: choose which ML models and naive benchmarks to train, and their hyperparameters."""

    st.header("Step 2 — Model Configuration")

    if not st.session_state.data_loaded:
        st.error("No data loaded yet — go back to Step 1.")
        if st.button("← Back to Data Configuration"):
            st.session_state.step = 1
            st.rerun()
        return

    with st.sidebar:
        st.header("Model Selection")

        available_models = get_all_ml_models()
        model_descriptions = get_model_descriptions()

        selected_ml_models = st.multiselect(
            "ML Models:",
            options=list(available_models.keys()),
            default=[]
        )

        available_benchmarks = get_all_naive_strategies()
        benchmark_descriptions = get_strategy_descriptions()

        selected_benchmarks = st.multiselect(
            "Benchmarks:",
            options=list(available_benchmarks.keys()),
            default=[]
        )

        st.subheader("Parameters")
        model_configs = {}
        
        for model_name in selected_ml_models:
            if model_name == 'Random Forest':
                with st.expander("Random Forest"):
                    n_estimators = st.slider("Trees:", 50, 500, 200, key=f"{model_name}_n_est")
                    max_depth = st.slider("Max Depth:", 5, 50, 6, key=f"{model_name}_depth")
                    min_samples_leaf = st.slider(
                        "Min Samples per Leaf:", 1, 100, 20, key=f"{model_name}_min_leaf",
                    )
                    st.caption(
                        "Higher values regularize harder — with ~35 features and a few "
                        "hundred training rows, the default of 1 lets trees memorize "
                        "individual days instead of learning anything general."
                    )
                    model_configs[model_name] = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
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
        
        st.markdown("---")
        train_models_btn = st.button("Train Models", type="primary", use_container_width=True)

    col1, col2 = st.columns([2, 1])

    model_blurbs = {
        'Random Forest': "Ensemble of decision trees. Handles non-linear features well and exposes feature importance.",
        'MLP': "Fully-connected neural network. More capacity for non-linear patterns, more prone to overfitting here.",
    }
    benchmark_blurbs = {
        'Bullish': "Always predicts up.",
        'Bearish': "Always predicts down.",
        'Random': "50/50 coin flip.",
        'Frequency': "Samples from the historical up/down ratio.",
        'Momentum (Last Direction)': "Repeats the last observed direction.",
        'Mean Reversion (Contrarian)': "Predicts the opposite of the last observed direction.",
    }

    with col1:
        st.subheader("Selected ML Models")
        if selected_ml_models:
            for model_name in selected_ml_models:
                blurb = model_blurbs.get(model_name, "Machine learning model for classification.")
                st.markdown(f"**{model_descriptions.get(model_name, model_name)}** — {blurb}")
        else:
            st.caption("No ML models selected.")

        st.subheader("Selected Benchmark Strategies")
        if selected_benchmarks:
            for bench_name in selected_benchmarks:
                blurb = benchmark_blurbs.get(bench_name, "Baseline strategy for comparison.")
                st.markdown(f"**{benchmark_descriptions.get(bench_name, bench_name)}** — {blurb}")
        else:
            st.caption("No benchmark strategies selected.")

    with col2:
        st.subheader("Data Summary")
        st.metric("Total Samples", len(st.session_state.X_train) + len(st.session_state.X_test))
        st.metric("Features", st.session_state.X_train.shape[1])
        st.metric("Training", len(st.session_state.X_train))
        st.metric("Testing", len(st.session_state.X_test))

        st.subheader("Selection Summary")
        st.write(f"**ML Models:** {len(selected_ml_models)}")
        st.write(f"**Benchmarks:** {len(selected_benchmarks)}")
        st.write(f"**Total:** {len(selected_ml_models) + len(selected_benchmarks)}")

    if not selected_ml_models and not selected_benchmarks:
        st.warning("Select at least one model or benchmark strategy in the sidebar.")

    if train_models_btn:
        if not selected_ml_models and not selected_benchmarks:
            st.error("Select at least one model or benchmark before training.")
            return

        with st.spinner("Training models..."):
            try:
                results = {}
                progress_bar = st.progress(0)
                total_models = len(selected_ml_models) + len(selected_benchmarks)
                current_model = 0

                for model_name in selected_ml_models:
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
                
                for bench_name in selected_benchmarks:
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

            except Exception as e:
                st.error(f"Training error: {str(e)}")
                logger.error(f"Training error: {e}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Data"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.session_state.results:
            if st.button("View Results →", type="primary"):
                st.session_state.step = 3
                st.rerun()


CHART_LAYOUT_DEFAULTS = dict(paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=PAPER))


def render_step_3_results_evaluation():
    """Step 3: rank everything trained in Step 2, and let the top models be inspected in detail."""

    st.header("Step 3 — Results & Evaluation")

    if not st.session_state.results:
        st.error("No results yet — go back and train something in Step 2.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Models"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("New Analysis"):
                reset_session_state()
                st.rerun()
        return

    st.subheader("Model Rankings")

    sorted_results = sorted(st.session_state.results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)

    results_data = []
    for model_name, result in sorted_results:
        results_data.append({
            'Model': get_short_name(result['name']),
            'Type': 'ML' if result['type'] == 'ml_model' else 'Benchmark',
            'Train Accuracy': f"{result['train_accuracy']:.3f}",
            'Test Accuracy': f"{result['test_accuracy']:.3f}",
            'Overfitting': f"{result['train_accuracy'] - result['test_accuracy']:.3f}"
        })

    results_df = pd.DataFrame(results_data)

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.subheader("Detailed Analysis")

    top_models = sorted_results[:3]

    for i, (model_name, result) in enumerate(top_models):
        short_name = get_short_name(result['name'])
        type_label = 'ML' if result['type'] == 'ml_model' else 'Benchmark'

        with st.expander(f"#{i + 1} · {short_name} ({type_label})"):

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

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, test_pred)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.imshow(
                    cm,
                    labels=dict(x="Prediction", y="Actual", color="Count"),
                    x=['Down', 'Up'],
                    y=['Down', 'Up'],
                    title="Confusion Matrix",
                    color_continuous_scale=[[0, PANEL], [1, BRASS]]
                )
                fig.update_traces(text=cm, texttemplate="%{text}")
                fig.update_layout(**CHART_LAYOUT_DEFAULTS)
                st.plotly_chart(fig, use_container_width=True, key=f"confusion_matrix_{model_name}_{i}")

            with col2:
                if len(test_pred) > 0:
                    correct_predictions = (test_pred == y_test).astype(int)
                    cumulative_accuracy = np.cumsum(correct_predictions) / np.arange(1, len(correct_predictions) + 1)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=cumulative_accuracy,
                        mode='lines',
                        name='Cumulative Accuracy',
                        line=dict(color=BRASS)
                    ))
                    fig.add_hline(y=result['test_accuracy'], line_dash="dash",
                                line_color=PAPER, annotation_text=f"Average: {result['test_accuracy']:.3f}")
                    fig.update_layout(
                        title="Accuracy Evolution",
                        xaxis_title="Sample",
                        yaxis_title="Cumulative Accuracy",
                        **CHART_LAYOUT_DEFAULTS
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"performance_time_{model_name}_{i}")

    st.subheader("Comparative Analysis")

    model_names = [get_short_name(result['name']) for _, result in sorted_results]
    train_accuracies = [result['train_accuracy'] for _, result in sorted_results]
    test_accuracies = [result['test_accuracy'] for _, result in sorted_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Training', x=model_names, y=train_accuracies, marker_color=LINE))
    fig.add_trace(go.Bar(name='Testing', x=model_names, y=test_accuracies, marker_color=BRASS))
    fig.update_layout(
        title="Training vs Testing Accuracy",
        xaxis_title="Models",
        yaxis_title="Accuracy",
        barmode='group',
        xaxis_tickangle=-45,
        **CHART_LAYOUT_DEFAULTS
    )
    st.plotly_chart(fig, use_container_width=True, key="performance_comparison_chart")

    overfitting_scores = [result['train_accuracy'] - result['test_accuracy'] for _, result in sorted_results]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=model_names,
            y=overfitting_scores,
            marker_color=[DOWN_COLOR if x > 0.1 else BRASS if x > 0.05 else UP_COLOR for x in overfitting_scores]
        ))
        fig.update_layout(
            title="Overfitting (Train − Test Accuracy)",
            xaxis_title="Models",
            yaxis_title="Train-Test Difference",
            xaxis_tickangle=-45,
            **CHART_LAYOUT_DEFAULTS
        )
        fig.add_hline(y=0.1, line_dash="dash", line_color=DOWN_COLOR, annotation_text="Alert Threshold")
        st.plotly_chart(fig, use_container_width=True, key="overfitting_analysis_chart")

    with col2:
        ml_models = [result for _, result in sorted_results if result['type'] == 'ml_model']
        benchmarks = [result for _, result in sorted_results if result['type'] == 'benchmark']

        if ml_models and benchmarks:
            ml_avg = np.mean([r['test_accuracy'] for r in ml_models])
            bench_avg = np.mean([r['test_accuracy'] for r in benchmarks])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['ML Models', 'Benchmarks'],
                y=[ml_avg, bench_avg],
                marker_color=[BRASS, LINE]
            ))
            fig.update_layout(
                title="Average Accuracy by Type",
                yaxis_title="Average Accuracy",
                **CHART_LAYOUT_DEFAULTS
            )
            st.plotly_chart(fig, use_container_width=True, key="model_type_performance_chart")

    st.subheader("Reading the Results")

    best_model = sorted_results[0]
    worst_model = sorted_results[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Best: {get_short_name(best_model[1]['name'])}**")
        st.write(f"Accuracy: {best_model[1]['test_accuracy']:.3f}")
        st.write(f"Type: {best_model[1]['type'].replace('_', ' ').title()}")

        if best_model[1]['type'] == 'ml_model':
            overfitting = best_model[1]['train_accuracy'] - best_model[1]['test_accuracy']
            if overfitting > 0.1:
                st.caption("Train/test gap over 0.1 — likely overfitting.")
            else:
                st.caption("Train/test gap is small — no strong overfitting signal.")

    with col2:
        st.markdown(f"**Weakest: {get_short_name(worst_model[1]['name'])}**")
        st.write(f"Accuracy: {worst_model[1]['test_accuracy']:.3f}")
        st.write(f"Type: {worst_model[1]['type'].replace('_', ' ').title()}")

    st.subheader("Export")

    col1, col2 = st.columns(2)

    with col1:
        results_csv = results_df.to_csv(index=False)
        st.download_button(
            label="Results (CSV)",
            data=results_csv,
            file_name=f"ml_trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
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
            label="Report (JSON)",
            data=report_json,
            file_name=f"ml_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Models"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("New Analysis"):
            reset_session_state()
            st.rerun()


def reset_session_state():
    """Reset all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


if __name__ == "__main__":
    main()
