"""
InfyVision - Clean Stock Price Prediction Dashboard
Author: Uttam Tripathi
Description: Streamlined dashboard with clean formatting
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from arch import arch_model
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Configuration
# ============================================
WINDOW_SIZE = 60
FEATURE_COLS = ['Close', 'GARCH_Vol', 'RSI', 'MACD', 'Volume_Change', 'Close_Open_Ratio']
SIM_DAYS = 30
SIM_RUNS = 500
RECENT_WINDOW = 252
TICKER = "INFY.NS"

MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'infy_lstm_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'infy_scaler.pkl')
CONFIG_PATH = os.path.join(MODEL_DIR, 'infy_config.pkl')

# ============================================
# Data Functions
# ============================================
@st.cache_data(ttl=3600)
def load_data(ticker=TICKER):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def add_technical_indicators(df):
    df = df.copy()

    # GARCH Volatility
    try:
        returns = 100 * df['Close'].pct_change().dropna()
        am = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        res = am.fit(disp='off')
        df['GARCH_Vol'] = res.conditional_volatility
    except:
        df['GARCH_Vol'] = df['Close'].pct_change().rolling(20).std() * 100

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rs = rs.fillna(100)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].clip(0, 100)

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2

    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()

    # Close/Open Ratio
    df['Close_Open_Ratio'] = df['Close'] / df['Open']

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df

def predict_next_day(model, scaler, df):
    last_sequence = df[FEATURE_COLS].values[-WINDOW_SIZE:]
    last_scaled = scaler.transform(last_sequence)
    X_pred = np.reshape(last_scaled, (1, WINDOW_SIZE, len(FEATURE_COLS)))
    pred_scaled = model.predict(X_pred, verbose=0)

    dummy = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, 0] = pred_scaled[0, 0]
    dummy[0, 1:] = last_scaled[-1, 1:]
    return scaler.inverse_transform(dummy)[0, 0]

def monte_carlo_simulation(df, days, runs):
    recent_returns = df['Close'].iloc[-RECENT_WINDOW:].pct_change().dropna()
    mu = recent_returns.mean()
    sigma = recent_returns.std()
    S0 = df['Close'].iloc[-1]

    np.random.seed(42)
    Z = np.random.normal(0, 1, (days, runs))
    daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * Z)

    paths = np.zeros((days + 1, runs))
    paths[0] = S0
    for t in range(1, days + 1):
        paths[t] = paths[t-1] * daily_returns[t-1]

    return paths, mu, sigma

# ============================================
# Load Artifacts
# ============================================
@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]):
        st.error("Model artifacts not found. Please run the Jupyter notebook first.")
        st.stop()
    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)
    return model, scaler, config

# ============================================
# Main Dashboard
# ============================================
def main():
    st.set_page_config(page_title="InfyVision", page_icon="📈", layout="wide")

    # ── Premium Light Theme ──────────────────────────────────────────────────
    st.markdown("""
        <style>
        /* ── Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

        /* ── Design Tokens ── */
        :root {
            --ink:          #0d1117;
            --ink-soft:     #374151;
            --ink-muted:    #6b7280;
            --accent:       #1a56db;
            --accent-light: #eff6ff;
            --accent-border:#bfdbfe;
            --amber:        #d97706;
            --amber-light:  #fffbeb;
            --amber-border: #fde68a;
            --green:        #059669;
            --red:          #dc2626;
            --surface:      #ffffff;
            --surface-2:    #f8fafc;
            --surface-3:    #f1f5f9;
            --border:       #e2e8f0;
            --border-light: #f1f5f9;
            --shadow-xs:    0 1px 2px rgba(13,17,23,0.05);
            --shadow-sm:    0 1px 3px rgba(13,17,23,0.07), 0 1px 2px rgba(13,17,23,0.04);
            --shadow-md:    0 4px 14px rgba(13,17,23,0.08), 0 2px 4px rgba(13,17,23,0.04);
            --shadow-lg:    0 12px 32px rgba(13,17,23,0.10), 0 4px 8px rgba(13,17,23,0.05);
            --shadow-xl:    0 24px 48px rgba(13,17,23,0.12), 0 8px 16px rgba(13,17,23,0.06);
            --r-sm:  8px;
            --r-md:  14px;
            --r-lg:  20px;
            --r-xl:  28px;
        }

        /* ── Base ── */
        html, body, [class*="st-"] {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* ── Background: dot-grid + ambient light blooms + INFY watermark ── */
        .stApp {
            background-color: #f9fafb;
            background-image: radial-gradient(circle, rgba(203,213,225,0.50) 1px, transparent 1px);
            background-size: 28px 28px;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse 90% 60% at 8%  4%,  rgba(219,234,254,0.65) 0%, transparent 55%),
                radial-gradient(ellipse 55% 45% at 90% 82%, rgba(254,243,199,0.55) 0%, transparent 50%),
                radial-gradient(ellipse 50% 40% at 52% 96%, rgba(220,252,231,0.38) 0%, transparent 55%),
                radial-gradient(ellipse 42% 35% at 96% 18%, rgba(237,233,254,0.32) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        /* INFY faint watermark */
        .stApp::after {
            content: "INFY";
            position: fixed;
            bottom: -2rem;
            right: -1rem;
            font-family: 'DM Serif Display', Georgia, serif;
            font-style: italic;
            font-size: 18rem;
            font-weight: 400;
            color: rgba(26, 86, 219, 0.028);
            pointer-events: none;
            z-index: 0;
            line-height: 1;
            user-select: none;
        }
        .main > div {
            position: relative;
            z-index: 1;
        }

        /* ── Topbar ── */
        header[data-testid="stHeader"] {
            background: rgba(249,250,251,0.94) !important;
            backdrop-filter: blur(14px) !important;
            -webkit-backdrop-filter: blur(14px) !important;
            border-bottom: 1px solid var(--border) !important;
            box-shadow: 0 1px 0 rgba(13,17,23,0.04) !important;
        }
        header[data-testid="stHeader"] a,
        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }
        .stDeployButton { background: transparent !important; }
        .stDeployButton svg { fill: var(--ink) !important; }

        /* ── Typography ── */
        h1, h2, h3, h4 {
            color: var(--ink) !important;
            letter-spacing: -0.01em !important;
        }
        h2, h3, h4 {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
        }
        .stMarkdown p {
            color: var(--ink-soft);
            font-size: 0.925rem;
            line-height: 1.65;
        }

        /* ── Hero Header ── */
        .infy-hero {
            display: flex;
            align-items: center;
            gap: 1.4rem;
            padding: 2.25rem 0 0.5rem 0;
        }
        .infy-hero-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 4rem;
            height: 4rem;
            background: linear-gradient(145deg, #1a56db 0%, #1d4ed8 55%, #1e40af 100%);
            border-radius: 20px;
            font-size: 1.85rem;
            box-shadow:
                0 8px 24px rgba(26,86,219,0.32),
                0 2px 6px rgba(26,86,219,0.18),
                inset 0 1px 0 rgba(255,255,255,0.22);
            flex-shrink: 0;
            position: relative;
        }
        .infy-hero-icon::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.20);
        }
        .infy-hero-wordmark {
            font-family: 'DM Serif Display', Georgia, serif;
            font-style: italic;
            font-weight: 400;
            font-size: 3.2rem;
            letter-spacing: -0.035em;
            color: var(--ink);
            line-height: 1;
            margin: 0;
        }
        .infy-hero-descriptor {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin-top: 0.38rem;
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }
        .infy-hero-descriptor .sep {
            width: 3px;
            height: 3px;
            border-radius: 50%;
            background: var(--ink-muted);
            opacity: 0.35;
            display: inline-block;
            flex-shrink: 0;
        }
        .infy-hero-byline {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.8rem;
            font-weight: 400;
            color: var(--ink-muted);
            margin-top: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            flex-wrap: wrap;
        }
        .infy-hero-byline strong {
            color: var(--ink-soft);
            font-weight: 600;
        }
        .infy-hero-byline .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            background: var(--accent-light);
            border: 1px solid var(--accent-border);
            border-radius: 20px;
            padding: 0.2rem 0.65rem;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--accent);
            letter-spacing: 0.02em;
            transition: background 0.15s, box-shadow 0.15s;
        }
        .infy-hero-byline .pill:hover {
            background: #dbeafe;
            box-shadow: 0 2px 6px rgba(26,86,219,0.14);
        }

        /* ── Animated gradient rule after hero ── */
        .infy-rule {
            height: 2px;
            border: none;
            margin: 1.5rem 0 1.75rem 0;
            background: linear-gradient(
                90deg,
                var(--accent-light) 0%,
                var(--accent) 20%,
                #60a5fa 40%,
                var(--amber) 65%,
                #fbbf24 80%,
                var(--amber-light) 100%
            );
            background-size: 200% 100%;
            animation: ruleSlide 5s linear infinite;
            border-radius: 2px;
            opacity: 0.65;
        }
        @keyframes ruleSlide {
            0%   { background-position: 0%   0%; }
            100% { background-position: 200% 0%; }
        }

        /* ── Standard dividers ── */
        hr {
            border: none !important;
            border-top: 1px solid var(--border) !important;
            margin: 1.75rem 0 !important;
            opacity: 1 !important;
        }

        /* ── Metric Cards ── */
        div[data-testid="metric-container"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--r-lg);
            padding: 1.5rem 1.75rem 1.4rem;
            box-shadow: var(--shadow-md);
            transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        /* Always-visible coloured top bar */
        div[data-testid="metric-container"]::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            border-radius: var(--r-lg) var(--r-lg) 0 0;
        }
        div[data-testid="metric-container"]:nth-of-type(1)::before {
            background: linear-gradient(90deg, #1a56db 0%, #60a5fa 100%);
        }
        div[data-testid="metric-container"]:nth-of-type(2)::before {
            background: linear-gradient(90deg, #d97706 0%, #fbbf24 100%);
        }
        div[data-testid="metric-container"]:nth-of-type(3)::before {
            background: linear-gradient(90deg, #059669 0%, #34d399 100%);
        }
        div[data-testid="metric-container"]:nth-of-type(4)::before {
            background: linear-gradient(90deg, #dc2626 0%, #f87171 100%);
        }
        div[data-testid="metric-container"]:hover {
            box-shadow: var(--shadow-xl);
            transform: translateY(-4px);
            border-color: var(--accent-border);
        }
        div[data-testid="metric-container"] label {
            color: var(--ink-muted) !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.09em !important;
        }
        div[data-testid="metric-container"] div[data-testid="metric-value"] {
            color: var(--ink) !important;
            font-size: 2.1rem !important;
            font-weight: 700 !important;
            font-family: 'DM Serif Display', Georgia, serif !important;
            font-style: italic !important;
            line-height: 1.15 !important;
            margin-top: 0.2rem !important;
            letter-spacing: -0.02em !important;
        }
        div[data-testid="metric-container"] div[data-testid="metric-delta"] {
            font-size: 0.84rem !important;
            font-weight: 600 !important;
            margin-top: 0.3rem !important;
        }
        div[data-testid="metric-container"] div[data-testid="metric-delta"] span:first-child {
            color: var(--green) !important;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
            box-shadow: 4px 0 28px rgba(13,17,23,0.06) !important;
        }
        /* Sidebar monogram */
        section[data-testid="stSidebar"] > div:first-child::before {
            content: "IV";
            display: block;
            font-family: 'DM Serif Display', Georgia, serif;
            font-style: italic;
            font-size: 1.4rem;
            font-weight: 400;
            color: var(--accent);
            letter-spacing: -0.03em;
            padding: 1.25rem 1rem 0.25rem 1rem;
            border-bottom: 1px solid var(--border-light);
            margin-bottom: 1rem;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stText {
            color: var(--ink-soft) !important;
            font-size: 0.875rem !important;
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 0.68rem !important;
            letter-spacing: 0.11em !important;
            text-transform: uppercase !important;
            color: var(--ink-muted) !important;
            font-weight: 700 !important;
            font-family: 'DM Sans', sans-serif !important;
            font-style: normal !important;
            margin-bottom: 0.6rem !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 1px solid var(--border-light) !important;
        }
        section[data-testid="stSidebar"] h4 {
            font-size: 0.64rem !important;
            letter-spacing: 0.09em !important;
            text-transform: uppercase !important;
            color: var(--ink-muted) !important;
            font-weight: 700 !important;
        }
        /* Sidebar control group inset panels */
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] .stCheckbox {
            background: var(--surface-2);
            border: 1px solid var(--border-light);
            border-radius: var(--r-sm);
            padding: 0.6rem 0.75rem;
            margin-bottom: 0.4rem;
        }
        .stSidebar .stButton button,
        section[data-testid="stSidebar"] .stButton button {
            background: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            color: var(--ink) !important;
            border-radius: var(--r-md) !important;
            padding: 0.65rem 1.5rem !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            transition: all 0.18s !important;
            width: 100% !important;
            box-shadow: var(--shadow-sm) !important;
            letter-spacing: 0.01em !important;
        }
        .stSidebar .stButton button:hover,
        section[data-testid="stSidebar"] .stButton button:hover {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            color: #fff !important;
            box-shadow: 0 4px 16px rgba(26,86,219,0.28) !important;
            transform: translateY(-1px) !important;
        }

        /* ── Slider accent ── */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }

        /* ── Tabs — pill segmented control ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0 !important;
            background: var(--surface-3) !important;
            padding: 5px !important;
            border-radius: var(--r-md) !important;
            border: 1px solid var(--border) !important;
            width: fit-content !important;
            box-shadow: var(--shadow-sm) !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--ink-muted) !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            padding: 0.5rem 1.3rem !important;
            border-radius: 10px !important;
            transition: all 0.18s !important;
            border: none !important;
            background: transparent !important;
            letter-spacing: 0.005em !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--ink) !important;
            background: rgba(255,255,255,0.65) !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--ink) !important;
            font-weight: 700 !important;
            background: var(--surface) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        /* Active tab coloured dot */
        .stTabs [aria-selected="true"]::before {
            content: "●";
            font-size: 0.38rem;
            color: var(--accent);
            vertical-align: middle;
            margin-right: 0.3rem;
            line-height: 1;
            position: relative;
            top: -1px;
        }
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* ── Expanders ── */
        [data-testid="stExpander"] {
            border-radius: var(--r-md) !important;
            overflow: hidden !important;
            margin-bottom: 0.8rem !important;
        }
        .streamlit-expanderHeader {
            background: var(--surface) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: var(--r-md) !important;
            color: var(--ink) !important;
            font-weight: 600 !important;
            padding: 1rem 1.4rem !important;
            transition: all 0.2s ease !important;
            font-size: 0.9rem !important;
            box-shadow: var(--shadow-sm) !important;
            margin-bottom: 0 !important;
        }
        /* Distribution expander — blue accent */
        [data-testid="stExpander"]:nth-of-type(1) .streamlit-expanderHeader {
            border-left: 4px solid var(--accent) !important;
            background: linear-gradient(to right, var(--accent-light) 0%, #ffffff 50%) !important;
        }
        [data-testid="stExpander"]:nth-of-type(1) .streamlit-expanderHeader:hover {
            background: linear-gradient(to right, #dbeafe 0%, #f8fafc 50%) !important;
            border-color: var(--accent) !important;
            box-shadow: 0 4px 18px rgba(26,86,219,0.13) !important;
        }
        /* Market Stats expander — amber accent */
        [data-testid="stExpander"]:nth-of-type(2) .streamlit-expanderHeader {
            border-left: 4px solid var(--amber) !important;
            background: linear-gradient(to right, var(--amber-light) 0%, #ffffff 50%) !important;
        }
        [data-testid="stExpander"]:nth-of-type(2) .streamlit-expanderHeader:hover {
            background: linear-gradient(to right, #fef3c7 0%, #f8fafc 50%) !important;
            border-color: var(--amber) !important;
            box-shadow: 0 4px 18px rgba(217,119,6,0.13) !important;
        }
        /* expand hint pill */
        .streamlit-expanderHeader::after {
            content: "expand ↓";
            margin-left: auto;
            font-size: 0.62rem;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: var(--ink-muted);
            opacity: 0.55;
            padding: 0.22rem 0.65rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--surface-3);
            transition: opacity 0.18s, background 0.18s;
            white-space: nowrap;
        }
        .streamlit-expanderHeader:hover::after {
            opacity: 1;
            background: var(--surface-2);
        }
        .streamlit-expanderContent {
            background: var(--surface-2) !important;
            border: 1.5px solid var(--border) !important;
            border-top: none !important;
            border-radius: 0 0 var(--r-md) var(--r-md) !important;
            padding: 1.3rem 1.4rem !important;
            box-shadow: var(--shadow-sm) !important;
        }

        /* ── Disabled ticker input — fully visible ── */
        .stTextInput input,
        .stTextInput input:disabled,
        .stTextInput input[disabled] {
            background: #f0f6ff !important;
            border: 1.5px solid var(--accent-border) !important;
            border-radius: var(--r-sm) !important;
            color: var(--accent) !important;
            font-size: 0.9rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.07em !important;
            -webkit-text-fill-color: var(--accent) !important;
            opacity: 1 !important;
        }

        /* ── Section labels with fading rule ── */
        .section-label {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.67rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin: 0.25rem 0 0.9rem 0;
            padding-left: 0.1rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .section-label::after {
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(to right, var(--border), transparent);
        }

        /* ── Chart wrapper cards ── */
        .chart-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--r-lg);
            box-shadow: var(--shadow-md);
            overflow: hidden;
            transition: box-shadow 0.22s;
            margin-bottom: 1rem;
        }
        .chart-card:hover { box-shadow: var(--shadow-xl); }
        .chart-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1.25rem 0.75rem;
            border-bottom: 1px solid var(--border-light);
            background: var(--surface-2);
        }
        .chart-card-title {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: var(--ink-soft);
        }
        .chart-card-legend {
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .legend-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.67rem;
            font-weight: 600;
            color: var(--ink-muted);
            padding: 0.15rem 0.5rem;
            background: var(--surface-3);
            border: 1px solid var(--border);
            border-radius: 20px;
        }
        .legend-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }

        /* ── Stats Panel (replaces old stat-boxes) ── */
        .stats-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--r-lg);
            overflow: hidden;
            box-shadow: var(--shadow-md);
        }
        .stats-panel-header {
            padding: 0.75rem 1rem;
            background: var(--surface-2);
            border-bottom: 1px solid var(--border-light);
            font-size: 0.67rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--ink-muted);
        }
        .stat-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.68rem 1rem;
            border-bottom: 1px solid var(--border-light);
            transition: background 0.15s;
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-row:hover { background: var(--surface-2); }
        .stat-row:nth-child(1) { border-left: 3px solid #94a3b8; }
        .stat-row:nth-child(2) { border-left: 3px solid var(--accent); }
        .stat-row:nth-child(3) { border-left: 3px solid var(--green); }
        .stat-row:nth-child(4) { border-left: 3px solid var(--red); }
        .stat-row:nth-child(5) { border-left: 3px solid var(--amber); }
        .stat-row:nth-child(6) { border-left: 3px solid #8b5cf6; }
        .stat-row:nth-child(7) { border-left: 3px solid #06b6d4; }
        .stat-label {
            font-size: 0.67rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--ink-muted);
        }
        .stat-value {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--ink);
            font-variant-numeric: tabular-nums;
        }

        /* ── Plotly Charts ── */
        .js-plotly-plot {
            border-radius: 0 !important;
            overflow: hidden !important;
            box-shadow: none !important;
            border: none !important;
            background: var(--surface) !important;
        }

        /* ── Caption & small text ── */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--ink-muted) !important;
            font-size: 0.77rem !important;
        }

        /* ── Links ── */
        a { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* ── Spinner ── */
        .stSpinner > div {
            border-color: var(--accent) transparent transparent transparent !important;
        }

        /* ── Footer — left monogram + right badge ── */
        .infy-footer {
            display: flex;
            align-items: center;
            padding: 1rem 0 0.5rem;
            font-size: 0.78rem;
            color: var(--ink-muted);
        }
        .infy-footer-left {
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }
        .infy-footer-monogram {
            font-family: 'DM Serif Display', Georgia, serif;
            font-style: italic;
            font-size: 1.1rem;
            color: var(--accent);
            letter-spacing: -0.02em;
            line-height: 1;
            padding: 0.2rem 0.6rem;
            background: var(--accent-light);
            border: 1px solid var(--accent-border);
            border-radius: var(--r-sm);
        }
        .infy-footer strong { color: var(--ink-soft); font-weight: 600; }
        .infy-footer-spacer { flex: 1; }
        .infy-footer-right {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .infy-footer-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.6rem;
            background: var(--surface-3);
            border: 1px solid var(--border);
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--ink-muted);
            letter-spacing: 0.04em;
        }

        /* ── Global text contrast ── */
        .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: var(--ink-soft) !important;
        }
        strong { color: var(--ink) !important; }
        </style>
    """, unsafe_allow_html=True)

    # ── Hero Header ─────────────────────────────────────────────────────────
    st.markdown("""
        <div class="infy-hero">
            <div class="infy-hero-icon">📈</div>
            <div class="infy-hero-text">
                <div class="infy-hero-wordmark">InfyVision</div>
                <div class="infy-hero-descriptor">
                    <span>Infosys</span>
                    <span class="sep"></span>
                    <span>LSTM Stock Predictor</span>
                </div>
                <div class="infy-hero-byline">
                    Crafted by <strong>Uttam Tripathi</strong>
                    <span class="pill">&#9889; LSTM</span>
                    <span class="pill">&#128201; GARCH</span>
                    <span class="pill">&#127922; Monte Carlo</span>
                </div>
            </div>
        </div>
        <div class="infy-rule"></div>
    """, unsafe_allow_html=True)

    # Load artifacts and data
    model, scaler, config = load_artifacts()

    with st.spinner("Loading market data..."):
        df_raw = load_data()
        df = add_technical_indicators(df_raw)

    # Calculate once - use everywhere
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    next_price = predict_next_day(model, scaler, df)

    # Extract metrics from config
    mape = config.get('test_mape', 0)
    mae = config.get('test_mae', 0)
    direction = config.get('direction_accuracy', 0)

    # ============================================
    # Sidebar - Controls Only
    # ============================================
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.text_input("Ticker", value=TICKER, disabled=True)

        st.markdown("#### Simulation")
        sim_days = st.slider("Days", 10, 60, SIM_DAYS)
        sim_runs = st.slider("Runs", 100, 1000, SIM_RUNS)

        st.markdown("#### Chart")
        show_ma = st.checkbox("Show 20-day MA", value=True)

        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"Data: {df.index[-1].strftime('%Y-%m-%d')}")

    # ============================================
    # Top Metrics - Only Current and Forecast
    # ============================================
    col1, col2 = st.columns(2)

    with col1:
        delta = ((current_price/prev_price)-1)*100
        st.metric("💰 Current Price", f"₹{current_price:.2f}", f"{delta:+.2f}%")

    with col2:
        delta_f = ((next_price/current_price)-1)*100
        st.metric("🔮 Next Day Forecast", f"₹{next_price:.2f}", f"{delta_f:+.2f}%")

    st.divider()

    # ============================================
    # Three Clean Tabs
    # ============================================
    tab1, tab2, tab3 = st.tabs([
        "📈 Price & Forecast",
        "📊 Model Performance",
        "🎲 Risk Analysis"
    ])

    # ============================================
    # TAB 1: Price & Forecast
    # ============================================
    with tab1:
        col_left, col_right = st.columns([3, 1])

        with col_left:
            # Build MA legend pill only when checkbox is on
            ma_legend_html = (
                '<span class="legend-pill">'
                '<span class="legend-dot" style="background:#94a3b8;"></span>'
                '20d MA'
                '</span>'
            ) if show_ma else ''

            # Chart card header — fully self-contained HTML, no open tags left unclosed
            st.markdown(
                f'<div class="chart-card">'
                f'<div class="chart-card-header">'
                f'<span class="chart-card-title">Price History &middot; Last 180 Days</span>'
                f'<div class="chart-card-legend">'
                f'<span class="legend-pill"><span class="legend-dot" style="background:#1e3a8a;"></span>Close</span>'
                f'{ma_legend_html}'
                f'<span class="legend-pill"><span class="legend-dot" style="background:#f59e0b;"></span>Forecast</span>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=df.index[-180:],
                y=df['Close'].iloc[-180:],
                name="Price",
                line=dict(color='#1e3a8a', width=2.5),
                hovertemplate='Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
            ))

            # Moving average
            if show_ma:
                ma20 = df['Close'].iloc[-180:].rolling(20).mean()
                fig.add_trace(go.Scatter(
                    x=df.index[-180:],
                    y=ma20,
                    name="20-day MA",
                    line=dict(color='#94a3b8', width=2, dash='dash'),
                    hovertemplate='Date: %{x}<br>MA20: ₹%{y:.2f}<extra></extra>'
                ))

            # Forecast point
            next_date = df.index[-1] + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[next_price],
                name="Forecast",
                mode='markers',
                marker=dict(color='#f59e0b', size=18, symbol='star', line=dict(color='#ffffff', width=2)),
                hovertemplate='Date: %{x}<br>Forecast: ₹%{y:.2f}<extra></extra>'
            ))

            fig.update_layout(
                template='plotly_white',
                height=450,
                hovermode='x unified',
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                font=dict(family="DM Sans, sans-serif", size=12, color='#374151'),
                plot_bgcolor='rgba(255,255,255,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                xaxis=dict(
                    gridcolor='#f1f5f9',
                    showgrid=True,
                    tickfont=dict(color='#6b7280', size=11),
                    title_font=dict(color='#6b7280', size=12)
                ),
                yaxis=dict(
                    gridcolor='#f1f5f9',
                    showgrid=True,
                    tickfont=dict(color='#6b7280', size=11),
                    title_font=dict(color='#6b7280', size=12)
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown(f"""
            <div class="stats-panel">
                <div class="stats-panel-header">Today's Stats</div>
                <div class="stat-row">
                    <span class="stat-label">Date</span>
                    <span class="stat-value">{df.index[-1].strftime('%b %d, %Y')}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Open</span>
                    <span class="stat-value">&#8377;{df['Open'].iloc[-1]:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">High</span>
                    <span class="stat-value">&#8377;{df['High'].iloc[-1]:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Low</span>
                    <span class="stat-value">&#8377;{df['Low'].iloc[-1]:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Volume</span>
                    <span class="stat-value">{df['Volume'].iloc[-1]:,.0f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">RSI</span>
                    <span class="stat-value">{df['RSI'].iloc[-1]:.1f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">MACD</span>
                    <span class="stat-value">{df['MACD'].iloc[-1]:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ============================================
    # TAB 2: Model Performance - FIXED DISPLAY
    # ============================================
    with tab2:
        st.markdown("### 📊 Jupyter Notebook Results")

        # Three columns with clean formatting
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Error Metrics")
            st.metric("MAE", f"₹{mae:.2f}")
            st.metric("MAPE", f"{mape:.2f}%")
            st.metric("Direction Acc", f"{direction:.1f}%")

        with col2:
            st.markdown("#### Model Specs")
            st.markdown(f"""
            **Window:** {WINDOW_SIZE} days  
            **Features:** {len(FEATURE_COLS)}  
            **LSTM:** 64 → 32 units  
            **Epochs:** 12 (early stop)  
            **Best Val Loss:** 0.0040
            """)

        with col3:
            st.markdown("#### Test Results")
            st.markdown(f"""
            **Test Samples:** 235  
            **RMSE:** ₹46.76  
            **Train/Test:** 80/20  
            **Training Date:** {config.get('training_date', 'unknown')[:16]}
            """)

        # Feature correlation
        st.markdown("### 📊 Feature Importance")
        features = FEATURE_COLS + ['Volume']
        corr_data = df[features].corr()['Close'].drop('Close').sort_values()

        fig = go.Figure(data=[
            go.Bar(
                x=corr_data.values,
                y=corr_data.index,
                orientation='h',
                marker_color=['#10b981' if x > 0 else '#f43f5e' for x in corr_data.values],
                text=[f'{x:.3f}' for x in corr_data.values],
                textposition='outside',
                textfont=dict(size=12, color='#0f172a'),
                hovertemplate='%{y}: %{x:.3f}<extra></extra>'
            )
        ])
        fig.update_layout(
            template='plotly_white',
            height=300,
            xaxis_title="Correlation with Price",
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(family="Inter, sans-serif", size=12, color='#1e293b'),
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(
                gridcolor='#e2e8f0',
                tickfont=dict(color='#1e293b', size=11),
                title_font=dict(color='#1e293b', size=12)
            ),
            yaxis=dict(
                gridcolor='#e2e8f0',
                tickfont=dict(color='#1e293b', size=11),
                title_font=dict(color='#1e293b', size=12)
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # TAB 3: Risk Analysis
    # ============================================
    with tab3:
        # Run simulation
        paths, mu, sigma = monte_carlo_simulation(df, sim_days, sim_runs)
        final_prices = paths[-1]

        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected", f"₹{np.mean(final_prices):.2f}")
        col2.metric("Median", f"₹{np.median(final_prices):.2f}")
        col3.metric("95% VaR", f"₹{np.percentile(final_prices, 5):.2f}")
        col4.metric("Loss Prob", f"{np.mean(final_prices < current_price)*100:.1f}%")

        # Confidence band chart with card header (self-contained, no split HTML)
        st.markdown(
            f'<div class="chart-card">'
            f'<div class="chart-card-header">'
            f'<span class="chart-card-title">Monte Carlo Simulation &middot; {sim_days} Days &middot; {sim_runs} Runs</span>'
            f'<div class="chart-card-legend">'
            f'<span class="legend-pill"><span class="legend-dot" style="background:rgba(30,58,138,0.30);"></span>95% Band</span>'
            f'<span class="legend-pill"><span class="legend-dot" style="background:#1e3a8a;"></span>Median</span>'
            f'<span class="legend-pill"><span class="legend-dot" style="background:#f59e0b;"></span>Current</span>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        x_vals = list(range(sim_days + 1))
        lower = np.percentile(paths, 5, axis=1)
        upper = np.percentile(paths, 95, axis=1)

        fig.add_trace(go.Scatter(
            x=x_vals + x_vals[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor='rgba(30, 58, 138, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=np.percentile(paths, 50, axis=1),
            name='Median',
            line=dict(color='#1e3a8a', width=2.5),
            hovertemplate='Day: %{x}<br>Median: ₹%{y:.2f}<extra></extra>'
        ))

        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color="#f59e0b",
            annotation_text=f"Current: ₹{current_price:.2f}",
            annotation_position="bottom right",
            annotation_font=dict(size=12, color='#1e293b')
        )

        fig.update_layout(
            template='plotly_white',
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="Inter, sans-serif", size=12, color='#1e293b'),
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            xaxis=dict(
                gridcolor='#e2e8f0',
                tickfont=dict(color='#1e293b', size=11),
                title_font=dict(color='#1e293b', size=12)
            ),
            yaxis=dict(
                gridcolor='#e2e8f0',
                tickfont=dict(color='#1e293b', size=11),
                title_font=dict(color='#1e293b', size=12)
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution - prominently highlighted expander
        with st.expander("📊 View Price Distribution"):
            fig_hist = go.Figure(data=[
                go.Histogram(
                    x=final_prices,
                    nbinsx=40,
                    marker_color='#1e3a8a',
                    hovertemplate='Price: ₹%{x:.2f}<br>Frequency: %{y}<extra></extra>'
                )
            ])
            fig_hist.add_vline(x=current_price, line_dash="dash", line_color="#f59e0b")
            fig_hist.update_layout(
                template='plotly_white',
                height=250,
                xaxis_title="Price (₹)",
                yaxis_title="Frequency",
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                font=dict(family="Inter, sans-serif", size=12, color='#1e293b'),
                plot_bgcolor='rgba(255,255,255,0)',
                paper_bgcolor='rgba(255,255,255,0)',
                xaxis=dict(
                    gridcolor='#e2e8f0',
                    tickfont=dict(color='#1e293b', size=11),
                    title_font=dict(color='#1e293b', size=12)
                ),
                yaxis=dict(
                    gridcolor='#e2e8f0',
                    tickfont=dict(color='#1e293b', size=11),
                    title_font=dict(color='#1e293b', size=12)
                )
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Market stats expander - also highlighted
        with st.expander("📈 Market Statistics"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Daily Return", f"{mu*100:.4f}%")
            col2.metric("Daily Vol", f"{sigma*100:.4f}%")
            col3.metric("Annual Vol", f"{sigma*np.sqrt(252)*100:.2f}%")

    # ============================================
    # Footer — left monogram + right badge
    # ============================================
    st.divider()
    st.markdown("""
    <div class="infy-footer">
        <div class="infy-footer-left">
            <span class="infy-footer-monogram">IV</span>
            Crafted with &#128153; by <strong>Uttam Tripathi</strong>
        </div>
        <div class="infy-footer-spacer"></div>
        <div class="infy-footer-right">
            <span class="infy-footer-badge">InfyVision v1.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()