# =============================================================================
# utils/app_utils.py — Shared helpers for the InfyVision Streamlit app
# =============================================================================
# Contains: authentication, CSS injection, model/data caching, prediction
# pipeline, Monte Carlo GBM engine, Plotly theming, metric-card HTML builder,
# and CSV export utilities.  Every protected page imports from here.
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import json
import hashlib
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# Guarantee the project root is on sys.path so `from config import *` works
# no matter which page Streamlit actually executes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TICKER, WINDOW_SIZE, FEATURE_COLS, SENTIMENT_COL, MODEL_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Authentication  (JSON-file backed, SHA-256 hashed passwords)
# ─────────────────────────────────────────────────────────────────────────────
USERS_FILE = PROJECT_ROOT / "users.json"


def _load_users() -> dict:
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(username: str, password: str):
    """Returns (success: bool, message: str)."""
    users = _load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = hash_password(password)
    _save_users(users)
    return True, "Registration successful!  Please log in."


def authenticate_user(username: str, password: str) -> bool:
    return _load_users().get(username) == hash_password(password)


def require_login():
    """Gate for protected pages — call at the very top."""
    if not st.session_state.get("logged_in", False):
        st.warning("🔒  Please log in to access this page.")
        if st.button("Go to Login →"):
            st.switch_page("pages/2_Login.py")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Global premium dark CSS  (inject once per page)
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
    :root{
        --bg0:#0a0e1a;--bg1:#111827;--bg2:#1a1f35;--bg2h:#222845;
        --brd:rgba(99,102,241,.15);
        --t1:#f1f5f9;--t2:#94a3b8;--t3:#64748b;
        --ai:#6366f1;--ap:#8b5cf6;--ac:#06b6d4;--ae:#10b981;--ar:#f43f5e;--aa:#f59e0b;
        --gp:linear-gradient(135deg,#6366f1,#8b5cf6);
        --gs:linear-gradient(135deg,#10b981,#06b6d4);
        --gd:linear-gradient(135deg,#f43f5e,#f59e0b);
        --sh:0 0 30px rgba(99,102,241,.12);
    }
    html,body,[data-testid="stAppViewContainer"]{background:var(--bg0)!important;color:var(--t1)!important;font-family:'Inter',sans-serif!important}
    [data-testid="stHeader"]{background:transparent!important}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1629,#111827)!important;border-right:1px solid var(--brd)!important}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span{color:var(--t2)!important}
    div[data-testid="stMetric"]{background:var(--bg2)!important;border:1px solid var(--brd)!important;border-radius:12px!important;padding:16px 20px!important;box-shadow:var(--sh)!important}
    div[data-testid="stMetric"] label{color:var(--t2)!important;font-weight:500!important}
    .stButton>button{background:var(--gp)!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:10px 28px!important;font-weight:600!important;font-family:'Inter',sans-serif!important;transition:all .3s!important;box-shadow:0 4px 15px rgba(99,102,241,.3)!important}
    .stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 25px rgba(99,102,241,.45)!important}
    input,textarea,select,div[data-baseweb="input"] input,div[data-baseweb="select"] div{background-color:var(--bg2)!important;color:var(--t1)!important;border-color:var(--brd)!important;border-radius:8px!important}
    .stTabs [data-baseweb="tab-list"]{gap:8px!important;background:transparent!important}
    .stTabs [data-baseweb="tab"]{background-color:var(--bg2)!important;color:var(--t2)!important;border-radius:8px 8px 0 0!important;border:1px solid var(--brd)!important;padding:8px 20px!important}
    .stTabs [aria-selected="true"]{background:var(--gp)!important;color:#fff!important}
    .stDownloadButton>button{background:var(--gs)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important}
    .stSelectbox label,.stSlider label,.stNumberInput label{color:var(--t2)!important;font-weight:500!important}
    div[data-testid="stExpander"],div[data-testid="stForm"]{background-color:var(--bg2)!important;border:1px solid var(--brd)!important;border-radius:12px!important}
    ::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:var(--bg0)}::-webkit-scrollbar-thumb{background:var(--ai);border-radius:3px}
    #MainMenu{visibility:hidden}footer{visibility:hidden}
    [data-testid="stSidebarNav"]{display:none!important}
    .watermark{position:fixed;bottom:12px;right:18px;font-size:11px;color:rgba(148,163,184,.35);font-family:'JetBrains Mono',monospace;z-index:9999;pointer-events:none;letter-spacing:.5px}
    </style>
    <div class="watermark">Built by Uttam Tripathi</div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (for authenticated pages)
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 20px">
            <span style="font-size:32px">📈</span>
            <h2 style="margin:4px 0 0;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                font-weight:800;font-size:22px">InfyVision</h2>
            <p style="margin:2px 0 0;font-size:11px;color:#64748b">AI-Powered Stock Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(f"👤  **{st.session_state.get('username','User')}**")
        st.markdown("---")
        # Navigation
        st.page_link("pages/1_Landing.py", label="🏠 Home", use_container_width=True)
        st.page_link("pages/3_Dashboard.py", label="📊 Dashboard", use_container_width=True)
        st.page_link("pages/4_Performance.py", label="🏆 Model Performance", use_container_width=True)
        st.page_link("pages/5_Risk_Analysis.py", label="⚠️ Risk Analysis", use_container_width=True)
        st.page_link("pages/6_News_Sentiment.py", label="📰 News & Sentiment", use_container_width=True)
        st.page_link("pages/7_Dataset_Explorer.py", label="🔬 Dataset Explorer", use_container_width=True)
        st.markdown("---")
        if st.button("🚪 Logout", key="sb_logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.switch_page("pages/1_Landing.py")
        st.markdown(
            '<p style="text-align:center;font-size:10px;color:#475569">'
            '© 2026 InfyVision · Uttam Tripathi</p>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data & model loading  (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📡  Fetching INFY.NS market data & computing indicators …")
def load_data():
    """Fetch price data, compute indicators, add sentiment, fit scaler.
    Returns the fully-initialised DataFetcher object."""
    from data.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    fetcher.fetch_data()
    fetcher.compute_indicators()
    fetcher.add_sentiment(days_back=7)
    # Fit scaler by calling prepare_sequences (necessary for inverse_transform)
    fetcher.prepare_sequences(prediction_days=5)
    return fetcher


@st.cache_resource(show_spinner="🧠  Loading pre-trained LSTM / GRU / Transformer …")
def load_models():
    """Load pre-trained .keras models from saved_models/.  Returns dict."""
    from models.lstm_model import LSTMModel
    from models.gru_model import GRUModel
    from models.transformer_model import TransformerModel

    input_shape = (WINDOW_SIZE, len(FEATURE_COLS) + 1)
    loaded = {}
    for tag, cls in [("LSTM", LSTMModel), ("GRU", GRUModel), ("Transformer", TransformerModel)]:
        p = os.path.join(PROJECT_ROOT, MODEL_DIR, f"infy_{tag.lower()}_5d.keras")
        if os.path.exists(p):
            m = cls(input_shape)
            m.load(p)
            loaded[tag] = m
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
def predict_5day(fetcher, models: dict):
    """Run every model + ensemble on the latest window.
    Returns (predictions_df, current_price, next_dates)."""
    X_pred = fetcher.prepare_last_sequence()
    current_price = float(fetcher.df["Close"].iloc[-1])
    last_date = fetcher.df.index[-1]

    next_dates = []
    d = last_date
    for _ in range(5):
        d += pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        next_dates.append(d)

    results = {}
    for name, mdl in models.items():
        s = mdl.predict(X_pred)[0].flatten()
        results[name] = fetcher.inverse_transform_prices(s)[:5]

    if results:
        results["Ensemble"] = np.mean(list(results.values()), axis=0)

    rows = []
    for i in range(5):
        row = {"Day": f"Day {i+1}", "Date": next_dates[i].strftime("%Y-%m-%d")}
        for n in results:
            row[n] = round(float(results[n][i]), 2)
        rows.append(row)

    return pd.DataFrame(rows), current_price, next_dates


def compute_model_metrics(fetcher, models: dict):
    """Evaluate every model on the held-out test split.  Returns a DataFrame
    with MAE, RMSE, MAPE, R², Direction Accuracy per model."""
    from sklearn.metrics import r2_score
    (_, _), (_, _), (X_test, y_test), _ = fetcher.prepare_sequences(prediction_days=5)

    rows = []
    for name, mdl in models.items():
        yp = mdl.predict(X_test)
        yp1 = yp[:, 0] if yp.ndim > 1 and yp.shape[-1] == 5 else yp.flatten()
        yt1 = y_test[:, 0] if y_test.ndim > 1 else y_test.flatten()
        ypa = fetcher.inverse_transform_prices(yp1)
        yta = fetcher.inverse_transform_prices(yt1)
        mae   = np.mean(np.abs(yta - ypa))
        rmse  = np.sqrt(np.mean((yta - ypa)**2))
        mape  = np.mean(np.abs((yta - ypa) / yta)) * 100
        r2    = r2_score(yta, ypa)
        dacc  = np.mean((np.diff(yta) > 0) == (np.diff(ypa) > 0)) * 100
        rows.append({
            "Model": name,
            "MAE (₹)": round(mae, 2),
            "RMSE (₹)": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R²": round(r2, 4),
            "Dir. Accuracy (%)": round(dacc, 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo GBM engine
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo_gbm(S0: float, mu: float, sigma: float,
                    T: int = 30, N: int = 1000, seed: int = 42) -> np.ndarray:
    """Geometric Brownian Motion Monte Carlo simulation.

    Parameters
    ----------
    S0    : current price
    mu    : annualised drift  (use historical mean daily return × 252)
    sigma : annualised volatility
    T     : forecast horizon in trading days
    N     : number of simulation paths
    seed  : random seed

    Returns
    -------
    np.ndarray of shape (N, T+1) — simulated price paths.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252
    paths = np.zeros((N, T + 1))
    paths[:, 0] = S0
    for t in range(1, T + 1):
        z = rng.standard_normal(N)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Plotly dark-theme helper
# ─────────────────────────────────────────────────────────────────────────────
def dark_layout(fig, title="", height=500):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#f1f5f9", family="Inter")),
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8"),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        xaxis=dict(gridcolor="rgba(99,102,241,.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(99,102,241,.08)", zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metric-card HTML builder
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str = "", color: str = "#6366f1"):
    delta_html = ""
    if delta:
        dc = "#10b981" if not delta.startswith("-") else "#f43f5e"
        delta_html = f'<span style="font-size:13px;color:{dc};font-weight:600">{delta}</span>'
    return f"""
    <div style="background:linear-gradient(135deg,#1a1f35,#222845);border:1px solid rgba(99,102,241,.18);
        border-radius:14px;padding:22px 26px;text-align:center;box-shadow:0 0 30px rgba(99,102,241,.1)">
        <p style="margin:0;font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:1.2px;font-weight:600">{label}</p>
        <p style="margin:6px 0 4px;font-size:28px;font-weight:800;
            background:linear-gradient(135deg,{color},#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{value}</p>
        {delta_html}
    </div>"""
