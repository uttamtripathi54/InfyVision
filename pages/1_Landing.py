# =============================================================================
# pages/1_Landing.py — Public hero / landing page
# =============================================================================

import streamlit as st
import sys, os
from pathlib import Path

# ── Path bootstrap ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.app_utils import inject_css

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfyVision — AI Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

# ── Hide sidebar on public pages ─────────────────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"]{display:none}
[data-testid="stSidebarCollapsedControl"]{display:none}
</style>
""", unsafe_allow_html=True)

# ── Hero section ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:60px 20px 30px">
    <p style="font-size:14px;letter-spacing:3px;color:#6366f1;font-weight:700;
       text-transform:uppercase;margin-bottom:12px">AI-POWERED STOCK INTELLIGENCE</p>
    <h1 style="font-size:clamp(36px,6vw,64px);font-weight:900;line-height:1.1;margin:0;
       background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 30%,#06b6d4 70%,#10b981 100%);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">
       InfyVision</h1>
    <p style="font-size:20px;color:#94a3b8;max-width:700px;margin:18px auto 0;line-height:1.6">
       Predict <span style="color:#8b5cf6;font-weight:700">Infosys (INFY.NS)</span> stock prices
       using deep-learning ensembles of LSTM, GRU &amp; Transformer models — backed by
       Monte&nbsp;Carlo risk simulations &amp; real-time sentiment analysis.</p>
</div>
""", unsafe_allow_html=True)

# ── CTA button ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🚀  Get Started — Free", use_container_width=True, key="hero_cta"):
        st.switch_page("pages/2_Login.py")

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature cards ────────────────────────────────────────────────────────────
def _card(icon, title, body, grad):
    return f"""
    <div style="background:linear-gradient(135deg,#1a1f35,#222845);
        border:1px solid rgba(99,102,241,.15);border-radius:16px;
        padding:30px 24px;text-align:center;height:100%;
        box-shadow:0 0 40px rgba(99,102,241,.08);
        transition:transform .3s,box-shadow .3s">
        <div style="width:56px;height:56px;border-radius:14px;
            background:{grad};display:flex;align-items:center;
            justify-content:center;margin:0 auto 16px;font-size:26px">{icon}</div>
        <h3 style="color:#f1f5f9;font-size:17px;font-weight:700;margin:0 0 8px">{title}</h3>
        <p style="color:#94a3b8;font-size:13.5px;line-height:1.55;margin:0">{body}</p>
    </div>"""

features = [
    ("🧠", "Deep-Learning Ensemble",
     "LSTM, GRU &amp; Transformer models trained on 5 years of data, averaged into a robust ensemble forecast.",
     "linear-gradient(135deg,#6366f1,#8b5cf6)"),
    ("📊", "5-Day Price Forecast",
     "Get next-5-trading-day predictions with confidence metrics, percentage changes &amp; actionable insights.",
     "linear-gradient(135deg,#8b5cf6,#06b6d4)"),
    ("⚠️", "Monte Carlo Risk Engine",
     "Geometric Brownian Motion simulations with VaR, Expected Shortfall &amp; interactive confidence intervals.",
     "linear-gradient(135deg,#f43f5e,#f59e0b)"),
    ("📰", "Live Sentiment Analysis",
     "Yahoo Finance RSS news feed scored with VADER NLP — sentiment trends mapped to trading signals.",
     "linear-gradient(135deg,#10b981,#06b6d4)"),
    ("🏆", "Model Benchmarking",
     "Side-by-side MAE, RMSE, MAPE, R² &amp; direction accuracy so you know which model to trust.",
     "linear-gradient(135deg,#06b6d4,#6366f1)"),
    ("🔬", "Full Dataset Explorer",
     "Interactive EDA — price trends, volatility, correlation heatmaps, technical indicators &amp; feature distributions.",
     "linear-gradient(135deg,#f59e0b,#f43f5e)"),
]

cols = st.columns(3)
for idx, (icon, title, body, grad) in enumerate(features):
    with cols[idx % 3]:
        st.markdown(_card(icon, title, body, grad), unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Tech stack ribbon ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:30px 0 10px">
    <p style="font-size:12px;letter-spacing:2px;color:#64748b;text-transform:uppercase;
       font-weight:600;margin-bottom:14px">Powered by</p>
    <div style="display:flex;justify-content:center;gap:28px;flex-wrap:wrap">
        <span style="color:#94a3b8;font-size:14px;font-weight:600">TensorFlow</span>
        <span style="color:#64748b">·</span>
        <span style="color:#94a3b8;font-size:14px;font-weight:600">Streamlit</span>
        <span style="color:#64748b">·</span>
        <span style="color:#94a3b8;font-size:14px;font-weight:600">Plotly</span>
        <span style="color:#64748b">·</span>
        <span style="color:#94a3b8;font-size:14px;font-weight:600">VADER NLP</span>
        <span style="color:#64748b">·</span>
        <span style="color:#94a3b8;font-size:14px;font-weight:600">GARCH</span>
        <span style="color:#64748b">·</span>
        <span style="color:#94a3b8;font-size:14px;font-weight:600">yFinance</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 20px;border-top:1px solid rgba(99,102,241,.1);margin-top:30px">
    <p style="color:#475569;font-size:12px;margin:0">
        © 2026 InfyVision · Built with ❤️ by <strong style="color:#8b5cf6">Uttam Tripathi</strong>
        · Infosys Springboard Project
    </p>
</div>
""", unsafe_allow_html=True)
