# =============================================================================
# pages/7_Dataset_Explorer.py — Dataset Explorer & EDA  (Protected)
# =============================================================================
# Unique content: dataset preview, technical indicator summary, sentiment
# feature distribution, price trend, volatility, correlation heatmap,
# features used in the project.
# =============================================================================

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.app_utils import (
    inject_css, render_sidebar, require_login,
    load_data, dark_layout, metric_card,
)
from config import FEATURE_COLS, SENTIMENT_COL, WINDOW_SIZE

st.set_page_config(page_title="Dataset Explorer — InfyVision", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
require_login()
render_sidebar()

# ── Load ─────────────────────────────────────────────────────────────────────
fetcher = load_data()
df = fetcher.df.copy()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 6px">
    <h1 style="font-size:28px;font-weight:800;margin:0">
       <span style="font-size:28px">🔬</span>
       <span style="background:linear-gradient(135deg,#06b6d4,#6366f1);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">Dataset Explorer &amp; EDA</span></h1>
    <p style="color:#64748b;font-size:13px;margin:4px 0 0">
       INFY.NS — 5-year historical data with engineered features</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Dataset overview cards ───────────────────────────────────────────────────
dc1, dc2, dc3, dc4 = st.columns(4)
dc1.markdown(metric_card("Trading Days", f"{len(df):,}"), unsafe_allow_html=True)
dc2.markdown(metric_card("Date Range",
    f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}",
    color="#06b6d4"), unsafe_allow_html=True)
dc3.markdown(metric_card("Features", f"{len(df.columns)}"), unsafe_allow_html=True)
dc4.markdown(metric_card("Window Size", f"{WINDOW_SIZE} days", color="#f59e0b"),
             unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Dataset preview ──────────────────────────────────────────────────────────
st.markdown("#### 📋 Dataset Preview (last 20 rows)")
st.dataframe(df.tail(20), use_container_width=True)

# ── Features used in the project ─────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🧩 Features Used in the Model")

features_info = {
    "Close": "Closing price — the primary prediction target.",
    "GARCH_Vol": "GARCH(1,1) conditional volatility — captures time-varying risk.",
    "RSI": "Relative Strength Index (14-day) — momentum oscillator (0–100).",
    "MACD": "Moving Average Convergence Divergence — trend-following signal.",
    "Volume_Change": "Day-over-day percentage change in trading volume.",
    "Close_Open_Ratio": "Close / Open ratio — intra-day directional strength.",
    "Sentiment": "VADER compound sentiment score from Yahoo Finance news.",
}

for feat, desc in features_info.items():
    present = "✅" if feat in df.columns else "❌"
    st.markdown(f"- {present} **{feat}** — {desc}")

# ── Technical indicators summary ─────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📊 Technical Indicators Summary")

indicator_cols = [c for c in ["Close", "GARCH_Vol", "RSI", "MACD", "Volume_Change",
                               "Close_Open_Ratio", "Sentiment"] if c in df.columns]
summary = df[indicator_cols].describe().T
summary = summary[["mean", "std", "min", "25%", "50%", "75%", "max"]]
summary = summary.round(4)
st.dataframe(summary, use_container_width=True)

# ── Price trend (unique EDA chart — candlestick-style OHLC) ──────────────────
st.markdown("---")
st.markdown("#### 📈 Historical Price Trend")

fig_price = go.Figure()
if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
    fig_price.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#10b981", decreasing_line_color="#f43f5e",
        name="OHLC",
    ))
else:
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines",
        line=dict(width=1.5, color="#8b5cf6"), name="Close",
    ))
dark_layout(fig_price, "INFY.NS — OHLC Price History", 450)
fig_price.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig_price, use_container_width=True)

# ── Volatility chart ─────────────────────────────────────────────────────────
st.markdown("#### 📉 Volatility (GARCH) Over Time")
if "GARCH_Vol" in df.columns:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=df.index, y=df["GARCH_Vol"], mode="lines",
        fill="tozeroy", fillcolor="rgba(243,63,94,.08)",
        line=dict(width=1.2, color="#f43f5e"), name="GARCH Vol",
    ))
    dark_layout(fig_vol, "Conditional Volatility (GARCH)", 350)
    st.plotly_chart(fig_vol, use_container_width=True)

# ── Sentiment feature distribution ───────────────────────────────────────────
st.markdown("#### 🎭 Sentiment Feature Distribution")
if "Sentiment" in df.columns:
    col_sh, col_sb = st.columns(2)
    with col_sh:
        fig_sh = go.Figure(go.Histogram(
            x=df["Sentiment"], nbinsx=40,
            marker_color="rgba(99,102,241,.55)",
            marker_line=dict(color="#8b5cf6", width=1),
        ))
        fig_sh.add_vline(x=0, line_dash="dash", line_color="#10b981")
        dark_layout(fig_sh, "Sentiment Score Histogram", 350)
        fig_sh.update_layout(xaxis_title="Sentiment", yaxis_title="Frequency")
        st.plotly_chart(fig_sh, use_container_width=True)
    with col_sb:
        fig_sb = go.Figure(go.Box(
            y=df["Sentiment"], name="Sentiment",
            marker_color="#06b6d4", boxpoints="outliers",
        ))
        dark_layout(fig_sb, "Sentiment Box Plot", 350)
        st.plotly_chart(fig_sb, use_container_width=True)

# ── Correlation heatmap ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🔥 Feature Correlation Heatmap")

corr_cols = [c for c in indicator_cols if c in df.columns]
corr = df[corr_cols].corr()

fig_hm = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns,
    colorscale=[[0, "#1a1f35"], [0.5, "#6366f1"], [1, "#f59e0b"]],
    zmin=-1, zmax=1,
    text=np.round(corr.values, 2), texttemplate="%{text}",
    textfont=dict(size=11, color="#f1f5f9"),
))
dark_layout(fig_hm, "Pearson Correlation Matrix", 500)
st.plotly_chart(fig_hm, use_container_width=True)

# ── RSI & MACD subplots ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🔧 RSI & MACD Indicators")
col_rsi, col_macd = st.columns(2)

with col_rsi:
    if "RSI" in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines",
                                      line=dict(width=1.2, color="#8b5cf6"), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f43f5e",
                          annotation_text="Overbought", annotation_font_color="#f43f5e")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981",
                          annotation_text="Oversold", annotation_font_color="#10b981")
        dark_layout(fig_rsi, "RSI (14-day)", 350)
        st.plotly_chart(fig_rsi, use_container_width=True)

with col_macd:
    if "MACD" in df.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines",
                                       line=dict(width=1.2, color="#06b6d4"), name="MACD"))
        if "Signal" in df.columns:
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"], mode="lines",
                                           line=dict(width=1, color="#f59e0b", dash="dot"),
                                           name="Signal"))
        if "MACD_Hist" in df.columns:
            colors_h = ["#10b981" if v >= 0 else "#f43f5e" for v in df["MACD_Hist"]]
            fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
                                       marker_color=colors_h, name="Histogram", opacity=0.4))
        dark_layout(fig_macd, "MACD", 350)
        st.plotly_chart(fig_macd, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
csv = df.to_csv().encode("utf-8")
st.download_button("⬇️  Export Full Dataset CSV", csv, "infy_dataset.csv",
                   "text/csv", use_container_width=False)
