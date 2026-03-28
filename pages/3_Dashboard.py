# =============================================================================
# pages/3_Dashboard.py — Main forecast overview  (Protected)
# =============================================================================
# Shows: 5-day ensemble forecast chart, key metric cards (predicted price,
# % change, confidence, trend), quick market snapshot.
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
from utils.app_utils import (
    inject_css, render_sidebar, require_login,
    load_data, load_models, predict_5day,
    dark_layout, metric_card,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard — InfyVision", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
require_login()
render_sidebar()

# ── Load data & models ───────────────────────────────────────────────────────
fetcher = load_data()
models  = load_models()
pred_df, current_price, next_dates = predict_5day(fetcher, models)

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 6px">
    <h1 style="font-size:28px;font-weight:800;margin:0">
       <span style="font-size:28px">📊</span>
       <span style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">Forecast Dashboard</span></h1>
    <p style="color:#64748b;font-size:13px;margin:4px 0 0">
       INFY.NS · 5-Day Ahead Ensemble Prediction · Updated in real time</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Key metric cards ─────────────────────────────────────────────────────────
ens_prices = pred_df["Ensemble"].values
day5_price = ens_prices[-1]
pct_change = ((day5_price - current_price) / current_price) * 100
trend      = "Bullish 📈" if pct_change > 0 else "Bearish 📉"
avg_pred   = np.mean(ens_prices)
spread     = ((max(ens_prices) - min(ens_prices)) / current_price) * 100

c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Current Price", f"₹{current_price:,.2f}"), unsafe_allow_html=True)
c2.markdown(metric_card("Day-5 Forecast", f"₹{day5_price:,.2f}",
                         f"{pct_change:+.2f}%",
                         "#10b981" if pct_change > 0 else "#f43f5e"),
            unsafe_allow_html=True)
c3.markdown(metric_card("5-Day Avg Forecast", f"₹{avg_pred:,.2f}"), unsafe_allow_html=True)
c4.markdown(metric_card("Trend Signal", trend,
                         f"Spread: {spread:.2f}%",
                         "#10b981" if pct_change > 0 else "#f43f5e"),
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Forecast chart (Ensemble default, all models shown) ─────────────────────
model_cols = [c for c in pred_df.columns if c not in ("Day", "Date")]
fig = go.Figure()

# Add all individual models as thin, semi-transparent lines
for col in model_cols:
    if col != "Ensemble":
        fig.add_trace(go.Scatter(
            x=pred_df["Date"], y=pred_df[col], name=col,
            mode="lines+markers",
            line=dict(width=1.5, dash="dot"),
            opacity=0.55,
        ))

# Add current price anchor
fig.add_trace(go.Scatter(
    x=[fetcher.df.index[-1].strftime("%Y-%m-%d")] + pred_df["Date"].tolist(),
    y=[current_price] + pred_df["Ensemble"].tolist(),
    name="Ensemble",
    mode="lines+markers",
    line=dict(width=3, color="#8b5cf6"),
    marker=dict(size=8, color="#8b5cf6", line=dict(width=2, color="#fff")),
))

# Horizontal reference line at current price
fig.add_hline(y=current_price, line_dash="dash", line_color="rgba(148,163,184,.35)",
              annotation_text=f"Current ₹{current_price:,.2f}",
              annotation_font_color="#94a3b8")

dark_layout(fig, "5-Day Price Forecast — INFY.NS", height=440)
st.plotly_chart(fig, use_container_width=True)

# ── Day-by-day prediction table ──────────────────────────────────────────────
st.markdown("#### 📋 Day-by-Day Predictions")
display_df = pred_df.copy()
for c in model_cols:
    display_df[f"{c} Δ%"] = ((display_df[c] - current_price) / current_price * 100).round(2)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Market snapshot ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📈 Market Snapshot")
df = fetcher.df
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Open", f"₹{float(df['Open'].iloc[-1]):,.2f}")
mc2.metric("High", f"₹{float(df['High'].iloc[-1]):,.2f}")
mc3.metric("Low", f"₹{float(df['Low'].iloc[-1]):,.2f}")
mc4.metric("Close", f"₹{float(df['Close'].iloc[-1]):,.2f}")
mc5.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️  Export Predictions CSV", csv, "infy_predictions.csv",
                   "text/csv", use_container_width=False)
