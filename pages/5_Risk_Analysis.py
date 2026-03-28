# =============================================================================
# pages/5_Risk_Analysis.py — Monte Carlo GBM Risk Analysis  (Protected)
# =============================================================================
# Unique content: interactive MC simulation, price distribution histogram,
# confidence bands, VaR, Expected Shortfall, risk interpretation.
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
    load_data, monte_carlo_gbm,
    dark_layout, metric_card,
)

st.set_page_config(page_title="Risk Analysis — InfyVision", page_icon="⚠️",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
require_login()
render_sidebar()

# ── Load data ────────────────────────────────────────────────────────────────
fetcher = load_data()
df = fetcher.df
current_price = float(df["Close"].iloc[-1])

# Historical statistics for GBM calibration
daily_returns = df["Close"].pct_change().dropna()
mu_annual    = float(daily_returns.mean()) * 252
sigma_annual = float(daily_returns.std()) * np.sqrt(252)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 6px">
    <h1 style="font-size:28px;font-weight:800;margin:0">
       <span style="font-size:28px">⚠️</span>
       <span style="background:linear-gradient(135deg,#f43f5e,#f59e0b);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">Monte Carlo Risk Analysis</span></h1>
    <p style="color:#64748b;font-size:13px;margin:4px 0 0">
       Geometric Brownian Motion simulation for INFY.NS — interactive controls</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Controls ─────────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    n_sims = st.slider("Number of Simulations", 100, 10000, 1000, 100, key="mc_sims")
with col_ctrl2:
    horizon = st.slider("Time Horizon (trading days)", 5, 120, 30, 5, key="mc_horizon")
with col_ctrl3:
    confidence = st.selectbox("Confidence Level", [90, 95, 99], index=1, key="mc_conf")

# ── Run simulation ───────────────────────────────────────────────────────────
paths = monte_carlo_gbm(current_price, mu_annual, sigma_annual,
                        T=horizon, N=n_sims, seed=42)

final_prices  = paths[:, -1]
returns_pct   = ((final_prices - current_price) / current_price) * 100
alpha         = 1 - confidence / 100

# Risk metrics
var_price     = np.percentile(final_prices, alpha * 100)
var_pct       = ((var_price - current_price) / current_price) * 100
es_mask       = final_prices <= var_price
es_price      = float(np.mean(final_prices[es_mask])) if es_mask.any() else var_price
es_pct        = ((es_price - current_price) / current_price) * 100
median_price  = float(np.median(final_prices))
mean_price    = float(np.mean(final_prices))
prob_profit   = float(np.mean(final_prices > current_price) * 100)
max_gain_pct  = float(np.max(returns_pct))
max_loss_pct  = float(np.min(returns_pct))

# ── Metric cards ─────────────────────────────────────────────────────────────
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.markdown(metric_card("Current Price", f"₹{current_price:,.2f}"), unsafe_allow_html=True)
r1c2.markdown(metric_card(f"VaR ({confidence}%)", f"₹{var_price:,.2f}",
                           f"{var_pct:+.2f}%", "#f43f5e"), unsafe_allow_html=True)
r1c3.markdown(metric_card("Expected Shortfall", f"₹{es_price:,.2f}",
                           f"{es_pct:+.2f}%", "#f43f5e"), unsafe_allow_html=True)
r1c4.markdown(metric_card("Prob. of Profit", f"{prob_profit:.1f}%",
                           "", "#10b981" if prob_profit > 50 else "#f43f5e"),
              unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
r2c1.markdown(metric_card("Median Outcome", f"₹{median_price:,.2f}"), unsafe_allow_html=True)
r2c2.markdown(metric_card("Mean Outcome", f"₹{mean_price:,.2f}"), unsafe_allow_html=True)
r2c3.markdown(metric_card("Max Upside", f"{max_gain_pct:+.2f}%", "", "#10b981"), unsafe_allow_html=True)
r2c4.markdown(metric_card("Max Downside", f"{max_loss_pct:+.2f}%", "", "#f43f5e"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Simulation fan chart ─────────────────────────────────────────────────────
st.markdown("#### 🎲 Simulation Paths & Confidence Bands")

percentiles = {
    "5th":  np.percentile(paths, 5, axis=0),
    "25th": np.percentile(paths, 25, axis=0),
    "50th": np.percentile(paths, 50, axis=0),
    "75th": np.percentile(paths, 75, axis=0),
    "95th": np.percentile(paths, 95, axis=0),
}
x_days = list(range(horizon + 1))

fig_fan = go.Figure()

# Show a sample of paths (max 200 for performance)
sample_n = min(n_sims, 200)
for i in range(sample_n):
    fig_fan.add_trace(go.Scatter(
        x=x_days, y=paths[i], mode="lines",
        line=dict(width=0.3, color="rgba(139,92,246,.08)"),
        showlegend=False, hoverinfo="skip",
    ))

# Confidence bands
fig_fan.add_trace(go.Scatter(x=x_days, y=percentiles["95th"], mode="lines",
                              line=dict(width=0), showlegend=False))
fig_fan.add_trace(go.Scatter(x=x_days, y=percentiles["5th"], mode="lines",
                              fill="tonexty", fillcolor="rgba(99,102,241,.12)",
                              line=dict(width=0), name="90% CI"))
fig_fan.add_trace(go.Scatter(x=x_days, y=percentiles["75th"], mode="lines",
                              line=dict(width=0), showlegend=False))
fig_fan.add_trace(go.Scatter(x=x_days, y=percentiles["25th"], mode="lines",
                              fill="tonexty", fillcolor="rgba(99,102,241,.22)",
                              line=dict(width=0), name="50% CI"))
fig_fan.add_trace(go.Scatter(x=x_days, y=percentiles["50th"], mode="lines",
                              line=dict(width=2.5, color="#8b5cf6"), name="Median"))
fig_fan.add_hline(y=current_price, line_dash="dash", line_color="rgba(148,163,184,.4)",
                  annotation_text=f"Current ₹{current_price:,.2f}",
                  annotation_font_color="#94a3b8")

dark_layout(fig_fan, f"Monte Carlo GBM — {n_sims:,} simulations × {horizon} days", 500)
fig_fan.update_layout(xaxis_title="Trading Days Ahead", yaxis_title="Price (₹)")
st.plotly_chart(fig_fan, use_container_width=True)

# ── Final price distribution histogram ───────────────────────────────────────
st.markdown("#### 📊 Terminal Price Distribution")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=final_prices, nbinsx=60,
    marker_color="rgba(99,102,241,.55)",
    marker_line=dict(color="#8b5cf6", width=1),
    name="Distribution",
))
fig_hist.add_vline(x=current_price, line_dash="dash", line_color="#10b981",
                   annotation_text="Current", annotation_font_color="#10b981")
fig_hist.add_vline(x=var_price, line_dash="dash", line_color="#f43f5e",
                   annotation_text=f"VaR {confidence}%", annotation_font_color="#f43f5e")
fig_hist.add_vline(x=median_price, line_dash="dot", line_color="#f59e0b",
                   annotation_text="Median", annotation_font_color="#f59e0b")

dark_layout(fig_hist, f"Terminal Price Distribution after {horizon} Days", 400)
fig_hist.update_layout(xaxis_title="Price (₹)", yaxis_title="Frequency")
st.plotly_chart(fig_hist, use_container_width=True)

# ── Risk interpretation ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💡 Risk Interpretation")

st.markdown(f"""
- **Value at Risk (VaR) at {confidence}%:** There is a **{alpha*100:.0f}%** chance the price could 
  fall below **₹{var_price:,.2f}** ({var_pct:+.2f}%) over the next **{horizon}** trading days.

- **Expected Shortfall (CVaR):** If the price *does* breach the VaR level, the average loss 
  scenario results in a price of **₹{es_price:,.2f}** ({es_pct:+.2f}%).

- **Probability of Profit:** Based on {n_sims:,} simulations, there is a **{prob_profit:.1f}%** 
  probability that the stock will be **above** the current price after {horizon} days.

- **Spread:** The 90% confidence interval for the terminal price is 
  **₹{percentiles['5th'][-1]:,.2f}** – **₹{percentiles['95th'][-1]:,.2f}**, implying a 
  potential range of **₹{percentiles['95th'][-1]-percentiles['5th'][-1]:,.2f}**.

- **GBM Parameters:** Annual drift (μ) = {mu_annual*100:.2f}%, 
  Annual volatility (σ) = {sigma_annual*100:.2f}%.
""")

risk_level = "Low" if sigma_annual < 0.2 else "Moderate" if sigma_annual < 0.35 else "High"
risk_color = "#10b981" if risk_level == "Low" else "#f59e0b" if risk_level == "Moderate" else "#f43f5e"
st.markdown(
    f'<p style="font-size:16px;font-weight:700">Overall Risk Assessment: '
    f'<span style="color:{risk_color}">{risk_level} Risk</span></p>',
    unsafe_allow_html=True,
)

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
sim_summary = pd.DataFrame({
    "Metric": ["Current Price", f"VaR ({confidence}%)", "Expected Shortfall",
               "Median Outcome", "Mean Outcome", "Prob. of Profit",
               "Max Upside %", "Max Downside %", "Annual Drift", "Annual Volatility"],
    "Value": [f"₹{current_price:,.2f}", f"₹{var_price:,.2f}", f"₹{es_price:,.2f}",
              f"₹{median_price:,.2f}", f"₹{mean_price:,.2f}", f"{prob_profit:.1f}%",
              f"{max_gain_pct:+.2f}%", f"{max_loss_pct:+.2f}%",
              f"{mu_annual*100:.2f}%", f"{sigma_annual*100:.2f}%"],
})
csv = sim_summary.to_csv(index=False).encode("utf-8")
st.download_button("⬇️  Export Risk Metrics CSV", csv, "infy_risk_analysis.csv",
                   "text/csv", use_container_width=False)
