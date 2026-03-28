# =============================================================================
# pages/4_Performance.py — Model Performance & Accuracy  (Protected)
# =============================================================================
# Unique content: detailed per-model metrics table, best-model highlight,
# actual-vs-predicted scatter per model, residual distribution.
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
    load_data, load_models, compute_model_metrics,
    dark_layout, metric_card,
)

st.set_page_config(page_title="Performance — InfyVision", page_icon="🏆",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
require_login()
render_sidebar()

# ── Load ─────────────────────────────────────────────────────────────────────
fetcher = load_data()
models  = load_models()
metrics_df = compute_model_metrics(fetcher, models)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 6px">
    <h1 style="font-size:28px;font-weight:800;margin:0">
       <span style="font-size:28px">🏆</span>
       <span style="background:linear-gradient(135deg,#6366f1,#8b5cf6);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">Model Performance &amp; Accuracy</span></h1>
    <p style="color:#64748b;font-size:13px;margin:4px 0 0">
       Detailed comparison of LSTM, GRU &amp; Transformer on the held-out test set</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Best model highlight ────────────────────────────────────────────────────
best_idx = metrics_df["MAPE (%)"].idxmin()
best = metrics_df.iloc[best_idx]

c1, c2, c3 = st.columns(3)
c1.markdown(metric_card("🥇 Best Model", best["Model"], f'MAPE {best["MAPE (%)"]}%', "#10b981"),
            unsafe_allow_html=True)
c2.markdown(metric_card("Lowest MAE", f'₹{best["MAE (₹)"]}', best["Model"], "#06b6d4"),
            unsafe_allow_html=True)
c3.markdown(metric_card("Best R²", f'{best["R²"]}', best["Model"], "#8b5cf6"),
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Metrics comparison table ─────────────────────────────────────────────────
st.markdown("#### 📊 Full Metrics Comparison")
st.dataframe(metrics_df.style.highlight_min(
    subset=["MAE (₹)", "RMSE (₹)", "MAPE (%)"], color="rgba(16,185,129,.25)"
).highlight_max(
    subset=["R²", "Dir. Accuracy (%)"], color="rgba(99,102,241,.25)"
), use_container_width=True, hide_index=True)

st.markdown("""
> **Interpretation:** Lower MAE/RMSE/MAPE = better accuracy.  Higher R² & Direction Accuracy = the model
> captures both magnitude and trend direction well.  The highlighted model has the best overall MAPE.
""")

# ── Bar chart: metrics side by side ──────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📐 Metrics Bar Comparison")

tab_mae, tab_rmse, tab_mape, tab_r2, tab_dir = st.tabs(
    ["MAE", "RMSE", "MAPE", "R²", "Dir. Accuracy"]
)
colors = ["#6366f1", "#8b5cf6", "#06b6d4"]

for tab, col, fmt in [
    (tab_mae,  "MAE (₹)",  "₹{:.2f}"),
    (tab_rmse, "RMSE (₹)", "₹{:.2f}"),
    (tab_mape, "MAPE (%)", "{:.2f}%"),
    (tab_r2,   "R²",       "{:.4f}"),
    (tab_dir,  "Dir. Accuracy (%)", "{:.2f}%"),
]:
    with tab:
        fig = go.Figure(go.Bar(
            x=metrics_df["Model"], y=metrics_df[col],
            marker_color=colors[:len(metrics_df)],
            text=[fmt.format(v) for v in metrics_df[col]],
            textposition="outside",
        ))
        dark_layout(fig, col, height=380)
        st.plotly_chart(fig, use_container_width=True)

# ── Actual vs Predicted scatter per model ────────────────────────────────────
st.markdown("---")
st.markdown("#### 🎯 Actual vs Predicted (Test Set — Day-1 predictions)")

(_, _), (_, _), (X_test, y_test), _ = fetcher.prepare_sequences(prediction_days=5)

selected_model = st.selectbox("Select model", list(models.keys()), key="perf_model_sel")
mdl = models[selected_model]
yp = mdl.predict(X_test)
yp1 = yp[:, 0] if yp.ndim > 1 and yp.shape[-1] == 5 else yp.flatten()
yt1 = y_test[:, 0] if y_test.ndim > 1 else y_test.flatten()
ypa = fetcher.inverse_transform_prices(yp1)
yta = fetcher.inverse_transform_prices(yt1)

col_a, col_b = st.columns(2)

# Scatter
with col_a:
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=yta, y=ypa, mode="markers",
                               marker=dict(size=4, color="#8b5cf6", opacity=0.6),
                               name="Predictions"))
    mn, mx = min(yta.min(), ypa.min()), max(yta.max(), ypa.max())
    fig_s.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                               line=dict(dash="dash", color="#10b981", width=2),
                               name="Perfect fit"))
    dark_layout(fig_s, f"{selected_model} — Actual vs Predicted", 400)
    st.plotly_chart(fig_s, use_container_width=True)

# Residual histogram
with col_b:
    residuals = yta - ypa
    fig_r = go.Figure(go.Histogram(
        x=residuals, nbinsx=40,
        marker_color="rgba(99,102,241,.6)",
        marker_line=dict(color="#8b5cf6", width=1),
    ))
    fig_r.add_vline(x=0, line_dash="dash", line_color="#10b981")
    dark_layout(fig_r, f"{selected_model} — Residual Distribution", 400)
    fig_r.update_layout(xaxis_title="Residual (₹)", yaxis_title="Count")
    st.plotly_chart(fig_r, use_container_width=True)

# ── Insights ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💡 Model Insights")
for _, row in metrics_df.iterrows():
    qual = "excellent" if row["MAPE (%)"] < 3 else "good" if row["MAPE (%)"] < 5 else "moderate"
    dir_q = "strong" if row["Dir. Accuracy (%)"] > 55 else "average"
    st.markdown(
        f"- **{row['Model']}** — {qual} accuracy (MAPE {row['MAPE (%)']}%), "
        f"R² of {row['R²']}, and {dir_q} direction accuracy ({row['Dir. Accuracy (%)']}%)."
    )

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
csv = metrics_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️  Export Metrics CSV", csv, "infy_model_metrics.csv",
                   "text/csv", use_container_width=False)
