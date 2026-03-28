# =============================================================================
# pages/6_News_Sentiment.py — News & Sentiment Feed  (Protected)
# =============================================================================
# Unique content: full news article list with per-article VADER scores,
# sentiment trend summary, polarity breakdown, daily sentiment chart.
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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.app_utils import (
    inject_css, render_sidebar, require_login,
    load_data, dark_layout, metric_card,
)
from sentiment.sentiment_analyzer import SentimentAnalyzer

st.set_page_config(page_title="News & Sentiment — InfyVision", page_icon="📰",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
require_login()
render_sidebar()

# ── Fetch news ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📰  Fetching latest Infosys news …", ttl=900)
def fetch_news_with_scores(days_back=14):
    sa = SentimentAnalyzer("INFY.NS")
    vader = SentimentIntensityAnalyzer()
    news = sa.fetch_news(days_back)
    if not news:
        return pd.DataFrame()
    df_news = pd.DataFrame(news)
    df_news["text"] = df_news["title"] + " " + df_news["summary"]
    # VADER compound + polarity breakdown
    compounds, positives, negatives, neutrals = [], [], [], []
    for t in df_news["text"]:
        sc = vader.polarity_scores(t)
        compounds.append(sc["compound"])
        positives.append(sc["pos"])
        negatives.append(sc["neg"])
        neutrals.append(sc["neu"])
    df_news["compound"]  = compounds
    df_news["positive"]  = positives
    df_news["negative"]  = negatives
    df_news["neutral"]   = neutrals
    df_news["label"] = df_news["compound"].apply(
        lambda x: "🟢 Positive" if x > 0.05 else ("🔴 Negative" if x < -0.05 else "🟡 Neutral")
    )
    df_news["date"] = pd.to_datetime(df_news["date"])
    return df_news.sort_values("date", ascending=False).reset_index(drop=True)

news_df = fetch_news_with_scores()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 6px">
    <h1 style="font-size:28px;font-weight:800;margin:0">
       <span style="font-size:28px">📰</span>
       <span style="background:linear-gradient(135deg,#06b6d4,#10b981);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent">News &amp; Sentiment Analysis</span></h1>
    <p style="color:#64748b;font-size:13px;margin:4px 0 0">
       Latest Infosys headlines scored with VADER NLP sentiment engine</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if news_df.empty:
    st.info("No recent news articles found.  Sentiment features in the model default to 0.")
    st.stop()

# ── Summary cards ────────────────────────────────────────────────────────────
avg_sent   = float(news_df["compound"].mean())
n_pos      = int((news_df["compound"] > 0.05).sum())
n_neg      = int((news_df["compound"] < -0.05).sum())
n_neu      = len(news_df) - n_pos - n_neg
overall    = "Bullish 📈" if avg_sent > 0.05 else ("Bearish 📉" if avg_sent < -0.05 else "Neutral ➡️")
sent_color = "#10b981" if avg_sent > 0.05 else ("#f43f5e" if avg_sent < -0.05 else "#f59e0b")

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.markdown(metric_card("Avg Sentiment", f"{avg_sent:+.3f}", overall, sent_color),
             unsafe_allow_html=True)
sc2.markdown(metric_card("Positive", str(n_pos), f"{n_pos/len(news_df)*100:.0f}%", "#10b981"),
             unsafe_allow_html=True)
sc3.markdown(metric_card("Negative", str(n_neg), f"{n_neg/len(news_df)*100:.0f}%", "#f43f5e"),
             unsafe_allow_html=True)
sc4.markdown(metric_card("Neutral", str(n_neu), f"{n_neu/len(news_df)*100:.0f}%", "#f59e0b"),
             unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Daily sentiment trend chart ──────────────────────────────────────────────
st.markdown("#### 📈 Daily Sentiment Trend")
daily = news_df.groupby(news_df["date"].dt.date)["compound"].mean().reset_index()
daily.columns = ["Date", "Avg Sentiment"]

fig_trend = go.Figure()
colors_bar = ["#10b981" if v > 0.05 else "#f43f5e" if v < -0.05 else "#f59e0b"
              for v in daily["Avg Sentiment"]]
fig_trend.add_trace(go.Bar(
    x=daily["Date"].astype(str), y=daily["Avg Sentiment"],
    marker_color=colors_bar, name="Sentiment",
))
fig_trend.add_hline(y=0, line_dash="dash", line_color="rgba(148,163,184,.3)")
dark_layout(fig_trend, "Daily Average Sentiment Score", 360)
fig_trend.update_layout(xaxis_title="Date", yaxis_title="VADER Compound")
st.plotly_chart(fig_trend, use_container_width=True)

# ── Sentiment distribution pie ───────────────────────────────────────────────
st.markdown("#### 🔵 Sentiment Breakdown")
col_pie, col_box = st.columns(2)

with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=["Positive", "Negative", "Neutral"],
        values=[n_pos, n_neg, n_neu],
        marker=dict(colors=["#10b981", "#f43f5e", "#f59e0b"]),
        hole=0.45,
        textinfo="label+percent",
        textfont=dict(color="#f1f5f9"),
    ))
    dark_layout(fig_pie, "Sentiment Distribution", 380)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_box:
    fig_box = go.Figure(go.Box(
        y=news_df["compound"], name="Compound",
        marker_color="#8b5cf6",
        boxpoints="all", jitter=0.4, pointpos=-1.5,
    ))
    dark_layout(fig_box, "Compound Score Distribution", 380)
    fig_box.update_layout(yaxis_title="VADER Compound")
    st.plotly_chart(fig_box, use_container_width=True)

# ── Full news list ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📋 All News Articles")

for _, row in news_df.iterrows():
    badge_color = "#10b981" if "Positive" in row["label"] else "#f43f5e" if "Negative" in row["label"] else "#f59e0b"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1f35,#222845);border:1px solid rgba(99,102,241,.12);
        border-radius:12px;padding:18px 22px;margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:11px;color:#64748b">{row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}</span>
            <span style="background:{badge_color};color:#fff;padding:2px 10px;border-radius:20px;
                font-size:11px;font-weight:600">{row['label']}</span>
        </div>
        <p style="color:#f1f5f9;font-weight:600;font-size:15px;margin:0 0 6px">{row['title']}</p>
        <p style="color:#94a3b8;font-size:13px;margin:0 0 8px;line-height:1.5">{row['summary'][:200]}{'…' if len(str(row['summary'])) > 200 else ''}</p>
        <div style="display:flex;gap:16px;font-size:11px;color:#64748b">
            <span>Compound: <strong style="color:#8b5cf6">{row['compound']:+.3f}</strong></span>
            <span>Pos: <strong style="color:#10b981">{row['positive']:.2f}</strong></span>
            <span>Neg: <strong style="color:#f43f5e">{row['negative']:.2f}</strong></span>
            <span>Neu: <strong style="color:#f59e0b">{row['neutral']:.2f}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Insights ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💡 Sentiment Insights")
st.markdown(f"""
- **Overall Mood:** The average VADER compound score across {len(news_df)} articles is **{avg_sent:+.3f}**, 
  indicating an overall **{overall.split()[0].lower()}** sentiment.
- **Positive/Negative Split:** {n_pos} positive vs {n_neg} negative articles 
  (ratio {'favours bulls' if n_pos > n_neg else 'favours bears' if n_neg > n_pos else 'is balanced'}).
- **Impact on Model:** The sentiment feature feeds into the LSTM/GRU/Transformer as a daily aggregate 
  score, influencing directional bias in the 5-day forecast.
""")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
export_df = news_df[["date", "title", "summary", "compound", "positive", "negative", "neutral", "label"]].copy()
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️  Export News & Sentiment CSV", csv, "infy_news_sentiment.csv",
                   "text/csv", use_container_width=False)
