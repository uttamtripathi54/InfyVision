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
    
    # Header
    st.title("📈 InfyVision - Infosys Stock Predictor")
    st.markdown("Built with 💙 by **Uttam Tripathi** · Based on LSTM Analysis")
    st.divider()

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
        st.header("⚙️ Controls")
        st.text_input("Ticker", value=TICKER, disabled=True)
        
        st.subheader("Simulation")
        sim_days = st.slider("Days", 10, 60, SIM_DAYS)
        sim_runs = st.slider("Runs", 100, 1000, SIM_RUNS)
        
        st.subheader("Chart")
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
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=df.index[-180:], 
                y=df['Close'].iloc[-180:],
                name="Price", 
                line=dict(color='#00d4ff', width=2)
            ))
            
            # Moving average
            if show_ma:
                ma20 = df['Close'].iloc[-180:].rolling(20).mean()
                fig.add_trace(go.Scatter(
                    x=df.index[-180:], 
                    y=ma20,
                    name="20-day MA", 
                    line=dict(color='orange', width=1, dash='dash')
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
                marker=dict(color='yellow', size=12, symbol='star')
            ))
            
            fig.update_layout(
                template='plotly_dark', 
                height=450, 
                hovermode='x unified',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 📋 Today's Stats")
            st.markdown(f"""
            **Date:** {df.index[-1].strftime('%Y-%m-%d')}
            
            **Open:** ₹{df['Open'].iloc[-1]:.2f}
            
            **High:** ₹{df['High'].iloc[-1]:.2f}
            
            **Low:** ₹{df['Low'].iloc[-1]:.2f}
            
            **Volume:** {df['Volume'].iloc[-1]:,.0f}
            
            **RSI:** {df['RSI'].iloc[-1]:.1f}
            
            **MACD:** {df['MACD'].iloc[-1]:.2f}
            """)
    
    # ============================================
    # TAB 2: Model Performance - FIXED DISPLAY
    # ============================================
    with tab2:
        st.subheader("📊 Jupyter Notebook Results")
        
        # Three columns with clean formatting
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Error Metrics")
            st.metric("MAE", f"₹{mae:.2f}", "Target: < ₹20")
            st.metric("MAPE", f"{mape:.2f}%", "Target: < 3%")
            st.metric("Direction Acc", f"{direction:.1f}%", "Target: > 55%")
        
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
        st.subheader("📊 Feature Importance")
        features = FEATURE_COLS + ['Volume']
        corr_data = df[features].corr()['Close'].drop('Close').sort_values()
        
        fig = go.Figure(data=[
            go.Bar(
                x=corr_data.values,
                y=corr_data.index,
                orientation='h',
                marker_color=['#00d4ff' if x > 0 else '#ff4444' for x in corr_data.values],
                text=[f'{x:.3f}' for x in corr_data.values],
                textposition='outside'
            )
        ])
        fig.update_layout(
            template='plotly_dark',
            height=300,
            xaxis_title="Correlation with Price",
            margin=dict(l=0, r=0, t=0, b=0)
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
        
        # Confidence band chart
        fig = go.Figure()
        
        x_vals = list(range(sim_days + 1))
        lower = np.percentile(paths, 5, axis=1)
        upper = np.percentile(paths, 95, axis=1)
        
        fig.add_trace(go.Scatter(
            x=x_vals + x_vals[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor='rgba(100,150,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_vals, 
            y=np.percentile(paths, 50, axis=1),
            name='Median',
            line=dict(color='cyan', width=2)
        ))
        
        fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color="yellow",
            annotation_text=f"Current: ₹{current_price:.2f}",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            title=f"Monte Carlo - {sim_days} Days",
            xaxis_title="Days Ahead",
            yaxis_title="Price (₹)",
            hovermode='x unified',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution
        with st.expander("📊 View Price Distribution"):
            fig_hist = go.Figure(data=[
                go.Histogram(
                    x=final_prices,
                    nbinsx=40,
                    marker_color='#00d4ff'
                )
            ])
            fig_hist.add_vline(x=current_price, line_dash="dash", line_color="yellow")
            fig_hist.update_layout(
                template='plotly_dark',
                height=250,
                xaxis_title="Price (₹)",
                yaxis_title="Frequency",
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Market stats
        with st.expander("📈 Market Statistics"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Daily Return", f"{mu*100:.4f}%")
            col2.metric("Daily Vol", f"{sigma*100:.4f}%")
            col3.metric("Annual Vol", f"{sigma*np.sqrt(252)*100:.2f}%")

    # ============================================
    # Footer with Personal Touch
    # ============================================
    st.divider()
    st.markdown("""
    <div style='text-align: center; opacity: 0.8;'>
        <span style='font-size: 12px;'>Crafted with 💙 by Uttam Tripathi · InfyVision v1.0</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()