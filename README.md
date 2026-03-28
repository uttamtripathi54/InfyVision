<p align="center">
  <h1 align="center">📈 InfyVision — AI-Powered Stock Intelligence</h1>
  <p align="center">
    <strong>Predict Infosys (INFY.NS) stock prices using deep-learning ensembles of LSTM, GRU & Transformer models — backed by Monte Carlo risk simulations & real-time sentiment analysis.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/TensorFlow-2.x-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
    <img src="https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Plotly-Interactive-3f4f75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
    <img src="https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge" alt="License">
  </p>
</p>

---

## 🌟 Overview

**InfyVision** is an end-to-end, institutional-grade stock intelligence SaaS application built as part of the **Infosys Springboard** program. It combines cutting-edge deep learning models with quantitative risk analysis and NLP-driven sentiment scoring to deliver actionable 5-day price forecasts for **Infosys Ltd (INFY.NS)** on the NSE.

The application features a premium dark-themed dashboard powered by **Streamlit**, with interactive **Plotly** visualizations, real-time data from **Yahoo Finance**, and an ensemble of three neural network architectures.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Deep Learning Ensemble** | LSTM, GRU & Transformer models trained on 5 years of data, averaged into a robust ensemble forecast |
| 📊 **5-Day Price Forecast** | Next-5-trading-day predictions with confidence metrics, percentage changes & actionable insights |
| ⚠️ **Monte Carlo Risk Engine** | Geometric Brownian Motion (GBM) simulations with Value at Risk (VaR), Expected Shortfall (CVaR) & interactive confidence intervals |
| 📰 **Live Sentiment Analysis** | Yahoo Finance RSS news feed scored with VADER NLP — sentiment trends mapped to trading signals |
| 🏆 **Model Benchmarking** | Side-by-side MAE, RMSE, MAPE, R² & direction accuracy comparison across all models |
| 🔬 **Full Dataset Explorer** | Interactive EDA with OHLC candlestick charts, volatility plots, correlation heatmaps, technical indicators & feature distributions |
| 🔐 **User Authentication** | Secure login system with SHA-256 password hashing |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI LAYER                        │
│  Landing → Login → Dashboard → Performance → Risk → Sentiment    │
│                                → Dataset Explorer                │
├──────────────────────────────────────────────────────────────────┤
│                      APPLICATION LAYER                           │
│  app_utils.py (shared helpers, caching, CSS injection)           │
├──────────────────────────────────────────────────────────────────┤
│                       MODEL LAYER                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────┐    │
│  │   LSTM   │  │   GRU    │  │ Transformer  │  │  Ensemble  │    │
│  │ (64→32)  │  │ (64→32)  │  │ (64d, 4h, 2L)│  │  (Average) │    │
│  └──────────┘  └──────────┘  └──────────────┘  └────────────┘    │
├──────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                │
│  DataFetcher (yFinance) → Technical Indicators (GARCH, RSI,      │
│  MACD) → Sentiment (VADER NLP) → MinMaxScaler → Sequences        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow / Keras |
| **Frontend** | Streamlit (multi-page app) |
| **Visualization** | Plotly (interactive charts) |
| **Data Source** | yFinance (Yahoo Finance API) |
| **Volatility Modelling** | ARCH (GARCH(1,1)) |
| **Sentiment Analysis** | VADER (vaderSentiment) |
| **News Feed** | feedparser (Yahoo Finance RSS) |
| **Numerical Computing** | NumPy, Pandas, SciPy, scikit-learn |

---

## 📁 Project Structure

```
infy_vision_pro/
├── app.py                      # Streamlit entry point — redirects to Landing
├── main.py                     # CLI for training, comparison & prediction
├── config.py                   # Hyperparameters, paths & feature configuration
├── requirements.txt            # Python dependencies
├── users.json                  # User credentials (SHA-256 hashed passwords)
│
├── data/
│   ├── __init__.py
│   └── data_fetcher.py         # yFinance downloader, indicator engine, scaler
│
├── models/
│   ├── __init__.py
│   ├── lstm_model.py           # Stacked LSTM (64 → 32 units)
│   ├── gru_model.py            # Stacked GRU (64 → 32 units)
│   ├── transformer_model.py    # Transformer encoder (4 heads, 2 layers)
│   └── ensemble.py             # Simple averaging ensemble
│
├── sentiment/
│   ├── __init__.py
│   └── sentiment_analyzer.py   # Yahoo RSS fetcher + VADER scoring
│
├── utils/
│   ├── __init__.py
│   ├── app_utils.py            # Shared Streamlit helpers (CSS, sidebar, caching)
│   ├── helpers.py              # Metric calculation utilities
│   └── logger.py               # Logging configuration
│
├── pages/                      # Streamlit multi-page app routes
│   ├── 1_Landing.py            # Public hero/landing page
│   ├── 2_Login.py              # Authentication page
│   ├── 3_Dashboard.py          # 5-day forecast dashboard (protected)
│   ├── 4_Performance.py        # Model benchmarking & accuracy (protected)
│   ├── 5_Risk_Analysis.py      # Monte Carlo GBM simulation (protected)
│   ├── 6_News_Sentiment.py     # News feed & sentiment analysis (protected)
│   └── 7_Dataset_Explorer.py   # EDA & feature exploration (protected)
│
└── saved_models/               # Persisted trained model weights
    ├── infy_lstm_5d.keras
    ├── infy_gru_5d.keras
    └── infy_transformer_5d.keras
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed
- **pip** package manager
- Internet connection (for fetching live stock data & news)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/infy_vision_pro.git
cd infy_vision_pro
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 💻 Usage

### Web Application (Streamlit)

1. **Landing Page** — Public hero page showcasing features and tech stack
2. **Login** — Authenticate with your credentials (default: `uttam` / `12345`)
3. **Dashboard** — View the 5-day ensemble forecast with key metrics and interactive charts
4. **Performance** — Compare LSTM, GRU & Transformer accuracy (MAE, RMSE, MAPE, R², Direction Accuracy)
5. **Risk Analysis** — Run Monte Carlo GBM simulations with configurable parameters (simulations, horizon, confidence level)
6. **News & Sentiment** — Browse latest Infosys headlines with VADER sentiment scores
7. **Dataset Explorer** — Full EDA with OHLC candlesticks, volatility, correlations, RSI & MACD charts

### CLI (Command Line)

The `main.py` script provides a CLI interface for training and prediction:

```bash
# Train all models (LSTM + GRU + Transformer)
python main.py --train all

# Train a specific model
python main.py --train lstm

# Compare all models
python main.py --compare

# Predict next 5 days with a specific model
python main.py --predict lstm
python main.py --predict gru
python main.py --predict transformer

# Predict using the ensemble (average of all models)
python main.py --predict ensemble

# Fetch data only (no training)
python main.py --fetch-only

# Custom prediction horizon
python main.py --train all --prediction-days 10
```

---

## 🧠 Model Details

### Input Features (7 features × 60-day window)

| Feature | Description |
|---|---|
| `Close` | Closing price — the primary prediction target |
| `GARCH_Vol` | GARCH(1,1) conditional volatility — captures time-varying risk |
| `RSI` | Relative Strength Index (14-day) — momentum oscillator (0–100) |
| `MACD` | Moving Average Convergence Divergence — trend-following signal |
| `Volume_Change` | Day-over-day percentage change in trading volume |
| `Close_Open_Ratio` | Close / Open ratio — intra-day directional strength |
| `Sentiment` | VADER compound sentiment score from Yahoo Finance news |

### Model Architectures

#### LSTM (Long Short-Term Memory)
- **Architecture:** Stacked LSTM → Dense
- **Units:** 64 → 32
- **Dropout:** 20%
- **Output:** 5 steps (next 5 trading days)

#### GRU (Gated Recurrent Unit)
- **Architecture:** Stacked GRU → Dense
- **Units:** 64 → 32
- **Dropout:** 20%
- **Output:** 5 steps

#### Transformer Encoder
- **Architecture:** Positional Encoding → Multi-Head Attention (×2) → Global Average Pooling → Dense
- **Embedding Dim:** 64
- **Attention Heads:** 4
- **Encoder Layers:** 2
- **Dropout:** 20%
- **Output:** 5 steps

#### Ensemble
- **Method:** Simple averaging of predictions from all loaded models
- **Benefit:** Reduced variance and improved robustness

### Training Configuration

| Parameter | Value |
|---|---|
| **Window Size** | 60 trading days |
| **Prediction Horizon** | 5 trading days |
| **Train/Test Split** | 80% / 20% |
| **Validation Split** | 90% of training set |
| **Epochs** | 30 (with early stopping, patience=5) |
| **Batch Size** | 32 |
| **Optimizer** | Adam |
| **Loss Function** | Mean Squared Error (MSE) |
| **Scaler** | MinMaxScaler (0–1) |

---

## ⚠️ Risk Analysis — Monte Carlo Simulation

The risk engine uses **Geometric Brownian Motion (GBM)** to simulate thousands of possible future price paths:

- **Configurable Simulations:** 100 – 10,000 paths
- **Adjustable Horizon:** 5 – 120 trading days
- **Confidence Levels:** 90%, 95%, 99%

### Risk Metrics Computed

| Metric | Description |
|---|---|
| **Value at Risk (VaR)** | Maximum expected loss at the chosen confidence level |
| **Expected Shortfall (CVaR)** | Average loss in worst-case scenarios beyond VaR |
| **Probability of Profit** | Percentage of simulations ending above current price |
| **Confidence Bands** | 5th–95th and 25th–75th percentile price corridors |

---

## 📰 Sentiment Analysis Pipeline

```
Yahoo Finance RSS Feed → feedparser → Raw Headlines + Summaries
     ↓
VADER SentimentIntensityAnalyzer → Compound Score (-1 to +1)
     ↓
Daily Aggregation → Forward-fill to Trading Days
     ↓
Input Feature for LSTM / GRU / Transformer Models
```

- **Source:** Yahoo Finance RSS (`feeds.finance.yahoo.com`)
- **Engine:** VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Lookback:** Configurable (default: 7 days)
- **Classification:** Positive (>0.05), Negative (<-0.05), Neutral

---

## 📊 Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **MAE** (Mean Absolute Error) | Average absolute prediction error in ₹ |
| **RMSE** (Root Mean Square Error) | Penalizes larger errors more heavily |
| **MAPE** (Mean Absolute Percentage Error) | Scale-independent accuracy percentage |
| **R²** (Coefficient of Determination) | Proportion of variance explained by the model |
| **Direction Accuracy** | Percentage of correctly predicted up/down moves |

---

## 🔐 Authentication

The app uses a simple JSON-based user store (`users.json`) with **SHA-256** hashed passwords. Protected pages (Dashboard, Performance, Risk, Sentiment, Explorer) require login.

To add a new user, generate the SHA-256 hash of the password and add it to `users.json`:

```python
import hashlib
hashlib.sha256("your_password".encode()).hexdigest()
```

---

## 🗂️ Data Source

- **Provider:** Yahoo Finance via the `yfinance` Python library
- **Ticker:** `INFY.NS` (Infosys Ltd — National Stock Exchange of India)
- **History:** 5 years of daily OHLCV data (auto-adjusted)
- **Refresh:** Data is fetched fresh on each app launch / CLI run

---

## 📋 Dependencies

```
numpy
pandas
yfinance
scikit-learn
tensorflow
arch
plotly
matplotlib
seaborn
feedparser
vaderSentiment
streamlit
scipy
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Uttam Tripathi**

Built with ❤️ as part of the **Infosys Springboard** program.

---

<p align="center">
  <sub>© 2026 InfyVision · AI-Powered Stock Intelligence</sub>
</p>
