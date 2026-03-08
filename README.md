# 📈 InfyVision

### AI‑Powered Infosys Stock Price Prediction using LSTM

**Author:** Uttam Tripathi\
**Domain:** Artificial Intelligence • Time‑Series Forecasting •
Financial Analytics

------------------------------------------------------------------------

## 🚀 Project Overview

**InfyVision** is an AI‑driven stock prediction system designed to
forecast the **next‑day closing price of Infosys (INFY.NS)** using a
**deep learning LSTM model**.

The project integrates **time‑series modeling, financial indicators, and
probabilistic risk simulation** to create a practical decision‑support
dashboard for market analysis.

A **Streamlit interactive dashboard** allows users to visualize price
trends, model predictions, performance metrics, and future risk
projections.

------------------------------------------------------------------------

## 🧠 Core Capabilities

-   Deep learning **LSTM model for time‑series forecasting**
-   **Next‑day stock price prediction**
-   Advanced **technical indicator feature engineering**
-   **Monte Carlo simulation** for future risk distribution
-   Interactive **Streamlit analytics dashboard**
-   **Feature correlation analysis** for model transparency
-   **Real‑time data retrieval** using Yahoo Finance API
-   Clean and responsive **Plotly visualizations**

------------------------------------------------------------------------

## 📊 Data Source

Historical market data is collected from **Yahoo Finance** using the
`yfinance` API.

Dataset characteristics:

-   Approximately **5 years of historical market data**
-   Infosys stock ticker: **INFY.NS**
-   Includes:

  Feature   Description
  --------- --------------------------
  Open      Opening price
  High      Highest price of the day
  Low       Lowest price of the day
  Close     Closing price
  Volume    Total trading volume

------------------------------------------------------------------------

## ⚙️ Feature Engineering

To improve model accuracy, multiple **technical indicators** are
generated from raw price data.

  Feature            Description
  ------------------ -----------------------------------
  Close              Closing stock price
  GARCH Volatility   Market volatility estimate
  RSI                Relative Strength Index
  MACD               Momentum indicator
  Volume Change      Daily percentage change in volume
  Close/Open Ratio   Intraday price movement indicator

------------------------------------------------------------------------

## 🏗 Model Architecture

The predictive model uses a **stacked LSTM neural network** designed for
sequential time‑series learning.

**Architecture:**

-   LSTM Layer --- 64 units\
-   LSTM Layer --- 32 units\
-   Dense Output Layer

**Training Configuration**

-   Window Size: **60 days**
-   Train/Test Split: **80 / 20**
-   Early Stopping for regularization

------------------------------------------------------------------------

## 📉 Model Performance

  Metric               Value
  -------------------- ---------
  MAE                  \~ ₹26
  MAPE                 \~ 1.7%
  RMSE                 \~ ₹34
  Direction Accuracy   \~ 47%

These results indicate **strong short‑term prediction capability with
low percentage error**.

------------------------------------------------------------------------

## 🎲 Risk Analysis -- Monte Carlo Simulation

To estimate potential market scenarios, the project includes **Monte
Carlo simulation**.

Simulation parameters:

-   Forecast Horizon: **30 days**
-   Simulation Runs: **500**
-   Based on **historical return mean and volatility**

Risk insights generated:

-   Expected future price
-   Median price estimate
-   **Value at Risk (95%)**
-   Probability of loss

------------------------------------------------------------------------

## 📊 Dashboard Features

The **Streamlit dashboard** provides three primary analytical views.

### 1️⃣ Price & Forecast

-   Historical price visualization
-   Moving average trend line
-   Next‑day prediction marker
-   Current market statistics

### 2️⃣ Model Performance

-   Error metrics and evaluation results
-   Model training configuration
-   Feature correlation insights

### 3️⃣ Risk Analysis

-   Monte Carlo simulation paths
-   Confidence interval bands
-   Price distribution visualization
-   Market volatility statistics

------------------------------------------------------------------------

## 📂 Project Structure

    InfyVision/
    │
    ├── InfyVision.ipynb        # Model training and experimentation
    ├── app.py                  # Streamlit dashboard
    │
    ├── saved_models/
    │   ├── infy_lstm_model.keras
    │   ├── infy_scaler.pkl
    │   └── infy_config.pkl
    │
    └── README.md

------------------------------------------------------------------------

## 🛠 Installation

Clone the repository:

    git clone https://github.com/uttamtripathi54/InfyVision.git
    cd InfyVision

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ Running the Dashboard

Launch the Streamlit application:

    streamlit run app.py

Open the generated **local Streamlit URL** in your browser.

------------------------------------------------------------------------

## 🔮 Future Improvements

Potential upgrades for the next version:

-   Multi‑stock prediction capability
-   Transformer‑based deep learning models
-   Sentiment analysis from financial news
-   Portfolio risk optimization
-   Automated trading signals

------------------------------------------------------------------------

## 👨‍💻 Author

**Uttam Tripathi**\
AI & Cloud Enthusiast\
Azure Certified \| Data Science Projects

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is developed **for educational and research purposes
only**.

Stock market predictions involve uncertainty.\
The results generated by this system **should not be considered
financial advice**.
