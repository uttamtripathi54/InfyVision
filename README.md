# InfyVision -- Infosys Stock Price Prediction using LSTM

**Author:** Uttam Tripathi\
**Project Type:** AI / Machine Learning -- Stock Market Prediction\
**Tech Stack:** Python, TensorFlow (LSTM), Streamlit, Plotly, yFinance,
Scikit‑learn

------------------------------------------------------------------------

## Overview

InfyVision is an AI-powered stock prediction system that forecasts the
**next-day price of Infosys (INFY.NS)** using a **deep learning LSTM
model**.

The project combines **time-series deep learning**, **technical
indicators**, and **Monte Carlo risk simulation** to create a complete
decision-support dashboard.

The system also includes a **Streamlit interactive dashboard** where
users can visualize price trends, predictions, model performance, and
market risk analysis.

------------------------------------------------------------------------

## Key Features

-   LSTM based deep learning model for time-series prediction
-   Next-day stock price forecasting
-   Technical indicators feature engineering
-   Monte Carlo simulation for risk analysis
-   Interactive dashboard using Streamlit
-   Feature correlation analysis
-   Real-time stock data using yFinance
-   Clean visualizations using Plotly

------------------------------------------------------------------------

## Project Architecture

The system follows a simple pipeline:

1.  Data Collection\
2.  Feature Engineering\
3.  Data Scaling & Sequence Creation\
4.  LSTM Model Training\
5.  Model Evaluation\
6.  Prediction Generation\
7.  Streamlit Dashboard Visualization

------------------------------------------------------------------------

## Dataset

Stock market data is collected using **Yahoo Finance API (yfinance)**.

Dataset contains approximately **5 years of historical Infosys stock
data**, including:

-   Open
-   High
-   Low
-   Close
-   Volume

------------------------------------------------------------------------

## Feature Engineering

The model uses multiple engineered features to improve prediction
accuracy:

  Feature            Description
  ------------------ ---------------------------------------
  Close              Closing stock price
  GARCH Volatility   Market volatility estimation
  RSI                Relative Strength Index
  MACD               Moving Average Convergence Divergence
  Volume Change      Daily percentage change in volume
  Close/Open Ratio   Price movement indicator

------------------------------------------------------------------------

## Model Architecture

The deep learning model is based on **Stacked LSTM layers**.

Architecture:

-   LSTM Layer (64 units)
-   LSTM Layer (32 units)
-   Dense Output Layer

Parameters:

-   Window Size: 60 days
-   Train/Test Split: 80 / 20
-   Early Stopping Enabled

------------------------------------------------------------------------

## Model Performance

  Metric               Value
  -------------------- ---------
  MAE                  \~ ₹26.21
  MAPE                 \~ 1.76%
  Direction Accuracy   \~ 47%
  RMSE                 \~ ₹34.09

These results show the model performs well for **short-term price
prediction**.

------------------------------------------------------------------------

## Risk Analysis (Monte Carlo Simulation)

The project includes **Monte Carlo simulation** to estimate future stock
price distributions.

Simulation parameters:

-   Simulation Horizon: 30 days
-   Runs: 500
-   Uses historical return mean and volatility

Risk metrics produced:

-   Expected Price
-   Median Price
-   Value at Risk (95%)
-   Probability of Loss

------------------------------------------------------------------------

## Dashboard Features

The Streamlit dashboard provides three main sections:

### 1. Price & Forecast

-   Historical price chart
-   Moving average visualization
-   Next-day forecast marker
-   Market statistics

### 2. Model Performance

-   Model metrics
-   Training configuration
-   Feature correlation visualization

### 3. Risk Analysis

-   Monte Carlo price simulation
-   Confidence interval bands
-   Probability of loss analysis

------------------------------------------------------------------------

## Project Structure

    Uttam_Tripathi_InfyVision/
    │
    ├── InfyVision.ipynb        # Model training notebook
    ├── app.py                  # Streamlit dashboard
    ├── saved_models/
    │   ├── infy_lstm_model.keras
    │   ├── infy_scaler.pkl
    │   └── infy_config.pkl
    │
    └── README.md

------------------------------------------------------------------------

## Installation

Clone the repository:

    git clone https://github.com/yourusername/infyvision.git
    cd infyvision

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

## Running the Dashboard

Run the Streamlit app:

    streamlit run app.py

Then open the local dashboard link in your browser.

------------------------------------------------------------------------

## Future Improvements

-   Multi-stock prediction support
-   Transformer based deep learning models
-   Portfolio optimization module
-   Sentiment analysis integration
-   Live trading signals

------------------------------------------------------------------------

## Author

**Uttam Tripathi**\
AI / Cloud Enthusiast \| Azure Certified \| Data Science Projects

------------------------------------------------------------------------

## Disclaimer

This project is created for **educational and research purposes only**.\
Stock market predictions are uncertain and should **not be considered
financial advice**.
