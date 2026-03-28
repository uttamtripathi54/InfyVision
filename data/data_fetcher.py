# data/data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime
from config import *
from sentiment.sentiment_analyzer import SentimentAnalyzer
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataFetcher:
    """Fetch and preprocess stock data with technical indicators and sentiment."""

    def __init__(self, ticker=TICKER, start_date=START_DATE, end_date=END_DATE):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sentiment_analyzer = SentimentAnalyzer(ticker)

    def fetch_data(self):
        """Download price data from yfinance."""
        logger.info(f"Fetching {self.ticker} from {self.start_date.date()} to {self.end_date.date()}")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        self.df = df
        logger.info(f"Downloaded {len(df)} trading days")
        return df

    def compute_indicators(self):
        """Add GARCH volatility, RSI, MACD, volume change, close/open ratio."""
        if self.df is None:
            raise ValueError("No data. Call fetch_data first.")
        df = self.df.copy()

        # GARCH volatility
        try:
            returns = 100 * df['Close'].pct_change().dropna()
            am = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
            res = am.fit(disp='off')
            df['GARCH_Vol'] = res.conditional_volatility
        except Exception as e:
            logger.warning(f"GARCH failed, using rolling std: {e}")
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
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        # Volume change
        df['Volume_Change'] = df['Volume'].pct_change()

        # Close/Open ratio
        df['Close_Open_Ratio'] = df['Close'] / df['Open']

        # Drop rows with NaN from indicators
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        self.df = df
        logger.info("Indicators computed.")
        return df

    def add_sentiment(self, days_back=7):
        """Add daily sentiment from news to the dataframe."""
        daily_sent = self.sentiment_analyzer.get_daily_sentiment(days_back)
        if daily_sent.empty:
            logger.warning("No sentiment data. Adding zero column.")
            self.df['Sentiment'] = 0.0
            return self.df

        # Align with trading days: forward fill missing
        self.df = self.df.reset_index()
        self.df['date'] = pd.to_datetime(self.df['Date']).dt.date
        daily_sent['date'] = pd.to_datetime(daily_sent['date']).dt.date
        merged = self.df.merge(daily_sent, on='date', how='left')
        merged['Sentiment'] = merged['sentiment'].fillna(0.0)
        merged = merged.drop(columns=['sentiment']).set_index('Date')
        self.df = merged
        logger.info("Sentiment added.")
        return self.df

    def prepare_sequences(self, feature_cols=None, prediction_days=1):
        """Create sequences for LSTM input and output.
        
        Args:
            feature_cols: columns to use for features
            prediction_days: number of days ahead to predict (1 or 5)
        """
        if feature_cols is None:
            feature_cols = FEATURE_COLS + [SENTIMENT_COL]

        data = self.df[feature_cols].copy()
        scaled = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(WINDOW_SIZE, len(scaled) - prediction_days + 1):
            X.append(scaled[i-WINDOW_SIZE:i])
            # Get next `prediction_days` close prices
            y_values = scaled[i:i+prediction_days, 0]
            y.append(y_values)

        X = np.array(X)
        y = np.array(y)

        split_idx = int(len(X) * TRAIN_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Further split training for validation
        val_split = int(len(X_train) * 0.9)
        X_train_final, X_val = X_train[:val_split], X_train[val_split:]
        y_train_final, y_val = y_train[:val_split], y_train[val_split:]

        return (X_train_final, y_train_final), (X_val, y_val), (X_test, y_test), self.scaler

    def prepare_last_sequence(self, feature_cols=None):
        """Prepare the last sequence for prediction."""
        if feature_cols is None:
            feature_cols = FEATURE_COLS + [SENTIMENT_COL]
        
        data = self.df[feature_cols].copy()
        scaled = self.scaler.transform(data)
        last_sequence = scaled[-WINDOW_SIZE:]
        return np.reshape(last_sequence, (1, WINDOW_SIZE, len(feature_cols)))

    def inverse_transform_prices(self, scaled_predictions):
        """Convert scaled price predictions back to actual prices.
        
        Args:
            scaled_predictions: scaled price values (can be 1D or multiple days)
        """
        num_features = len(self.scaler.feature_names_in_)
        
        if len(scaled_predictions.shape) == 1:
            scaled_predictions = scaled_predictions.reshape(-1, 1)
        
        num_days = scaled_predictions.shape[0]
        dummy = np.zeros((num_days, num_features))
        dummy[:, 0] = scaled_predictions.flatten()
        
        return self.scaler.inverse_transform(dummy)[:, 0]