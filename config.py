# config.py
import os
from datetime import datetime, timedelta

# Data
TICKER = 'INFY.NS'
START_DATE = datetime.now() - timedelta(days=5*365)
END_DATE = datetime.now()
WINDOW_SIZE = 60
FEATURE_COLS = ['Close', 'GARCH_Vol', 'RSI', 'MACD', 'Volume_Change', 'Close_Open_Ratio']
SENTIMENT_COL = 'Sentiment'

# Model
LSTM_UNITS = [64, 32]
GRU_UNITS = [64, 32]
TRANSFORMER_DIM = 64
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
DROPOUT_RATE = 0.2
EPOCHS = 30
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8

# Paths
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# News sentiment
NEWS_DAYS_BACK = 7