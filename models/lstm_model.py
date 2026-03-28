# models/lstm_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from config import LSTM_UNITS, DROPOUT_RATE, EPOCHS, BATCH_SIZE
from utils.logger import setup_logger

logger = setup_logger(__name__)

class LSTMModel:
    """LSTM model for time series prediction."""

    def __init__(self, input_shape, name='lstm'):
        self.input_shape = input_shape
        self.name = name
        self.model = None
        self.history = None

    def build(self, output_steps=1):
        """Build LSTM model.
        
        Args:
            output_steps: number of prediction steps (1 for next day, 5 for next 5 days)
        """
        model = Sequential([
            LSTM(LSTM_UNITS[0], return_sequences=True, input_shape=self.input_shape),
            Dropout(DROPOUT_RATE),
            LSTM(LSTM_UNITS[1], return_sequences=False),
            Dropout(DROPOUT_RATE),
            Dense(output_steps)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        self.output_steps = output_steps
        logger.info(f"LSTM model built with {output_steps} output steps.")
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
        if self.model is None:
            self.build()
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        logger.info("LSTM training completed.")
        return self.history

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def save(self, path):
        self.model.save(path)
        logger.info(f"LSTM model saved to {path}")

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        logger.info(f"LSTM model loaded from {path}")