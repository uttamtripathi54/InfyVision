# main.py
import argparse
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from pathlib import Path
from config import *
from data.data_fetcher import DataFetcher
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from models.transformer_model import TransformerModel
from models.ensemble import Ensemble
from utils.logger import setup_logger
from utils.helpers import calculate_metrics

logger = setup_logger(__name__, log_file='infy_vision.log')

def train_model(model_type, fetcher, prediction_days=5, save=True):
    """Train a single model for multi-day prediction."""
    logger.info(f"Training {model_type} model for {prediction_days}-day prediction...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = fetcher.prepare_sequences(prediction_days=prediction_days)
    input_shape = (X_train.shape[1], X_train.shape[2])

    if model_type == 'lstm':
        model = LSTMModel(input_shape)
    elif model_type == 'gru':
        model = GRUModel(input_shape)
    elif model_type == 'transformer':
        model = TransformerModel(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.build(output_steps=prediction_days)
    model.train(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    if y_pred.shape[-1] == 1:
        y_pred = y_pred.reshape(-1)
    if y_test.ndim > 1 and y_test.shape[-1] == prediction_days:
        y_pred_actual = fetcher.inverse_transform_prices(y_pred[:, 0] if y_pred.ndim > 1 else y_pred)
        y_test_actual = fetcher.inverse_transform_prices(y_test[:, 0])
    else:
        y_pred_actual = fetcher.inverse_transform_prices(y_pred.flatten())
        y_test_actual = fetcher.inverse_transform_prices(y_test.flatten())

    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    rmse = np.sqrt(np.mean((y_test_actual - y_pred_actual) ** 2))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    dir_acc = np.mean((np.diff(y_test_actual) > 0) == (np.diff(y_pred_actual) > 0)) * 100
    
    logger.info(f"{model_type.upper()} results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, Dir Acc: {dir_acc:.2f}%")

    if save:
        model_path = os.path.join(MODEL_DIR, f'infy_{model_type}_{prediction_days}d.keras')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

    return model, (mae, rmse, mape, dir_acc)

def predict_next_5_days(model_type, fetcher, prediction_days=5):
    """Predict next 5 trading days price using a trained model."""
    try:
        input_shape = (WINDOW_SIZE, len(FEATURE_COLS) + 1)
        
        if model_type == 'lstm':
            model = LSTMModel(input_shape)
        elif model_type == 'gru':
            model = GRUModel(input_shape)
        elif model_type == 'transformer':
            model = TransformerModel(input_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_path = os.path.join(MODEL_DIR, f'infy_{model_type}_{prediction_days}d.keras')
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None
        
        model.load(model_path)
        
        # Prepare sequences to fit scaler properly
        _ = fetcher.prepare_sequences(prediction_days=prediction_days)
        
        # Prepare last sequence
        X_pred = fetcher.prepare_last_sequence()
        pred_scaled = model.predict(X_pred)[0]
        
        # Handle output shape
        if isinstance(pred_scaled, np.ndarray):
            pred_scaled = pred_scaled.flatten()
        
        pred_prices = fetcher.inverse_transform_prices(pred_scaled)
        current_price = fetcher.df['Close'].iloc[-1]
        
        # Generate dates for next 5 trading days
        next_dates = []
        current_date = fetcher.df.index[-1]
        for i in range(1, prediction_days + 1):
            next_date = current_date + pd.Timedelta(days=i)
            while next_date.weekday() >= 5:  # Skip weekends
                next_date += pd.Timedelta(days=1)
            next_dates.append(next_date)
        
        logger.info(f"\n{model_type.upper()} Next {prediction_days} Days Prediction:")
        logger.info(f"Current Price: ₹{current_price:.2f}")
        for i, (date, price) in enumerate(zip(next_dates, pred_prices[:prediction_days]), 1):
            logger.info(f"Day {i} ({date.date()}): ₹{price:.2f} (Change: {((price-current_price)/current_price)*100:+.2f}%)")
        
        return pred_prices[:prediction_days]
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(fetcher, prediction_days=5):
    """Train and evaluate all models, print comparison."""
    results = {}
    for model_type in ['lstm', 'gru', 'transformer']:
        try:
            _, metrics = train_model(model_type, fetcher, prediction_days, save=True)
            results[model_type] = metrics
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            results[model_type] = (float('inf'), float('inf'), float('inf'), 0)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON - 5-DAY PREDICTION")
    print("="*70)
    print(f"{'Model':<12} {'MAE (₹)':<12} {'RMSE (₹)':<12} {'MAPE (%)':<12} {'Dir Acc (%)':<12}")
    print("-"*70)
    for name, (mae, rmse, mape, acc) in results.items():
        print(f"{name:<12} {mae:<12.2f} {rmse:<12.2f} {mape:<12.2f} {acc:<12.2f}")
    print("="*70 + "\n")
    return results

def ensemble_predict(fetcher, prediction_days=5):
    """Load all models and predict with averaging for 5 days."""
    try:
        input_shape = (WINDOW_SIZE, len(FEATURE_COLS)+1)
        models = []
        
        for model_type in ['lstm', 'gru', 'transformer']:
            try:
                if model_type == 'lstm':
                    model = LSTMModel(input_shape)
                elif model_type == 'gru':
                    model = GRUModel(input_shape)
                elif model_type == 'transformer':
                    model = TransformerModel(input_shape)
                else:
                    continue
                
                model_path = os.path.join(MODEL_DIR, f'infy_{model_type}_{prediction_days}d.keras')
                if os.path.exists(model_path):
                    model.load(model_path)
                    models.append(model)
                else:
                    logger.warning(f"Model {model_path} not found, skipping...")
            except Exception as e:
                logger.warning(f"Could not load {model_type}: {e}")
        
        if not models:
            logger.error("No models available for ensemble prediction")
            return None
        
        # Prepare sequences to fit scaler
        _ = fetcher.prepare_sequences(prediction_days=prediction_days)
        
        ensemble = Ensemble(models)
        
        # Prepare last sequence
        X_pred = fetcher.prepare_last_sequence()
        pred_scaled = ensemble.predict(X_pred)[0]
        
        if isinstance(pred_scaled, np.ndarray):
            pred_scaled = pred_scaled.flatten()
        
        pred_prices = fetcher.inverse_transform_prices(pred_scaled)
        current_price = fetcher.df['Close'].iloc[-1]
        
        # Generate dates for next 5 trading days
        next_dates = []
        current_date = fetcher.df.index[-1]
        for i in range(1, prediction_days + 1):
            next_date = current_date + pd.Timedelta(days=i)
            while next_date.weekday() >= 5:  # Skip weekends
                next_date += pd.Timedelta(days=1)
            next_dates.append(next_date)
        
        logger.info(f"\nENSEMBLE Next {prediction_days} Days Prediction (Average of {len(models)} models):")
        logger.info(f"Current Price: ₹{current_price:.2f}")
        for i, (date, price) in enumerate(zip(next_dates, pred_prices[:prediction_days]), 1):
            logger.info(f"Day {i} ({date.date()}): ₹{price:.2f} (Change: {((price-current_price)/current_price)*100:+.2f}%)")
        
        return pred_prices[:prediction_days]
    
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction with LSTM, GRU, Transformer - 5-Day Forecast')
    parser.add_argument('--train', choices=['lstm', 'gru', 'transformer', 'all'], default='all',
                        help='Model(s) to train')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    parser.add_argument('--predict', choices=['lstm', 'gru', 'transformer', 'ensemble'],
                        help='Predict next 5 days using specified model')
    parser.add_argument('--fetch-only', action='store_true', help='Only fetch data and exit')
    parser.add_argument('--prediction-days', type=int, default=5, help='Number of days to predict (default: 5)')
    args = parser.parse_args()

    # Initialize data fetcher
    fetcher = DataFetcher()
    fetcher.fetch_data()
    fetcher.compute_indicators()
    fetcher.add_sentiment(days_back=NEWS_DAYS_BACK)

    if args.fetch_only:
        logger.info("Data fetching complete.")
        print(f"Data prepared: {len(fetcher.df)} trading days ready for prediction")
        return

    if args.compare:
        compare_models(fetcher, args.prediction_days)
        return

    if args.predict:
        logger.info(f"Running prediction for {args.prediction_days} days ahead...")
        if args.predict == 'ensemble':
            ensemble_predict(fetcher, args.prediction_days)
        else:
            predict_next_5_days(args.predict, fetcher, args.prediction_days)
        return

    if args.train == 'all':
        logger.info(f"Training all models for {args.prediction_days}-day prediction...")
        for model_type in ['lstm', 'gru', 'transformer']:
            try:
                train_model(model_type, fetcher, args.prediction_days)
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                import traceback
                traceback.print_exc()
    else:
        train_model(args.train, fetcher, args.prediction_days)

    logger.info("All tasks completed.")
    print("\n✓ Training complete! Use --predict flag to make predictions.")

if __name__ == '__main__':
    main()