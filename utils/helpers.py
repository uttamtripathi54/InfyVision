# utils/helpers.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """Compute MAE, RMSE, MAPE, direction accuracy."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    direction_actual = np.diff(y_true) > 0
    direction_pred = (y_pred[1:] > y_true[:-1])
    direction_acc = np.mean(direction_actual == direction_pred) * 100
    return mae, rmse, mape, direction_acc