"""Evaluation metrics for volume forecasting.

This module provides functions to calculate common forecast evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error.

    MAE = mean(|y_true - y_pred|)

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        Mean Absolute Error as a float.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error.

    RMSE = sqrt(mean((y_true - y_pred)^2))

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        Root Mean Square Error as a float.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Args:
        y_true: Array of actual values. Should not contain zeros.
        y_pred: Array of predicted values.

    Returns:
        Mean Absolute Percentage Error as a percentage (0-100+).
    """
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error.

    sMAPE = mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|)) * 100

    This metric is symmetric: swapping y_true and y_pred gives the same result.
    It is bounded between 0% and 200%.

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        Symmetric Mean Absolute Percentage Error as a percentage (0-200).
    """
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(numerator / denominator) * 100)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate all evaluation metrics.

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        Dictionary with keys 'mae', 'rmse', 'mape', 'smape' and their values.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
