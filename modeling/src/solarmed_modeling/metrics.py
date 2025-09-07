"""
    Module to calculate metrics for evaluating the performance of the model to experimental data
"""

from typing import Literal, get_args, Optional
import numpy as np
import pandas as pd

supported_metrics_type = Literal['ITAE', 'ISE', 'IAE', 'RMSE', 'MAE', 'MSE', 'R2', 'NRMSE', 'MAPE']

def calculate_itae(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the Integral Time-weighted Absolute Error (ITAE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.
        time (array-like): Time values.

    Returns:
        float: ITAE value.
    """
    time = np.arange(0, len(predicted)).ravel()

    error = np.abs(predicted - actual)
    itae = np.nansum(error * time[:, np.newaxis])

    return itae


def calculate_ise(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the Integral Square Error (ISE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: ISE value.
    """
    error = predicted - actual
    ise = np.nansum(np.square(error))

    return ise


def calculate_iae(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the Integral Absolute Error (IAE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: IAE value.
    """

    error = np.abs(predicted - actual)
    iae = np.nansum(error)

    return iae

def calculate_rmse(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: RMSE value.
    """

    error = predicted - actual
    rmse = np.sqrt(np.nanmean(np.square(error)))

    return rmse


def calculate_mae(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the Mean Absolute Error (MAE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: MAE value.
    """

    error = np.abs(predicted - actual)
    mae = np.nanmean(error)

    return mae


def calculate_mse(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the Mean Square Error (MSE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: MSE value.
    """

    error = predicted - actual
    mse = np.nanmean(np.square(error))

    return mse

def calculate_r2(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the R2 score.

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: R2 score.
    """

    mean_actual = np.nanmean(actual)
    ss_tot = np.nansum(np.square(actual - mean_actual))
    ss_res = np.nansum(np.square(actual - predicted))

    r2 = 1 - ss_res / ss_tot

    return r2

def calculate_nrmse(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the Normalized Root Mean Square Error (NRMSE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: NRMSE value.
    """

    error = predicted - actual
    nrmse = np.sqrt(np.nanmean(np.square(error))) / np.nanstd(actual)

    return nrmse

def calculate_mape(predicted: np.ndarray[float], actual: np.ndarray[float]) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.

    Returns:
        float: MAPE value.
    """

    error = np.abs((predicted - actual) / actual)
    mape = np.nanmean(error) * 100  # Convert to percentage

    return mape


def calculate_metrics(
    predicted: np.ndarray[float] | pd.DataFrame, 
    actual: np.ndarray[float] | pd.DataFrame, 
    metrics: Optional[list[supported_metrics_type]] = None
) -> dict[str, float]:
    """
    Calculate the metrics for evaluating the performance of the model to experimental data.

    Args:
        predicted (array-like): Predicted values.
        actual (array-like): Actual values.
        metrics (list): List of metrics to calculate. Leave unspecified to calculate all metrics.

    Returns:
        dict: Dictionary with the calculated metrics.
    """

    if metrics is not None:
        assert all(metric in get_args(supported_metrics_type) for metric in metrics), \
            f"Metrics not supported. Supported metrics: {get_args(supported_metrics_type)}"
    else:
        metrics = get_args(supported_metrics_type)
        
    if isinstance(predicted, pd.DataFrame):
        predicted = predicted.to_numpy()
    if isinstance(actual, pd.DataFrame):
        actual = actual.to_numpy()

    calculated_metrics = {}

    if 'ITAE' in metrics:
        calculated_metrics['ITAE'] = calculate_itae(predicted, actual)
    if 'ISE' in metrics:
        calculated_metrics['ISE'] = calculate_ise(predicted, actual)
    if 'IAE' in metrics:
        calculated_metrics['IAE'] = calculate_iae(predicted, actual)
    if 'RMSE' in metrics:
        calculated_metrics['RMSE'] = calculate_rmse(predicted, actual)
    if 'MAE' in metrics:
        calculated_metrics['MAE'] = calculate_mae(predicted, actual)
    if 'MSE' in metrics:
        calculated_metrics['MSE'] = calculate_mse(predicted, actual)
    if 'R2' in metrics:
        calculated_metrics['R2'] = calculate_r2(predicted, actual)
    if 'NRMSE' in metrics:
        calculated_metrics['NRMSE'] = calculate_nrmse(predicted, actual)
    if 'MAPE' in metrics:
        calculated_metrics['MAPE'] = calculate_mape(predicted, actual)

    return calculated_metrics