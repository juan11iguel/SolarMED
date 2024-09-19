"""
    Module to calculate metrics for evaluating the performance of the model to experimental data
"""

import numpy as np
from typing import Literal, get_args

supported_metrics_type = Literal['ITAE', 'ISE', 'IAE']

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
    time = np.arange(0, len(predicted))

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


def calculate_metrics(predicted: np.ndarray[float], actual: np.ndarray[float], metrics: list[supported_metrics_type] = None) -> dict[str, float]:
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
        assert all(metric in get_args(supported_metrics_type) for metric in metrics), f"Metrics not supported. Supported metrics: {get_args(supported_metrics_type)}"
    else:
        metrics = get_args(supported_metrics_type)

    calculated_metrics = {}

    if 'ITAE' in metrics:
        calculated_metrics['ITAE'] = calculate_itae(predicted, actual)
    if 'ISE' in metrics:
        calculated_metrics['ISE'] = calculate_ise(predicted, actual)
    if 'IAE' in metrics:
        calculated_metrics['IAE'] = calculate_iae(predicted, actual)

    return calculated_metrics