import numpy as np
import pandas as pd
import plotly.graph_objs as go
from loguru import logger

from solarmed_modeling.curve_fitting.curves import *


def evaluate_fit(x, fit_type, params):

    if fit_type == 'linear_fit':
        return linear_fit(x, *params)
    elif fit_type == 'spline_fit':
        return spline_fit(x, *params)
    elif fit_type == 'exponential_curve':
        return exponential_curve(x, *params)
    elif fit_type == 'quadratic_curve':
        return quadratic_curve(x, *params)
    elif fit_type == 'logarithmic_curve':
        return logarithmic_curve(x, *params)
    else:
        raise ValueError(f'Unknown fit type: {fit_type}')


def ensure_monotony(data: np.array, gap=0.01) -> np.array:
    """
    This function ensures that the given data is monotonically increasing. If a value is found to be less than its
    predecessor, it is replaced with the value of its predecessor plus a small gap.

    Parameters:
    data (np.array): The input data array that needs to be checked for monotonicity.
    gap (float, optional): The minimum difference between consecutive values in the output array. Default is 0.01.

    Returns:
    np.array: The input data array adjusted to ensure monotonicity.
    """

    # Iterate over the data array starting from the second element
    for i in range(1, len(data)):
        # If the current element is less than its predecessor
        if data[i] < data[i - 1]:
            # Replace it with the value of its predecessor plus the gap
            data[i] = data[i - 1] + gap

    return data


def fit_data(df: pd.DataFrame, x_var: str, y_var: str, degree: int = 2, x_var_unit: str = None,
             y_var_unit: str = None) -> tuple[np.polynomial.Polynomial, go.Figure]:
    """
    This is copied from [juan11iguel/phd-utils](https://github.com/juan11iguel/phd-utils/blob/main/src/phd_utils/curve_fitting/__init__.py)
    Fit data to a polynomial and visualize the fit

    Args:
        df (pd.DataFrame): Dataframe with the data
        x_var (str): Independent variable id in the dataframe
        y_var (str): Dependent variable id in the dataframe
        degree (int, optional): Order of the polynomial. Defaults to 2.
        x_var_unit (str, optional): Independent variable units for the plot. Defaults to None.
        y_var_unit (str, optional): Dependent variable units for the plot. Defaults to None.

    Returns:
        tuple[np.polynomial.Polynomial, go.Figure]: Polynomial fit and plotly figure
    """

    fit: np.polynomial.Polynomial = np.polynomial.Polynomial.fit(df[x_var], df[y_var], degree)

    logger.debug(f"Fit: {fit}")
    if len(df) < 20:
        logger.debug(f"x data: {df[x_var].to_numpy()} \n y data: {df[y_var].to_numpy()}")

    # Visualize the fit
    x = np.linspace(df[x_var].min(), df[x_var].max(), 100)
    y = fit(x)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df[x_var], y=df[y_var], mode='markers', name='Data', marker=dict(size=10, color='black', ), )
    )

    fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines', name='Fit', line=dict(color='black', width=2, ), )
    )

    fig.add_trace(
        go.Scatter(x=x, y=np.polynomial.Polynomial(fit.convert().coef)(x), mode='lines', name='Fit from coeff',
                   line=dict(color='blue', width=1, ), )
    )

    r2 = np.corrcoef(df[y_var], fit(df[x_var]))[0, 1] ** 2

    fig.update_layout(
        title=f"{x_var} vs {y_var}",
        xaxis_title=f'{x_var} ({x_var_unit})',
        yaxis_title=f'{y_var} ({y_var_unit})',
        showlegend=True,
        template='ggplot2',
        annotations=[
            dict(
                x=0.05,
                y=0.9,
                xref='paper',
                yref='paper',
                showarrow=False,
                text=f"R<sup>2</sup>: {r2:.2f}<br>y = {str(fit.convert())}</br>",
                font=dict(size=12, color='black'),
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='black',
                borderwidth=1,
                borderpad=4,
                opacity=0.8
            )
        ],
        width=800,
        height=600
    )

    np.set_printoptions(precision=8)
    print(f"[{y_var}=f({x_var})] Coefficients to copy: {fit.convert().coef}")
    np.set_printoptions(precision=2)

    return fit, fig
