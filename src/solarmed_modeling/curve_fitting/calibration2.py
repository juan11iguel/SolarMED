import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

def fit_data(df: pd.DataFrame, x_var: str, y_var: str, degree: int = 2, x_var_unit: str = None, y_var_unit: str = None) -> tuple[np.polynomial.Polynomial, go.Figure]:
    """
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
    logger.debug(f"x data: {df[x_var].to_numpy()} \n y data: {df[y_var].to_numpy()}")

    # Visualize the fit
    x = np.linspace(df[x_var].min(), df[x_var].max(), 100)
    y = fit(x)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df[x_var], y=df[y_var], mode='markers', name='Data', marker=dict( size=10, color='black', ),)
    )

    fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines', name='Fit', line=dict(color='black', width=2,),)
    )

    fig.add_trace(
        go.Scatter(x=x, y=np.polynomial.Polynomial(fit.convert().coef)(x), mode='lines', name='Fit from coeff', line=dict(color='blue',width=1,),)
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
                font=dict(size=12,color='black'),
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

    return fit, fig