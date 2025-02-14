import plotly.graph_objects as go
import pandas as pd
from solarmed_optimization import ProblemParameters
from solarmed_optimization.utils.operation_plan import generate_update_datetimes

def plot_op_mode_change_candidates(I_series: pd.Series, pp: ProblemParameters, updates_per_action: int = 3) -> go.Figure:
    """
    Visualize the irradiance and the candidates using plotly
    
    Args:
        I_series (pd.Series): Irradiance series
        pp (ProblemParameters): Problem parameters
        updates_per_action (int, optional): Number of updates per action. Defaults to 3.
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()

    index_start = I_series.index[0]
    while index_start < I_series.index[-1]:
        index_end = index_start + pd.Timedelta(days=1)

        I_exp = I_series.loc[index_start:index_end]
        I_opt = (
            I_exp.resample(f"{pp.sample_time_opt}s", origin="start")
            .interpolate()
            .resample(f"{10}min", origin="start")
            .interpolate()
        )
        startup_candidates = generate_update_datetimes(I_opt, updates_per_action, "startup")
        shutdown_candidates = generate_update_datetimes(I_opt, updates_per_action, "shutdown")
        showlegend = True if index_start == I_series.index[0] else False

        fig.add_trace(
            go.Scatter(
                x=I_exp.index,
                y=I_exp.values,
                mode="lines",
                name="Experimental irradiance",
                showlegend=showlegend,
                line=dict(width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=I_opt.index,
                y=I_opt.values,
                mode="lines",
                name="Optimization irradiance",
                showlegend=showlegend,
                line=dict(width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=startup_candidates,
                y=[I_opt.loc[time] for time in startup_candidates],
                mode="markers",
                name="Startup candidates",
                marker_color="green",
                showlegend=showlegend,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=shutdown_candidates,
                y=[I_opt.loc[time] for time in shutdown_candidates],
                mode="markers",
                name="Shutdown candidates",
                marker_color="red",
                showlegend=showlegend,
            )
        )

        index_start = index_end

    fig.update_layout(
        title="Irradiance and operation start/stop candidates",
        xaxis_title="Time",
        yaxis_title="Irradiance (W/m²)",
    )
    
    return fig