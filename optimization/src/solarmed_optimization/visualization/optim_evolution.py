from pathlib import Path
from typing import Literal
from collections.abc import Iterable
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors
from plotly.colors import hex_to_rgb

from phd_visualizations import save_figure

from solarmed_optimization.utils import flatten_list
from solarmed_optimization.problems.minlp import OptimToFsmsVarIdsMapping
from solarmed_optimization.problems.minlp import BaseProblem

# Constants
plt_colors = plotly.colors.qualitative.Plotly
gray_colors = plotly.colors.sequential.Greys[2:][::-1]
green_colors = plotly.colors.sequential.Greens[2:][::-1]

def plot_dec_vars_evolution(problem: BaseProblem, df_hors_: list[pd.DataFrame] | list[list[pd.DataFrame]], df_mod: pd.DataFrame = None, full_xaxis_range: bool = True, episode_samples: int = None) -> go.Figure:
    # df_aux: pd.DataFrame | None = None,
    # TODO: Use grid_specs to define the layout (spacing between plots)
    # TODO: Add support for df_hors being a list of lists of dataframes. 
    # If it's just a list, make it into a list of lists with a single element
    
    def get_bounds_values(df: pd.DataFrame, var_id: str) -> tuple[np.ndarray, np.ndarray]:
        # var_id = OptimVarIdstoModelVarIdsMapping(var).name
        
        _var_id_ub = f"upper_bounds_{var_id}"
        _var_id_lb = f"lower_bounds_{var_id}"
        if df.iloc[0][var_id].dtype in [bool, int]:
            yval_upper = df[_var_id_ub].astype(int) + 0.25
            yval_lower = df[_var_id_lb].astype(int) - 0.25
        else:
            yval_upper = df[_var_id_ub]
            yval_lower = df[_var_id_lb]
            
        return yval_lower, yval_upper
    
    # if full_xaxis_range:
    #     assert df_aux is not None or episode_samples is not None, "If full_xaxis_range is True, df_aux or episode_samples should be provided"
    # end_idx = episode_samples if episode_samples is not None else len(df_aux)
    if full_xaxis_range:
        assert episode_samples is not None, "If full_xaxis_range is True, df_aux or episode_samples should be provided"
    end_idx = episode_samples #if episode_samples is not None else len(df_aux)  
    
    # Make sure df_hors is a list of lists
    for idx, item in enumerate(df_hors_):
        if not isinstance(item, list):
            df_hors_[idx] = [item]
    
    sample_time_mod: int = problem.sample_time_mod
    optim_window_size: int = problem.n_evals_mod_in_hor_window
    optim_step_size: int = problem.n_evals_mod_in_opt_step
            
    if len(df_hors_) > 0:
        xtick_vals: np.ndarray[int] = pd.RangeIndex(start=0, step=1, stop=df_hors_[-1][-1].index[-1] + 1).to_numpy()
            
        # upper_bounds_plt = [forward_fill_resample(upper_bound, target_size=len(df_hors_[-1])) for upper_bound in upper_bounds] 
        # lower_bounds_plt = [forward_fill_resample(lower_bound, target_size=len(df_hors_[-1])) for lower_bound in lower_bounds]
    else:
        xtick_vals: np.ndarray[int] = np.arange(start=0, stop=optim_window_size+1, step=1)

    if full_xaxis_range:
        xtick_vals: np.ndarray[int] = np.arange(start=0, stop=end_idx, step=1)

    time_deltas = pd.to_timedelta(xtick_vals * sample_time_mod, unit='s')
    xtick_vals: list[int] = xtick_vals.tolist()
    xtick_labels = [
        f'{x.components.hours:02d}:{x.components.minutes:02d}' if not pd.isnull(x) else ''
        for x in time_deltas
    ]
    xaxes_label_alias = {val: label for val, label in zip(xtick_vals, xtick_labels) for val, label in zip(xtick_vals, xtick_labels)}


    # Create subplots with shared x-axis
    subplot_titles = [None] * problem.n_dec_vars
    subplot_titles[0] = "Decision variables"
    subplot_titles.append("Fitness evolution")
    fig = make_subplots(rows=problem.n_dec_vars +1, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.01, subplot_titles=subplot_titles)

    # var_ids = [getattr(OptimToFsmsVarIdsMapping(), dec_var_id) for dec_var_id in problem.dec_var_int_ids]
    # var_ids = flatten_list(var_ids)
    # var_ids.extend(problem.dec_var_real_ids)

    # Iterate over decision variables to create individual plots
    for i, var_id in enumerate(problem.dec_var_ids):
        var = f"dec_var_{var_id}"
        # Temporary
        # if df_mod is not None:
        # if df_aux is not None:
        #     fig.add_trace(
        #         go.Scatter(
        #             x=np.arange(start=0, stop=len(df_aux), step=1),
        #             y=df_aux[var_id].astype(int) if df_aux.iloc[0][var_id].dtype == bool else df_aux[var_id],
        #             mode='markers+lines',
        #             name=f"experimental {var_id}",
        #             line=dict(color=plt_colors[i], width=1),
        #             marker=dict(color=plt_colors[i], size=4, symbol='x'),
        #         ),
        #         row=i+1, col=1
        #     )
        
        if len(df_hors_) > 0:
            marker_symbols: np.ndarray[str] = np.full((len(df_hors_[-1]),), 'circle', dtype='<U15')  # <U15 for max string length
            # marker_sizes: np.ndarray[int] = np.full((len(df_hors_[-1]), ), 5)
        
        # Horizon trace(s)
        for hor_idx, df_hor in enumerate(df_hors_):
            # if hor_idx == len(df_hors)-1:
            #     update_idxs = np.insert(np.where(np.abs(np.diff(df_hor[var])) > 0)[0] + 1, 0, 0)
            #     marker_symbols[update_idxs] = 'circle-open-dot'
            #     marker_sizes[update_idxs] = 10
                # print(f"{marker_symbols=}")
                # print(f"{marker_sizes=}")
            indexer = len(df_hors_)-hor_idx-1

            if indexer >= len(gray_colors):
                color_idx = 0
                marker_size = 3
            else:
                color_idx = indexer
            opacity = np.max([ 0.1, 1-0.1*(indexer) ])
            color = f"{green_colors[color_idx]}".replace(")", f",{opacity})").replace("rgb", "rgba")
            width = np.max([0.1, 1-0.2*( indexer )])
            marker_size = np.max([3, 5-2*( indexer )])
            best_idx: int = np.argmax([df_h["net_profit"].sum() for df_h in df_hor])
            # if i == 0:
            #     print(f"Prediction step {hor_idx}: {color_idx=}, {width=}, {opacity=}")
        
            # Display only the most fit individual for past evaluations, and the
            # whole population for the last one
            for h_idx, df_h in enumerate([df_hor[best_idx]] if hor_idx != len(df_hors_)-1 else df_hor):
                color_ = color
                zorder=1
                width_=width
                marker_size_=marker_size
                if hor_idx == len(df_hors_)-1:
                    if h_idx==best_idx:
                        color_ = "seagreen"
                        zorder=2
                        width_=1.5*width
                    else:
                        marker_size_=0.1
                        width_=.5*width
                        color_ = f"{gray_colors[color_idx]}".replace(")", f",{.5})").replace("rgb", "rgba")
                        
                # if i==0:
                #     print(f"{len(df_hors_)-1=}, {hor_idx=} | {h_idx=} {best_idx=} | {color_=}")
                
                fig.add_trace(
                    go.Scatter(
                        x=df_h.index,
                        y=df_h[var].astype(int) if df_h.iloc[0][var].dtype == bool else df_h[var],
                        mode='markers+lines',
                        name=f"Predicted {var_id} at step {hor_idx}",
                        # fill="tozeroy",
                        line=dict(
                            color=color_, 
                            width=width_,
                        ),
                        # marker=dict(color=gray_colors[len(df_hors_)-hor_idx]),
                        marker_symbol=marker_symbols.tolist(),
                        marker_size=marker_size_, #marker_sizes.tolist(),
                        marker_color=color_,
                        zorder=zorder
                    ),
                    row=i+1, col=1
                )
            ## Fitness
            if i == 0: # Just once
                fig.add_trace(
                    go.Scatter(
                        x=df_hor[best_idx].index,
                        y=df_hor[best_idx]["net_profit"],
                        mode='markers+lines',
                        name=f"Predicted fitness at step {hor_idx}",
                        # fill="tozeroy" if hor_idx == len(df_hors_)-1 else None,
                        # fillcolor=f"{color[:-4]}0.3)",
                        line=dict(
                            color=color, 
                            width=width,
                        ),
                        # marker=dict(color=gray_colors[len(df_hors_)-hor_idx]),
                        marker_symbol="triangle-up",
                        marker_size=marker_size, #marker_sizes.tolist(),
                        marker_color=color,
                    ),
                    row=problem.n_dec_vars+1, col=1
                ) 
        
        ## Add upper and lower bounds as dashed lines for the best last evaluation
        if len(df_hors_) > 0: #and hor_idx == len(df_hors_):
            lb, ub = get_bounds_values(df_hor[best_idx], var_id)
            fig.add_trace(
                go.Scatter(
                    x=df_hor[best_idx].index,
                    y=lb,
                    mode='lines',
                    line=dict(#dash='dash', 
                            #color=gray_colors[len(df_hors)-hor_idx], 
                            width=0, color=gray_colors[0]),
                    name=f'upper bound {var_id}',
                    # fill="tonexty"
                    # hoverinfo='skip',
                ),
                row=i+1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_hor[best_idx].index,
                    y=ub,
                    mode='lines',
                    line=dict(width=0, color=gray_colors[0]), #dash='dashdot', color=gray_colors[len(df_hors)-hor_idx], width=0),
                    name=f'lower bound {var_id}',
                    # hoverinfo='skip'
                    fill="tonexty",
                    fillcolor=f"{gray_colors[0]}".replace(")", ",0.1)").replace("rgb", "rgba"),
                ),
                row=i+1, col=1
            )
        
            
        # Experimental/simulated data trace
        if df_mod is None:
            # Add an empty trace so the subplots are created
            fig.add_trace(go.Scatter(name=var_id), row=i+1, col=1)
        else:
            fig.add_trace(
                go.Scatter(
                    x=df_mod.index,
                    y=df_mod[var].astype(int) if df_mod.iloc[0][var].dtype == bool else df_mod[var_id],
                    mode='markers+lines',
                    name=var_id,
                    # fill="tozeroy",
                    line=dict(color=plt_colors[i], width=5),
                    marker=dict(color=plt_colors[i], size=8),
                    zorder=3,
                ),
                row=i+1, col=1
            )
            
            ## Add upper and lower bounds as dashed lines
            lb, ub = get_bounds_values(df_mod, var_id)
            fig.add_trace(
                go.Scatter(
                    x=df_mod.index,
                    y=lb,
                    mode='lines',
                    line=dict(width=0, color=plt_colors[i]), #dash='dash'),
                    name=f'upper bound {var_id}',
                    # fill="tonexty"
                    # hoverinfo='skip',
                ),
                row=i+1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_mod.index,
                    y=ub,
                    mode='lines',
                    line=dict(width=0, color=plt_colors[i]), #dash='dashdot', ),
                    name=f'lower bound {var_id}',
                    # hoverinfo='skip'
                    fill="tonexty",
                    fillcolor=f"rgba{hex_to_rgb(plt_colors[i]) + (0.2,)}"
                ),
                row=i+1, col=1
            )
        
        if len(df_hors_) > 0:
            fig.update_yaxes(title_text=var_id.replace("_", ","), row=i+1, col=1,) #autorange="reversed"
            if  df_hors_[-1][-1].iloc[0][var_id].dtype == bool:
                fig.update_yaxes(
                    tickvals=[0, 1],
                    ticktext=["False", "True"],
                    range=[-0.5, 1.5],
                    row=i+1, col=1,
                )
        
    # Add fitness evolution plot
    var_id: str = "net_profit"
    # Experimental/simulated data trace
    if df_mod is None:
        # Add an empty trace so the subplots are created
        fig.add_trace(go.Scatter(name=var_id), row=problem.n_dec_vars+1, col=1)
    else:
        # TODO: We should fill gaps with nans, otherwise the plot will connect the dots
        indexer = df_mod[var_id] >= 0
        fig.add_trace(
            go.Scatter(
                x=df_mod.index[indexer],
                y=df_mod.loc[indexer, var_id],
                mode='markers+lines',
                name=var_id,
                fill="tozeroy",
                fillcolor=f"rgba{str(hex_to_rgb(plt_colors[7]))[:-1]}, 0.5)",
                marker=dict(color=plt_colors[2], symbol="triangle-up", size=8),
                line_color=plt_colors[2],
            ),
            row=problem.n_dec_vars+1, col=1
        )
        indexer = df_mod[var_id] < 0
        fig.add_trace(
            go.Scatter(
                x=df_mod.index[indexer],
                y=df_mod.loc[indexer, var_id],
                mode='markers+lines',
                name=var_id,
                fill="tozeroy",
                fillcolor=f"rgba{str(hex_to_rgb(plt_colors[1]))[:-1]}, 0.5)",
                marker=dict(color=plt_colors[1], symbol="triangle-down", size=8),
                showlegend=False,
                line_color=plt_colors[1],
            ),
            row=problem.n_dec_vars+1, col=1
        )
            
    # Add optimization horizon window
    start = 0
    end = optim_window_size
    hor_shift = 0
    if len(df_hors_) > 0:
        start = df_hors_[-1][-1].index.to_numpy()[0]
        end = df_hors_[-1][-1].index.to_numpy()[-1]
        if df_mod is not None:
            if df_mod.index.stop - df_hors_[-1][-1].index.start >= optim_step_size:
                hor_shift = optim_step_size
            
    start += hor_shift
    end += hor_shift

    fig.add_shape(
        type="rect",
        x0=start,
        x1=end,
        yref="paper",
        y0=1.02,
        y1=-0.001,
        line=dict(width=2, color="Blue"),
        fillcolor="LightSkyBlue",
        opacity=0.1,
        layer="between",
        name="Optimization window",
        label=dict(text="Optimization window", textposition="top center", font=dict(color="MediumSlateBlue", size=10)),
        showlegend=True
    )

    # Update layout to show x-axis only on the bottom plot
    fig.update_layout(
        height=100 * problem.n_dec_vars,  # Adjust height based on number of subplots
        showlegend=False,
        title=dict(
            text='Optimization variables evolution',
            # x=0.5,
            # xanchor='center',
            font = (dict(size=16, weight='bold'))
        ),
        hoversubplots="axis",
        hovermode='x',
        template="ggplot2"
    )

    # setattr(
    #     fig.layout, 
    #     f"xaxis{problem.n_dec_vars}", 
    #     dict(
    #     #   tickvals=xtick_vals,
    #     #   ticktext=xtick_labels,
    #     #   showticklabels=True,
    #     ),
    # )
    fig.update_xaxes(
        # tickvals=xtick_vals,
        labelalias=xaxes_label_alias,
        showticklabels=True,
        title="Time (hh:mm)",
        # TODO: We can't depend on df_aux for defining the axis range
        range=[0, end_idx] if full_xaxis_range else [0, len(xtick_vals)],
        row=problem.n_dec_vars+1, col=1,
        # tick0=df_mod.index.start if df_mod is not None else 0,
        # dtick=optim_step_size,
    )
    [
        fig.update_xaxes(row=i, col=1,
                         tick0=df_mod.index.start if df_mod is not None else 0,
                         dtick=optim_step_size,)
        for i in range(problem.n_dec_vars+2) # TODO: Should use some parameter
    ]

    # for data in fig.data:
    #     data.xaxis = 'x'
    
    return fig


def generate_animation(output_path: Path, df_hors: list[pd.DataFrame], df_sim: pd.DataFrame, 
                       df_aux: pd.DataFrame, problem: BaseProblem,
                       output_formats: list[Literal["png", "html"]] = ["png"]) -> None:

    assert all(out_format in ["png", "html"] for out_format in output_formats), "Invalid output format"
    
    # output_path = Path("./results/dec_vars_evolution")
    output_path.mkdir(exist_ok=True)
    
    # Delete all files in the output directory
    for file in output_path.iterdir():
        file.unlink()

    for i in range(len(df_hors)):
        fig = plot_dec_vars_evolution(
            df_hors_=[] if i==0 else df_hors[0:i+1], 
            df_mod=None if i==0 else df_sim.iloc[0:df_hors[i].index.start-problem.n_evals_mod_in_opt_step+1],
            df_aux=df_aux,
            problem=problem,
            full_xaxis_range=True,
        )
        save_figure(fig, figure_name=f"step{i:03d}_0_pre_eval_dec_var_evol", 
                    figure_path=output_path, formats=["png"])#, "html"])
        
        fig = plot_dec_vars_evolution(
            df_hors_=df_hors[0:i+1], 
            df_mod=None if i==0 else df_sim.iloc[0:df_hors[i].index.start+1],
            df_aux=df_aux,
            problem=problem,
            full_xaxis_range=True,
        )
        save_figure(fig, figure_name=f"step{i:03d}_1_post_eval_dec_var_evol", 
                    figure_path=output_path, formats=["png"])#, "html"])
        
        
def plot_obj_scape_comp_1d(fitness_history_list: list[np.ndarray[float]], algo_ids: list[str], highlight_best: int = 1, **kwargs) -> go.Figure:
    
    assert len(fitness_history_list) == len(algo_ids), "fitness_history_list and algo_ids should have the same length"
    
    # First create the base plot calling plot_obj_space_1d_no_animation
    fig = plot_obj_space_1d_no_animation(fitness_history_list[0], algo_id=algo_ids[0])
    
    # And then add the other fitness histories
    best_fit_idxs = []
    for algo_id, fitness_history in zip(algo_ids[1:], fitness_history_list[1:]):
        avg_fitness = [np.mean(x) for x in fitness_history]
        generation = np.arange(len(fitness_history))
        
        fig.add_trace(go.Scatter(x=generation, y=avg_fitness, mode="lines", name=algo_id))
        
        # Store the best highlight_best avg_fitness indexes
        best_fit_idxs.extend(np.argsort(avg_fitness)[:highlight_best])
        
    # Increase the line width for the best fitness values
    for idx in best_fit_idxs:
        fig.data[idx].line.width = 2
        
    fig.update_layout(**kwargs)
    
    return fig
        
"""
From here is basically copied from EvoX: https://github.com/EMI-Group/evox/blob/main/src/evox/vis_tools/plot.py#L4
Have to find a better way to import this without having to install the whole package nor copying code
"""
        
def plot_obj_space_1d(fitness_history: list[np.ndarray[float]], animation: bool = True, **kwargs) -> go.Figure:
    if animation:
        return plot_obj_space_1d_animation(fitness_history, **kwargs)
    else:
        return plot_obj_space_1d_no_animation(fitness_history, **kwargs)


def plot_obj_space_1d_no_animation(fitness_history: list[np.ndarray[float]], algo_id: str = None, **kwargs) -> go.Figure:

    avg_fitness = [np.mean(x) for x in fitness_history]
    generation = np.arange(len(fitness_history))

    additional_scatters = []
    if isinstance(fitness_history[0], Iterable):
        min_fitness = [np.min(x) for x in fitness_history]
        max_fitness = [np.max(x) for x in fitness_history]
        median_fitness = [np.median(x) for x in fitness_history]
        
        additional_scatters = [
            go.Scatter(x=generation, y=min_fitness, mode="lines", name="Min"),
            go.Scatter(x=generation, y=max_fitness, mode="lines", name="Max"),
            go.Scatter(x=generation, y=median_fitness, mode="lines", name="Median"),
        ]
        
    # Layout defaults
    kwargs.setdefault("yaxis_title", "Fitness")
    kwargs.setdefault("xaxis_title", "Number of objective function evaluations")
    kwargs.setdefault("title_text", "<b>Fitness evolution</b><br>comparison between different algorithms")
        
    fig = go.Figure(
        [
            *additional_scatters,
            go.Scatter(x=generation, y=avg_fitness, mode="lines", name="Average" if algo_id is None else algo_id),
        ],
        layout=go.Layout(
            showlegend=True,
            # legend={
            #     "x": 1,
            #     "y": 1,
            #     "xanchor": "auto",
            # },
            # margin={"l": 0, "r": 0, "t": 0, "b": 0},
            **kwargs
        ),
    )

    return fig

def plot_obj_space_1d_animation(fitness_history: list[np.ndarray[float]], **kwargs) -> go.Figure:
    """

    Args:
        fitness_history (list[np.ndarray[float]]): List of fitness values for each individual per generation

    Returns:
        go.Figure: Figure object
        
    Example:
    # This is the last population, after evolution
    # pop = isl.get_population()
    # Properties
    # - best_idx
    # - worst_idx
    # - champion_f
    # - champion_x
    log = isl.get_algorithm().extract(type(algorithm)).get_log()

    # We only have information from the best individual per generation
    fitness_history = [l[2] for l in log]
    
    fig = plot_obj_space_1d_animation(fitness_history=fitness_history, title="Fitness evolution")
    fig

    """

    min_fitness = [np.min(x) for x in fitness_history]
    max_fitness = [np.max(x) for x in fitness_history]
    median_fitness = [np.median(x) for x in fitness_history]
    avg_fitness = [np.mean(x) for x in fitness_history]
    generation = np.arange(len(fitness_history))

    frames = []
    steps = []
    for i in range(len(fitness_history)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=generation[: i + 1],
                        y=min_fitness[: i + 1],
                        mode="lines",
                        name="Min",
                        showlegend=True,
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=max_fitness[: i + 1],
                        mode="lines",
                        name="Max",
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=median_fitness[: i + 1],
                        mode="lines",
                        name="Median",
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=avg_fitness[: i + 1],
                        mode="lines",
                        name="Average",
                    ),
                ],
                name=str(i),
            )
        )

        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 200},
                },
            ],
        }
        steps.append(step)

    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},
            "pad": {"b": 1, "t": 10},
            "len": 0.8,
            "x": 0.2,
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]
    lb = min(min_fitness)
    ub = max(max_fitness)
    fit_range = ub - lb
    lb = lb - 0.05 * fit_range
    ub = ub + 0.05 * fit_range
    fig = go.Figure(
        data=frames[-1].data,
        layout=go.Layout(
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            sliders=sliders,
            xaxis={"range": [0, len(fitness_history)], "autorange": False},
            yaxis={"range": [lb, ub], "autorange": False},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "x": 0.2,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 30},
                },
            ],
            **kwargs,
        ),
        frames=frames,
    )

    return fig

def plot_dec_space(population_history, **kwargs,) -> go.Figure:
    """A Built-in plot function for visualizing the population of single-objective algorithm.
    Use plotly internally, so you need to install plotly to use this function.

    If the problem is provided, we will plot the fitness landscape of the problem.
    """

    all_pop = np.concatenate(population_history, axis=0)
    x_lb = np.min(all_pop[:, 0])
    x_ub = np.max(all_pop[:, 0])
    x_range = x_ub - x_lb
    x_lb = x_lb - 0.1 * x_range
    x_ub = x_ub + 0.1 * x_range
    y_lb = np.min(all_pop[:, 1])
    y_ub = np.max(all_pop[:, 1])
    y_range = y_ub - y_lb
    y_lb = y_lb - 0.1 * y_range
    y_ub = y_ub + 0.1 * y_range

    frames = []
    steps = []
    for i, pop in enumerate(population_history):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=pop[:, 0],
                        y=pop[:, 1],
                        mode="markers",
                        marker={"color": "#636EFA"},
                    ),
                ],
                name=str(i),
            )
        )
        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 200},
                },
            ],
        }
        steps.append(step)

    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},
            "pad": {"b": 1, "t": 10},
            "len": 0.8,
            "x": 0.2,
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            sliders=sliders,
            xaxis={"range": [x_lb, x_ub]},
            yaxis={"range": [y_lb, y_ub]},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": 200,
                                        "easing": "linear",
                                    },
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "x": 0.2,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 30},
                },
            ],
            **kwargs,
        ),
        frames=frames,
    )

    return fig