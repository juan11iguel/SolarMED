from pathlib import Path
from collections import Counter
from typing import Sequence, Optional, Literal
from dataclasses import asdict
import numpy as np
import pandas as pd
from dataclasses import dataclass
import hjson
from loguru import logger
import plotly.graph_objects as go

from phd_visualizations import save_figure
from phd_visualizations.test_timeseries import experimental_results_plot
from phd_visualizations.constants import plt_colors, color_palette, symbols
from phd_visualizations.utils import hex_to_rgba_str
from phd_visualizations.optimization import plot_obj_scape_comp_1d
# from solarmed_optimization.visualization.optim_evolution import plot_obj_scape_comp_1d

from solarmed_optimization import ProblemParameters, IntegerDecisionVariables
from solarmed_optimization.problems.nnlp import OperationPlanResults
from solarmed_optimization.utils import condition_result_dataframe
from solarmed_optimization.utils.operation_plan import generate_operation_datetimes, OperationPlanner

plt_colors = plt_colors*10

def compact_repr(lst):
    """ Compact representation (e.g., for repeated tuples) """
    counts = Counter(tuple(lst[i:i+3]) for i in range(0, len(lst), 3))
    return " + ".join([f"{v}×{list(k)}" for k, v in counts.items()])

def plot_op_mode_change_candidates(I_series: pd.Series, pp: ProblemParameters, include_experimental: bool = True) -> go.Figure:
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

    I_opt = I_series.resample(f"{pp.sample_time_opt}s", origin="start").first()
    I_opt_resampled = I_opt.resample(f"{pp.sample_time_mod}s", origin="start").interpolate()

    operation_datetimes = generate_operation_datetimes(I_opt_resampled, pp.operation_actions, pp.irradiance_thresholds)

    if include_experimental:
        fig.add_trace(
            go.Scatter(
                x=I_series.index,
                y=I_series.values,
                mode="lines",
                name="Experimental irradiance",
                showlegend=True,
                line=dict(width=2),
                line_color=color_palette["plotly_orange"],
            )
    )
    fig.add_trace(
        go.Scatter(
            x=I_opt.index,
            y=I_opt.values,
            mode="lines",
            name=f"Optimization irradiance (Downsampled to {pp.sample_time_opt/3600:.1f} hours)",
            showlegend=True,
            line=dict(width=2, color=color_palette["plotly_orange"], dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=I_opt_resampled.index,
            y=I_opt_resampled.values,
            mode="lines",
            name=f"Optimization irradiance (Resampled to {pp.sample_time_mod} seconds)",
            showlegend=True,
            line=dict(width=1, dash="dot"),
            fill="tozeroy",
            line_color="rgba(255, 211, 0, 1)",
            fillcolor="rgba(255, 211, 0, 0.3)"
        )
    )
    # Add horizontal lines for thresholds
    for name, value in asdict(pp.irradiance_thresholds).items():
        fig.add_hline(
            y=value,
            line=dict(color=color_palette["bg_gray"], dash="dash"),
            annotation_text=name,
            annotation_position="bottom right",
        )

    action_types = np.unique([action_tuple[0] for action_tuple in list(pp.operation_actions.values())[0]])
            
    for subsystem_idx, (subsystem_id, action_tuples) in enumerate(operation_datetimes.items()):
        for action_type in action_types:
            
            if action_type == "startup":
                symbols_ = [symb for symb in symbols if symb.startswith("triangle-up")]
            elif action_type == "shutdown":
                symbols_ = [symb for symb in symbols if symb.startswith("triangle-down")]
            else:
                symbols_ = symbols
            
            action_dts = np.concatenate([action_tuple[1] for action_tuple in action_tuples if action_tuple[0] == action_type])
            fig.add_trace(
                go.Scatter(
                    x=action_dts,
                    y = [I_opt_resampled.iloc[I_opt_resampled.index.get_indexer([time], method="nearest")[0]] for time in action_dts],
                    mode="markers",
                    name=f"{subsystem_id} - {action_type}",
                    marker_color=plt_colors[subsystem_idx],
                    marker_symbol=symbols_[subsystem_idx],
                    marker_size=10,
                )
            )
    # Build compact subtitle lines
    subtitle_lines = [
        f"<b>{key}</b>: {compact_repr(value)}" for key, value in pp.operation_actions.items()
    ]
    subtitle_str = "<br>".join(subtitle_lines)

    fig.update_layout(
        title=dict(
            text="<b>Operation mode change candidates</b>",
            # x=0.5,
            # xanchor="center",
            subtitle=dict(
                text=subtitle_str,
                font=dict(size=12, color="gray")
            ),
            automargin=True,
        ),
        xaxis_title="Time",
        yaxis_title="Irradiance (W/m²)",
        template="plotly_white",
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0)",
            font_size=12,
            bordercolor="rgba(0,0,0,0)"
            # font_family="Rockwell"
        ),
    )
    # fig
    return fig

def plot_operation_plans(
    int_dec_vars_list: Sequence[IntegerDecisionVariables],
    pp: Optional[ProblemParameters] = None, 
    I: Optional[pd.Series] = None,
    problem_idxs: Optional[Sequence[int]] = None,
) -> go.Figure:

    subsystem_ids = list(asdict(int_dec_vars_list[0]).keys())
    if len(subsystem_ids) == 2:
        colors = [color_palette["plotly_orange"], color_palette["wct_purple"]]
    else:
        colors = plt_colors
    problem_idxs = problem_idxs or range(len(int_dec_vars_list))
    problem_labels = [f"Problem {i:03d}" for i in problem_idxs]
    days = np.unique(list(asdict(int_dec_vars_list[0]).values())[0].index.day)

    fig = go.Figure()

    # Add a filled trace on secondary y-axis
    if I is not None:
        fig.add_trace(go.Scatter(
            x=I.index[I.index.day <= days[-1]],
            y=I.values[I.index.day <= days[-1]],
            fill='tozeroy',
            fillcolor='rgba(255, 211, 0, 0.3)',
            mode='lines',
            line=dict(color='rgba(255, 211, 0, 0.3)', width=3),
            name='Irradiance ❯',
            yaxis='y'
        ))

    if pp is not None and I is not None:
        operation_datetimes = generate_operation_datetimes(I, pp.operation_actions, pp.irradiance_thresholds)
        action_dts = np.concatenate([
            action_tuple[1]
            for action_tuples in operation_datetimes.values()
            for action_tuple in action_tuples
        ])
        for idx, op_dt in enumerate(action_dts):
            fig.add_vline(
                x=op_dt,
                name="Operation datetimes" if idx==0 else None,
                line=dict(color=hex_to_rgba_str(color_palette["gray"], alpha=0.5), dash="dash"),
            )

    for problem_idx, int_dec_vars in enumerate(int_dec_vars_list):
        # Y-positions
        base_y = problem_idx * 0.5
        # print(base_y)
        for subsystem_idx, subsystem_id in enumerate(subsystem_ids):
            subsystem_y_ = base_y + (subsystem_idx+1)*0.1

            # Get start and end times
            subsystem_start = []
            subsystem_end = []
            subsystem_y = []
            symbols = []
            for day in days:
                start, end = int_dec_vars.get_start_and_end_datetimes(var_id=subsystem_id, day=day)
                subsystem_start.append(start)
                subsystem_end.append(end)
                symbols.extend(['triangle-up', 'triangle-down', 'x'])
                subsystem_y.extend([subsystem_y_, subsystem_y_, None])
            subsystem_x = np.concatenate([[start, end, None] for start, end in zip(subsystem_start, subsystem_end)])

            # Add line + markers for sfts_mode
            fig.add_trace(go.Scatter(
                x=subsystem_x,
                y=subsystem_y,
                mode="lines+markers",
                line=dict(color=colors[subsystem_idx]),
                marker=dict(symbol=symbols, size=8),
                name=problem_labels[problem_idx] if problem_idx > 0 else subsystem_id.replace("_", " "),  # Show legend only once
                showlegend=(problem_idx == 0),
                yaxis="y2",
            ))

        # Add annotation
        fig.add_annotation(
            xref="paper", yref="y2",
            x=0, y=base_y + 0.1*len(subsystem_ids) / 2,
            text=problem_labels[problem_idx],
            showarrow=False,
            font=dict(size=12),
            xanchor="right"
        )
        
    # Build compact subtitle lines
    if pp is not None:
        subtitle_lines = [
            f"<b>{key}</b>: {compact_repr(value)}" for key, value in pp.operation_actions.items()
        ]
        subtitle_str = "<br>".join(subtitle_lines)
    else:
        subtitle_str = "Segments represent the active operation span"
    
    # Final figure layout
    fig.update_layout(
        title=dict(text="Operation plan for the different alternatives",
                   automargin=True,
                #    y=0.95, yanchor="top", automargin=True, yref="container",
                   subtitle=dict(text=subtitle_str, font=dict(size=12, color="gray"))),
        xaxis=dict(title="", nticks=10),
        yaxis=dict(  # Now the irradiance axis (on the right)
            title='Irradiance (W/m²)' if I is not None else "",
            side='right',
            showgrid=False
        ),
        yaxis2=dict(  # Now the operation axis (on the left)
            overlaying='y',
            side='left',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=(0.0, subsystem_y_+0.2)
        ),
        height=200 + 30*len(int_dec_vars_list),
        width=600,
        margin=dict(l=100, r=20, t=20, b=80),
        template="plotly_white",
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0)",
            font_size=12,
            bordercolor="rgba(0,0,0,0)"
            # font_family="Rockwell"
        ),
        legend=dict(orientation="v", y=1, yanchor="top", yref="container") #, y=-0.2, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)")
    )
    
# fig
    return fig


@dataclass
class OperationPlanVisualizer:
    op_plan_results: OperationPlanResults
    output_path: Optional[Path] = None
    data_path: Path = Path("/workspaces/SolarMED/optimization/data")
    results_plot_simplified_config: Optional[dict] = None
    results_plot_config: Optional[dict] = None
    vars_config: Optional[dict] = None
    best_problem_idxs: Optional[list[int]] = None
    
    def __post_init__(self):
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

        if self.data_path is not None:
            if self.results_plot_simplified_config is None:
                with open(self.data_path / "plot_config_simplified.hjson") as f:
                    self.results_plot_simplified_config = hjson.load(f)
        
            if self.results_plot_config is None:
                with open(self.data_path / "plot_config.hjson") as f:
                    self.results_plot_config = hjson.load(f)
                
            if self.vars_config is None:
                with open(self.data_path / "variables_config.hjson") as f:
                    self.vars_config = hjson.load(f)
                    
        self.best_problem_idxs = self.op_plan_results.fitness.argsort().tolist()
    
    def check_output_path_defined(self):
        if self.output_path is None:
            raise ValueError("Output path not defined")
    
    def plot_fitness_history(self, save: bool = False, highlight_best: Optional[int] = 3) -> go.Figure:
        if save:
            self.check_output_path_defined()
            
        df = self.op_plan_results.fitness_history.ffill()
        fitness_history_list = [df[col].tolist() for col in df.columns]
        problem_ids = [f"problem {problem_id:03d}" for problem_id in df.columns.tolist()]
            
        fig = plot_obj_scape_comp_1d(
            fitness_history_list=fitness_history_list, 
            algo_ids=[problem_ids[0].replace("000", f"0-{len(fitness_history_list)}"), *problem_ids[1:]],
            highlight_best=highlight_best,
            title_text="<b>Fitness evolution</b><br>comparison between different problems", # algorithms
            width=600,
            legend=dict(x=0.72, y=1),
            showlegend=False,
        )
        
        return fig

    def plot_result_timeseries(self, save: bool = False, version: Literal["simplified", "complete"] = "simplified") -> go.Figure:
        if save:
            self.check_output_path_defined()
            
            
        if version == "simplified":
            plot_config = self.results_plot_simplified_config
        else:
            plot_config = self.results_plot_config
        assert plot_config is not None, "plot config should be defined"
        assert self.vars_config is not None, "asdasdasd"
        
        df = self.op_plan_results.results_df
        df = condition_result_dataframe(df)
        plot_config["plots"]["fitness_cumulative"]["title"] = f"Total acummulated benefit: <b>{df['cumulative_net_profit'].iloc[-1]:.2f}</b> (u.m.)"

        fig = experimental_results_plot(
            plot_config, 
            df=df, # best_dfs[0], 
            # df_comp=best_dfs[1:2] if max_n_cases > 1 else None,
            vars_config=self.vars_config, resample=False
        )

        return fig
    
    def plot_operation_plans(self, save: bool = False, n_best_problems: Optional[int] = None) -> go.Figure:
        if save:
            self.check_output_path_defined()
            
        pp = self.op_plan_results.problem_params
        I = self.op_plan_results.results_df["I"]
            
        int_dec_vars_list = OperationPlanner.initialize(
                operation_actions = pp.operation_actions,
                irradiance_thresholds = pp.irradiance_thresholds
            ).generate_decision_series(I=I)
            
        if n_best_problems is not None:
            int_dec_vars_list = int_dec_vars_list[:n_best_problems]
            
        fig = plot_operation_plans(
            int_dec_vars_list, 
            I=I,
            pp=pp,
            problem_idxs=self.best_problem_idxs[:n_best_problems] if n_best_problems is not None else None,
        )
        
        return fig
    
    def plot_op_mode_change_candidates(self, save: bool = False, I_exp: Optional[pd.Series] = None) -> go.Figure:
        
        if save:
            self.check_output_path_defined()
            
        fig = plot_op_mode_change_candidates(
            I_series=I_exp if I_exp is not None else self.op_plan_results.results_df["I"], 
            pp=self.op_plan_results.problem_params,
            include_experimental = True if I_exp is not None else False,
        )
        
        return fig