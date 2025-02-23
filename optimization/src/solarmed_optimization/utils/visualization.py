import plotly.graph_objects as go
import pandas as pd
from solarmed_optimization import ProblemData, PopulationResults
from solarmed_optimization.problems.minlp import BaseProblem
from solarmed_optimization.visualization.optim_evolution import (plot_dec_vars_evolution,
                                                                 plot_obj_space_1d_animation)

def generate_visualizations(problem: BaseProblem, df_hors: list[pd.DataFrame],
                            df_sim: pd.DataFrame, problem_data: ProblemData,
                            metadata: dict[str, str], pop_results: PopulationResults) -> list[go.Figure]:
    
    figs: list[go.Figure] = []
    
    figs.append( plot_dec_vars_evolution(problem=problem, df_hors_=df_hors, df_mod=df_sim, 
                                            full_xaxis_range=False, episode_samples=problem_data.problem_samples.episode_samples) )
    
    figs.append( plot_obj_space_1d_animation(fitness_history=pop_results.fitness_per_gen) )
    figs[-1].update_layout(title=f"{metadata['date_str']} - Fitness evolution - {metadata['algo_id'].upper()}",
                           margin=dict(t=50, b=100), height=600)
    
    return figs


def condition_result_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Condition the optimization results DataFrame for visualization purposes."""
    
    df["med_active"] = df["med_active"].astype(bool).fillna(False)
    df["sf_active"] = df["sf_active"].astype(bool).fillna(False)
    df["sf_ts_state"] = df["sf_ts_state"].astype(float).fillna(0).astype(int)
    df["med_state"] = df["med_state"].astype(float).fillna(0).astype(int)
    df["Pth_hx_p"] = df["Pth_hx_p"].fillna(0.0)
    df["Pth_hx_s"] = df["Pth_hx_s"].fillna(0.0)
    df["Pth_ts_dis"] = df["Pth_ts_dis"].fillna(0.0)
    df["net_profit"] = df["net_profit"].fillna(0.0)
    df["net_loss"] = df["net_loss"].fillna(0.0)
    df["Jtotal"] = df["Jtotal"].fillna(0.0)
    

    # Infer objects to avoid silent downcasting
    # df = df.infer_objects(copy=False)

    # New plot variables
    df["cumulative_net_profit"] = df["net_profit"].cumsum()
    # df["sfts_mode"] = df["sf_ts_state"] + 1
    # df["med_mode"] = df["med_state"].apply(lambda x: 2 if 1 <= x < 5 or x == 5 else 1 if x == 0 else x)
    
    return df