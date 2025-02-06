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