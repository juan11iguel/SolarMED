import numpy as np
import pandas as pd

from solarmed_modeling.solar_med import SolarMED

from solarmed_optimization import (EnvironmentVariables,
                                   ProblemData,)
from solarmed_optimization.utils import (decision_vector_to_decision_variables,
                                         evaluate_model,
                                         add_bounds_to_dataframe)
from solarmed_optimization.problems import BaseMinlpProblem

def evaluate_optimization(df_sim: pd.DataFrame, pop: list[np.ndarray[float | int]], 
                          env_vars: EnvironmentVariables, problem: BaseMinlpProblem,
                          model: SolarMED, problem_data: ProblemData, idx_mod: int, 
                          best_idx: int = 0,) -> tuple[pd.DataFrame, pd.DataFrame, SolarMED]:
    
    pp = problem_data.problem_params
    ps = problem_data.problem_samples
    # model = problem_data.model # noo!!
    
    
    # lower_bounds, upper_bounds = problem.get_bounds(readable_format=True)

    """ Evaluate horizon to validate inputs and compute outputs for every individual in the population """
    df_hor: list[pd.DataFrame] = []
    
    for x in pop:
        dec_vars = decision_vector_to_decision_variables(x = x,
                                                         dec_var_updates = pp.dec_var_updates,
                                                         span = "optim_window",
                                                         sample_time_mod = pp.sample_time_mod, 
                                                         optim_window_time = pp.optim_window_time,
                                                         sample_time_opt = pp.sample_time_opt)
        df_hor.append( evaluate_model(model = SolarMED(**model.dump_instance()), # Copy the model instance to avoid modifying the original
                                      n_evals_mod = ps.n_evals_mod_in_hor_window,
                                      mode = "evaluation",
                                      dec_vars = dec_vars, 
                                      env_vars = env_vars,
                                      df_start_idx=idx_mod) )
        df_hor[-1] = add_bounds_to_dataframe(df_hor[-1], problem=problem, 
                                             target="optim_window",
                                             df_idx=df_hor[-1].index[0])


    """ Simulate one optimization step for the best individual """
    
    dec_vars = decision_vector_to_decision_variables(x = pop[best_idx],
                                                     dec_var_updates = pp.dec_var_updates,
                                                     span = "optim_step",
                                                     sample_time_mod = pp.sample_time_mod, 
                                                     optim_window_time = pp.optim_window_time,
                                                     sample_time_opt = pp.sample_time_opt)
    df_sim = evaluate_model(model = model, # model instance will be modified
                            n_evals_mod = ps.n_evals_mod_in_opt_step,
                            mode = "evaluation",
                            dec_vars = dec_vars, 
                            env_vars = env_vars,
                            model_dec_var_ids=problem.dec_var_model_ids,
                            df_mod = df_sim,
                            df_start_idx=pp.idx_start)
    df_sim = add_bounds_to_dataframe(df_sim, problem=problem, 
                                     target="optim_step",
                                     df_idx=idx_mod)
    
    return df_hor, df_sim, model # model is already updated, but return it anyway