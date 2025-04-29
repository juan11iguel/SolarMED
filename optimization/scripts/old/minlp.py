import json
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import pygmo as pg

from solarmed_optimization import (EnvironmentVariables,
                                   ProblemParameters,
                                   ProblemSamples)
from solarmed_optimization.utils import (decision_vector_to_decision_variables,
                                         validate_dec_var_updates)
from solarmed_optimization.utils.evaluation import evaluate_model
from solarmed_optimization.utils.initialization import problem_initialization
from solarmed_optimization.problems.pygmo import MinlpProblem

logger.disable("phd_visualizations")

#%% Constants
# Paths definition
output_path: Path = Path("./results")
data_path: Path = Path("./data")
fsm_data_path: Path = Path("./results")
date_str: str = "20230703" # "20230707_20230710" # '20230630' '20230703'
pop_size: int = 10

# Either load parameters from json or create a new instance
with open(output_path / "problem_params.json") as f:
    problem_params = ProblemParameters(**json.load(f))
# problem_params = ProblemParameters()

problem_data = problem_initialization(problem_params=problem_params,
                                      date_str=date_str,
                                      data_path=data_path)

ps: ProblemSamples = problem_data.problem_samples
pp: ProblemParameters = problem_data.problem_params
df: pd.DataFrame = problem_data.df
model = problem_data.model

# df_mods: list[pd.DataFrame] = []
df_hors: list[pd.DataFrame] = []
df_sim: pd.DataFrame = None

# Fill missing data
# df['med_vac_state'] = 2

opt_step_idx: int = 0
# for opt_step_idx in range(0, max_opt_steps):
idx_mod = pp.idx_start
# for opt_step_idx in range(0, max_opt_steps):
hor_span = (idx_mod+1, idx_mod+1+ps.n_evals_mod_in_hor_window)

# Optimization step `opt_step_idx`

# 1. Initialize the problem instance
## Intialize model instance
ds = df.iloc[idx_mod]

print("")
print(f"Optimization step {opt_step_idx+1}/{ps.max_opt_steps}")

## Environment variables predictions
ds = df.iloc[hor_span[0]:hor_span[1]]
env_vars: EnvironmentVariables = EnvironmentVariables(
    I=ds['I'].values,
    Tamb=ds['Tamb'].values,
    Tmed_c_in=ds['Tmed_c_in'].values,
    cost_w=np.ones((ps.n_evals_mod_in_hor_window, )) * pp.env_params.cost_w,
    cost_e=np.ones((ps.n_evals_mod_in_hor_window, )) * pp.env_params.cost_e,
)

## Initialize problem
problem = MinlpProblem(
    model=model, 
    sample_time_opt=pp.sample_time_opt,
    optim_window_time=pp.optim_window_time,
    env_vars=env_vars,
    dec_var_updates=pp.dec_var_updates,
    fsm_valid_sequences=pp.fsm_valid_sequences,
    fsm_data_path=fsm_data_path
)

# PyGMO specific code
prob = pg.problem(problem)
print(prob)

# isl = pg.island(algo = pg.gaco(gen=2, ker=pop_size, q=1.0, seed=23), 
#                 prob = problem, 
#                 size=pop_size, 
#                 udi=pg.mp_island())
# print(f"Number of processes in the pool (should be equal to the number of logical CPUs): {isl.extract(pg.mp_island).get_pool_size()}")
# isl.evolve()
# print(f"After initiating evolution: \n{isl}")
# # isl.wait()
# isl.wait_check()
# pg.mp_island.shutdown_pool()
# pg.mp_bfe.shutdown_pool()
# print("After finishing evolution:")
# print(isl)

# print(isl.get_population().champion_f[0])
# print(isl.get_population().champion_x)

algo = pg.algorithm(pg.gaco(gen=10, ker=pop_size))
print(f"Running {algo.get_name()}")
algo.set_verbosity(1) # regulates both screen and log verbosity

# Initialize population and evolve population
pop = pg.population(prob, size=pop_size)
print(f"Initial population: {pop}\nStarting evolution...")
pop = algo.evolve(pop)

# Extract results of evolution
uda=algo.extract(pg.gaco)
print(f"Completed evolution, best fitness: {pop.champion_f[0]}, \nbest decision vector: {pop.champion_x}")

print(uda.get_log())

for iter_log in uda.get_log():
    print(iter_log)

# if __name__ == "__main__":
#     main()