import inspect
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import os
import hjson
from typing import Literal, Annotated
from loguru import logger

import pygad
from solarMED_optimization.custom_GA import MyGA

from solarMED_modeling.solar_med import SolarMED
from solarMED_modeling import MedVacuumState
from solarMED_modeling.utils.matlab_environment import set_matlab_environment
from solarMED_optimization import EnvVarsSolarMED, CostVarsSolarMED, fitness_function

from phd_visualizations import save_figure
from phd_visualizations.test_timeseries import experimental_results_plot
from solarMED_optimization.visualization import plot_episode_state_evolution
from solarMED_modeling import SF_TS_State, MedState

set_matlab_environment()

#%% Parameters

date_str: str = '20230707_20230710'
filenames_process_data = [f'{date_str}_solarMED.csv', f'{date_str}_MED.csv']

data_path: Path = Path(f'{os.getenv("HOME")}/Nextcloud/Juanmi_MED_PSA/EURECAT/data')
config_path: Path = Path(f'{os.getenv("HOME")}/development_psa/SolarMED-modeling/data')
# Create output_path to store results
output_path: Path = Path("results")
output_path_docs: Path = Path("docs/attachments")

sample_rate = '300s'
sample_rate_numeric = int(sample_rate[:-1])

df_mod = pd.DataFrame()
df_list = [] # List of dataframes with the results of the simulation of the prediction horizon on each iteration

idx_start = 0
model_sample_rate = sample_rate_numeric # seconds
n_of_dec_vars_updates: int = 10 # Number of decision variables updates per prediction horizon
prediction_horizon_duration = 24*3600 # seconds
Np = prediction_horizon_duration//model_sample_rate
default_cost_w = 3 # €/m3
default_cost_e = 0.05 # €/kWh
ga_instance = None

# Algorithm hyper-parameters

num_generations = 1  # Number of generations.
num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 10  # Number of solutions/individuals in the population on each generation (constant population).

# Gene space
# As of Python 3.7: "the insertion-order preservation nature of dict objects has been declared to be an official part of the Python language spec."
# np.arange(start, stop, step)
# np.linspace(start, stop, num=50)
init_vars = dict(Tts_h=[], Tts_c=[], Tsf_in_ant=np.array([0]), msf_ant=np.array([0]), resolution_mode="simple")
model = SolarMED(**init_vars)

## Thermal storage
mts_src_sol_space = [0]
mts_src_sol_space.extend(np.linspace(start=model.lims_mts_src[0], stop=model.lims_mts_src[1], num=4))

## Solar field
Tsf_out_sol_space = [0]
Tsf_out_sol_space.extend(np.linspace(start=model.lims_Tsf_out[0], stop=model.lims_Tsf_out[1], num=6))
## MED
mmed_s_sol_space = [0]
mmed_s_sol_space.extend(np.linspace(start=model.lims_mmed_s[0], stop=model.lims_mmed_s[1], num=3))
mmed_f_sol_space = np.linspace(start=model.lims_mmed_f[0], stop=model.lims_mmed_f[1], num=4)
Tmed_s_in_sol_space = np.linspace(start=model.lims_Tmed_s_in[0], stop=model.lims_Tmed_s_in[1], num=4)
Tmed_c_out_sol_space = np.linspace(start=20, stop=34, num=4)
med_vacuum_state_sol_space = [state.value for state in MedVacuumState]

# UPDATE: Since now gen_names is used, it's enough with keeping the order between gen_names and gene_space
# IMPORTANT! Keep the order of the model.step() method
base_names = ["mts_src", "Tsf_out", "mmed_s", "mmed_f", "Tmed_s_in", "Tmed_c_out", "med_vacuum_state"]
gen_names: Annotated[list[str], n_of_dec_vars_updates * len(base_names)] = [f"{base_name}_{i}" for i in
                                                                            range(n_of_dec_vars_updates) for base_name
                                                                            in base_names]
gene_space = [
                 # Decision variables. Important to keep order in sync with `step` model's method
                 # Options: list of values, dict(low, high, step),
                 mts_src_sol_space, Tsf_out_sol_space, mmed_s_sol_space, mmed_f_sol_space, Tmed_s_in_sol_space,
                 Tmed_c_out_sol_space, med_vacuum_state_sol_space
             ] * n_of_dec_vars_updates

# Number of decision variables
num_genes = len(gene_space)

# Mutation type
"""
    From the [docs]():

    The problem with constant mutation rate across all genes is that "The weak point of “classical” GAs is the total randomness of mutation, which is applied equally to all chromosomes, irrespective of their fitness. Thus a very good chromosome is equally likely to be disrupted by mutation as a bad one."

    Adaptive mutation works as follows:

    1. Calculate the average fitness value of the population (f_avg).
    2. For each chromosome, calculate its fitness value (f).
    3. If f<f_avg, then this solution is regarded as a low-quality solution and thus the mutation rate should be kept high because this would increase the quality of this solution.

       If f>f_avg, then this solution is regarded as a high-quality solution and thus the mutation rate should be kept low to avoid disrupting this high quality solution.

"""
mutation_type: Literal["adaptive", "random"] = "adaptive"
# If mutation type is set to adaptive, then two mutation_probability are required, the probaility for the worse solutions (first element) and the probability for the better solutions (second element)
mutation_probability: Annotated[list[float], 2] = [0.25, 0.1]

#%% Load data

# Process experimental data
from solarMED_modeling.utils import data_preprocessing, data_conditioning

data_paths = [data_path / filename_process_data for filename_process_data in filenames_process_data]

# 20230707_20230710 data does not include solar irradiation
# An alternative source from http://heliot.psa.es/meteo_psa_2022 is used for the solar irradiance
data_paths.append(data_path / "environment_data/env_20230707_20230710.csv")

# 20230707_20230710 does not include continuous seawater temperature and salinity
# An alternative source from https://doi.org/10.25423/CMCC/MEDSEA_ANALYSISFORECAST_PHY_006_013_EAS8 is used for seawater temperature and salinity
data_paths.append(data_path / "external_data/env_20220524_20240524.csv")

with open(config_path / "variables_config.hjson") as f:
    vars_config = hjson.load(f)

# Load data and preprocess data
df = data_preprocessing(data_paths, vars_config, sample_rate_key=sample_rate, fill_nans=True)

# Condition data
df = data_conditioning(df, sample_rate_numeric=sample_rate_numeric, vars_config=vars_config)

# 20230707_20230710 data does not include solar irradiation
# An alternative source from http://heliot.psa.es/meteo_psa_2022 is used for irradiance and ambient temperature

# 20230707_20230710 does not include continuos seawater temperature and salinity
# An alternative source from https://doi.org/10.25423/CMCC/MEDSEA_ANALYSISFORECAST_PHY_006_013_EAS8 is used for seawater temperature and salinity

df.rename(columns={
    # First rename the original columns
    "I": "I_orig", "Tamb": "Tamb_orig",
    "Tmed_c_in": "Tmed_c_in_orig", "wmed_f": "wmed_f_orig",

    # Then rename the new columns
    "GHI (W/m²)": "I", "Temperature (ºC)": "Tamb",
    "DNI (W/m²)": "DNI", "DHI (W/m²)": "DHI",
    "so": "wmed_f", "thetao": "Tmed_c_in"
}, inplace=True)

# There should be no duplicates
for col in df.columns:
    print(col) if col in ["Tmed_c_in", "Tamb", "I", "wmed_f"] else None

#%% Evaluate "optimal" operation for the episode

# Initialize model
model = SolarMED(
    use_models=True,
    use_finite_state_machine=True,
    resolution_mode="simple",
    sample_time=model_sample_rate,

    # If a slow sample time is used, the solar field internal PID needs to be detuned
    # Ki_sf=-0.0001,
    # Kp_sf=-0.005,

    # Initial states
    ## Thermal storage
    Tts_h=[df['Tts_h_t'].iloc[idx_start],
           df['Tts_h_m'].iloc[idx_start],
           df['Tts_h_b'].iloc[idx_start]],
    Tts_c=[df['Tts_c_t'].iloc[idx_start],
           df['Tts_c_m'].iloc[idx_start],
           df['Tts_c_b'].iloc[idx_start]],
    ## Solar field
    Tsf_in_ant=np.full((model_sample_rate * 10, 1), df['Tsf_in'].iloc[idx_start]),
    msf_ant=np.full((model_sample_rate * 10, 1), df['qsf'].iloc[idx_start]),
)

# Simulation loop
# for step_idx in range(idx_start, len(df)-(idx_start+Np)):
step_idx = idx_start

# Predict environment variables
# TODO - For now just use the available data

# Generate decision variables update samples
# TODO - For now just distribute them uniformly
# With numpy it's simpler to work with row vectors than column vectors
samples_opt: np.ndarray[bool] = np.full((Np, 1), False, dtype=bool)
samples_opt[::Np // n_of_dec_vars_updates] = True

# Setup additional required variables
env_vars: EnvVarsSolarMED = EnvVarsSolarMED(
    Tmed_c_in=df["Tmed_c_in"].iloc[step_idx:step_idx + Np].values,
    Tamb=df["Tamb"].iloc[step_idx:step_idx + Np].values,
    I=df["I"].iloc[step_idx:step_idx + Np].values,
    wmed_f=df["wmed_f"].iloc[step_idx:step_idx + Np].values if "wmed_f" in df.columns else None
)

cost_vars: CostVarsSolarMED = CostVarsSolarMED(
    costs_w=df["costs_w"].iloc[step_idx:step_idx + Np].values if "costs_w" in df.columns else np.full((1, Np),
                                                                                                      default_cost_w),
    costs_e=df["costs_e"].iloc[step_idx:step_idx + Np].values if "costs_e" in df.columns else np.full((1, Np),
                                                                                                      default_cost_e)
)

# Initialize the population from the previous step
if ga_instance is not None:
    # Should retrieve not only the best candidate but the X best candidates, the rest of the population should be randomly generated
    pass

# Update decision variables
ga_instance = MyGA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=gene_space,
    gene_names=gen_names,
    on_generation=None,
    mutation_type=mutation_type,
    mutation_probability=mutation_probability,
    # parallel_processing=["thread", 4],
    save_best_solutions=False,
    keep_elitism=10,
)
ga_instance.additional_vars = dict(
    env_vars=env_vars, cost_vars=cost_vars, samples_opt=samples_opt,
    sample_rate=model_sample_rate, Np=Np, Nc=np.sum(samples_opt), model=model,
    episode_idx=step_idx
)
## Reset additional outputs, it should be set at each .run() call
ga_instance.additional_outputs = dict(best_sol_df=None, solution_idx=None, best_fitness=-np.inf)

if step_idx == idx_start:
    ga_instance.summary()  # Print summary of the GA instance, only once

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()

# Repliace the steps already implemented in the fitness_function
# Simulate system with new decision variables
# model.step(**solution[0], **env_vars[0], )
# model.evaluate_fitness_function(cost_e=cost_vars.costs_e[0], cost_w=cost_vars.costs_w[0])

# # Retrieve results for the whole prediction horizon from fitness_function evaluation
# if ga_instance.additional_outputs["best_sol_df"] is None:
#     raise ValueError("Dataframe from best solution was not saved in the additional outputs")
# if ga_instance.additional_outputs["solution_idx"] != solution_idx:
#     raise ValueError("Solution index from best solution does not match the one from the additional outputs")
#
# df_list.append(
#     ga_instance.additional_outputs["best_sol_df"])  # Save results of current and future steps from current iteration
# df_mod = model.to_dataframe(df_mod)

# # Visualize results of iteration
# ga_instance.grouped_plot(figure_layout="vertical");

solution, solution_fitness, solution_idx = ga_instance.best_solution()
logger.info(f"Parameters of the best solution : {solution}")
logger.info(f"Fitness value of the best solution = {solution_fitness}")
logger.info(f"Index of the best solution : {solution_idx}")

if ga_instance.best_solution_generation != -1:
    logger.info(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

genat = f"{datetime.datetime.now():%Y%m%d%H%M}"


#%%
import cloudpickle

dump = cloudpickle.dumps(ga_instance)

ga_instance_loaded = cloudpickle.loads(dump)

#%% Export results (This should be part of the loop)
ga_instance.save(output_path / f"gen_at_{genat}_{df.index[step_idx].strftime('%Y%m%d')}_step_{step_idx}_ga_instance")

#%%
import cloudpickle

def find_non_picklable_attrs(obj):
    non_picklable_attrs = []
    for attr_name, attr_value in obj.__dict__.items():
        try:
            cloudpickle.dumps(attr_value)
        except Exception:
            non_picklable_attrs.append(attr_name)
    return non_picklable_attrs

print(find_non_picklable_attrs(ga_instance))


#%% Visualization

# Since at this point is not feasible to compute the decision variables for the whole episode, compute only the first iteration and
# simulate the system along the prediction horizon using the decision variables from the first iteration

from solarMED_modeling.solar_med import SolarMED
from solarMED_modeling.utils.matlab_environment import set_matlab_environment
from solarMED_optimization.simulation import simulate_episode
from solarMED_optimization import DecVarsSolarMED

set_matlab_environment()
logger.enable("solarMED_modeling")

# Initialize model
model = SolarMED(
    use_models=True,
    use_finite_state_machine=True,
    resolution_mode="simple",
    sample_time=model_sample_rate,

    # If a slow sample time is used, the solar field internal PID needs to be detuned
    # Ki_sf=-0.0001,
    # Kp_sf=-0.005,

    # Initial states
    ## Thermal storage
    Tts_h=[df['Tts_h_t'].iloc[idx_start],
           df['Tts_h_m'].iloc[idx_start],
           df['Tts_h_b'].iloc[idx_start]],
    Tts_c=[df['Tts_c_t'].iloc[idx_start],
           df['Tts_c_m'].iloc[idx_start],
           df['Tts_c_b'].iloc[idx_start]],
    ## Solar field
    Tsf_in_ant=np.full((model_sample_rate * 10, 1), df['Tsf_in'].iloc[idx_start]),
    msf_ant=np.full((model_sample_rate * 10, 1), df['qsf'].iloc[idx_start]),
)

step_idx = idx_start

samples_opt: np.ndarray[bool] = np.full((Np, 1), False, dtype=bool)
samples_opt[::Np // n_of_dec_vars_updates] = True

# Setup additional required variables
env_vars: EnvVarsSolarMED = EnvVarsSolarMED(
    Tmed_c_in=df["Tmed_c_in"].iloc[step_idx:step_idx + Np].values,
    Tamb=df["Tamb"].iloc[step_idx:step_idx + Np].values,
    I=df["I"].iloc[step_idx:step_idx + Np].values,
    wmed_f=df["wmed_f"].iloc[step_idx:step_idx + Np].values if "wmed_f" in df.columns else None
)

cost_vars: CostVarsSolarMED = CostVarsSolarMED(
    costs_w=df["costs_w"].iloc[step_idx:step_idx + Np].values if "costs_w" in df.columns else np.full((1, Np),
                                                                                                      default_cost_w),
    costs_e=df["costs_e"].iloc[step_idx:step_idx + Np].values if "costs_e" in df.columns else np.full((1, Np),
                                                                                                      default_cost_e)
)

num_dec_vars = len(DecVarsSolarMED.model_fields)  # Could be taken from previous definition, no need to reevaluate here
avail_dec_var_updates = solution[0::num_dec_vars]

if np.sum(samples_opt) > len(avail_dec_var_updates):
    logger.warning(
        f"The number of samples to update the decision variables ({np.sum(samples_opt)}) is greater than the number of available decision variables updates ({len(avail_dec_var_updates)}). Copying last decision variable update to avoid error. This needs to be fixed!")

    for i in range(0, np.sum(samples_opt) - len(avail_dec_var_updates)):
        solution = np.concatenate([solution, solution[-num_dec_vars:]])

dec_vars: DecVarsSolarMED = DecVarsSolarMED(
    mts_src=solution[0::num_dec_vars],
    Tsf_out=solution[1::num_dec_vars],
    mmed_s=solution[2::num_dec_vars],
    mmed_f=solution[3::num_dec_vars],
    Tmed_s_in=solution[4::num_dec_vars],
    Tmed_c_out=solution[5::num_dec_vars],
    med_vacuum_state=solution[6::num_dec_vars]
)

df_mod = simulate_episode(model_instance=model, env_vars=env_vars, cost_vars=cost_vars, dec_vars=dec_vars,
                          samples_opt=samples_opt)
# Sync model index with measured data
df_mod.index = df.index[
               idx_start:len(df_mod)]  # idx_start-1 because now we are adding one element after the initialization

# Split the df between the past and current data and the future/predicted data
df_current = df_mod.iloc[:idx_start + 10]
df_future = df_mod.iloc[idx_start + 10:]

#%% System variables evolution viz

with open(config_path / "plot_config.hjson") as f:
    plt_config = hjson.load(f)

plt_config["plots"]["costs"]["traces_left"][0]["var_id"] = 'net_profit'
plt_config["plots"]["heat_exchanger_flows"]["traces_left"].pop(-1)
plt_config["plots"]["thermal_storage_flows"]["traces_left"].pop(-1)
plt_config["plots"]["thermal_storage_flows"]["traces_left"].pop(-1)
plt_config["plots"]["med_temperatures"].pop("traces_right", None)

comp_trace_style: dict = {
    "line": {"dash": "dash"},
    "marker": {"color": "rgba(0,0,0,0)"}

}

fig = experimental_results_plot(plt_config, df=df_current, df_comp=df_future, vars_config=vars_config, resample=False,
                                index_adaptation_policy='combine', )

# Save figure
save_figure(
    figure_name=f"SolarMED_validation_{df.index[0].strftime('%Y%m%d')}_genat_{genat}",
    figure_path=output_path,
    fig=fig, formats=('png', 'html'),
    width=fig.layout.width, height=fig.layout.height, scale=2
)

#%% State evolution viz

fig = plot_episode_state_evolution(df_mod, subsystems_state_cls=[SF_TS_State, MedState], show_edges=False)

save_figure(
    figure_name=f"SolarMED_state_evolution_{df.index[0].strftime('%Y%m%d')}_genat_{genat}",
    figure_path=output_path,
    fig=fig, formats=('png', 'html'),
    width=fig.layout.width, height=fig.layout.height, scale=2
)