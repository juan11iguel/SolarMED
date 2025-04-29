from pathlib import Path
import time
import datetime
import numpy as np
import pandas as pd
from loguru import logger
import pygmo as pg

from solarmed_optimization import (EnvironmentVariables,
                                   ProblemParameters,
                                   ProblemSamples,
                                   AlgorithmParameters,
                                   PopulationResults,)
from solarmed_optimization.utils.initialization import (problem_initialization, 
                                                        InitialStates, 
                                                        generate_population,
                                                        initialize_problem_instance_minlp)
from solarmed_optimization.utils.evaluation import evaluate_optimization
from solarmed_optimization.utils.serialization import MinlpOptimizationResults
from solarmed_optimization.utils.visualization import generate_visualizations
# from solarmed_optimization.problems.minlp import Problem

logger.disable("phd_visualizations")

#%% Constants
# Paths definition
base_output_path: Path = Path("./results/b")
data_path: Path = Path("./data")
fsm_data_path: Path = Path("./results/fsm_data")
# n_islands: int = 10e

if not base_output_path.exists():
    base_output_path.mkdir()
# Either load parameters from json or create a new instance
# with open(output_path / "problem_params.json") as f:
#     problem_params = ProblemParameters(**json.load(f))
problem_params: ProblemParameters = ProblemParameters(
    optim_window_time=8*3600, # 8 hours
    episode_duration=3600 * 12 # To only have one optimization step
)
optim_params: AlgorithmParameters = AlgorithmParameters(
    pop_size=16, # 32
    n_gen=100,
    seed_num=32
)

algorithms_to_eval: dict[str, dict[str, int]] = {
    # "sga": {
    #     "gen": optim_params.n_gen,
    #     "seed": optim_params.seed_num
    # },
    # "gaco": {
    #     "gen": optim_params.n_gen, 
    #     "ker": optim_params.pop_size, 
    #     "seed": optim_params.seed_num
    # },
    "nsga2": { # Does not parallelize
        "gen":optim_params.n_gen,
        "seed":optim_params.seed_num           
    },
    # "ihs": { # Does not parallelize
    #     "gen":optim_params.n_gen,
    #     "seed":optim_params.seed_num           
    # },
}
dates_to_eval: list[str] = ["20180921_20180928", ] # 20180921_20180928 20230703 "20230707_20230710" # '20230630' '20230703'
initial_states: InitialStates = InitialStates(Tts_h=[90, 80, 70], 
                                              Tts_c=[70, 60, 50]) # Set to None to use values from data


def simulate_episode(algo_id: str, algo_params: dict[str, int], date_str: str,  output_path: Path):
    
    print(f"Running simulation for {algo_id} algorithm on {date_str} data")
    
    problem_data = problem_initialization(problem_params=problem_params,
                                          date_str=date_str,
                                          data_path=data_path,
                                          initial_states=initial_states)

    ps: ProblemSamples = problem_data.problem_samples
    pp: ProblemParameters = problem_data.problem_params
    df: pd.DataFrame = problem_data.df
    model = problem_data.model

    df_sim: pd.DataFrame = None
    df_hors: list[pd.DataFrame | list[pd.DataFrame]] = []
    isl: pg.island = None

    # Setup optimization algorithm / computation strategy
    algorithm = getattr(pg, algo_id)(**algo_params)
    algo = pg.algorithm(algorithm)
    algo.set_verbosity(1) # regulates both screen and log verbosity

    # island = pg.ipyparallel_island()
    metadata: dict[str, str] = {"date_str": date_str, "algo_id": algo_id}

    opt_step_idx: int = 0
    max_opt_steps: int = (ps.episode_samples-pp.idx_start-ps.optim_window_samples) // ps.n_evals_mod_in_opt_step - 1
    idx_mod = pp.idx_start
    initial_time = time.time()
    for opt_step_idx in range(0, max_opt_steps):
        hor_span = (idx_mod+1, idx_mod+1+ps.n_evals_mod_in_hor_window)

        # Optimization step `opt_step_idx`
        print("")
        print(f"Optimization step {opt_step_idx+1}/{max_opt_steps}")
        print(f"Range: {hor_span}")

        # 1. Initialize the problem instance
        problem, env_vars = initialize_problem_instance_minlp(problem_data=problem_data, idx_mod=idx_mod,
                                                        fsm_data_path=fsm_data_path, return_env_vars=True)
        ## Environment variables predictions
        # ds = df.iloc[hor_span[0]:hor_span[1]]
        # env_vars: EnvironmentVariables = EnvironmentVariables(
        #     I=ds['I'].values,
        #     Tamb=ds['Tamb'].values,
        #     Tmed_c_in=ds['Tmed_c_in'].values,
        #     cost_w=np.ones((ps.n_evals_mod_in_hor_window, )) * pp.env_params.cost_w,
        #     cost_e=np.ones((ps.n_evals_mod_in_hor_window, )) * pp.env_params.cost_e,
        # )

        # ## Initialize problem
        # problem = MinlpProblem(
        #     model=model, 
        #     sample_time_opt=pp.sample_time_opt,
        #     optim_window_time=pp.optim_window_time,
        #     env_vars=env_vars,
        #     dec_var_updates=pp.dec_var_updates,
        #     fsm_valid_sequences=pp.fsm_valid_sequences,
        #     fsm_data_path=fsm_data_path,
        #     use_inequality_contraints=False
        # )
        prob = pg.problem(problem)

        # Manually set initial population
        paths_df = problem.fsm_med_data.paths_df
        if opt_step_idx == 0: # problem.model_dict["current_state"] == "000" SolarMedState.sf_IDLE_ts_IDLE_med_OFF            
            pop_dec_vec = generate_population(model=model, pp=pp, problem=problem,
                                              pop_size=optim_params.pop_size, prob=prob, 
                                              return_decision_vector=True,
                                              paths_from_state_df=paths_df[paths_df["0"] == model.med_state.value])
            
            # Inneficient since it will trigger a fitness evaluation for each individual, Ideally I would like to provide the population upon initialization
            # TODO: Create post in pygmo github discussions
            pop = pg.population(prob, size=optim_params.pop_size, seed=optim_params.seed_num)
            [pop.set_x(i, x) for i, x in enumerate(pop_dec_vec)]
            # for i, dec_vars in enumerate(pop_dec_vars):
            #     # Initialize random decision vectors
            #     x = pg.random_decision_vector(prob)
            #     dec_vars_ = decision_vector_to_decision_variables(x, dec_var_updates=problem.dec_var_updates, span='none', )
            #     # Set the real decision variables values in the manually generated decision variables
            #     [setattr(dec_vars, var_id, getattr(dec_vars_, var_id)) for var_id in problem.dec_var_real_ids]
            #     # Validate the real decision variables values using the manually established logical variables values
            #     dec_vars = validate_real_dec_vars(dec_vars, problem.real_dec_vars_box_bounds)
            #     pop.set_x(i, decision_variables_to_decision_vector(dec_vars), )
                
        else:
            # Set initial population from previous step last generation
            # for i, (x, f) in enumerate(zip(isl.get_population().get_x(), isl.get_population().get_f())):
            #     # pop.set_xf(i, x, f)
            #     # TODO: Not so simple, needs to take into account the sample times of optimization steps and model 
            #     x = np.roll(x, -1) # Remove the first element which is the decision vector for the last optimization step
            #     x[-1] = x[-2] # Set the last element to the penultimate element
            #     pop.set_x(i, x)
                
            pop_dec_vec = generate_population(model=model, pp=pp, problem=problem,
                                              pop_size=optim_params.pop_size,
                                              dec_vec=pop.champion_x,
                                              return_decision_vector=True,
                                              paths_from_state_df=paths_df[paths_df["0"] == model.med_state.value])
            [pop.set_x(i, x) for i, x in enumerate(pop_dec_vec)]
                
        print(f"{pop=}")

        isl = pg.island(algo=algo, pop=pop) # udi=island
        # isl = pg.island(algo=algo,  prob=prob, size=optim_params.pop_size, seed=optim_params.seed_num)
        isl.evolve()
        print(isl)

        # Archipielago variant
        # archi = pg.archipelago(n=n_islands,algo=algo, prob=prob, pop_size=pop_size, seed=seed_num)
        # archi.evolve() 
        # print(archi)

        start_time = time.time()
        while isl.status == pg.evolve_status.busy:
            time.sleep(5)
            print(f"Elapsed time: {time.time() - start_time:.0f}")
            # print(f"Current evolution results | Best fitness: {pop_current.champion_f[0]}, \nbest decision vector: {pop_current.champion_x}")
        optim_eval_elapsed_time = int(time.time() - start_time)
        print(f"Completed evolution! Took {time.time() - start_time:.0f} seconds") #| Best fitness: {pop_current.champion_f[0]}, \nbest decision vector: {pop_current.champion_x}")

        # Extract updated problem
        problem = isl.get_population().problem.extract(object)

        # Evaluate optimization and move `model` to the next step
        # Update, only evaluate the best in the population, otherwise it takes too much space
        df_hor, df_sim, model = evaluate_optimization(
            df_sim=df_sim, 
            pop=[ isl.get_population().get_x()[isl.get_population().best_idx()] ],
            best_idx=0,
            model=model,
            env_vars=env_vars, problem=problem,
            problem_data=problem_data, idx_mod=idx_mod
        )
        
        print(f"After moving model to the next step, {model.current_sample=}")
        
        # if opt_step_idx > 0:
        #     df_hors[-1] = df_hors[-1][step_results.best_idx_per_gen[-1]] # Only keep the dataframe from the best individual from past steps
        df_hors.append(df_hor[0])
        
        pop_results: PopulationResults = PopulationResults.initialize(
            problem=problem,
            pop_size=optim_params.pop_size,
            n_gen=optim_params.n_gen,
            elapsed_time=optim_eval_elapsed_time,
        )
        
        # Export results
        MinlpOptimizationResults(
            metadata=metadata,
            problem_params=problem_params,
            initial_states=initial_states,
            algo_log=isl.get_algorithm().extract( getattr(pg, algo_id) ).get_log(),
            df_hor=df_hor[0],
            df_sim=df_sim,
            pop_results=pop_results,
            algo_params=algo_params,
            # figs=generate_visualizations(problem=problem, df_hors=df_hors, df_sim=df_sim, 
            #                              problem_data=problem_data, metadata=metadata, 
            #                              pop_results=pop_results)
        ).dump(output_path=output_path, step_idx=opt_step_idx)

        # Finally, increase counter
        idx_mod += ps.n_evals_mod_in_opt_step
        
        print(f"Current system state: {model.current_state.name}, integer inputs: med_active={model.qmed_s}, vacuum={model.med_vacuum_state}, ts_active={model.ts_active}, sf_active={model.sf_active}")
        print(f"Elapsed time: {time.time() - initial_time:.0f}")
        print("")
        
    print(f"Completed simulation! Cumulative elapsed time: {time.time() - initial_time:.0f}")


def main():
    for date_str in dates_to_eval:
        for algo_id, algo_params in algorithms_to_eval.items():
            output_path = base_output_path / f"{date_str}_eval_at_{datetime.datetime.now(tz=datetime.timezone.utc):%Y%m%d}" / algo_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            simulate_episode(algo_id=algo_id, algo_params=algo_params, date_str=date_str, output_path=output_path)
        
if __name__ == "__main__":
    main()