import copy
from pathlib import Path
import time
import numpy as np
import pandas as pd
from loguru import logger
import pygmo as pg

from phd_visualizations import save_figure

from solarmed_optimization import (
    EnvironmentVariables,
    ProblemParameters,
    ProblemSamples,
    RealDecisionVariablesUpdatePeriod,
    InitialDecVarsValues,
)
from solarmed_optimization.problems.nlp import Problem

from solarmed_optimization.utils.initialization import (
    problem_initialization,
    InitialStates,
)
from solarmed_optimization.utils.operation_plan import OperationPlanner
from solarmed_optimization.utils.serialization import export_algo_comparison
from solarmed_optimization.visualization.nlp import plot_op_mode_change_candidates
from solarmed_optimization.utils import extract_prefix

logger.disable("phd_visualizations")

# TODOs:
# - [ ] Verify all visualizations can be generated
# - [ ] Verify outputs are correctly stored in OptimizationResults

# 1. Define parameters
# %% Constants
# Paths definition
base_output_path: Path = Path("../results") / "nNLP_algo_comp"
data_path: Path = Path("../data")
date_str: str = "20180921_20180928"  # "20230707_20230710" # '20230630' '20230703'

# Parameters
max_n_obj_fun_evals: int = 5_000
pop_sizes: list[int] = [1] # [10, 20, 50] #, 150]

problem_params: ProblemParameters = ProblemParameters(
    optim_window_time=36 * 3600,  # 1d12h
    sample_time_opt=3600,  # 1h, In NLP-operation plan just used to resample environment variables
    operation_actions= {
        # Day 1 -----------------------  # Day 2 -----------------------
        "sfts": [("startup", 3), ("shutdown", 3), ("startup", 1), ("shutdown", 1)],
        "med": [("startup", 3), ("shutdown", 3), ("startup", 1), ("shutdown", 1)],
    },
    initial_states = InitialStates(Tts_h=[90, 80, 70], Tts_c=[70, 60, 50])
)

# optim_params: AlgorithmParameters = AlgorithmParameters(
#     pop_size=16, # 32
#     n_gen=100,
#     seed_num=32
# )        

metadata: dict = {
    "date_str": date_str,
    "max_n_obj_fun_evals": max_n_obj_fun_evals,
    "pop_sizes": pop_sizes,
}

if not base_output_path.exists():
    base_output_path.mkdir()

algorithms_to_eval: dict[str, dict[str, int]] = {
    # "sade": {}, # at least 7 individuals # Checked
    # "gaco": {}, # requires pop_size > len(x) # Checked
    # "cmaes": {"force_bounds": True}, # Checked
    # "pso_gen": {}, # Checked
    "sea": {}, # Checked
    # "de": {}, # Checked
    # "simulated_annealing": {"Ts": 10, "Tf": 0.01},
}
    
def main() -> None:
    
    # 2. Setup environment
    problem_data = problem_initialization(
        problem_params=problem_params,
        date_str=date_str,
        data_path=data_path,
    )
    ps: ProblemSamples = problem_data.problem_samples
    pp: ProblemParameters = problem_data.problem_params
    df: pd.DataFrame = problem_data.df
    model = problem_data.model

    # Setup environment
    idx_mod = pp.idx_start

    hor_span = (idx_mod + 1, idx_mod + 1 + ps.n_evals_mod_in_hor_window)
    ds = problem_data.df.iloc[hor_span[0] : hor_span[1]]

    env_vars: EnvironmentVariables = EnvironmentVariables(
        I=ds["I"],
        Tamb=ds["Tamb"],
        Tmed_c_in=ds["Tmed_c_in"],
        cost_w=pd.Series(
            data=np.ones((ps.n_evals_mod_in_hor_window,)) * pp.env_params.cost_w,
            index=ds.index,
        ),
        cost_e=pd.Series(
            data=np.ones((ps.n_evals_mod_in_hor_window,)) * pp.env_params.cost_e,
            index=ds.index,
        ),
    )
    # For operation plan, environment variables are only available with a one hour resolution
    env_vars_opt = env_vars.resample(f"{pp.sample_time_opt}s", origin="start")

    print(f"{env_vars.I.index[0]=}, {env_vars.I.index[-1]=}, {env_vars.I.index.freq=}")

    # 3. Build operation plan
    operation_planner = OperationPlanner.initialize(pp.operation_actions)
    print(operation_planner)

    I = [
        env_vars_opt.I.loc[
            df.index[0] + pd.Timedelta(days=n_day) : df.index[0]
            + pd.Timedelta(days=n_day + 1)
        ]
        .resample(f"{10}min", origin="start")
        .interpolate()
        for n_day in range(pp.optim_window_days)
    ]
    int_dec_vars_list = operation_planner.generate_decision_series(I)

    # Generate and save visualization of operation mode change candidates
    # save_figure(
    #     plot_op_mode_change_candidates(I_series=df["I"], pp=pp),
    #     figure_name="op_mode_change_candidates",
    #     figure_path=base_output_path,
    #     formats=["html", "png", "svg"],
    # )

    # 4. Initialize problem instance for one candidate
    candidate_idx: int = 0
    int_dec_vars = int_dec_vars_list[candidate_idx]

    problem = Problem(
        int_dec_vars=int_dec_vars,
        # Planner layer should get time imprecise weather forecasts
        env_vars=env_vars_opt,
        real_dec_vars_update_period=pp.real_dec_vars_update_period,
        model=model,
        initial_dec_vars_values=pp.initial_dec_vars_values,
        sample_time_ts=pp.sample_time_ts,
        
        store_x=False,
        store_fitness=False,
    )

    prob = pg.problem(problem)
    
    # 5. Build archipielago
    archi = pg.archipelago()
    
    results_dict: dict = {}

    for pop_idx, pop_size in enumerate(pop_sizes):
        n_gen = max_n_obj_fun_evals // pop_size
        # Generate the same intial population for all algorithms
        logger.info(f"Started generating initial population {pop_size=}")
        pop0 = pg.population(prob, size=pop_size)
        logger.info("Initial population generated")
        
        for algo_id, algo_params in algorithms_to_eval.items():
            if algo_id == "gaco":
                if pop_size <= problem.size_dec_vector:
                    logger.warning(f"Skipping {algo_id} with pop_size={pop_size} <= {problem.size_dec_vector}")
                    continue
                
            if algo_id == "sade":
                if pop_size < 7:
                    logger.warning(f"Skipping {algo_id} with pop_size={pop_size} < 7")
                    continue
            
            if algo_id in ["sea"]:
                if pop_idx > 0:
                    continue
                pop_size_ = 1
                
                # Only one individual, so as many generations as obj fun evals available
                n_gen_ = max_n_obj_fun_evals
                algo_params["gen"] = n_gen_
                
            elif algo_id == "simulated_annealing":
                n_gen_ = n_gen
                pop_size_ = pop_size
                algo_params.update({
                    "bin_size": pop_size_,
                    "n_T_adj": n_gen_,   
                })
                
            else:
                # Defaults
                n_gen_ = n_gen
                pop_size_ = pop_size
                algo_params["gen"] = n_gen_
                
            results_dict[f"{algo_id}_pop_{pop_size_}_gen_{n_gen_}"] = {
                "x0": pop0.champion_x,
                "fitness0": pop0.champion_f,
                "pop_size": pop_size_,
                "n_gen": n_gen_,
            }
            
            algo = pg.algorithm(getattr(pg, algo_id)(**algo_params))
            # Does this make the memory footprint grow too much?
            # No, it also grows when deactivated
            algo.set_verbosity(1) 
            
            archi.push_back(
                # Setting use_pool=True results in ever-growing memory footprint for the sub-processes
                # https://github.com/esa/pygmo2/discussions/168#discussioncomment-10269386
                pg.island(udi=pg.mp_island(use_pool=False), algo=algo, pop=copy.deepcopy(pop0), )
            )
            
    # 6. Evaluate the archipielago
    archi.evolve()
    print(archi)

    start_time = time.time()
    while archi.status == pg.evolve_status.busy:
        time.sleep(5)
        print(f"Elapsed time: {time.time() - start_time:.0f}")
        # print(f"Current evolution results | Best fitness: {pop_current.champion_f[0]}, \nbest decision vector: {pop_current.champion_x}")
    metadata["evaluation_time"] = int(time.time() - start_time)
    print(f"Completed evolution! Took {metadata['evaluation_time'] - start_time:.0f} seconds") 

    # 7. Generate results
    # For now just extract evolved populations
    algo_logs = []
    algo_ids = []
    for isl, result_key in zip(archi, results_dict.keys()):
        results_dict[result_key]["x"] = isl.get_population().champion_x
        results_dict[result_key]["fitness"] = isl.get_population().champion_f
        algo_id = extract_prefix(result_key)
        
        algo_ids.append(algo_id)
        algo_logs.append( isl.get_algorithm().extract( getattr(pg, algo_id) ).get_log() )
                
    # 8. Save results
    export_algo_comparison(
        results_dict=results_dict, output_path=base_output_path,
        metadata=metadata, problem_params=problem_params,
        algo_logs=algo_logs, algo_ids=algo_ids, table_ids=list(results_dict.keys())
    )
    
    
    
    
if __name__ == "__main__":
    main()