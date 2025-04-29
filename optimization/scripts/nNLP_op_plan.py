# Evaluate operation plan - startup / shutdown

import argparse
import gzip
import shutil
from typing import Literal
import copy
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import pandas as pd
from loguru import logger
import pygmo as pg
import threading
import datetime
from tqdm import tqdm

from solarmed_optimization.utils.progress import update_bar_every
from solarmed_optimization import (
    EnvironmentVariables,
    ProblemParameters,
    ProblemSamples,
    RealDecisionVariablesUpdatePeriod,
    InitialDecVarsValues,
)
# from solarmed_optimization.problems.nlp import AlgoParams
from solarmed_optimization.utils.initialization import (
    problem_initialization,
    InitialStates,
    initialize_problem_instances_nNLP
)
        
debug_mode = False
        
# Temporary class definition --------------------------------------
from typing import Literal, Optional
import math
from dataclasses import dataclass
import pandas as pd
from solarmed_optimization.utils.evaluation import evaluate_optimization_nlp
from solarmed_optimization.utils.serialization import get_fitness_history
from solarmed_optimization.problems import BaseNlpProblem
from solarmed_modeling.solar_med import SolarMED

@dataclass
class OperationPlanResults:
    date_str: str # Date in YYYYMMDD format
    action: Literal["startup", "shutdown"] # Operation plan action identifier
    x: pd.DataFrame # Decision vector (columns) for each problem (rows)
    # int_dec_vars: list[pd.DataFrame] # Integer decision variables for each problem
    fitness: pd.Series # Fitness values for each problem
    fitness_history: pd.DataFrame # Optimization algorithm evolution fitness history (rows) for each problem (columns)
    environment_df: pd.DataFrame # Environment data
    best_problem_idx: Optional[int] = None # Index of the best performing problem
    results_df: Optional[pd.DataFrame] = None # Simulation timeseries results for the best performing problem
    evaluation_time: Optional[float] = None # Time, in seconds, taken to evaluate layer
    
    def __post_init__(self):
        if self.best_problem_idx is None:
            self.best_problem_idx = int(self.fitness.idxmin())
        
    def evaluate_best_problem(self, problems: list[BaseNlpProblem] | BaseNlpProblem, model: SolarMED) -> pd.DataFrame:
        
        self.results_df = evaluate_optimization_nlp(
            x=self.x.iloc[self.best_problem_idx].values, 
            problem=problems[self.best_problem_idx] if isinstance(problems, list) else problems,
            model=SolarMED(**model.dump_instance())
        )
        return self.results_df

@dataclass
class ProblemsEvaluationParameters:
    """
    Parameters for the problems evaluation process.
    """
    drop_fraction: float # Fraction of problems to drop per update (0 to 1)
    max_n_obj_fun_evals: int # Total maximum number of objective function evaluations
    max_n_parallel_problems: int = 50 # Maximum number of problems to evaluate in parallel
    n_updates: Optional[int] = None # Number of (problem drop) updates to perform
    n_obj_fun_evals_per_update: Optional[int] = None # Number of objective function evaluations between updates
    
    def __post_init__(self):
        assert self.n_updates is not None or self.n_obj_fun_evals_per_update is not None, "Either n_updates or n_obj_fun_evals_per_update must be provided"
        assert self.drop_fraction >= 0 and self.drop_fraction <= 1, "Fraction of problems to drop per update must be between 0 and 1"
        
        if self.n_obj_fun_evals_per_update is None:
            self.n_obj_fun_evals_per_update = self.max_n_obj_fun_evals // self.n_updates
        elif self.n_updates is None:
            self.n_updates = self.max_n_obj_fun_evals // self.n_obj_fun_evals_per_update
            
    def update_problems(self, problems_fitness: list[float]) -> tuple[list[int], list[int]]:
        """
        Drop the worst performing problems based on the drop_fraction.
        Returns the indices of the problems to keep and to drop.
        NaN values in problems_fitness are ignored in the decision process,
        but their positions are preserved in index calculations.
        """
                
        # Get the indices of non-NaN entries
        valid_indices = [i for i, val in enumerate(problems_fitness) if not math.isnan(val)]
        # Get the fitness values of non-NaN entries
        valid_fitness = [(i, problems_fitness[i]) for i in valid_indices]
        # Sort valid fitness values by value (ascending = worst first)
        valid_fitness_sorted = sorted(valid_fitness, key=lambda x: x[1])
        
        # Determine number to drop
        n_to_drop = int(len(valid_fitness_sorted) * self.drop_fraction)
        # Extract the global indices to drop
        drop_indices = [i for i, _ in valid_fitness_sorted[:n_to_drop]]
        # Extract the global indices to keep (remaining non-NaNs not in drop)
        keep_indices = [i for i in valid_indices if i not in drop_indices]
        
        return keep_indices, drop_indices
              
@dataclass
class AlgoParams:
    algo_id: str = "sea"
    max_n_obj_fun_evals: int = 1_000 # When debugging, change to a lower value
    max_n_logs: int = 300
    pop_size: int = 1
    
    params_dict: dict = None
    log_verbosity: int = None
    gen: int = None

    def __post_init__(self, ):

        if self.algo_id in ["gaco", "sga", "pso_gen"]:
            self.gen = self.max_n_obj_fun_evals // self.pop_size
            self.params_dict = {
                "gen": self.gen,
            }
        elif self.algo_id == "simulated_annealing":
            self.gen = self.max_n_obj_fun_evals // self.pop_size
            self.params_dict = {
                "bin_size": self.pop_size,
                "n_T_adj": self.gen
            }
        else:
            self.pop_size = 1
            self.gen = self.max_n_obj_fun_evals
            self.params_dict = { "gen": self.max_n_obj_fun_evals // self.pop_size }
        
        if self.log_verbosity is None:
            self.log_verbosity = math.ceil( self.gen / self.max_n_logs)


def get_initial_states(sim_df: Optional[pd.DataFrame] = None) -> InitialStates:
    
    if sim_df is not None:
        Tts_h = [sim_df.iloc[-1][f"Tts_h_{key}"] for key in ["t", "m", "b"]]
        Tts_c = [sim_df.iloc[-1][f"Tts_c_{key}"] for key in ["t", "m", "b"]]
    else:
        Tts_h=[90, 80, 70]
        Tts_c=[70, 60, 50]
        
    return InitialStates(Tts_h=Tts_h, Tts_c=Tts_c)

def problem_parameters_definition(action: str, initial_states: InitialStates) -> ProblemParameters:
    
    if action == "startup":
        if debug_mode:
            # Simplify the combinations to have a reduced number of them
            operation_actions = {
                # Day 1 -----------------------  # Day 2 -----------------------
                "sfts": [("startup", 2), ("shutdown", 2), ("startup", 1), ("shutdown", 1)],
                "med": [("startup", 2), ("shutdown", 2), ("startup", 1), ("shutdown", 1)],
            }
        else:
            operation_actions = {
                # Day 1 -----------------------  # Day 2 -----------------------
                "sfts": [("startup", 3), ("shutdown", 3), ("startup", 1), ("shutdown", 1)],
                "med": [("startup", 3), ("shutdown", 3), ("startup", 1), ("shutdown", 1)],
            }
    
    elif action == "shutdown":
        # Shutdown operation updates
        if debug_mode:
            operation_actions= {
                # Day 1 ---------------  # Day 2 -----------------------
                "sfts": [("shutdown", 2), ("startup", 1), ("shutdown", 1)],
                "med":  [("shutdown", 2), ("startup", 1), ("shutdown", 1)],
            }
        else:
            # Shutdown operation updates
            operation_actions= {
                # Day 1 ---------------  # Day 2 -----------------------
                "sfts": [("shutdown", 3), ("startup", 2), ("shutdown", 2)],
                "med":  [("shutdown", 3), ("startup", 2), ("shutdown", 2)],
            }
        
    else:
        raise ValueError(f"Unknown action {action}")

    return ProblemParameters(
        optim_window_time=36 * 3600,  # 1d12h
        sample_time_opt=3600,  # 1h, In NLP-operation plan just used to resample environment variables
        operation_actions=operation_actions,
        initial_states=initial_states,
        real_dec_vars_update_period=RealDecisionVariablesUpdatePeriod(),
        initial_dec_vars_values=InitialDecVarsValues(), # Defaults valid for startup, not shutdown
        on_limits_violation_policy="penalize",
    )

def build_archipielago(problems: list[BaseNlpProblem], algo_params: AlgoParams, pop0: list[pg.population] = None) -> pg.archipelago:
    
    if pop0 is not None:
        assert len(problems) == len(pop0), f"Number of initial populations ({len(pop0)}) should match number of problems ({len(problems)})"
    
    archi = pg.archipelago()
    for problem_idx, problem in enumerate(problems):

        # Initialize problem instance
        prob = pg.problem(problem)
        
        # Initialize population
        if pop0 is not None and pop0[problem_idx] is not None:
            pop = pop0[problem_idx]
        else:
            pop = pg.population(prob, size=algo_params.pop_size, seed=0)
        
        algo = pg.algorithm(getattr(pg, algo_params.algo_id)(**algo_params.params_dict))
        algo.set_verbosity( algo_params.log_verbosity )
        
        # 6. Build up archipielago
        archi.push_back(
            # Setting use_pool=True results in ever-growing memory footprint for the sub-processes
            # https://github.com/esa/pygmo2/discussions/168#discussioncomment-10269386
            pg.island(udi=pg.mp_island(use_pool=False), algo=algo, pop=pop, )
        )
        
    return archi

def evaluate_problems(problems: list[BaseNlpProblem], algo_params: AlgoParams, problems_eval_params: ProblemsEvaluationParameters, action: str):
    # This function should sequentially, build the archipielagos, 
    # evolve them, drop poorly performing problems and repeat until
    # the best performing problems are evolved completely
    
    def update_fitness_history(isl: pg.island, fit_his: pd.Series | None, initial: bool = False) -> pd.Series:
        
        if initial:
            fit = pd.Series(isl.get_population().champion_f[0], index=[0])
        else:
            fit = get_fitness_history(algo_params.algo_id, isl.get_algorithm() )
            fit_last = pd.Series(isl.get_population().champion_f[0], 
                                 index=[problems_eval_params.n_obj_fun_evals_per_update])
            if len(fit) == 0:
                fit = fit_last
            elif fit.index[-1] < problems_eval_params.n_obj_fun_evals_per_update:
                # Append the last fitness
                fit = pd.concat([fit, fit_last])
        
        if fit_his is None or len(fit_his) == 0:
            fit_his = fit
        else:
            fit.index = fit.index + fit_his.index[-1] +1
            fit_his = pd.concat([fit_his, fit])
            
        return fit_his
    
    date_str = list(asdict(problems[0].env_vars).values())[0].index[0].strftime("%Y%m%d")
    start_time = time.time()
    
    algo_params = AlgoParams(
        algo_id=algo_params.algo_id,
        max_n_obj_fun_evals=problems_eval_params.n_obj_fun_evals_per_update,
        pop_size=algo_params.pop_size,
        max_n_logs=algo_params.max_n_logs,
    )
    
    x = [None] * len(problems)
    fitness = [None] * len(problems)
    fitness_history = [None] * len(problems)
    droped_problem_idxs = []
    kept_problem_idxs = np.arange(len(problems))
    for update_idx in range(problems_eval_params.n_updates):
        
        # Evaluate problems for the update
        yet_to_eval_idxs = kept_problem_idxs
        batch_idx=1
        log_header_str = f"{date_str} Op.Plan - {action} | Evaluation step {update_idx+1}/{problems_eval_params.n_updates}"
        
        while len(yet_to_eval_idxs) > 0:
            
            batch_size = min(problems_eval_params.max_n_parallel_problems, len(yet_to_eval_idxs))
            
            archi = build_archipielago(
                problems=[problems[idx] for idx in yet_to_eval_idxs][:batch_size], 
                algo_params=algo_params,
                pop0=[x[idx] for idx in yet_to_eval_idxs[:batch_size]]
            )
            logger.info(f"{log_header_str} | Initialized archipelago of problems of size {batch_size}")
            
            # Add initial fitness to fitness history
            for idx, isl in enumerate(archi):
                problem_idx = yet_to_eval_idxs[idx]
                fitness_history[problem_idx] = update_fitness_history(isl, fitness_history[problem_idx], initial=True)
            
            start_time2 = time.time()
            archi.evolve()
            logger.info(archi)
            archi.wait_check()
            # while archi.status == pg.evolve_status.busy:
            #     time.sleep(5)
            #     print(f"Elapsed time: {time.time() - start_time:.0f}")
                # print(f"Current evolution results | Best fitness: {pop_current.champion_f[0]}, \nbest decision vector: {pop_current.champion_x}")
            
            # Update output objects
            for idx, isl in enumerate(archi):
                problem_idx = yet_to_eval_idxs[idx]
                x[problem_idx] = isl.get_population().champion_x
                fitness[problem_idx] = isl.get_population().champion_f[0]
                fitness_history[problem_idx] = update_fitness_history(isl, fitness_history[problem_idx])
            
            yet_to_eval_idxs = yet_to_eval_idxs[batch_size:]
            logger.info(f"{log_header_str} | Completed evolution of batch {batch_idx}/{len(yet_to_eval_idxs)//batch_size+1}!. Took {int(time.time() - start_time2):.0f} seconds") 
            batch_idx+=1
        
        # Retain only the best performing problems
        fitness_current_update = np.array(copy.deepcopy(fitness))
        fitness_current_update[droped_problem_idxs] = np.nan
        kept_problem_idxs, drop_idxs = problems_eval_params.update_problems(fitness_current_update)
        droped_problem_idxs += drop_idxs
    
    evaluation_time = int(time.time() - start_time)
    longest_problem_x_idx = np.argmax([len(x_) for x_ in x])
    len_longest_x = len(x[longest_problem_x_idx])
    best_problem_idx = np.argmin(fitness)
    # dec_vec = [ for x_, problem in zip(x, problems)] # Including integer part
    # x should be padded with nans to match the length of the longest problem
    # fitness_history should be padded with nans to match lengths
    op_plan_results = OperationPlanResults(
        date_str=date_str, # Date in YYYYMMDD format
        action=action,
        x = pd.DataFrame(
            np.array([np.pad(item, (0, len_longest_x - len(item)), constant_values=np.nan) for item in x]), 
            columns = [
                f"{var_id}_step_{step_idx:03d}"
                for var_id, num_steps in asdict(problems[longest_problem_x_idx].dec_var_updates).items() if var_id not in problems[0].dec_var_int_ids
                for step_idx in range(num_steps)
            ]
        ),
        # int_dec_vars = [problem.int_dec_vars.to_dataframe() for problem in problems],
        fitness = pd.Series(fitness),
        fitness_history = pd.concat(fitness_history, axis=1).sort_index(),
        environment_df = problems[0].env_vars.to_dataframe(),
        best_problem_idx=best_problem_idx,
        evaluation_time=evaluation_time
    )
    
    logger.info(f"Completed evolution process! Took {evaluation_time/60:.0f} minutes") 
    
    return op_plan_results

def evaluate_operation_plan_layer(date: datetime.datetime, env_date_span_str: str, initial_states: InitialStates, action: Literal["startup", "shutdown"], data_path: Path, uncertainty_factor: float = 0., ) -> list[OperationPlanResults]:
    
    if uncertainty_factor > 0:
        unc_factors = [uncertainty_factor, 0, -uncertainty_factor]
    else:
        unc_factors = [0]
        
    op_plan_results_list: list[OperationPlanResults] = []
    problem_params = problem_parameters_definition(action, initial_states=initial_states)
    selected_date_span = [date, date + datetime.timedelta(days=problem_params.optim_window_days)]
    problem_data = problem_initialization(
        problem_params=problem_params, 
        date_str=env_date_span_str, 
        data_path=data_path,
        selected_date_span=selected_date_span,
    )
    
    if debug_mode:
        algo_params = AlgoParams(max_n_obj_fun_evals=20,)
        problems_eval_params = ProblemsEvaluationParameters(
            drop_fraction=0.5,
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            n_obj_fun_evals_per_update=5
        )
    else:
        # Set default values for the algorithm and evaluation parameters
        algo_params = AlgoParams()
        problems_eval_params = ProblemsEvaluationParameters()
    
    for unc_factor in unc_factors:
        
        # Modify environment
        problem_data_copy = copy.deepcopy(problem_data)
        problem_data_copy.df["I"] = problem_data_copy.df["I"] * (1 + np.random.rand(len(problem_data.df)) * unc_factor)
        
        problems = initialize_problem_instances_nNLP(
            problem_data=problem_data,        
            store_x=False,
            store_fitness=False,
        )
        op_plan_results = evaluate_problems(problems, algo_params=algo_params, problems_eval_params=problems_eval_params, action=action)
        # op_plan_results.evaluate_best_problem(problems=problems, model=problem_data_copy.model)
        op_plan_results_list.append(op_plan_results)
    
    return op_plan_results_list

def evaluate_operation_optimization_layer() -> None:
    raise NotImplementedError("Operation optimization layer evaluation not implemented yet")

def main(date_span: tuple[str, str], data_path: Path, env_date_span: tuple[str, str], output_path: Path, uncertainty_factor: bool, operation_optimization_layer: bool) -> None:
    logger.info(f"Evaluating nNLP-operation plan optimization for date span {date_span[0]}-{date_span[-1]}")

    env_date_span_str: str = f"{env_date_span[0]}_{env_date_span[1]}"

    start_date = datetime.datetime.strptime(date_span[0], "%Y%m%d").replace(hour=0)
    end_date = datetime.datetime.strptime(date_span[1], "%Y%m%d").replace(hour=23)
    all_dates = list(pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC'))

    # TODO: Here we should initialize once the problem and the model, to then simulate it after each evaluation of the
    # optimization layers (operation plan and operation optimization).
    # The environment (irradiance) should be randomly modified to account for uncertainty if uncertainty_factor > 0
    # problem_data = problem_initialization(...)
    # problem = None
    
    with tqdm(total=len(all_dates), desc="Evaluating days", unit="day", leave=True, ) as pbar:
        # Start the parallel thread to update the main progress bar
        status_update_thread = threading.Thread(target=update_bar_every, args=[pbar, 20], daemon=True)
        status_update_thread.start()
        
        sim_df = None
        for date in all_dates:            
            # Evaluate operation plan - startup
            op_plan_results = evaluate_operation_plan_layer(
                date, 
                env_date_span_str, 
                initial_states = get_initial_states(sim_df), 
                uncertainty_factor=uncertainty_factor, 
                action="startup",
                data_path=data_path
            )
            
            # Evaluate operation optimization
            # if operation_optimization_layer:
            #     evaluate_operation_optimization_layer()
            
            # Evaluate operation plan - shutdown
            # initial_states = get_initial_states(sim_df)
            # op_plan_results = evaluate_operation_plan_layer(uncertainty_factor=uncertainty_factor, action="shutodwn")
            
            # Simulate until start of next action
            # This is where the operation optimization layer should take over and fill the gap
            # Here just assume the prediction is perfect and we will reach the next action
            # if problem is None:
            #     problem = initialize_problem_instances_nNLP(problem_data,)[0]
                
            # Simulate until next action
            # sim_df = # evaluate_optimization_nlp(
            #     x=self.x.iloc[self.best_problem_idx].values, 
            #     problem=problems[self.best_problem_idx] if isinstance(problems, list) else problems,
            #     model=SolarMED(**model.dump_instance())
            # )

            pbar.update(1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--date_span', nargs=2, default=["20180921", "20180921"], help="Date span to evaluate in YYYYMMDD format")
    parser.add_argument('--env_date_span', nargs=2, default=["20180921", "20180928"], help="Date span for environment data in YYYYMMDD format")
    parser.add_argument('--evaluation_id', type=str, default="")
    parser.add_argument('--base_path', type=str, default="/workspaces/SolarMED", help="Base path for the project")
    parser.add_argument('--data_path', type=str, default="optimization/data", help="Path to the environment data folder")
    parser.add_argument('--full_export', action='store_true', help="Export full version of the results (increases file size)")
    parser.add_argument('--uncertainty_factor', type=float, default=0, help="Uncertainty factor [0,1) where 0 disables uncertainty and only one scenario is considered")
    parser.add_argument('--operation_optimization_layer', action='store_true', help="Evaluate operation optimization layer")
    parser.add_argument('--debug', action='store_true', help="Debug mode")

    args = parser.parse_args()
        
    # Set the global variable
    debug_mode = args.debug
        
    # Manage paths
    file_id = f"results_nNLP_op_plan_eval_at_{datetime.datetime.now():%Y%m%dT%H%M}_{args.evaluation_id}"
    output_path = Path(args.base_path) / f"optimization/results/{args.date_span[0]}_{args.date_span[1]}/{file_id}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Call the main function with the parsed arguments
    main(
        date_span=args.date_span,       
        data_path=Path(args.base_path) / args.data_path,
        env_date_span=args.env_date_span,
        output_path=output_path,
        uncertainty_factor=args.uncertainty_factor,
        operation_optimization_layer=args.operation_optimization_layer,
    )