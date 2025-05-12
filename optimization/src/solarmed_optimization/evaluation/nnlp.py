from typing import Optional
import copy
from pathlib import Path
import time
from dataclasses import asdict

from loguru import logger
import numpy as np
import pandas as pd
import pygmo as pg
from tqdm.auto import tqdm

from solarmed_optimization import AlgoParams, ProblemData, ProblemsEvaluationParameters, OpPlanActionType
from solarmed_optimization.problems import BaseNlpProblem
from solarmed_optimization.problems.nnlp import OperationPlanResults
from solarmed_optimization.utils import select_best_alternative
from solarmed_optimization.utils.initialization import initialize_problem_instances_nNLP
from solarmed_optimization.utils.operation_plan import build_archipielago
from solarmed_optimization.utils.serialization import get_fitness_history


def evaluate_problems(problems: list[BaseNlpProblem], algo_params: AlgoParams, problems_eval_params: ProblemsEvaluationParameters, action: OpPlanActionType) -> OperationPlanResults:
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

    # TODO: Here x should be providable from a previous evaluation
    x = [None] * len(problems)
    fitness = [None] * len(problems)
    fitness_history = [None] * len(problems)
    droped_problem_idxs = []
    kept_problem_idxs = np.arange(len(problems))

    progress_bar = tqdm(range(problems_eval_params.n_updates), desc="Candidate problems evaluation", leave=True, position=2)

    for update_idx in progress_bar:

        # Evaluate problems for the update
        yet_to_eval_idxs = kept_problem_idxs
        batch_idx=1
        # log_header_str = f"{date_str} | Op.Plan - {action} | Evaluation step {update_idx+1}/{problems_eval_params.n_updates} | Active problems {len(kept_problem_idxs)}/{len(problems)}"

        while len(yet_to_eval_idxs) > 0:

            batch_size = min(problems_eval_params.max_n_parallel_problems, len(yet_to_eval_idxs))

            progress_bar.set_postfix(
                {"Active problems": f"{len(kept_problem_idxs)}/{len(problems)}"},
                {"Batch": f"{batch_idx}/{len(yet_to_eval_idxs)//batch_size+1}"},
            )

            archi = build_archipielago(
                problems=[problems[idx] for idx in yet_to_eval_idxs][:batch_size],
                algo_params=algo_params,
                x0=[x[idx] for idx in yet_to_eval_idxs[:batch_size]],
                fitness0=[fitness[idx] for idx in yet_to_eval_idxs[:batch_size]],
            )
            # logger.info(f"{log_header_str} | Initialized archipelago of problems of size {batch_size}")

            # Add initial fitness to fitness history
            for idx, isl in enumerate(archi):
                problem_idx = yet_to_eval_idxs[idx]
                fitness_history[problem_idx] = update_fitness_history(isl, fitness_history[problem_idx], initial=True)

            # start_time2 = time.time()
            archi.evolve()
            # logger.info(archi)
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
            # logger.info(f"{log_header_str} | Completed evolution of batch {batch_idx}/{len(yet_to_eval_idxs)//batch_size+1}!. Took {int(time.time() - start_time2):.0f} seconds") 
            batch_idx+=1

        # Retain only the best performing problems
        fitness_current_update = np.array(copy.deepcopy(fitness))
        fitness_current_update[droped_problem_idxs] = np.nan
        kept_problem_idxs, drop_idxs = problems_eval_params.update_problems(fitness_current_update)
        droped_problem_idxs += drop_idxs

    evaluation_time = int(time.time() - start_time)
    longest_problem_x_idx = np.argmax([len(x_) for x_ in x])
    len_longest_x = len(x[longest_problem_x_idx])
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
        # environment_df = problems[0].env_vars.to_dataframe(),
        evaluation_time=evaluation_time,
        algo_params=algo_params,
        problems_eval_params=problems_eval_params,
    )

    # logger.info(f"Completed evolution process! Took {evaluation_time/60:.0f} minutes") 

    return op_plan_results


def evaluate_operation_plan_layer(
    problem_data: ProblemData,
    action: OpPlanActionType,
    uncertainty_factor: float = 0.,
    stored_results: Optional[Path] = None,
    debug_mode: bool = False,
) -> tuple[list[OperationPlanResults], BaseNlpProblem, int]:

    if uncertainty_factor > 0:
        unc_factors = [uncertainty_factor, 0, -uncertainty_factor]
    else:
        unc_factors = [0]

    op_plan_results_list: list[OperationPlanResults] = []

    if debug_mode:
        algo_params = AlgoParams(max_n_obj_fun_evals=10,)
        problems_eval_params = ProblemsEvaluationParameters(
            drop_fraction=0.5,
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            n_obj_fun_evals_per_update=5
        )
    else:
        # Set default values for the algorithm and evaluation parameters
        algo_params = AlgoParams()
        # TODO: Here, except for the first evaluation, less function evaluations should be used
        # since we will provide the best solution from the previous evaluation
        problems_eval_params = ProblemsEvaluationParameters(n_updates=3, drop_fraction=0.5)

    progress_bar = tqdm(
        unc_factors,
        desc=f"Op.Plan - {action} | Scenarios evaluation",
        leave=True,
        position=1    # Position 1, main loop is at position 0
    )

    for scenario_idx, unc_factor in enumerate(progress_bar):
        progress_bar.set_postfix({"Uncertainty factor": f"{unc_factor:.2f}"})

        # Modify environment
        problem_data_copy = copy.deepcopy(problem_data)
        problem_data_copy.df["I"] = problem_data_copy.df["I"] * (1 + np.random.rand(len(problem_data.df)) * unc_factor)

        problems = initialize_problem_instances_nNLP(
            problem_data=problem_data_copy,
            store_x=False,
            store_fitness=False,
            log=debug_mode
        )
        if stored_results is not None:
            # Skip evaluting the layer if results exists
            date_str = list(asdict(problems[0].env_vars).values())[0].index[0].strftime("%Y%m%d")
            try:
                op_plan_results = OperationPlanResults.initialize(
                    input_path=stored_results,
                    date_str=date_str,
                    action=action,
                    scenario_idx=scenario_idx,
                )
            except KeyError:
                logger.info(f"Stored results provided but not found for action {action} in {stored_results}, falling back to computing layer")
                stored_results = None
            else:
                # Check retrieved results params match the current ones
                # TODO: We should check that problem_params, algo_params, problem_eval_params are the same
                # assert unc_factor == op_plan_results.uncertainty_factor, f"Uncertainty factor {unc_factor} does not match stored results {op_plan_results.uncertainty_factor}"
                logger.info(f"Retrieved previously computed solutions for action {action} in {stored_results}")

        if stored_results is None:
            op_plan_results = evaluate_problems(
                problems,
                algo_params=algo_params,
                problems_eval_params=problems_eval_params,
                action=action,
            )
            op_plan_results.evaluate_best_problem(problems=problems, model=problem_data_copy.model)
            op_plan_results.scenario_idx = scenario_idx
            op_plan_results.problem_params = problem_data_copy.problem_params

        op_plan_results_list.append(op_plan_results)

    fitness_df = pd.concat([op_plan_results.fitness for op_plan_results in op_plan_results_list], axis=1)
    # Choose best alternative
    best_alternative_idx, _ = select_best_alternative(fitness_df)

    return op_plan_results_list, problems[best_alternative_idx], best_alternative_idx