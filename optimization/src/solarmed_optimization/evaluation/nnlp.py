from typing import Optional
import copy
from pathlib import Path
import time
from dataclasses import asdict
import datetime

from loguru import logger
import numpy as np
import pandas as pd
import pygmo as pg
from tqdm.auto import tqdm

from solarmed_optimization import (
    AlgoParams, 
    ProblemData, 
    ProblemsEvaluationParameters, 
    OpPlanActionType,
    DecisionVariables,
    IntegerDecisionVariables
)
from solarmed_optimization.problems import BaseNlpProblem
from solarmed_optimization.problems.nnlp import OperationPlanResults, OperationOptimizationResults
from solarmed_optimization.utils import select_best_alternative, decision_vectors_to_dataframe
from solarmed_optimization.utils.initialization import (initialize_problem_instances_nNLP, 
                                                        initialize_problem_instance_NLP)
from solarmed_optimization.utils.operation_plan import build_archipielago
from solarmed_optimization.utils.serialization import get_fitness_history

def update_fitness_history(
    isl: pg.island,
    algo_id: str,
    n_obj_fun_evals_per_update: int,
    fit_his: Optional[pd.Series] = None, 
    initial: bool = False,
) -> pd.Series:

    if initial:
        fit = pd.Series(isl.get_population().champion_f[0], index=[0])
    else:
        fit = get_fitness_history(algo_id, isl.get_algorithm() )
        fit_last = pd.Series(isl.get_population().champion_f[0],
                                index=[n_obj_fun_evals_per_update])
        if len(fit) == 0:
            fit = fit_last
        elif fit.index[-1] < n_obj_fun_evals_per_update:
            # Append the last fitness
            fit = pd.concat([fit, fit_last])

    if fit_his is None or len(fit_his) == 0:
        fit_his = fit
    else:
        fit.index = fit.index + fit_his.index[-1] +1
        fit_his = pd.concat([fit_his, fit])

    return fit_his

def evaluate_problems(
    problems: list[BaseNlpProblem], 
    algo_params: AlgoParams, 
    problems_eval_params: ProblemsEvaluationParameters,
    x0: Optional[list[np.ndarray]] = None, 
    fitness0: Optional[list[float]] = None,
) -> tuple[str, pd.DataFrame, pd.Series, pd.DataFrame, int] :
    # date_str, x, fitness, fitness_history, evaluation_time
    
    # TODO: Find a way to have a common evaluate_problems(...) that can be used for both
    # evaluate_operation_optimization_problem and evaluate_operation_plan_problems
    date_str = list(asdict(problems[0].env_vars).values())[0].index[0].strftime("%Y%m%d")
    start_time = time.time()

    # TODO: Here x should be providable from a previous evaluation
    x = [x0_ for x0_ in x0] if x0 is not None else [None] * len(problems)
    fitness = [f0 for f0 in fitness0] if fitness0 is not None else [None] * len(problems)
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
                topology=problems_eval_params.archipelago_topology,
            )
            # logger.info(f"{log_header_str} | Initialized archipelago of problems of size {batch_size}")

            # Add initial fitness to fitness history
            for idx, isl in enumerate(archi):
                problem_idx = yet_to_eval_idxs[idx]
                fitness_history[problem_idx] = update_fitness_history(
                    isl, 
                    fit_his=fitness_history[problem_idx], 
                    algo_id=algo_params.algo_id,
                    n_obj_fun_evals_per_update=problems_eval_params.n_obj_fun_evals_per_update,
                    initial=True
                )
            for idx, isl in enumerate(archi):
                fitness_history[idx] = update_fitness_history(
                    isl, 
                    fit_his=fitness_history[idx], 
                    algo_id=algo_params.algo_id,
                    n_obj_fun_evals_per_update=problems_eval_params.n_obj_fun_evals_per_update,
                    initial=True
                )

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
                fitness_history[problem_idx] = update_fitness_history(
                    isl, 
                    fit_his=fitness_history[problem_idx],
                    algo_id=algo_params.algo_id,
                    n_obj_fun_evals_per_update=problems_eval_params.n_obj_fun_evals_per_update,
                )

            yet_to_eval_idxs = yet_to_eval_idxs[batch_size:]
            # logger.info(f"{log_header_str} | Completed evolution of batch {batch_idx}/{len(yet_to_eval_idxs)//batch_size+1}!. Took {int(time.time() - start_time2):.0f} seconds") 
            batch_idx+=1

        # Retain only the best performing problems
        fitness_current_update = np.array(copy.deepcopy(fitness))
        fitness_current_update[droped_problem_idxs] = np.nan
        
        # Add the last fitness to the fitness history
        new_fit_history = [
            pd.concat([
                fitness_history[idx], 
                pd.Series(
                    [fitness[idx]], 
                    index=[max(fitness_history[idx].index[-1] + 1, 
                           problems_eval_params.n_obj_fun_evals_per_update)]
                )
            ]) 
            if fitness[idx] < fitness_history[idx].iloc[-1] else fitness_history[idx]
            for idx in kept_problem_idxs
        ]
        for idx, new_fit_his in zip(kept_problem_idxs, new_fit_history):
            fitness_history[idx] = new_fit_his
            
        kept_problem_idxs, drop_idxs = problems_eval_params.update_problems(fitness_current_update)
        droped_problem_idxs += drop_idxs

    evaluation_time = int(time.time() - start_time)
    x = decision_vectors_to_dataframe(x, problems)
        
    fitness = pd.Series(fitness)
    fitness_history = pd.concat(fitness_history, axis=1).sort_index()
    
    return date_str, x, fitness, fitness_history, evaluation_time

def evaluate_operation_plan_problems(
    problems: list[BaseNlpProblem], 
    algo_params: AlgoParams, 
    problems_eval_params: ProblemsEvaluationParameters, 
    action: OpPlanActionType,
    x0: Optional[list[np.ndarray]] = None, 
    fitness0: Optional[list[float]] = None,
) -> OperationPlanResults:
    # This function should sequentially, build the archipielagos, 
    # evolve them, drop poorly performing problems and repeat until
    # the best performing problems are evolved completely

    algo_params = AlgoParams(
        algo_id=algo_params.algo_id,
        max_n_obj_fun_evals=problems_eval_params.n_obj_fun_evals_per_update,
        pop_size=algo_params.pop_size,
        max_n_logs=algo_params.max_n_logs,
    )
    
    date_str, x, fitness, fitness_history, evaluation_time = evaluate_problems(
        problems=problems, 
        algo_params=algo_params, 
        problems_eval_params=problems_eval_params, 
        x0=x0,
        fitness0=fitness0
    )
    
    return OperationPlanResults(
        date_str=date_str,
        action=action,
        x=x,
        fitness=fitness,
        fitness_history=fitness_history,
        evaluation_time=evaluation_time,
        algo_params=algo_params,
        problems_eval_params=problems_eval_params,
    )

def evaluate_operation_plan_layer(
    problem_data: ProblemData,
    action: OpPlanActionType,
    algo_params: AlgoParams,
    problems_eval_params: ProblemsEvaluationParameters,
    uncertainty_factor: float = 0.,
    results_df: Optional[pd.DataFrame] = None,
    stored_results: Optional[Path] = None,
    debug_mode: bool = False,
) -> tuple[list[OperationPlanResults], BaseNlpProblem, int]:

    if uncertainty_factor > 0:
        # Important, neutral must be the first element
        unc_factors = [0, uncertainty_factor, -uncertainty_factor]
    else:
        unc_factors = [0]

    op_plan_results_list: list[OperationPlanResults] = []

    progress_bar = tqdm(
        unc_factors,
        desc=f"Op.Plan - {action} | Scenarios evaluation",
        leave=True,
        position=1    # Position 1, main loop is at position 0
    )

    for scenario_idx, unc_factor in enumerate(progress_bar):
        progress_bar.set_postfix({"Uncertainty factor": f"{unc_factor:.2f}"})

        # Modify environment
        problem_data_copy = problem_data.copy()
        problem_data_copy.df["I"] = problem_data_copy.df["I"] * (1 + np.random.rand(len(problem_data.df)) * unc_factor)

        problems = initialize_problem_instances_nNLP(
            problem_data=problem_data_copy,
            store_x=False,
            store_fitness=False,
            log=debug_mode
        )
        
        # Setup initial decision variables from prior decision variables
        if results_df is None:
            x0 = None
        else:
            x0 = [
                problem.adapt_dec_vars_to_problem(
                    dec_vars0=DecisionVariables.from_dataframe(results_df),
                    return_dec_vector=True
                )
                for problem in problems
            ]
            
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
            op_plan_results = evaluate_operation_plan_problems(
                problems,
                algo_params=algo_params,
                problems_eval_params=problems_eval_params,
                action=action,
                x0=x0,
            )
            op_plan_results.evaluate_best_problem(problems=problems, model=problem_data_copy.model)
            op_plan_results.scenario_idx = scenario_idx
            op_plan_results.problem_params = problem_data_copy.problem_params

        op_plan_results_list.append(op_plan_results)

    fitness_df = pd.concat([op_plan_results.fitness for op_plan_results in op_plan_results_list], axis=1)
    # Choose best alternative
    best_alternative_idx, _ = select_best_alternative(fitness_df)
            
    return op_plan_results_list, problems[best_alternative_idx], best_alternative_idx

def evaluate_operation_optimization_problems(
    problem: BaseNlpProblem, 
    time_str: str,
    algo_params: AlgoParams, 
    problems_eval_params: ProblemsEvaluationParameters,
    x0: Optional[np.ndarray] = None, 
    fitness0: Optional[float] = None,
) -> OperationOptimizationResults:
    
    n_instances = problems_eval_params.n_instances
    
    date_str, x, fitness, fitness_history, evaluation_time = evaluate_problems(
        problems=[problem] * n_instances,
        algo_params=algo_params,
        problems_eval_params=problems_eval_params,
        x0=[x0]*n_instances,
        fitness0=[fitness0]*n_instances
    )
    
    return OperationOptimizationResults(
        date_str=date_str,
        time_str=time_str,
        x=x,
        fitness=fitness,
        fitness_history=fitness_history,
        evaluation_time=evaluation_time,
        algo_params=algo_params,
        problems_eval_params=problems_eval_params,
    )

def evaluate_operation_optimization_layer(
    problem_data: ProblemData,
    int_dec_vars: IntegerDecisionVariables,
    start_dt: datetime.datetime,
    algo_params: AlgoParams,
    problems_eval_params: ProblemsEvaluationParameters,
    results_df: Optional[pd.DataFrame] = None,
    stored_results: Optional[Path] = None,
    debug_mode: bool = False,
) -> tuple[OperationOptimizationResults, BaseNlpProblem]:

    problem_data_copy = problem_data.copy()
    
    problem = initialize_problem_instance_NLP(problem_data, int_dec_vars=int_dec_vars, start_dt=start_dt)

    # Setup initial decision variables from prior decision variables
    if results_df is None:
        x0 = None
    else:
        dec_vars = (
            DecisionVariables.from_dataframe(results_df).
            dump_in_span(span=(start_dt, None), return_format="series", align_first=True, resampling_method="nearest")
        )
        x0 = problem.decision_variables_to_decision_vector(dec_vars)
        problem.adapt_dec_vars_to_problem(
            dec_vars0=DecisionVariables.from_dataframe(results_df),
            return_dec_vector=True
        )
        
    if stored_results is not None:
        # Skip evaluting the layer if results exists
        date_str = problem.env_vars.get_date_str()
        time_str = f"{start_dt.strftime('%H_%M')}"
        try:
            op_optim_results = OperationOptimizationResults.initialize(input_path=stored_results, date_str=date_str, time_str=time_str)
        except KeyError:
            logger.info(f"Stored results provided but not found in {stored_results}, falling back to computing layer")
            stored_results = None
        else:
            # Check retrieved results params match the current ones
            # TODO: We should check that problem_params, algo_params, problem_eval_params are the same
            # assert unc_factor == op_optim_results.uncertainty_factor, f"Uncertainty factor {unc_factor} does not match stored results {op_optim_results.uncertainty_factor}"
            logger.info(f"Retrieved previously computed solutions in {stored_results}")

    if stored_results is None:
        op_optim_results = evaluate_operation_optimization_problems(
            problem,
            time_str=f"{start_dt.strftime('%H_%M')}",
            algo_params=algo_params,
            problems_eval_params=problems_eval_params,
            x0=x0,
        )
        op_optim_results.evaluate_best_problem(problems=problem, model=problem_data_copy.model)
        op_optim_results.problem_params = problem_data_copy.problem_params

    return op_optim_results, problem