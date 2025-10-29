# Evaluate operation plan - startup / shutdown
from typing import get_args, Optional
import copy
import argparse
from dataclasses import fields
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
import concurrent.futures
import datetime
from tqdm.auto import tqdm  # notebook compatible
# To export simulation dataframe
import shutil
import gzip

from solarmed_modeling.solar_med import InitialStates

from solarmed_optimization import (
    EnvironmentVariables,
    IntegerDecisionVariables, 
    DecisionVariables,
    OpPlanActionType,
    ProblemParameters,
    RealDecisionVariablesUpdatePeriod,
    InitialDecVarsValues,
    ProblemData,
    IrradianceThresholds,
    OptimToFsmsVarIdsMapping,
    OptimizationParams,
    ProblemsEvaluationParameters, 
    AlgoParams
)
from solarmed_optimization.problems import BaseNlpProblem
from solarmed_optimization.problems.nnlp import (
    OperationPlanResults,
    OperationOptimizationResults,
    batch_export,
)
from solarmed_optimization.evaluation.nnlp import (evaluate_operation_plan_layer,
                                                   evaluate_operation_optimization_layer)
from solarmed_optimization.utils import times_to_samples
from solarmed_optimization.utils.progress import update_bar_every
from solarmed_optimization.utils.initialization import problem_initialization
from solarmed_optimization.utils.evaluation import (evaluate_idle_thermal_storage,
                                                    evaluate_model)
from solarmed_optimization.utils.operation_plan import generate_update_datetimes



def get_optim_params_dict(debug_mode: bool = False) -> dict[str, OptimizationParams]:
    """
    Function to generate the optimization parameters for the different optimization problems.
    
    Returns
    -------
    dict[str, OptimizationParams]
        Dictionary with the optimization parameters for the different optimization problems.
    """

    optim_params_dict: dict[str, OptimizationParams] = {}

    algo_id: str = "sea"
    pop_size: int = 1

    # op_plan_first_run
    if debug_mode:
        algo_params = AlgoParams(algo_id=algo_id, pop_size=pop_size, max_n_obj_fun_evals=10,)
        problems_eval_params = ProblemsEvaluationParameters(
            drop_fraction=0.5,
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            n_obj_fun_evals_per_update=5
        )
    else:
        algo_params = AlgoParams(algo_id=algo_id, pop_size=pop_size, max_n_obj_fun_evals=500,)
        problems_eval_params = ProblemsEvaluationParameters(
            n_updates=3, 
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            drop_fraction=0.5
        )
    optim_params_dict["op_plan_first_run"] = OptimizationParams(
        algo_params=algo_params,
        problems_eval_params=problems_eval_params
    )

    # op_plan
    if debug_mode:
        algo_params = AlgoParams(algo_id=algo_id, pop_size=pop_size, max_n_obj_fun_evals=10,)
        problems_eval_params = ProblemsEvaluationParameters(
            drop_fraction=0.5,
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            n_obj_fun_evals_per_update=5
        )
    else:
        algo_params = AlgoParams(algo_id=algo_id, pop_size=pop_size, max_n_obj_fun_evals=250,)
        problems_eval_params = ProblemsEvaluationParameters(
            n_updates=3, 
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            drop_fraction=0.5
        )
    optim_params_dict["op_plan"] = OptimizationParams(
        algo_params=algo_params,
        problems_eval_params=problems_eval_params
    )

    # op_optim_standalone
    if debug_mode:
        algo_params = AlgoParams(algo_id=algo_id, pop_size=pop_size, max_n_obj_fun_evals=5,)
        problems_eval_params = ProblemsEvaluationParameters(
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            archipelago_topology="fully_connected",
            n_instances=3,
        )
    else:
        algo_params = AlgoParams(algo_id=algo_id, pop_size=pop_size, max_n_obj_fun_evals=100,)
        problems_eval_params = ProblemsEvaluationParameters(
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            archipelago_topology="fully_connected",
            n_instances=5,
        )
    optim_params_dict["op_optim_standalone"] = OptimizationParams(
        algo_params=algo_params,
        problems_eval_params=problems_eval_params
    )

    # op_optim_shared
    if debug_mode:
        algo_params = AlgoParams(algo_id=algo_id, pop_size=pop_size, max_n_obj_fun_evals=5,)
        problems_eval_params = ProblemsEvaluationParameters(
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            archipelago_topology="fully_connected",
            n_instances=1,
        )
    else:
        algo_params = AlgoParams(algo_id=algo_id, pop_size=pop_size, max_n_obj_fun_evals=100,)
        problems_eval_params = ProblemsEvaluationParameters(
            max_n_obj_fun_evals=algo_params.max_n_obj_fun_evals,
            archipelago_topology="fully_connected",
            n_instances=1,
        )
    optim_params_dict["op_optim_shared"] = OptimizationParams(
        algo_params=algo_params,
        problems_eval_params=problems_eval_params
    )

    return optim_params_dict

def problem_parameters_definition(action: OpPlanActionType, initial_states: Optional[InitialStates] = None) -> ProblemParameters:
    
    if action == "startup":
        if debug_mode:
            # Simplify the combinations to have a reduced number of them
            operation_actions = {
                # Day 1 -----------------------  # Day 2 -----------------------
                "sfts": [("startup", 2), ("shutdown", 1), ("startup", 1), ("shutdown", 1)],
                "med": [("startup", 2), ("shutdown", 1), ("startup", 1), ("shutdown", 1)],
            }
        else:
            operation_actions = {
                # Day 1 -----------------------  # Day 2 -----------------------
                "sfts": [("startup", 3), ("shutdown", 2), ("startup", 1), ("shutdown", 1)],
                "med": [("startup", 3), ("shutdown", 2), ("startup", 1), ("shutdown", 1)],
            }
        irradiance_thresholds = IrradianceThresholds(lower=300, upper=600)
    
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
        irradiance_thresholds = IrradianceThresholds(lower=50, upper=400)
        
    else:
        raise ValueError(f"Unknown action {action}. Options are: {get_args(OpPlanActionType)}")

    if debug_mode:
        # Increase the time period to evaluate the operation optimization
        # so we have to evaluate less iterations
        op_optim_eval_period = datetime.timedelta(minutes=30)
    else:
        op_optim_eval_period = ProblemParameters.op_optim_eval_period
        
    return ProblemParameters(
        optim_window_time=36 * 3600,  # 1d12h
        sample_time_opt=3600,  # 1h, In NLP-operation plan just used to resample environment variables
        operation_actions=operation_actions,
        initial_states=initial_states,
        real_dec_vars_update_period=RealDecisionVariablesUpdatePeriod(),
        # initial_dec_vars_values=None,
        on_limits_violation_policy="penalize",
        irradiance_thresholds=irradiance_thresholds,
        op_optim_eval_period=op_optim_eval_period
    )

def initialize_simulation(env_date_span_str: str, start_date: datetime.datetime, data_path: Path) -> tuple[ProblemData, EnvironmentVariables]:
    problem_params = problem_parameters_definition(action="startup", initial_states=InitialStates.initialize_from_inactive_state())
    selected_date_span = (start_date, None)#start_date + datetime.timedelta(days=problem_params.optim_window_days)]

    problem_data = problem_initialization(
        problem_params=problem_params, 
        date_str=env_date_span_str, 
        data_path=data_path,
        selected_date_span=selected_date_span,
    )

    # Environment variables used for the simulation
    # TODO: Add option to modify them so that they are not the same to the predictions
    env_vars = EnvironmentVariables.from_dataframe(
        problem_data.df, 
        cost_w=problem_params.env_params.cost_w,
        cost_e=problem_params.env_params.cost_e
    )
    
    return problem_data, env_vars

def update_problem_data(
    pdata: ProblemData, 
    start_dt: datetime.datetime, 
    action: OpPlanActionType, 
    sim_df: pd.DataFrame, 
    end_dt: Optional[datetime.datetime] = None
) -> ProblemData:
    
    # Update environment predictions
    # If not in simulation, this would be the opportunity to read new predictions
    if end_dt is None:
        pdata.df = pdata.df.loc[start_dt:]
    else:
        # Also trim end
        pdata.df = pdata.df.loc[start_dt:end_dt]
        
    # Ensure start_dt is compatible with data
    start_dt = pdata.df.index[0]
    
    pdata.problem_params = problem_parameters_definition(
        action=action, 
        initial_states=InitialStates.initialize_from_inactive_state(sim_df)
    )
    
    pp = pdata.problem_params
    pp.episode_duration = len(pdata.df) * pp.sample_time_mod
    pdata.problem_samples = times_to_samples(pp)
    
    # Get initial decision variables values
    init_dec_var_vals_dict = {}
    for fld in fields(InitialDecVarsValues):
        if fld.name in OptimToFsmsVarIdsMapping()._asdict().keys():
            val = all([sim_df[var_id].iloc[-1] > 0 for var_id in getattr( OptimToFsmsVarIdsMapping(), fld.name )])
        else:
            val = sim_df[fld.name].iloc[-1]
        val = 0 if np.isnan(val) else val
        
        init_dec_var_vals_dict[fld.name] = pd.Series([val], index=[start_dt])
    
    pp.initial_dec_vars_values = InitialDecVarsValues(**init_dec_var_vals_dict)
    
    return pdata

def get_op_plan_evaluation_datetime(
    action: OpPlanActionType, 
    computation_time: datetime.timedelta, 
    I: pd.Series,
    additional_dt: Optional[datetime.datetime] = None
) -> tuple[datetime.datetime, list[datetime.datetime]]:
    
    pp = problem_parameters_definition(action=action)
    op_action_tuple = list(pp.operation_actions.values())[0][0]
    shutdown_dts = generate_update_datetimes(
        I, n=op_action_tuple[1], action_type=action, 
        irradiance_thresholds=pp.irradiance_thresholds
    )
    if additional_dt is not None:
        shutdown_dts_ = [additional_dt] + shutdown_dts
    else:
        shutdown_dts_ = shutdown_dts
    
    return pd.Series(shutdown_dts_).min() - computation_time, shutdown_dts

def export_simulation_results(sim_df: pd.DataFrame, ) -> None: # output_path: Path compress: bool = False
    """
    Export simulation results DataFrame to an HDF5 file under key 'sim_df'.
    Preserves other keys. Handles .gz compressed files by decompressing before writing.
    
    Parameters:
    - sim_df: Simulation results DataFrame to export.
    - output_path: Base path for the result file (without suffix).
    - compress: Whether to gzip-compress the final output file.
    """
    # if not isinstance(output_path, Path):
    #     output_path = Path(output_path)

    h5_path = output_path.with_suffix(".h5")
    gz_path = output_path.with_suffix(".gz")

    # If only the compressed version exists, decompress it first
    if not h5_path.exists() and gz_path.exists():
        with gzip.open(gz_path, 'rb') as f_in, open(h5_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        logger.info(f"Decompressed existing {gz_path} to {h5_path} for updating")

    # Overwrite the "sim_df" key only, keep others
    sim_df.to_hdf(h5_path, key="sim_df", mode="a", format="table")

    if compress_results:
        with open(h5_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        h5_path.unlink()  # Delete the uncompressed version
        final_path = gz_path
    else:
        final_path = h5_path

    logger.info(f"Simulation results exported to {final_path}. Span {sim_df.index[0]} - {sim_df.index[-1]}")

def operation_plan_startup_block(
    problem_data: ProblemData, 
    env_vars: EnvironmentVariables, 
    sim_df: pd.DataFrame, 
    last_operation_end_dt: datetime.datetime,
    current_day_dt: datetime.datetime,
    uncertainty_factor: float,
    optim_params: OptimizationParams,
    results_df: Optional[pd.DataFrame] = None,
    stored_results: Optional[Path] = None,
) -> tuple[list[OperationPlanResults], BaseNlpProblem, pd.DataFrame, ProblemData]:
    """
    Evaluate operation plan - startup block
    """
    model = problem_data.model
    
    # Simulate idle system from last shutdown (if not first day) until start of operation plan evaluation
    current_day_dt = current_day_dt.replace(hour=0, minute=0, second=0)
    startup_eval_datetime, startup_candidate_dts = get_op_plan_evaluation_datetime(
        action="startup",
        computation_time = problem_data.problem_params.op_plan_startup_computation_time,
        # Only current operation day, otherwise the last day in the environment will be chosen
        I=env_vars.I.loc[env_vars.I.index > current_day_dt] 
    )
    
    _, _, out_df = evaluate_idle_thermal_storage(
        model=model, 
        dt_span=(last_operation_end_dt, startup_eval_datetime), 
        env_vars=env_vars, 
        df=sim_df,
        debug_mode=debug_mode,
    )
    assert out_df is not None
    sim_df = out_df
    
    if debug_mode:
        logger.debug(f"Current time: {sim_df.index[-1]} | After simulating from last day {operation_end=} up to {startup_eval_datetime=}")
    
    # Evaluate operation plan - startup
    # Update horizon end before updating available data
    horizon_end = (
        startup_eval_datetime + 
        datetime.timedelta(seconds=problem_data.problem_params.optim_window_time) + 
        problem_data.problem_params.system_shutdown_duration
    )
    problem_data = update_problem_data(problem_data, start_dt=startup_eval_datetime, end_dt=horizon_end, action="startup", sim_df=sim_df)
    
    # return problem_data

    op_plan_results_list, problem, best_problem_idx = evaluate_operation_plan_layer(
        problem_data,
        uncertainty_factor=uncertainty_factor, 
        action="startup",
        stored_results=stored_results,
        debug_mode=debug_mode,
        results_df=results_df,
        
        algo_params=optim_params.algo_params,
        problems_eval_params=optim_params.problems_eval_params,
    )
    batch_export(output_path, op_plan_results_list, compress=compress_results)
    
    logger.info(f"Out of {startup_candidate_dts} startup candidates, chosen startup datetime: {problem.operation_span[0]}")

    # Simulate up to operation optimization evaluation
    dt_span=(startup_eval_datetime, problem.operation_span[0])
    _, _, out_df = evaluate_idle_thermal_storage(
        model=model, 
        dt_span=dt_span, 
        env_vars=env_vars, 
        df=sim_df,
        debug_mode=debug_mode,
    )
    assert out_df is not None
    sim_df = out_df
    export_simulation_results(sim_df)
    
    if debug_mode:
        logger.debug(f"Current time: {sim_df.index[-1]} | After simulating from {startup_eval_datetime=} up to {problem.operation_span[0]=}")
    
    return op_plan_results_list, problem, sim_df, problem_data

def operation_plan_shutdown_block(
    problem_data: ProblemData, 
    sim_df: pd.DataFrame, 
    shutdown_eval_datetime: datetime.datetime,
    optim_params: OptimizationParams,
    shutdown_candidate_dts: list[datetime.datetime],
    results_df: Optional[pd.DataFrame] = None,
    stored_results: Optional[Path] = None,
) -> tuple[list[OperationPlanResults], BaseNlpProblem]:
    """
    Evaluate operation plan - shutdown block
    """
    
    # The layer is evaluated from its evaluation completion onwards
    start_dt = shutdown_eval_datetime + problem_data.problem_params.op_plan_shutdown_computation_time
    problem_data = update_problem_data(problem_data, start_dt=start_dt, action="shutdown", sim_df=sim_df)
    
    if debug_mode:
        print(f"Current time in problem_data.df: {problem_data.df.index[0]} | After updating problem data in order to evaluate op.plan-shutdown")
    
    # return problem_data

    op_plan_results_list, problem, best_problem_idx = evaluate_operation_plan_layer(
        problem_data=problem_data,
        uncertainty_factor=0, 
        action="shutdown",
        stored_results=stored_results,
        debug_mode=debug_mode,
        results_df=results_df,
        
        algo_params=optim_params.algo_params,
        problems_eval_params=optim_params.problems_eval_params,
    )
    batch_export(output_path, op_plan_results_list, compress=compress_results)
    
    logger.info(f"Out of {shutdown_candidate_dts} shutdown candidates, chosen shutdown datetime: {problem.operation_span[-1]}")
    
    return op_plan_results_list, problem

def operation_optimization_block(
    problem_data: ProblemData,
    int_dec_vars: IntegerDecisionVariables,
    current_sim_dt: datetime.datetime,
    stop_dt: datetime.datetime,
    sim_df: pd.DataFrame,
    optim_params: OptimizationParams,
    results_df: Optional[pd.DataFrame] = None,
    stored_results: Optional[Path] = None,
) -> tuple[OperationOptimizationResults, BaseNlpProblem, pd.DataFrame]:
    """
    Evaluate operation optimization
    """
    op_optim_computation_time = problem_data.problem_params.op_optim_computation_time
    op_optim_eval_period = problem_data.problem_params.op_optim_eval_period
    model = problem_data.model
    
    # Calculate the number of iterations
    total_iterations = int((stop_dt - (current_sim_dt+op_optim_eval_period)).total_seconds() / op_optim_eval_period.total_seconds())+1

    if total_iterations < 1:
        raise ValueError("Operation optimization evaluation period is greater than the provided time span")

    with tqdm(total=total_iterations, desc="Operation Optimization") as pbar:
        while current_sim_dt + op_optim_eval_period < stop_dt:
            pbar.set_postfix({"Current": current_sim_dt.strftime("%H:%M"), "Until": stop_dt.strftime("%H:%M")})
            
            # update_problem_data(...)

            # Evaluate
            op_optim_results, problem = evaluate_operation_optimization_layer(
                problem_data=problem_data,
                int_dec_vars=int_dec_vars,
                # From all the environment scenarios evaluated, take the first which is the nominal one
                # The rests are over or under estimations of the nominal predicted irradiance
                results_df=results_df,
                start_dt=current_sim_dt,
                
                debug_mode=debug_mode,
                algo_params=optim_params.algo_params,
                problems_eval_params=optim_params.problems_eval_params,
                stored_results=stored_results
            )
            # Export results
            op_optim_results.export(output_path, compress=compress_results)
            
            op_optim_eval_time = datetime.timedelta(seconds=int(op_optim_results.evaluation_time))
            # Setup new decision variables
            new_dec_vars = DecisionVariables.from_dataframe(op_optim_results.results_df)
            results_df = op_optim_results.results_df # This will update the initial solution for the next iteration

            if op_optim_eval_time > op_optim_eval_period:
                logger.error(f"Operation optimization evaluation time (={op_optim_eval_time/60:.1f} min) is greater than the evaluation period {op_optim_eval_period}")

            if op_optim_eval_time > op_optim_computation_time:
                # Computation time was underestimated, 
                # New decision vector needs to use old values in the excess period
                logger.warning(f"Operation optimization evaluation time {op_optim_eval_time} min is greater than the excepted time {op_optim_computation_time}")
                
                # Join the new decision variables with the old ones for the excess period
                excess_time = op_optim_eval_time - op_optim_computation_time
                dec_vars = (
                    dec_vars.dump_in_span(span=(current_sim_dt, current_sim_dt + excess_time), return_format="series").
                    append( new_dec_vars.dump_in_span(span=(current_sim_dt + excess_time, None), return_format="series") )
                )
            else:
                # Computation time was overestimated
                # Nothing needs to be done, the simulation still is performed with the old
                # decision variables since new values are generated only after the estimated computation time
                dec_vars = new_dec_vars
                
            # Simulate
            dt_span = (current_sim_dt , current_sim_dt + op_optim_eval_period)
            sim_df = evaluate_model(
                model=model,
                mode="evaluation",
                dec_vars=dec_vars.dump_in_span(span=dt_span, return_format="series"),
                env_vars=problem.env_vars.dump_in_span(span=dt_span, return_format="series"),
                # problem.env_vars are resampled to model sample time, maybe env_vars too?
                df_mod=sim_df,
                debug_mode=debug_mode,
            )
            current_sim_dt += op_optim_eval_period
            pbar.update(1)
            
            if debug_mode:
                logger.debug(f"Current time: {sim_df.index[-1]} | After simulating span {dt_span=} | Operation span {problem.operation_span}")
    
    export_simulation_results(sim_df)

    return op_optim_results, problem, sim_df

# -----------------------------------------------------------------------------------------------------
def main(
    date_span: tuple[str, str], 
    data_path: Path, 
    env_date_span: tuple[str, str], 
    uncertainty_factor: bool, 
    operation_optimization_layer_enabled: bool
) -> None:
    logger.info(f"Evaluating SolarMED optimization for date span {date_span[0]}-{date_span[-1]}")
    executor = concurrent.futures.ThreadPoolExecutor()

    # Setup environment
    env_date_span_str: str = f"{env_date_span[0]}_{env_date_span[1]}"
    start_date = datetime.datetime.strptime(date_span[0], "%Y%m%d").replace(hour=0).astimezone(tz=datetime.timezone.utc)
    end_date = datetime.datetime.strptime(date_span[1], "%Y%m%d").replace(hour=23).astimezone(tz=datetime.timezone.utc)
    if date_span[0] == date_span[1]:
        all_dates = [start_date]
    else:
        all_dates = list(pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC'))
    problem_data, env_vars = initialize_simulation(env_date_span_str=env_date_span_str, start_date=start_date, data_path=data_path)
    model = problem_data.model
    model_step_time = datetime.timedelta(seconds=model.sample_time)
    env_df = problem_data.df.copy()
    # env_vars0 = copy.deepcopy(env_vars)
    operation_end: datetime.datetime = env_vars.I.index[0]
    sim_df = model.to_dataframe(index=operation_end)
    optim_params_dict = get_optim_params_dict(debug_mode)
    current_optim_results_df: pd.DataFrame | None = None # Initialize prior optimization results

    # Setup progress bar
    progress_bar = tqdm(all_dates, desc="SolarMED | Evaluating episode", unit="day", leave=True, position=0)
    # status_update_thread = threading.Thread(target=update_bar_every, args=[progress_bar, 20], daemon=True)
    # status_update_thread.start()
    pbar_future = executor.submit(update_bar_every, progress_bar, 20)
        
    for date in progress_bar:
        progress_bar.set_postfix({"Current date": date.strftime("%Y%m%d")})
        
        # 0. Setup environment
        # Update environment
        problem_data.df = env_df # Reset problem environment timeseries
        # env_vars = copy.deepcopy(env_vars0)
        env_vars = env_vars.dump_in_span(span=(operation_end, None), return_format="series")
        model.reset_fsms_cooldowns()
        current_sim_dt = operation_end

        # 1. Set operation start (integer part)
        optim_params_key = "op_plan_first_run" if current_optim_results_df is None else "op_plan"
        # Evaluate operation plan - startup
        op_plan_results_list, op_plan_problem, sim_df, problem_data = operation_plan_startup_block(
            problem_data=problem_data.copy(),
            env_vars=env_vars,
            sim_df=sim_df,
            last_operation_end_dt=operation_end,
            current_day_dt=date,   
            uncertainty_factor=uncertainty_factor,
            optim_params=optim_params_dict[optim_params_key],
            stored_results=stored_results,
            results_df=current_optim_results_df,
        )
        current_sim_dt = op_plan_problem.operation_span[0]
        current_optim_results_df=op_plan_results_list[0].results_df
        op_optim_eval_datetime = op_plan_problem.operation_span[0] - problem_data.problem_params.op_optim_computation_time

        # Get operation plan - shutdown evaluation datetime
        shutdown_eval_datetime, shutdown_candidate_dts = get_op_plan_evaluation_datetime(
            action="shutdown",
            computation_time = 2 * problem_data.problem_params.op_plan_shutdown_computation_time,
            # Include current shutdown estimation to make sure we complete the 
            # evaluation of the op.plan layer-shutdown before it takes action
            additional_dt = op_plan_problem.operation_span[1],
            I=env_vars.I.loc[:date.replace(hour=23, minute=59, second=59)] # Only current day, otherwise the last day in the environment will be chosen
        )
        
        # 2. Set process variables (real part)
        if not operation_optimization_layer_enabled:
            # From operation start to shutdown evaluation
            dt_span = [op_plan_problem.operation_span[0], shutdown_eval_datetime]
            dec_vars = op_plan_problem.decision_vector_to_decision_variables(
                # uncertainty scenario to be used: 0
                x=op_plan_results_list[0].x.iloc[ op_plan_results_list[0].best_problem_idx ].values
            )
            
            sim_df = evaluate_model(
                model=model,
                mode="evaluation",
                dec_vars=dec_vars.dump_in_span(span=dt_span, return_format="series"),
                env_vars=env_vars.dump_in_span(span=dt_span, return_format="series"),
                df_mod=sim_df,
                debug_mode=debug_mode,
            )
            current_sim_dt = shutdown_eval_datetime
            
            if debug_mode:
                print(f"Current time: {sim_df.index[-1]} | After simulating from operation start ({op_optim_eval_datetime}) to shutdown evaluation ({shutdown_eval_datetime})")
        else:
            # Evaluate operation optimization
            op_optim_results, problem, sim_df = operation_optimization_block(
                problem_data=problem_data,
                int_dec_vars=op_plan_problem.int_dec_vars,
                results_df=current_optim_results_df,
                current_sim_dt=current_sim_dt,
                stop_dt=shutdown_eval_datetime,
                sim_df=sim_df,
                optim_params=optim_params_dict["op_optim_standalone"],
                stored_results=stored_results,
            )
            current_sim_dt = sim_df.index[-1]
            current_optim_results_df = op_optim_results.results_df
        
        # 3. Set operation end (integer part)
        # Evaluate operation plan - shutdown
        # Open a thread and offload the evaluation of the operation plan - shutdown to it
        asdasd
        future = executor.submit(
            operation_plan_shutdown_block,
            problem_data=problem_data, 
            sim_df=sim_df,
            shutdown_eval_datetime=shutdown_eval_datetime,
            shutdown_candidate_dts=shutdown_candidate_dts,
            optim_params=optim_params_dict["op_plan"],
            results_df=current_optim_results_df,
            stored_results=stored_results,
        )

        if not operation_optimization_layer_enabled:
            # Wait for the thread to finish
            op_plan_results_list, op_plan_problem = future.result()
            
            dt_span = [shutdown_eval_datetime + model_step_time, op_plan_problem.operation_span[-1]]
            dec_vars = op_plan_problem.decision_vector_to_decision_variables(
                x=op_plan_results_list[0].x.iloc[ op_plan_results_list[0].best_problem_idx ].values
            )
            # Alternative method
            # dec_vars = DecisionVariables.from_dataframe(op_plan_results_list[0].results_df)
        
        else:
            # 4. Set process variables (real part)
            
            # Keep evaluating operation optimization one optim.step at a time
            # until the op.plan - shutdown evaluation is completed
            period = problem_data.problem_params.op_optim_eval_period + model_step_time
            logger.debug(f"Performing up to {(shutdown_candidate_dts[0]-current_sim_dt)//period} operation optimization evaluations prior to shutdown evaluation completion")
            # not future.done() and 
            while (current_sim_dt + period) < shutdown_candidate_dts[0]:
                # Run the operation optimization layer
                op_optim_results, problem, sim_df = operation_optimization_block(
                    problem_data=problem_data,
                    int_dec_vars=op_plan_problem.int_dec_vars,
                    results_df=current_optim_results_df,
                    current_sim_dt=current_sim_dt + model_step_time,
                    stop_dt=(
                        # Add a second so it performs at least one iteration
                        current_sim_dt + period +  datetime.timedelta(seconds=1) 
                    ),
                    sim_df=sim_df,
                    optim_params=optim_params_dict["op_optim_shared"],
                    stored_results=stored_results,
                )
                current_sim_dt = sim_df.index[-1]
                current_optim_results_df = op_optim_results.results_df
            
            # Retrieve shutdown evaluation results
            op_plan_results_list, op_plan_problem = future.result()
            current_optim_results_df = op_plan_results_list[0].results_df
            logger.info("Completed evaluation of operation plan - shutdown. Updating operation end.")
            # jajaaaaaa
            
            # Update integer part and continue evaluating until operation end
            try:
                op_optim_results, problem, sim_df = operation_optimization_block(
                    problem_data=problem_data,
                    int_dec_vars=op_plan_problem.int_dec_vars,
                    results_df=current_optim_results_df,
                    current_sim_dt=current_sim_dt + model_step_time,
                    stop_dt=op_plan_problem.operation_span[-1],
                    sim_df=sim_df,
                    optim_params=optim_params_dict["op_optim_standalone"],
                    stored_results=stored_results,
                )
                current_sim_dt = sim_df.index[-1]
                current_optim_results_df = op_optim_results.results_df
            except ValueError:
                logger.info("Not enough time left in the operation to perform more operation optimization evaluations")
                pass

            dt_span = [current_sim_dt + model_step_time, op_plan_problem.operation_span[-1]]
            dec_vars = problem.decision_vector_to_decision_variables(
                x=op_optim_results.x.iloc[ op_optim_results.best_problem_idx ].values
            )
        
        # Simulate remaining of the operation + shutdown
        sim_df = evaluate_model(
            model=model,
            mode="evaluation",
            dec_vars=dec_vars.dump_in_span(span=dt_span, return_format="series"),
            env_vars=env_vars.dump_in_span(span=dt_span, return_format="series"),
            df_mod=sim_df,
            evaluate_shutdown=True,
            debug_mode=debug_mode,
        )
        current_sim_dt = sim_df.index[-1] # Do not use operation_span[-1] to include shutdown completion
        
        duplicates = sim_df.index.duplicated().sum()
        if duplicates > 0:
            logger.error(f"Duplicates in sim_df {duplicates} | {sim_df.index[sim_df.index.duplicated()]}")

        # Update operation end to actual value after simulation 
        # (plus a sample to avoid duplicates when evaluating idle thermal storage in next day/step)
        operation_end = sim_df.index[-1] + env_df.index.freq

        # Before continuing to the next day, export final day simulation results
        export_simulation_results(sim_df)

    executor.shutdown(cancel_futures=True, wait=False)
    logger.info(f"Completed evaluation of SolarMED optimization for date span {date_span[0]}-{date_span[-1]}")
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--date_span', nargs=2, default=["20180921", "20180922"], help="Date span to evaluate in YYYYMMDD format")
    parser.add_argument('--env_date_span', nargs=2, default=["20180921", "20180928"], help="Date span for environment data in YYYYMMDD format")
    parser.add_argument('--evaluation_id', type=str, default="dev")
    parser.add_argument('--base_path', type=str, default="/workspaces/SolarMED", help="Base path for the project")
    parser.add_argument('--data_path', type=str, default="optimization/data", help="Path to the environment data folder")
    # parser.add_argument('--full_export', action='store_true', help="Export full version of the results (increases file size)")
    parser.add_argument('--uncertainty_factor', type=float, default=0, help="Uncertainty factor [0,1) where 0 disables uncertainty and only one scenario is considered")
    parser.add_argument('--operation_optimization_layer', action='store_true', help="Evaluate operation optimization layer")
    parser.add_argument('--debug', action='store_true', help="Debug mode")

    args = parser.parse_args()
                
    stored_results: Optional[Path] = Path("/workspaces/SolarMED/optimization/results/20180921_20180922/results_nNLP_op_plan_eval_at_20250525T1902_dev.h5")

    file_id: str = f"results_nNLP_op_plan_eval_at_{datetime.datetime.now():%Y%m%dT%H%M}_{args.evaluation_id}"

    # Global variables
    debug_mode: bool = args.debug
    output_path: Path = Path(args.base_path) / f"optimization/results/{args.date_span[0]}_{args.date_span[1]}/{file_id}.h5"
    compress_results: bool = False

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

