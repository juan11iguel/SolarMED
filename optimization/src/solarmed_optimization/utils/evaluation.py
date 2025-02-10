from dataclasses import asdict
from datetime import datetime
from typing import Literal
from iapws import IAPWS97 as w_props
import numpy as np
import pandas as pd
from loguru import logger

from solarmed_modeling.fsms.med import FsmInputs as MedFsmInputs
from solarmed_modeling.fsms.sfts import FsmInputs as SfTsFsmInputs
from solarmed_modeling.solar_med import SolarMED
from solarmed_modeling.thermal_storage import thermal_storage_two_tanks_model

from solarmed_optimization import (DecisionVariables, EnvironmentVariables, MedMode,
                                   ProblemData, SfTsMode, med_fsm_inputs_table, sfts_fsm_inputs_table,)
from solarmed_optimization.utils import (decision_vector_to_decision_variables,
                                         add_bounds_to_dataframe,
                                         add_dec_vars_to_dataframe, resample_timeseries_range_and_fill)
from solarmed_optimization.utils.operation_plan import get_start_and_end_datetimes
from solarmed_optimization.problems import BaseNlpProblem, BaseMinlpProblem


def evaluate_idle_thermal_storage(Tamb: pd.Series, dt_span: tuple[datetime, datetime], model: SolarMED,
								  sample_time: int = 3600, ) -> tuple[np.ndarray[float], np.ndarray[float]]:
	"""Evaluate the thermal storage when the only factor are the thermal losses
	to the environment (idle operation). This function evaluates the thermal 
	storage with the given sample time, but only returns the temperature profile
	at the last step.

	Args:
		sample_time (int): Time precision of the steps to evaluate
		Tamb (pd.Series): Ambient temperature
		dt_span (tuple[datetime, datetime]): Start and end datetimes to evaluate
		model (_type_, optional): Complete system model instance, used to extract
		initial temperatures and model parameters.

	Returns:
		tuple[np.ndarray[float], np.ndarray[float]]: Final temperature profile for the
		hot and cold tanks.
	"""

	Tamb = resample_timeseries_range_and_fill(Tamb, sample_time=sample_time, dt_span=dt_span)
	elapsed_time_sec = (dt_span[1] - dt_span[0]).total_seconds()

	N = int( elapsed_time_sec // sample_time )
	Tts_h = model.Tts_h
	Tts_c = model.Tts_c

	wprops = (w_props(P=1.5, T=Tts_h[1]+273.15),
			  w_props(P=1.5, T=Tts_c[1]+273.15))

	for idx in range(N):

		Tts_h, Tts_c = thermal_storage_two_tanks_model(
			Ti_ant_h=Tts_h, Ti_ant_c=Tts_c,  # [ºC], [ºC]
			Tt_in=0,  # ºC
			Tb_in=0,  # ºC
			Tamb=Tamb.iloc[idx],  # ºC

			qsrc=0,  # m³/h
			qdis=0,  # m³/h

			model_params=model.model_params.ts,
			water_props=wprops,
			sample_time=sample_time,
		)

	remaining_time = elapsed_time_sec - N * sample_time
	if remaining_time <= 0:
		return Tts_h, Tts_c
	else:
		return thermal_storage_two_tanks_model(
			Ti_ant_h=Tts_h, Ti_ant_c=Tts_c,  # [ºC], [ºC]
			Tt_in=0,  # ºC
			Tb_in=0,  # ºC
			Tamb=Tamb.iloc[-1],  # ºC

			qsrc=0,  # m³/h
			qdis=0,  # m³/h

			model_params=model.model_params.ts,
			water_props=wprops,
			sample_time=remaining_time,
		)

def evaluate_model(model: SolarMED,
                   dec_vars: DecisionVariables,
                   env_vars: EnvironmentVariables,
                   n_evals_mod: int,
                   mode: Literal["optimization", "evaluation"] = "optimization",
                   model_dec_var_ids: list[str] = None,
                   df_mod: pd.DataFrame = None,
                   df_start_idx: int = None) -> pd.DataFrame | float:
    """ Evaluate the model for a given decision vector and environment variables
        n_evals_mod is the number of model evaluations, whose value depends on what
        is being evaluated:
        - If mode is optimization, n_evals_mod should be the number of model evaluations in the optimization window 
        (optim_window_time // sample_time_mod)
        - If mode is evaluation, n_evals_mod should be the number of model evaluations in one optimization step 
        (sample_time_opt // sample_time_mod)
        - Though an arbitrary number of evaluations can be performed, make sure that `n_evals_mod` is lower or equal 
        to the number of elements in the decision vector and environment variables.
    """
    if mode not in ["optimization", "evaluation"]:
        raise ValueError(f"Invalid mode: {mode}")
    
    # if mode == "optimization":
    #     assert model_dec_var_ids is not None, "`model_dec_var_ids` is required if `mode` is set to 'optimization'"

    # if df_mod is None and mode == "evaluation":
    #     df_mod = model.to_dataframe()

    if mode == "optimization":
        fitness: np.ndarray[float] = np.zeros((n_evals_mod, ))
        if model_dec_var_ids is not None:
            ics: np.ndarray[float] = np.zeros((n_evals_mod, len(model_dec_var_ids)))
        else:
            ics = None

    # dec_var_ids = list(asdict(dec_vars).keys())
    for step_idx in range(n_evals_mod):
        dv: DecisionVariables = dec_vars.dump_at_index(step_idx)
        ev: EnvironmentVariables = env_vars.dump_at_index(step_idx)

        # print(f"{dv.med_vac_state=}")

        # Get the MED FSM inputs for the current MED mode and state
        med_fsm_inputs: MedFsmInputs = med_fsm_inputs_table[ (MedMode(dv.med_mode), model.med_state) ]
        sfts_fsm_inputs: SfTsFsmInputs = sfts_fsm_inputs_table[ (SfTsMode(dv.sfts_mode), model.sf_ts_state) ]

        # TODO: Add here some low-level control/validation
        # - qts_src should be zero if Tsf_out is below Tts_c_b? Tts_h_t? Which temperature should be the threshold?
        Tsf_out = model.Tsf_out if model.Tsf_out is not None else 0
        if Tsf_out < model.Tts_h[1]:
            dv.qts_src = 0.0

        model.step(
            # Decision variables
            ## Thermal storage
            qts_src = dv.qts_src * sfts_fsm_inputs.ts_active,

            ## Solar field
            qsf = dv.qsf * sfts_fsm_inputs.sf_active,

            ## MED
            qmed_s = dv.qmed_s * med_fsm_inputs.med_active,
            qmed_f = dv.qmed_f * med_fsm_inputs.med_active,
            Tmed_s_in = dv.Tmed_s_in,
            Tmed_c_out = dv.Tmed_c_out,
            med_vacuum_state = med_fsm_inputs.med_vacuum_state,

            ## Environment
            I=ev.I,
            Tamb=ev.Tamb,
            Tmed_c_in=ev.Tmed_c_in,
            wmed_f=ev.wmed_f if ev.wmed_f is not None else None,

            # Additional parameters
            compute_fitness=True if mode == "evaluation" else False
        )

        if mode == "optimization":
            # Inequality contraints, decision variables should be the same after model evaluation: |dec_vars-dec_vars_model| < tol
            # TODO: Fix this after change of Med decision variable to an indirect one
            # ics[step_idx, :] = None
            # ics[step_idx, :] = compute_dec_var_differences(dec_vars=asdict(dv), 
            #                                                model_dec_vars=model.model_dump(include=model_dec_var_ids),
            #                                                model_dec_var_ids=model_dec_var_ids)
            fitness[step_idx] = model.evaluate_fitness_function(
                cost_e=ev.cost_e,
                cost_w=ev.cost_w,
                objective_type='minimize'
            )
        if mode == "evaluation":
            df_mod = model.to_dataframe(df_mod, )

    if mode == "optimization":
        return np.sum(fitness), ics.mean(axis=0) if ics is not None else None
    elif mode == "evaluation":
        if df_start_idx is not None:
            df_mod.index = pd.RangeIndex(start=df_start_idx, stop=len(df_mod)+df_start_idx)
        return df_mod#, ics # ic temporary to validate


def evaluate_model_multi_day(model: SolarMED,
                             dec_vars: DecisionVariables,
                             env_vars: EnvironmentVariables,
                             evaluation_range: tuple[datetime, datetime],
                             mode: Literal["optimization", "evaluation"] = "optimization",
                             dec_var_int_ids: list[str] = None,
                             sample_time_ts: int = None,
                             debug_mode: bool = False) -> pd.DataFrame | float:
    # TODO: Tener a esta función llamando a evaluate_model es un poco chapuza, 
    # la idea sería modificar evaluate_model para que sea válida tanto como para 
    # días individuales como para múltiples días
    
    operation_end0: datetime = evaluation_range[0]
    fitness_total = 0
    n_days = (evaluation_range[1] - evaluation_range[0]).days +1
    df_mod = None
    if mode == "evaluation":
        df_columns = model.to_dataframe().columns
    
    env_vars_index = list(asdict(env_vars).values())[0].index
    for day_idx in range(n_days):
		# day_idx = 0
        day = env_vars_index[0].day + day_idx
        # Compute operation start and end datetimes for the current day using integer decision variables
        daily_data = []
        for var_id in dec_var_int_ids:
            var_values = getattr(dec_vars, var_id)
            daily_data.append(
                var_values[(var_values.index.day >= day) & (var_values.index.day < day+1)]
            )
        operation_start, operation_end = get_start_and_end_datetimes(series=daily_data)
        
        if debug_mode:
            logger.info(f"{day_idx=} | {operation_start=}, {operation_end=}")

        # Evaluate thermal storage until start of operation
        Tts_h0 = model.Tts_h
        Tts_c0 = model.Tts_c
        if debug_mode:
            logger.info(f"{day_idx=} | Before idle evaluation: {model.Tts_h=}, {model.Tts_c=}")
        Tts_h, Tts_c = evaluate_idle_thermal_storage( 
            sample_time=sample_time_ts,
            Tamb=env_vars.Tamb,
            dt_span=(operation_end0, operation_start),
            model=model
        )
        if mode == "evaluation":
            # Prepend data prior to operation start using nans and environment variables
            # Absolutamente terrible
            data = {
                **asdict(env_vars), 
                **{name: pd.Series(data=[val], index=[operation_end0], name=name) 
                    for name, val in zip(["Tts_h_t", "Tts_h_m", "Tts_h_b", "Tts_c_t", "Tts_c_m", "Tts_c_b"], [*Tts_h0, *Tts_c0])}
            }
            df = pd.DataFrame(data, index=[operation_end0], columns=df_columns)
            if df_mod is None:
                df_mod = df
            else:
                df_mod = pd.concat([df_mod, df])
        
        if debug_mode:
            logger.info(f"{day_idx=} | After idle evaluation: {Tts_h=}, {Tts_c=}")
        model.Tts_h = Tts_h
        model.Tts_c = Tts_c

        # Evaluate fitness for the current day
        dec_vars_day = dec_vars.dump_in_span(span=(operation_start, operation_end), return_format="values")
        env_vars_day = env_vars.dump_in_span(span=(operation_start, operation_end), return_format="series")
        env_vars_day_index = list(asdict(env_vars_day).values())[0].index
        # The model instance is updated within the evaluate_model function
        outputs = evaluate_model(model=model,
                                 dec_vars=dec_vars_day,
                                 env_vars=env_vars_day,
                                 n_evals_mod=env_vars_day_index.shape[0],
                                 mode=mode)
        
        if mode == "optimization":
            fitness = outputs[0]
            fitness_total += fitness
        
            if debug_mode:
                logger.info(f"{day_idx=} | {fitness=}, {fitness_total=}")        
        elif mode == "evaluation":
            # Concatenate active operation for current day to previous data
            # Also regularize the index
            df_mod = pd.concat([
                df_mod, 
                outputs.set_index(
                    env_vars_day_index
                    # pd.date_range(
                    # start=operation_start, 
                    # end=operation_end - env_vars_index.freq, 
                    # freq=env_vars_index.freq, )
                )
            ]).resample(env_vars_index.freq, origin="start").ffill()
        
        operation_end0 = operation_end
    
    if mode == "optimization":
        return fitness_total
    elif mode == "evaluation":
        return df_mod


def evaluate_optimization(df_sim: pd.DataFrame, pop: list[np.ndarray[float | int]], 
                          env_vars: EnvironmentVariables, problem: BaseMinlpProblem,
                          model: SolarMED, problem_data: ProblemData, idx_mod: int, 
                          best_idx: int = 0,) -> tuple[pd.DataFrame, pd.DataFrame, SolarMED]:
    
    pp = problem_data.problem_params
    ps = problem_data.problem_samples
    # model = problem_data.model # noo!! or maybe it did not matter? only god knows with python mutables
    
    
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
        df_hor[-1] = add_dec_vars_to_dataframe(df_hor[-1], dec_vars, df_idx=df_hor[-1].index[0])
        


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
    df_sim = add_dec_vars_to_dataframe(df_sim, dec_vars, df_idx=idx_mod)
    
    return df_hor, df_sim, model # model is already updated, but return it anyway


def evaluate_optimization_nlp(x: np.ndarray[float], # env_vars: EnvironmentVariables, 
                              problem: BaseNlpProblem, model: SolarMED) -> pd.DataFrame:
    
    # Sanitize decision vector, sometimes float values are negative even though they are basically zero (float precision?)
    x[np.abs(x) < 1e-6] = 0
    
    dec_vars: DecisionVariables = problem.decision_vector_to_decision_variables(x=x)
    df_hor =  evaluate_model_multi_day(
        model=model,
        dec_vars=dec_vars,
        env_vars=problem.env_vars,
        evaluation_range=problem.episode_range,
        mode="evaluation",
        dec_var_int_ids=problem.dec_var_int_ids,
        sample_time_ts=problem.sample_time_ts,
        
    )
    df_hor = add_dec_vars_to_dataframe(df_hor, dec_vars, df_idx=df_hor.index[0])
    
    # Change index
    # df_hor.index = list(asdict(dec_vars).values())[0].index
    
    return df_hor