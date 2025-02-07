from dataclasses import fields
from typing import get_args
from dataclasses import asdict
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger
from iapws import IAPWS97 as w_props
import pygmo as pg

from solarmed_modeling.solar_med import SolarMED
from solarmed_modeling.fsms import SolarMedState
from solarmed_modeling.thermal_storage import thermal_storage_two_tanks_model

from solarmed_optimization import (RealDecVarsBoxBounds, 
                                   DecisionVariablesUpdates,
                                   EnvironmentVariables,
                                   InitialDecVarsValues,
                                   DecisionVariables,
                                   IntegerDecisionVariables,
                                   RealDecisionVariablesUpdatePeriod,
                                   RealDecisionVariablesUpdateTimes,
                                   RealLogicalDecVarDependence)
from solarmed_optimization.utils import evaluate_model, resample_timeseries_range_and_fill
from solarmed_optimization.utils.operation_plan import get_start_and_end_datetimes

def generate_real_dec_vars_times(real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod, 
                                 int_dec_vars_list: list[IntegerDecisionVariables]) -> RealDecisionVariablesUpdateTimes:
	"""Generate the real decision variables times for each integer decision variables element in the list.

	Args:
		real_dec_vars_update_period (RealDecisionVariablesUpdatePeriod): Real decision variables update period
		int_dec_vars_list (list[IntegerDecisionVariables]): List of integer decision variables

	Returns:
		RealDecisionVariablesUpdateTimes: Real decision variables times
  
	Example:
		real_dec_vars_times = generate_real_dec_vars_times(
			real_dec_vars_update_period=RealDecisionVariablesUpdatePeriod(),
			int_dec_vars_list=int_dec_vars
		)

		print(f"{int_dec_vars[0].sfts_mode=}")
		print(f"{int_dec_vars[0].med_mode=}")
		print(f"{real_dec_vars_times[0].qsf=}")
		print(f"{real_dec_vars_times[0].qmed_s=}")
	"""
 
	real_dec_var_times = []
	for int_dec_vars in int_dec_vars_list: # For each operation plan
		temp_dict = {}
		for real_dec_var_id, update_period in asdict(real_dec_vars_update_period).items():
			# The integer decision vars are used to mark the start and end of the datetimes
			aux_int_dec_var_id = RealLogicalDecVarDependence[real_dec_var_id].value
			aux_int_dec_var_series = getattr(int_dec_vars, aux_int_dec_var_id)
			
			# Generate the index for the real dec var series
			unique_days = aux_int_dec_var_series.index.normalize().unique()
			# Generate separate np.arange arrays for each day and concatenate
			daily_ranges = []

			for day in unique_days:
				# Get the range for the current day
				day_start = day
				day_end = day + pd.Timedelta(days=1)  # End of the day
				
				# Filter data for the current day
				daily_data = aux_int_dec_var_series[
					(aux_int_dec_var_series.index >= day_start) & 
					(aux_int_dec_var_series.index < day_end)
				]
				# Generate np.arange for the day's data 
				# From when the system is activated
				# Until it is deactivated
				start_idx = np.argmax(daily_data != 0)
				end_idx = len(daily_data) - np.argmax(daily_data[::-1] != 0) # - 1
				# print(f"{start_idx=}, {end_idx=}")
				# daily_range = np.arange(
				#     start=daily_data.index[ start_idx ],
				#     stop=daily_data.index[ end_idx if end_idx < len(daily_data)-1 else len(daily_data)-1 ],
				#     step=np.timedelta64(update_period, 's')
				# )
				# Generate the timezone-aware datetime index
				daily_range = pd.date_range(
					start=daily_data.index[start_idx],
					end=daily_data.index[min(end_idx, len(daily_data)-1)],  # Ensure valid index
					freq=pd.Timedelta(seconds=update_period),
					tz=daily_data.index.tz
				)
				daily_ranges.append(daily_range)

			temp_dict[real_dec_var_id] = np.concatenate(daily_ranges)
			# temp_dict[real_dec_var_id] = np.arange(start=aux_int_dec_var_series.index[0], 
			#                                        stop=aux_int_dec_var_series.index[-1], 
			#                                        step=np.timedelta64(update_period, 's'))
		real_dec_var_times.append(RealDecisionVariablesUpdateTimes(**temp_dict))

	return real_dec_var_times

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

# class NlpProblem
class Problem:
	sample_time_mod: int # Model sample time
	sample_time_ts: int  # Thermal storage sample time
	env_vars: EnvironmentVariables # Environment variables
	initial_values: InitialDecVarsValues
	int_dec_vars: IntegerDecisionVariables
	dec_var_updates: DecisionVariablesUpdates # Decision variables updates
	size_dec_vector: int # Size of the decision vector
	real_dec_vars_box_bounds: RealDecVarsBoxBounds
	real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod
	real_dec_vars_times: RealDecisionVariablesUpdateTimes
	episode_range: tuple[datetime, datetime] # Episode range
	initial_state: SolarMedState # System initial state
	model_dict: dict # SolarMED model dumped instance
	box_bounds_lower: list[np.ndarray[float]] # Lower bounds for the decision variables (in list of arrays format).
	box_bounds_upper: list[np.ndarray[float]] # Upper bounds for the decision variables (in list of arrays format).
	x_evaluated: list[list[float]] # Decision variables vector evaluated (i.e. sent to the fitness function)
	fitness_history: list[float] # Fitness record of decision variables sent to the fitness function
    
	def __init__(self, int_dec_vars: IntegerDecisionVariables, 
              	 initial_dec_vars_values: InitialDecVarsValues, 
                 env_vars: EnvironmentVariables, 
                 real_dec_vars_update_period: RealDecisionVariablesUpdatePeriod,
                 model: SolarMED,
                 sample_time_ts: int
                ) -> None:
     
		int_dec_vars = IntegerDecisionVariables(**asdict(int_dec_vars)) # Avoid modifying the original instance

		self.sample_time_mod = model.sample_time
		self.sample_time_ts = sample_time_ts
		self.env_vars = env_vars.resample(f"{self.sample_time_mod}s", origin="start")
		self.initial_values = initial_dec_vars_values
		self.real_dec_vars_update_period = real_dec_vars_update_period

		self.episode_range = env_vars.I.index[0], env_vars.I.index[-1]
		self.dec_var_ids, self.dec_var_dtypes = zip(*[(field.name, get_args(field.type)[0]) for field in fields(DecisionVariables)])
		self.dec_var_int_ids: list[str] = [var_id for var_id, var_type in zip(self.dec_var_ids, self.dec_var_dtypes) if var_type in [bool, int]]
		self.dec_var_real_ids: list[str] = [var_id for var_id, var_type in zip(self.dec_var_ids, self.dec_var_dtypes) if var_type is float]
        
        # Store the model instance
		self.model_dict = model.dump_instance()
		# Add fields not included in the dump_instance method
		self.model_dict.update(dict(
			# Initial states
			## FSM states
			fsms_internal_states=model.fsms_internal_states,
				med_state=model.med_state,
				sf_ts_state=model.sf_ts_state,
		))
		self.initial_state: SolarMedState = model.get_state()
			
		# Initialize real decision variables times
		self.real_dec_vars_times = generate_real_dec_vars_times(
			real_dec_vars_update_period=real_dec_vars_update_period,
			int_dec_vars_list=[int_dec_vars]
		)[0]
		self.size_dec_vector = sum([len(times) for times in asdict(self.real_dec_vars_times).values()])
		self.dec_var_updates = DecisionVariablesUpdates(
			**{name: values.size for name, values in asdict(int_dec_vars).items()},
			**{name: len(values) for name, values in asdict(self.real_dec_vars_times).items()},
        )
  
  		# Get real variables limits
		self.real_dec_vars_box_bounds = RealDecVarsBoxBounds.initialize(
			fmp=model.fixed_model_params, 
			Tmed_c_in=env_vars.Tmed_c_in.mean()
		)
		self.box_bounds_lower, self.box_bounds_upper = generate_bounds(self, int_dec_vars)

		# Initialize decision vector history
		self.x_evaluated = []
		self.fitness_history = []

		# Setup integer decision variables
		for int_dec_var_id, int_dec_var_values in asdict(int_dec_vars).items():
			# from env_vars.index[0] to int_dec_var.index[0] -> some initial provided value
			int_dec_var_values = pd.concat([
        			pd.Series(index=[self.episode_range[0]], 
                     		  data=[getattr(initial_dec_vars_values, int_dec_var_id)]), 
           			int_dec_var_values
              	])
			# From int_dec_var.index[-1] to env_vars.index[-1] the last value is kept
			# Actually, we don't keep simulating after the last integer decision variable value is set to zero
   			# setattr(
			# 	int_dec_vars, 
			# 	int_dec_var_id, 
			# 	pd.concat([
           	# 		int_dec_var_values.iloc[-1],
        	# 		pd.Series(index=[int_dec_var_values.index[-1]], data=getattr(initial_dec_vars_values, int_dec_var_id)), 
            #   	])
			# )
			# Resample integer decision variables to model sample time. Do it here to avoid doing it every time in the fitness function
			setattr(
				int_dec_vars, 
				int_dec_var_id, 
				int_dec_var_values.resample(f"{self.sample_time_mod}s").ffill()
			)
		self.int_dec_vars = int_dec_vars

	def get_bounds(self, ) -> tuple[np.ndarray, np.ndarray]:
		return np.concatenate(self.box_bounds_lower), np.concatenate(self.box_bounds_upper)

	def __post_init__(self, ) -> None:
		logger.info(f"""{self.get_name()} initialized.
					{self.get_extra_info()}""")

	def get_name(self) -> str:
		""" Get problem’s name """
		return "SolarMED NLP problem"

	def get_extra_info(self) -> str:
		""" Get problem’s extra info. """
		
		return f"""
	-\t Size of decision vector: {self.size_dec_vector} elements
	-\t Decision variable ids: {self.dec_var_ids}
	-\t Decision variable types: {self.dec_var_dtypes}
	-\t Model sample time: {self.sample_time_mod} seconds
	-\t System initial state: {self.initial_state.name}
	-\t Lower bounds: {self.box_bounds_lower}
	-\t Upper bounds: {self.box_bounds_upper}"""
 
	def get_nic(self) -> int:
		""" Get number of inequality constraints """
		return 0

	def get_nix(self) -> int:
		""" Get integer dimension """
		return 0

	def decision_vector_to_decision_variables(self, x: np.ndarray[float], resample: bool = True) -> DecisionVariables:
		""" Convert decision vector to decision variables """
  
		decision_dict: dict[str, np.ndarray] = {}
		cnt = 0
		for var_id in self.dec_var_real_ids:
			num_updates = getattr(self.dec_var_updates, var_id)
			dec_var_times = getattr(self.real_dec_vars_times, var_id)
			# from env_vars.index[0] to real_dec_var_time.index[0] -> some initial provided value
			data = np.insert(x[cnt:cnt+num_updates], 0, getattr(self.initial_values, var_id))
			index = np.insert(dec_var_times, 0, self.episode_range[0])
			decision_dict[var_id] = pd.Series(data=data, index=index)
			if resample:
				decision_dict[var_id] = decision_dict[var_id].resample(f"{self.sample_time_mod}s", origin="start").ffill()
			cnt += num_updates
			
		return DecisionVariables(
			**decision_dict,
			**asdict(self.int_dec_vars)
		)
		
	def fitness(self, x: np.ndarray[float],  store_x: bool = True, debug_mode: bool = False) -> list[float]:
            
		# Model initialization logic after external evaluation of thermal storage
		model: SolarMED = SolarMED(**self.model_dict)
		operation_end0: datetime = self.episode_range[0]
		fitness_total = 0
  
		# Sanitize decision vector, sometimes float values are negative even though they are basically zero (float precision?)
		x[np.abs(x) < 1e-6] = 0

		# Convert complete decision (whole episode) vector to decision variables with model sample time
		dec_vars_episode: DecisionVariables = self.decision_vector_to_decision_variables(x=x)

		n_days = (self.episode_range[-1] - self.episode_range[0]).days
		for day_idx in range(n_days+1):
		# day_idx = 0
			day = self.episode_range[0].day + day_idx    
			# Compute operation start and end datetimes for the current day
			daily_data = []
			for var_id in self.dec_var_int_ids:
				var_values = getattr(dec_vars_episode, var_id)
				daily_data.append(
					var_values[(var_values.index.day >= day) & (var_values.index.day < day+1)]
				)
			operation_start, operation_end = get_start_and_end_datetimes(series=daily_data)

			if debug_mode:
				logger.info(f"{day_idx=} | {operation_start=}, {operation_end=}")
   
			# Evaluate thermal storage until start of operation
			if debug_mode:
				logger.info(f"{day_idx=} | Before idle evaluation: {model.Tts_h=}, {model.Tts_c=}")
			Tts_h, Tts_c = evaluate_idle_thermal_storage( 
				sample_time=self.sample_time_ts,
				Tamb=self.env_vars.Tamb,
				dt_span=(operation_end0, operation_start),
				model=model
			)
			if debug_mode:
				logger.info(f"{day_idx=} | After idle evaluation: {Tts_h=}, {Tts_c=}")
			model.Tts_h = Tts_h
			model.Tts_c = Tts_c
  
			# Evaluate fitness for the current day
			dec_vars = dec_vars_episode.dump_in_span(span=(operation_start, operation_end), return_format= "values")
			# The model instance is updated within the evaluate_model function
			fitness, _ = evaluate_model(model = model, 
										n_evals_mod = len(dec_vars.qmed_f),
										mode = "optimization",
										dec_vars = dec_vars, 
										env_vars = self.env_vars)
			fitness_total += fitness
			operation_end0 = operation_end
			
			if debug_mode:
				logger.info(f"{day_idx=} | {fitness=}, {fitness_total=}")
    
		# Store decision vector and fitness value
		if store_x:
			self.x_evaluated.append(x.tolist())
			self.fitness_history.append(fitness_total)
  
		print(f"Evaluation {len(self.fitness_history)} | {fitness_total=}")
  
		return [fitness_total]

	def gradient(self, x):
		return pg.estimate_gradient(lambda x: self.fitness(x), x)


def generate_bounds(problem_instance: Problem, int_dec_vars: IntegerDecisionVariables) -> tuple[np.ndarray, np.ndarray]:
		# Lower box-bounds
		lb: list[np.ndarray[float]] = [
			np.full((len(timestamps), ), np.nan, dtype=float) for timestamps in asdict(problem_instance.real_dec_vars_times).values()
		]
		# Upper box-bounds
		ub: list[np.ndarray[float]] = lb.copy()
		
		for var_idx, var_id in enumerate(problem_instance.dec_var_real_ids):
			lower_limit = getattr(problem_instance.real_dec_vars_box_bounds, var_id)[0],
			upper_limit = getattr(problem_instance.real_dec_vars_box_bounds, var_id)[1],
			update_period = getattr(problem_instance.real_dec_vars_update_period, var_id) # seconds
			
			# Get integer decision variable values
			int_var_id = RealLogicalDecVarDependence[var_id].value
   			# Either zeros or ones, otherwise some logic should be included to translate the values
			int_var_values: pd.Series = getattr(int_dec_vars, int_var_id)
			# Resample integer decision variable to real decision variable sample time
			aux_logical_var_values = (
       			int_var_values
          		.resample(f"{update_period}s", origin=int_var_values.index[0])
            	.ffill()
             	# .bfill()
             	.values
			)

			lb[var_idx] = lower_limit * aux_logical_var_values
			ub[var_idx] = upper_limit * aux_logical_var_values
   
		return lb, ub